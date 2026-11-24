"""우선순위 2 · 1단계: 후보 미생성(자연·계절/일상·추상 그룹) 원인 진단 — 읽기 전용.

목적: `style_ambiguity_probe`(180문장, 48절)에서 "후보 없음"이 집중된 자연·계절(13단어)·일상·추상
(13단어) 26개 단어에 대해, (1) PERSON 문맥 템플릿 3개 중 어느 것에서 후보가 안 생기는지, (2) 같은
단어의 비PERSON 문맥에서는 모델이 뭐라고 예측하는지(원시 라벨)를 템플릿·단어 단위로 재구성한다.
results.json(48절)에는 그룹 집계와 상위 20개 예시만 있어 템플릿별 분해가 안 되므로, 같은 진단셋
문장으로 `ner.spans_for`/`ner.detect_with_ner`를 다시 호출해 전수 기록한다.

모델 재학습 없음 · ner.py/detectors.py/모델 파일/DB/설정 전부 미수정 · 외부 LLM 미호출.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from app.shared.config import settings  # noqa: E402
from app.shared.privacy import ner  # noqa: E402
from app.shared.privacy.types import Kind  # noqa: E402

PROBE = ROOT / "app" / "shared" / "privacy" / "eval" / "style_ambiguity_probe" / "ambiguity_probe.jsonl"
EXPECT_FP12 = {"service": "46210943c45f", "r1_s7": "c536eadeb5a1"}
TARGET_GROUPS = {"자연계절", "일상추상"}
TEMPLATE_LABEL = {1: "T1(~이/가 오늘 피아노학원)", 2: "T2(~한테 전화)", 3: "T3(~가 제일 웃겨)"}


def load_rows() -> list[dict]:
    return [json.loads(line) for line in PROBE.read_text(encoding="utf-8").splitlines() if line.strip()]


def raw_label_at(raw_spans: list, g: dict) -> tuple[str, float] | None:
    overlapping = [s for s in raw_spans if max(s.start, g["start"]) < min(s.end, g["end"])]
    if not overlapping:
        return None
    best = max(overlapping, key=lambda s: s.score)
    return best.label, round(best.score, 3)


def diagnose_cause(raw_spans: list, g: dict) -> str:
    overlapping = [s for s in raw_spans if max(s.start, g["start"]) < min(s.end, g["end"])]
    person_overlap = [s for s in overlapping if s.label in ("PERSON", "PERSON_ALIAS")]
    if not overlapping:
        return "후보 없음"
    if not person_overlap:
        return f"오분류({'/'.join(sorted({s.label for s in overlapping}))})"
    if all(s.score < ner.NER_MIN_SCORE for s in person_overlap):
        return "낮은 신뢰도"
    return "후처리 제거(merge_ner)"


def template_index(row_id: str) -> int:
    return int(row_id.rsplit("-", 1)[-1])


def run_model(model_key: str, model_dir: str, rows: list[dict]) -> dict:
    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = model_dir
    settings.privacy_ner_threads = 2
    status = ner.ner_status()
    fp12 = status["fingerprint"][:12]
    if EXPECT_FP12.get(model_key) and fp12 != EXPECT_FP12[model_key]:
        raise SystemExit(f"[{model_key}] 지문 불일치: got {fp12}")
    ner._cache.clear()

    person_rows = [r for r in rows if r["context_type"] == "PERSON" and r["group"] in TARGET_GROUPS]
    nonperson_rows = [r for r in rows if r["context_type"] == "NONPERSON" and r["group"] in TARGET_GROUPS]

    cause_by_template: dict[int, Counter] = defaultdict(Counter)
    per_word_template: dict[str, dict[int, str]] = defaultdict(dict)
    nonperson_raw_by_word: dict[str, list] = defaultdict(list)

    for r in person_rows:
        g = r["spans"][0]
        raw = ner.spans_for(r["text"]) or []
        final = ner.detect_with_ner(r["text"], ())
        final_exact = any(s.start == g["start"] and s.end == g["end"] and s.kind is Kind.PERSON for s in final)
        ti = template_index(r["id"])
        cause = "정확탐지" if final_exact else diagnose_cause(raw, g)
        cause_by_template[ti][cause] += 1
        per_word_template[r["word"]][ti] = cause

    for r in nonperson_rows:
        raw = ner.spans_for(r["text"]) or []
        g = r["spans"][0] if r["spans"] else {"start": 0, "end": len(r["word"])}
        hit = raw_label_at(raw, g) if r["spans"] else (raw_label_at(raw, {"start": r["text"].find(r["word"]), "end": r["text"].find(r["word"]) + len(r["word"])}))
        nonperson_raw_by_word[r["word"]].append(hit)

    words_all_3_fail = sorted(w for w, t in per_word_template.items() if all(t.get(i) == "후보 없음" for i in (1, 2, 3)))
    words_template_sensitive = sorted(
        w for w, t in per_word_template.items()
        if len({t.get(i) for i in (1, 2, 3)}) > 1
    )

    return {
        "fingerprint12": fp12,
        "cause_by_template": {TEMPLATE_LABEL[k]: dict(v) for k, v in cause_by_template.items()},
        "per_word_template_cause": {w: {TEMPLATE_LABEL[i]: c for i, c in t.items()} for w, t in per_word_template.items()},
        "words_all_3_templates_no_candidate": words_all_3_fail,
        "words_template_sensitive": words_template_sensitive,
        "nonperson_raw_prediction_by_word": {w: hits for w, hits in nonperson_raw_by_word.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("service_model_dir")
    ap.add_argument("r1_model_dir")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = load_rows()
    out = {
        "service": run_model("service", args.service_model_dir, rows),
        "r1_s7": run_model("r1_s7", args.r1_model_dir, rows),
    }

    for key, r in out.items():
        print(f"\n== {key} ==")
        print(" 템플릿별 원인:", r["cause_by_template"])
        print(" 3개 템플릿 전부 '후보 없음':", r["words_all_3_templates_no_candidate"])
        print(" 템플릿에 따라 결과가 갈리는 단어:", r["words_template_sensitive"])

    out_path = args.out or (ROOT / "app" / "shared" / "privacy" / "eval" / "style_ambiguity_probe" / "candidate_gap_step1.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
