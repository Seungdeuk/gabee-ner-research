"""의미적 중의성 진단셋(180문장) 평가 — 읽기 전용, 서비스 모델 vs R1 시드7.

ambiguity_probe.jsonl(180문장, build_ambiguity_probe.py 로 생성)을 쓴다. `known_names` 는 전부 비워
모델 자체의 문맥 구분 능력만 본다(조건 7). `ner.py`·모델 파일·DB·설정 미수정, 외부 LLM 미호출.

측정 두 층(조건 5):
- **원시 예측**: `ner.spans_for`(merge_ner 이전 1차 후보, 라벨·점수 그대로)
- **최종 가명처리 경로**: `ner.detect_with_ner`(서비스가 실제 쓰는 함수, 시범 후처리 미적용)

지표: PERSON Recall(원시/최종, 그룹별), 비PERSON 오탐률(최종 경로 기준), PERSON 문맥에서 틀렸을 때
ORG/LOC 오분류가 원인인 비율, 이름 전체 노출률(leaked=n).

성씨 효과는 별도 서브셋(단어마다 "김"+단어, PERSON 템플릿1)으로 분리 측정한다(조건 8) — 메인 180문장
결과와 섞지 않는다.

사용: ``python scripts/analyze_ambiguity_probe.py <서비스 모델 폴더> <R1 모델 폴더> [--out 결과.json]``
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
KIND_BY_LABEL = {"PERSON": Kind.PERSON, "PERSON_ALIAS": Kind.PERSON}


def has_batchim(ch: str) -> bool:
    return (ord(ch) - 0xAC00) % 28 != 0


def josa(word: str) -> str:
    return word + ("이" if has_batchim(word[-1]) else "가")


def load_rows() -> list[dict]:
    return [json.loads(line) for line in PROBE.read_text(encoding="utf-8").splitlines() if line.strip()]


def surname_rows(rows: list[dict]) -> list[dict]:
    out = []
    seen = set()
    for r in rows:
        if r["context_type"] != "PERSON" or r["word"] in seen:
            continue
        seen.add(r["word"])
        full = "김" + r["word"]
        text = f"{josa(full)} 오늘 피아노 학원 다녀왔대"
        s = text.find(full)
        out.append(
            {
                "id": f"amb-{r['word']}-surname", "word": r["word"], "group": r["group"],
                "context_type": "PERSON", "gold_label": "PERSON", "text": text,
                "spans": [{"kind": "PERSON", "start": s, "end": s + len(full)}], "known_names": [],
            }
        )
    return out


def final_person_spans(preds: list) -> list:
    return [s for s in preds if s.kind is Kind.PERSON]


def raw_label_at(raw_spans: list, g: dict) -> tuple[str, float] | None:
    overlapping = [s for s in raw_spans if max(s.start, g["start"]) < min(s.end, g["end"])]
    if not overlapping:
        return None
    best = max(overlapping, key=lambda s: s.score)
    return best.label, best.score


def covered(g: dict, preds: list) -> set[int]:
    cov: set[int] = set()
    for s in preds:
        if s.kind is Kind.PERSON:
            cov.update(range(max(s.start, g["start"]), min(s.end, g["end"])))
    return cov


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


def run_model(model_key: str, model_dir: str, rows: list[dict], surname_rows_: list[dict]) -> dict:
    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = model_dir
    settings.privacy_ner_threads = 2
    status = ner.ner_status()
    fp12 = status["fingerprint"][:12]
    if EXPECT_FP12.get(model_key) and fp12 != EXPECT_FP12[model_key]:
        raise SystemExit(f"[{model_key}] 지문 불일치: got {fp12}")
    if not status["active"]:
        raise SystemExit(f"[{model_key}] NER 엔진 비활성")
    ner._cache.clear()

    person_rows = [r for r in rows if r["context_type"] == "PERSON"]
    nonperson_rows = [r for r in rows if r["context_type"] == "NONPERSON"]

    by_group_recall_raw: dict[str, Counter] = defaultdict(Counter)
    by_group_recall_final: dict[str, Counter] = defaultdict(Counter)
    cause_counts: dict[str, Counter] = defaultdict(Counter)
    full_exposure_examples = []
    raw_examples = []

    for r in person_rows:
        g = r["spans"][0]
        n = g["end"] - g["start"]
        text = r["text"]
        raw = ner.spans_for(text) or []
        final = ner.detect_with_ner(text, ())
        raw_hit = raw_label_at(raw, g)
        raw_correct = raw_hit is not None and raw_hit[0] in ("PERSON", "PERSON_ALIAS") and any(
            s.start == g["start"] and s.end == g["end"] and s.label in ("PERSON", "PERSON_ALIAS") for s in raw
        )
        final_person = final_person_spans(final)
        final_exact = any(s.start == g["start"] and s.end == g["end"] for s in final_person)
        leaked = n - len(covered(g, final))

        by_group_recall_raw[r["group"]]["total"] += 1
        by_group_recall_raw[r["group"]]["correct"] += int(raw_correct)
        by_group_recall_final[r["group"]]["total"] += 1
        by_group_recall_final[r["group"]]["correct"] += int(final_exact)
        if leaked >= n:
            by_group_recall_final[r["group"]]["완전노출"] += 1

        if not final_exact:
            cause = diagnose_cause(raw, g)
            cause_counts[r["group"]][cause] += 1
            if leaked >= n and len(full_exposure_examples) < 20:
                full_exposure_examples.append(
                    {
                        "id": r["id"], "word": r["word"], "group": r["group"], "text": text,
                        "raw_prediction": raw_hit, "final_masked": _mask(text, final_person), "cause": cause,
                    }
                )
        if len(raw_examples) < 200:
            raw_examples.append({"id": r["id"], "word": r["word"], "text": text, "raw": raw_hit, "final_exact": final_exact})

    neg_by_group: dict[str, dict] = defaultdict(lambda: {"total": 0, "flagged": 0})
    org_loc_label_match: dict[str, dict] = defaultdict(lambda: {"total": 0, "label_correct": 0})
    for r in nonperson_rows:
        text = r["text"]
        final = ner.detect_with_ner(text, ())
        final_person = final_person_spans(final)
        flagged = bool(final_person)
        neg_by_group[r["group"]]["total"] += 1
        neg_by_group[r["group"]]["flagged"] += int(flagged)
        if r["gold_label"] in ("ORGANIZATION", "LOCATION"):
            raw = ner.spans_for(text) or []
            g = r["spans"][0]
            hit = raw_label_at(raw, g)
            org_loc_label_match[r["word"]]["total"] += 1
            org_loc_label_match[r["word"]]["label_correct"] += int(hit is not None and hit[0] == r["gold_label"])

    # ---- 성씨 서브셋(별도) ----
    surname_result = {"total": len(surname_rows_), "exact": 0, "details": []}
    for r in surname_rows_:
        g = r["spans"][0]
        final = ner.detect_with_ner(r["text"], ())
        exact = any(s.start == g["start"] and s.end == g["end"] for s in final_person_spans(final))
        surname_result["exact"] += int(exact)
        surname_result["details"].append({"word": r["word"], "text": r["text"], "exact": exact})

    def recall_rate(c: Counter) -> float:
        return round(c["correct"] / c["total"], 4) if c["total"] else 0.0

    return {
        "fingerprint12": fp12,
        "person_recall_raw_by_group": {k: {"total": v["total"], "correct": v["correct"], "rate": recall_rate(v)} for k, v in by_group_recall_raw.items()},
        "person_recall_final_by_group": {
            k: {"total": v["total"], "correct": v["correct"], "rate": recall_rate(v), "완전노출": v["완전노출"], "완전노출률": round(v["완전노출"] / v["total"], 4)}
            for k, v in by_group_recall_final.items()
        },
        "nonperson_fpr_by_group": {k: {**v, "fpr": round(v["flagged"] / v["total"], 4) if v["total"] else 0.0} for k, v in neg_by_group.items()},
        "failure_cause_by_group": {k: dict(v) for k, v in cause_counts.items()},
        "org_loc_label_match": dict(org_loc_label_match),
        "full_exposure_examples": full_exposure_examples,
        "surname_subprobe": surname_result,
    }


def _mask(text: str, preds: list) -> str:
    out = list(text)
    for s in preds:
        for i in range(max(0, s.start), min(len(out), s.end)):
            out[i] = "●"
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("service_model_dir")
    ap.add_argument("r1_model_dir")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = load_rows()
    sub = surname_rows(rows)
    print(f"진단셋 {len(rows)}문장(PERSON 90, 비PERSON 90) + 성씨 서브셋 {len(sub)}문장")

    out = {
        "service": run_model("service", args.service_model_dir, rows, sub),
        "r1_s7": run_model("r1_s7", args.r1_model_dir, rows, sub),
    }
    for key, r in out.items():
        print(f"\n== {key} ==")
        print("  최종경로 PERSON Recall(그룹별):", r["person_recall_final_by_group"])
        print("  비PERSON 오탐률(그룹별):", r["nonperson_fpr_by_group"])
        print("  실패 원인(그룹별):", r["failure_cause_by_group"])
        print(f"  성씨 서브셋 exact: {r['surname_subprobe']['exact']}/{r['surname_subprobe']['total']}")

    out_path = args.out or (ROOT / "app" / "shared" / "privacy" / "eval" / "style_ambiguity_probe" / "results.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
