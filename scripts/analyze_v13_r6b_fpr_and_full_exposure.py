"""63절 R6+B v13 결과에서 나온 두 가지(비PERSON 오탐 8.33%, 완전노출 14%/28건)를 토큰 단위로
진단한다 — 읽기 전용, R7 재학습 전에 먼저 수행하는 분석. `ner.py`/`merge_ner`/모델/DB/설정
미수정(실험 B는 이 스크립트 안에서만 디코딩을 바꾼 사본), v13 미수정, 재학습 없음, 외부 LLM
미호출.

1. 비PERSON 오탐: v13의 60개 부정 문장(nonperson 40 + background 20) 전부에 R6+B를 돌려 실제로
   무엇이 PERSON으로 잘못 잡혔는지(문장, 예측 span, 토큰별 태그·확률)를 전부 기록한다.
2. 완전노출 28건: 60절의 "완전노출 원인 4분류" 방식 그대로, 28건 각각의 정답 PERSON 구간에 겹치는
   원시 토큰 예측을 전부 뽑아 "후보 미생성"이 정말 전체 "O"인지, 다른 원인과 섞여 있는지 확인한다.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from app.shared.config import settings  # noqa: E402
from app.shared.privacy import detectors, ner  # noqa: E402
from app.shared.privacy.types import Kind  # noqa: E402

import experiment_b_relaxed_merge as eb  # noqa: E402

R6_DIR = r"C:\Users\<user>\venvs\ner-runs\export\r6-candidate-s42"
EXPECT_FP12 = "c59b174ba126"
V13 = ROOT / "app" / "shared" / "privacy" / "eval" / "style_v13_locked" / "v13_locked.jsonl"


def load_rows():
    return [json.loads(l) for l in V13.read_text(encoding="utf-8").splitlines() if l.strip()]


def gold_span(row):
    for sp in row.get("spans", []):
        if sp.get("kind") in ("PERSON", "PERSON_ALIAS"):
            return sp["start"], sp["end"]
    return None


def run_pipeline(text, insp):
    toks = insp.token_predictions(text)
    relaxed = eb.decode_relaxed(toks)
    ner_spans = eb.to_nerspans(relaxed, text)
    rule = detectors.detect(text, ())
    final = ner.merge_ner(text, rule, ner_spans)
    return toks, ner_spans, final


def main():
    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = R6_DIR
    settings.privacy_ner_threads = 2
    status = ner.ner_status()
    assert status["fingerprint"][:12] == EXPECT_FP12, status["fingerprint"][:12]
    insp = eb.TokenInspector(R6_DIR)

    rows = load_rows()

    # ---------- 1. 비PERSON 오탐 60건 전수 ----------
    neg_rows = [r for r in rows if gold_span(r) is None]
    assert len(neg_rows) == 60, len(neg_rows)
    fpr_cases = []
    for r in neg_rows:
        text = r["text"]
        toks, ner_spans, final = run_pipeline(text, insp)
        final_person = [(s.start, s.end, text[s.start:s.end]) for s in final if s.kind is Kind.PERSON]
        if final_person:
            fpr_cases.append({
                "id": r["id"], "category": r.get("category"), "style": r.get("style"),
                "text": text,
                "flagged_spans": final_person,
                "raw_ner_person_family": [(s.label, s.start, s.end, round(s.score, 3), text[s.start:s.end]) for s in ner_spans if s.label in ("PERSON", "PERSON_ALIAS")],
                "tokens": [(cs, ce, tag, round(p, 3), text[cs:ce]) for cs, ce, tag, p in toks],
            })

    # ---------- 2. 완전노출 28건 토큰 단위 ----------
    full_exposure_ids = {
        "v13-0001", "v13-0003", "v13-0010", "v13-0015", "v13-0023", "v13-0038", "v13-0041",
        "v13-0043", "v13-0046", "v13-0050", "v13-0055", "v13-0081", "v13-0090", "v13-0098",
        "v13-0101", "v13-0103", "v13-0109", "v13-0110", "v13-0115", "v13-0138", "v13-0158",
        "v13-0181", "v13-0183", "v13-0185", "v13-0189", "v13-0190", "v13-0195", "v13-0198",
    }
    by_id = {r["id"]: r for r in rows}
    full_cases = []
    for rid in sorted(full_exposure_ids):
        r = by_id[rid]
        text = r["text"]
        g = gold_span(r)
        toks, ner_spans, final = run_pipeline(text, insp)
        overlapping_tokens = [(cs, ce, tag, round(p, 3), text[cs:ce]) for cs, ce, tag, p in toks if max(cs, g[0]) < min(ce, g[1])]
        overlapping_raw_spans = [(s.label, s.start, s.end, round(s.score, 3), text[s.start:s.end]) for s in ner_spans if max(s.start, g[0]) < min(s.end, g[1])]
        full_cases.append({
            "id": rid, "word": r.get("word"), "category": r.get("category"), "style": r.get("style"),
            "text": text, "gold": (g[0], g[1], text[g[0]:g[1]]),
            "overlapping_tokens": overlapping_tokens,
            "overlapping_raw_spans": overlapping_raw_spans,
        })

    out = {"fpr_cases": fpr_cases, "full_exposure_cases": full_cases, "neg_total": 60, "fpr_count": len(fpr_cases)}
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "v13_r6b_fpr_and_full_exposure.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"fpr_count={len(fpr_cases)} / neg_total=60")
    print(f"full_exposure analyzed: {len(full_cases)}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
