"""v11 "기본" 문형 오류 사례 — 서비스/R4/R5 비교(읽기 전용, 사후 분석). R6 재학습 전 진단용.

56절에서 "기본" 문형이 R5의 다른 보강 문형(비교서술)보다 덜 좋아졌다는 게 확인돼, 그 원인을
문장 단위로 직접 본다. 각 사례를 50·52·54절과 동일한 4분류(후보 미생성 / ORG,LOC 오분류 /
신뢰도 부족 / 후처리 제거)로 나눈다. ner.py·모델·DB·설정 미수정, v11 데이터 수정 없음, 재학습 없음.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from app.shared.config import settings  # noqa: E402
from app.shared.privacy import ner  # noqa: E402
from app.shared.privacy.types import Kind  # noqa: E402

V11 = ROOT / "app" / "shared" / "privacy" / "eval" / "style_v11_locked" / "v11_locked.jsonl"
EXPECT_FP12 = {"service": "46210943c45f", "r4_candidate": "56249473d8d9", "r5_candidate": "a270ea570419"}


def load_rows():
    rows = [json.loads(l) for l in V11.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("category") == "기본"]


def gold_span(row):
    for sp in row.get("spans", []):
        if sp.get("kind") in ("PERSON", "PERSON_ALIAS"):
            return sp["start"], sp["end"]
    return None


def covered(preds, g):
    cov = set()
    for s in preds:
        if s.kind is Kind.PERSON:
            cov.update(range(max(s.start, g[0]), min(s.end, g[1])))
    return cov


def raw_hit(raw_spans, g):
    overlapping = [s for s in raw_spans if max(s.start, g[0]) < min(s.end, g[1])]
    if not overlapping:
        return None
    best = max(overlapping, key=lambda s: s.score)
    return best.label, round(best.score, 3)


def diagnose_cause(raw_spans, g):
    overlapping = [s for s in raw_spans if max(s.start, g[0]) < min(s.end, g[1])]
    person_overlap = [s for s in overlapping if s.label in ("PERSON", "PERSON_ALIAS")]
    if not overlapping:
        return "후보 미생성"
    if not person_overlap:
        return f"ORG/LOC 오분류({'/'.join(sorted({s.label for s in overlapping}))})"
    if all(s.score < ner.NER_MIN_SCORE for s in person_overlap):
        return "신뢰도 부족"
    return "후처리 제거(merge_ner)"


def run_model(model_key, model_dir, rows):
    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = model_dir
    settings.privacy_ner_threads = 2
    status = ner.ner_status()
    fp12 = status["fingerprint"][:12]
    if EXPECT_FP12.get(model_key) and fp12 != EXPECT_FP12[model_key]:
        raise SystemExit(f"[{model_key}] 지문 불일치: got {fp12}")
    ner._cache.clear()

    detail = {}
    causes = Counter()
    for r in rows:
        text = r["text"]
        g = gold_span(r)
        raw = ner.spans_for(text) or []
        final = ner.detect_with_ner(text, ())
        final_person = [(s.start, s.end) for s in final if s.kind is Kind.PERSON]
        n = g[1] - g[0]
        leaked = n - len(covered(final, g))
        exact = g in final_person
        full = leaked >= n
        cause = None if exact else diagnose_cause(raw, g)
        if not exact:
            causes[cause] += 1
        detail[r["id"]] = {
            "word": r.get("word"), "word_type": r.get("word_type"), "style": r.get("style"), "text": text,
            "exact": exact, "leaked": leaked, "n": n, "full_exposure": full,
            "raw_hit": raw_hit(raw, g), "cause": cause,
            "final_person_spans": [(s.start, s.end, round(s.score, 3)) for s in final if s.kind is Kind.PERSON],
        }
    return {"fingerprint12": fp12, "cause_counts": dict(causes), "detail": detail}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--service", required=True)
    ap.add_argument("--r4-candidate", required=True)
    ap.add_argument("--r5-candidate", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows = load_rows()
    print(f"v11 '기본' 문형 문장 수: {len(rows)}")

    out = {
        "service": run_model("service", a.service, rows),
        "r4_candidate": run_model("r4_candidate", a.r4_candidate, rows),
        "r5_candidate": run_model("r5_candidate", a.r5_candidate, rows),
    }

    # R4->R5 비교: 교차표 + 사례별 변화
    ids = [r["id"] for r in rows]
    r4, r5 = out["r4_candidate"]["detail"], out["r5_candidate"]["detail"]
    cross = Counter()
    improved, regressed, both_fail, both_ok = [], [], [], []
    for rid in ids:
        a4, a5 = r4[rid]["full_exposure"], r5[rid]["full_exposure"]
        cross[(a4, a5)] += 1
        if a4 and not a5:
            improved.append(rid)
        elif not a4 and a5:
            regressed.append(rid)
        elif a4 and a5:
            both_fail.append(rid)
        else:
            both_ok.append(rid)

    out["_r4_vs_r5_comparison"] = {
        "cross_table": {f"r4_full={k[0]}_r5_full={k[1]}": v for k, v in cross.items()},
        "improved_ids": improved, "regressed_ids": regressed, "both_fail_ids": both_fail, "both_ok_ids": both_ok,
    }

    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    for key in ("service", "r4_candidate", "r5_candidate"):
        print(f"{key}: cause_counts={out[key]['cause_counts']}")
    print(f"R4->R5: improved={len(improved)}, regressed={len(regressed)}, both_fail={len(both_fail)}, both_ok={len(both_ok)}")
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
