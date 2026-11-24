"""R4 v10 완전노출 사후 오류 분석 + R3→R4 역행 2건(48절 오탐·v8 완전노출) 원인 분석 — 읽기 전용.

v10은 이미 소비된 최종 테스트이므로 이 스크립트는 **설명만** 한다(재튜닝·재학습 없음).
ner.py/detectors.py/모델/DB/설정 미수정, 외부 LLM 미호출, v10 데이터 수정 없음.

측정:
A) v10 PERSON 문장 전체에 대해 서비스/R3/R4의 원시 예측(`ner.spans_for`)과 최종 결과
   (`ner.detect_with_ner`)를 기록하고, R4 완전노출 27건을 원인 4종으로 분류 + 문형·단어유형 교차표.
B) 48절(ambiguity_probe) 비PERSON 문장에서 R3→R4로 새로 flag된 사례의 원시 점수 비교.
C) v8_independent PERSON 문장에서 R3→R4로 완전노출이 늘어난 사례의 원시 점수 비교.
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

EVAL_ROOT = ROOT / "app" / "shared" / "privacy" / "eval"
V10 = EVAL_ROOT / "style_v10_locked" / "v10_locked.jsonl"
PROBE48 = EVAL_ROOT / "style_ambiguity_probe" / "ambiguity_probe.jsonl"
V8 = EVAL_ROOT / "style_v8_independent" / "v8_independent.jsonl"
EXPECT_FP12 = {"service": "46210943c45f", "r3_candidate": "14f297c7d5f5", "r4_candidate": "56249473d8d9"}


def load_rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def gold_span(row: dict) -> tuple[int, int] | None:
    for sp in row.get("spans", []):
        if sp.get("kind") in ("PERSON", "PERSON_ALIAS"):
            return sp["start"], sp["end"]
    return None


def raw_hit(raw_spans: list, g: tuple[int, int]) -> tuple[str, float] | None:
    overlapping = [s for s in raw_spans if max(s.start, g[0]) < min(s.end, g[1])]
    if not overlapping:
        return None
    best = max(overlapping, key=lambda s: s.score)
    return best.label, round(best.score, 3)


def diagnose_cause(raw_spans: list, g: tuple[int, int]) -> str:
    overlapping = [s for s in raw_spans if max(s.start, g[0]) < min(s.end, g[1])]
    person_overlap = [s for s in overlapping if s.label in ("PERSON", "PERSON_ALIAS")]
    if not overlapping:
        return "후보 미생성"
    if not person_overlap:
        return f"ORG/LOC 오분류({'/'.join(sorted({s.label for s in overlapping}))})"
    if all(s.score < ner.NER_MIN_SCORE for s in person_overlap):
        return "신뢰도 부족"
    return "후처리 제거(merge_ner)"


def covered(preds: list, g: tuple[int, int]) -> set[int]:
    cov: set[int] = set()
    for s in preds:
        if s.kind is Kind.PERSON:
            cov.update(range(max(s.start, g[0]), min(s.end, g[1])))
    return cov


def set_model(model_key: str, model_dir: str) -> str:
    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = model_dir
    settings.privacy_ner_threads = 2
    status = ner.ner_status()
    fp12 = status["fingerprint"][:12]
    if EXPECT_FP12.get(model_key) and fp12 != EXPECT_FP12[model_key]:
        raise SystemExit(f"[{model_key}] 지문 불일치: got {fp12}, expected {EXPECT_FP12.get(model_key)}")
    ner._cache.clear()
    return fp12


def eval_v10_detail(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        text = r["text"]
        raw = ner.spans_for(text) or []
        final = ner.detect_with_ner(text, ())
        g = gold_span(r)
        entry = {"id": r["id"], "category": r.get("category"), "word_type": r.get("word_type"), "word": r.get("word"), "text": text}
        if g is not None:
            n = g[1] - g[0]
            leaked = n - len(covered(final, g))
            exact = any(s.start == g[0] and s.end == g[1] and s.kind is Kind.PERSON for s in final)
            entry.update({
                "is_positive": True, "gold": text[g[0]:g[1]], "n": n, "leaked": leaked, "exact": exact,
                "full_exposure": leaked >= n, "raw_hit": raw_hit(raw, g),
                "cause_if_not_exact": None if exact else diagnose_cause(raw, g),
                "final_person_spans": [(s.start, s.end, round(s.score, 3)) for s in final if s.kind is Kind.PERSON],
            })
        else:
            flagged = any(s.kind is Kind.PERSON for s in final)
            entry.update({
                "is_positive": False, "flagged": flagged,
                "raw_person_candidates": [(s.start, s.end, s.label, round(s.score, 3)) for s in raw if s.label in ("PERSON", "PERSON_ALIAS")],
            })
        out[r["id"]] = entry
    return out


def part_a(a: argparse.Namespace) -> dict:
    rows = load_rows(V10)
    results = {}
    for key, model_dir in (("service", a.service), ("r3_candidate", a.r3_candidate), ("r4_candidate", a.r4_candidate)):
        set_model(key, model_dir)
        results[key] = eval_v10_detail(rows)

    ids_pos = [r["id"] for r in rows if gold_span(r) is not None]
    r4 = results["r4_candidate"]

    full_cases = [r4[rid] for rid in ids_pos if r4[rid]["full_exposure"]]
    cause_counts = Counter(c["cause_if_not_exact"] for c in full_cases)
    by_category = defaultdict(lambda: Counter())
    by_word_type = defaultdict(lambda: Counter())
    for c in full_cases:
        by_category[c["category"]][c["cause_if_not_exact"]] += 1
        by_word_type[c["word_type"]][c["cause_if_not_exact"]] += 1

    # 문형 x 단어유형 교차 완전노출 건수(전체 모수 대비)
    cross_cat_type = defaultdict(lambda: Counter())
    for rid in ids_pos:
        e = r4[rid]
        cross_cat_type[(e["category"], e["word_type"])]["total"] += 1
        cross_cat_type[(e["category"], e["word_type"])]["full_exposure"] += int(e["full_exposure"])

    # 호격 1건 역행: R3는 full_exposure 전무, R4에서 생긴 것 찾기
    vocative_regression = []
    for rid in ids_pos:
        s3, s4 = results["r3_candidate"][rid], r4[rid]
        if s3["category"] == "호격" and not s3["full_exposure"] and s4["full_exposure"]:
            vocative_regression.append({"id": rid, "r3": s3, "r4": s4})

    # 서비스 대비 R3 대비 R4 전체 교차(완전노출만)
    cross3 = Counter()
    for rid in ids_pos:
        key = (results["service"][rid]["full_exposure"], results["r3_candidate"][rid]["full_exposure"], r4[rid]["full_exposure"])
        cross3[key] += 1

    return {
        "full_exposure_cases": full_cases,
        "cause_counts": dict(cause_counts),
        "cause_by_category": {k: dict(v) for k, v in by_category.items()},
        "cause_by_word_type": {k: dict(v) for k, v in by_word_type.items()},
        "cross_category_x_wordtype_full_exposure": {f"{k[0]}/{k[1]}": dict(v) for k, v in cross_cat_type.items()},
        "vocative_regression_cases": vocative_regression,
        "three_way_full_exposure_cross": {str(k): v for k, v in cross3.items()},
    }


def part_b(a: argparse.Namespace) -> dict:
    rows = load_rows(PROBE48)
    neg_rows = [r for r in rows if gold_span(r) is None]
    results = {}
    for key, model_dir in (("r3_candidate", a.r3_candidate), ("r4_candidate", a.r4_candidate)):
        set_model(key, model_dir)
        out = {}
        for r in neg_rows:
            text = r["text"]
            raw = ner.spans_for(text) or []
            final = ner.detect_with_ner(text, ())
            flagged = any(s.kind is Kind.PERSON for s in final)
            out[r["id"]] = {
                "word": r.get("word"), "group": r.get("group"), "text": text, "flagged": flagged,
                "raw_person_candidates": [(s.start, s.end, s.label, round(s.score, 3)) for s in raw if s.label in ("PERSON", "PERSON_ALIAS")],
            }
        results[key] = out

    newly_flagged = []
    for rid in results["r3_candidate"]:
        r3e, r4e = results["r3_candidate"][rid], results["r4_candidate"][rid]
        if not r3e["flagged"] and r4e["flagged"]:
            newly_flagged.append({"id": rid, "word": r3e["word"], "group": r3e["group"], "text": r3e["text"],
                                   "r3_raw": r3e["raw_person_candidates"], "r4_raw": r4e["raw_person_candidates"]})
    return {"newly_flagged_by_r4": newly_flagged, "r3_total_flagged": sum(v["flagged"] for v in results["r3_candidate"].values()),
            "r4_total_flagged": sum(v["flagged"] for v in results["r4_candidate"].values())}


def part_c(a: argparse.Namespace) -> dict:
    rows = load_rows(V8)
    pos_rows = [r for r in rows if gold_span(r) is not None]
    results = {}
    for key, model_dir in (("r3_candidate", a.r3_candidate), ("r4_candidate", a.r4_candidate)):
        set_model(key, model_dir)
        out = {}
        for r in pos_rows:
            text = r["text"]
            g = gold_span(r)
            n = g[1] - g[0]
            raw = ner.spans_for(text) or []
            final = ner.detect_with_ner(text, ())
            leaked = n - len(covered(final, g))
            out[r["id"]] = {
                "name": r.get("name"), "category": r.get("category"), "position": r.get("position"), "text": text,
                "gold": text[g[0]:g[1]], "full_exposure": leaked >= n,
                "raw_hit": raw_hit(raw, g), "cause_if_full": diagnose_cause(raw, g) if leaked >= n else None,
            }
        results[key] = out

    newly_full = []
    for rid in results["r3_candidate"]:
        r3e, r4e = results["r3_candidate"][rid], results["r4_candidate"][rid]
        if not r3e["full_exposure"] and r4e["full_exposure"]:
            newly_full.append({"id": rid, **{f"r3_{k}": v for k, v in r3e.items()}, **{f"r4_{k}": v for k, v in r4e.items()}})
    return {"newly_full_exposure_by_r4": newly_full}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--service", required=True)
    ap.add_argument("--r3-candidate", required=True)
    ap.add_argument("--r4-candidate", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    out = {"A_v10_full_exposure": part_a(a), "B_probe48_new_fpr": part_b(a), "C_v8_new_full_exposure": part_c(a)}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    A = out["A_v10_full_exposure"]
    print("=== A: v10 R4 완전노출 원인 ===")
    print(A["cause_counts"])
    print("문형별:", A["cause_by_category"])
    print("단어유형별:", A["cause_by_word_type"])
    print("호격 역행 건수:", len(A["vocative_regression_cases"]))
    print("\n=== B: 48절 신규 오탐(R4) ===")
    print(f"R3 flagged={out['B_probe48_new_fpr']['r3_total_flagged']}, R4 flagged={out['B_probe48_new_fpr']['r4_total_flagged']}, 신규={len(out['B_probe48_new_fpr']['newly_flagged_by_r4'])}")
    print("\n=== C: v8 신규 완전노출(R4) ===")
    print(f"신규 완전노출 {len(out['C_v8_new_full_exposure']['newly_full_exposure_by_r4'])}건")
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
