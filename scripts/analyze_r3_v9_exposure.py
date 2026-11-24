"""R3 vs 기존 서비스 모델 — v9 완전노출 사례 사후 오류 분석(읽기 전용, 사후 분석).

v9는 이미 소비된 최종 테스트라 이 스크립트는 그 결과를 **설명만** 한다(재튜닝·재학습 없음).
ner.py/detectors.py/모델/DB/설정 미수정, 외부 LLM 미호출.

측정: 문장별 원시 예측(`ner.spans_for`, merge_ner 이전)과 최종 가명처리(`ner.detect_with_ner`)를
서비스/R3 두 모델에서 모두 기록하고, 완전노출(leaked>=n) 사례를 원인 4종으로 분류한다(50절과 동일한
cause 분류 로직). 서비스↔R3 교차표(개선/역행/둘다실패/둘다성공)와 단어·문형·조사별 집중도도 본다.
비PERSON 오탐 감소 원인은 A_neg/D_background_neg의 원시 예측 변화(점수 추이·후보 유무)로 본다.
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

V9 = ROOT / "app" / "shared" / "privacy" / "eval" / "style_v9_locked" / "v9_locked.jsonl"
EXPECT_FP12 = {"service": "46210943c45f", "r3_candidate": "14f297c7d5f5"}


def load_rows() -> list[dict]:
    return [json.loads(l) for l in V9.read_text(encoding="utf-8").splitlines() if l.strip()]


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


def run_model(model_key: str, model_dir: str) -> dict:
    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = model_dir
    settings.privacy_ner_threads = 2
    status = ner.ner_status()
    fp12 = status["fingerprint"][:12]
    if EXPECT_FP12.get(model_key) and fp12 != EXPECT_FP12[model_key]:
        raise SystemExit(f"[{model_key}] 지문 불일치: got {fp12}, expected {EXPECT_FP12.get(model_key)}")
    ner._cache.clear()

    rows = load_rows()
    per_sentence = {}
    for r in rows:
        text = r["text"]
        raw = ner.spans_for(text) or []
        final = ner.detect_with_ner(text, ())
        g = gold_span(r)
        entry = {
            "id": r["id"], "group": r.get("group"), "category": r.get("category"), "word": r.get("word"),
            "text": text, "raw_hit": None, "final_person_spans": [(s.start, s.end, round(s.score, 3)) for s in final if s.kind is Kind.PERSON],
        }
        if g is not None:
            n = g[1] - g[0]
            leaked = n - len(covered(final, g))
            exact = any(s.start == g[0] and s.end == g[1] and s.kind is Kind.PERSON for s in final)
            entry.update({
                "is_positive": True, "gold": text[g[0]:g[1]], "gold_span": g, "n": n, "leaked": leaked,
                "exact": exact, "full_exposure": leaked >= n,
                "raw_hit": raw_hit(raw, g),
                "cause_if_not_exact": None if exact else diagnose_cause(raw, g),
            })
        else:
            flagged = any(s.kind is Kind.PERSON for s in final)
            entry.update({"is_positive": False, "flagged": flagged, "raw_person_candidates": [(s.start, s.end, round(s.score, 3)) for s in raw if s.label in ("PERSON", "PERSON_ALIAS")]})
        per_sentence[r["id"]] = entry
    return {"fingerprint12": fp12, "by_id": per_sentence}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--service", required=True)
    ap.add_argument("--r3-candidate", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    svc = run_model("service", a.service)
    r3 = run_model("r3_candidate", a.r3_candidate)

    rows = load_rows()
    ids_pos = [r["id"] for r in rows if gold_span(r) is not None]
    ids_neg = [r["id"] for r in rows if gold_span(r) is None]

    # ---- 완전노출 사례(둘 중 하나라도 full_exposure) ----
    full_exposure_cases = []
    for rid in ids_pos:
        s, r = svc["by_id"][rid], r3["by_id"][rid]
        if s["full_exposure"] or r["full_exposure"]:
            full_exposure_cases.append({"id": rid, "word": s["word"], "category": s["category"], "text": s["text"],
                                         "service": {"full_exposure": s["full_exposure"], "exact": s["exact"], "raw_hit": s["raw_hit"], "cause": s["cause_if_not_exact"], "final_person_spans": s["final_person_spans"]},
                                         "r3": {"full_exposure": r["full_exposure"], "exact": r["exact"], "raw_hit": r["raw_hit"], "cause": r["cause_if_not_exact"], "final_person_spans": r["final_person_spans"]}})

    # ---- 교차표: 서비스 완전노출 여부 x R3 완전노출 여부 ----
    cross = Counter()
    improved, regressed, both_full = [], [], []
    for rid in ids_pos:
        s, r = svc["by_id"][rid], r3["by_id"][rid]
        key = (s["full_exposure"], r["full_exposure"])
        cross[key] += 1
        if s["full_exposure"] and not r["full_exposure"]:
            improved.append(rid)
        elif not s["full_exposure"] and r["full_exposure"]:
            regressed.append(rid)
        elif s["full_exposure"] and r["full_exposure"]:
            both_full.append(rid)

    # ---- R3 완전노출 원인 분류 집계 ----
    r3_cause_counts = Counter(r3["by_id"][rid]["cause_if_not_exact"] for rid in ids_pos if r3["by_id"][rid]["full_exposure"])
    svc_cause_counts = Counter(svc["by_id"][rid]["cause_if_not_exact"] for rid in ids_pos if svc["by_id"][rid]["full_exposure"])

    # ---- 단어별/문형별 R3 완전노출 집중도 ----
    by_word = Counter(r3["by_id"][rid]["word"] for rid in ids_pos if r3["by_id"][rid]["full_exposure"])
    by_category = Counter(r3["by_id"][rid]["category"] for rid in ids_pos if r3["by_id"][rid]["full_exposure"])
    by_word_total = Counter(r3["by_id"][rid]["word"] for rid in ids_pos)
    by_category_total = Counter(r3["by_id"][rid]["category"] for rid in ids_pos)

    # ---- 비PERSON 오탐 변화(A_neg + D_background_neg) ----
    neg_changes = []
    for rid in ids_neg:
        s, r = svc["by_id"][rid], r3["by_id"][rid]
        if s["flagged"] != r["flagged"]:
            neg_changes.append({"id": rid, "word": s["word"], "text": s["text"],
                                 "service_flagged": s["flagged"], "service_raw": s["raw_person_candidates"],
                                 "r3_flagged": r["flagged"], "r3_raw": r["raw_person_candidates"]})

    out = {
        "full_exposure_cases": full_exposure_cases,
        "cross_table": {f"service_full={k[0]}_r3_full={k[1]}": v for k, v in cross.items()},
        "improved_ids": improved, "regressed_ids": regressed, "both_still_full_ids": both_full,
        "r3_full_exposure_cause_counts": dict(r3_cause_counts),
        "service_full_exposure_cause_counts": dict(svc_cause_counts),
        "r3_full_exposure_by_word": dict(by_word), "r3_pos_total_by_word": dict(by_word_total),
        "r3_full_exposure_by_category": dict(by_category), "r3_pos_total_by_category": dict(by_category_total),
        "nonperson_flag_changes": neg_changes,
    }
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"완전노출(둘 중 하나) 사례: {len(full_exposure_cases)}")
    print(f"개선(서비스 완전노출→R3 해결): {len(improved)}, 역행(서비스 성공→R3 완전노출): {len(regressed)}, 둘다 완전노출: {len(both_full)}")
    print(f"R3 완전노출 원인: {dict(r3_cause_counts)}")
    print(f"R3 완전노출 단어별: {dict(by_word)}")
    print(f"R3 완전노출 문형별: {dict(by_category)}")
    print(f"비PERSON 오탐 변화 건수: {len(neg_changes)}")
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
