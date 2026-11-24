"""R6 후보를 서비스·R3·R4·R5와 함께 v12(최종 1회)·회귀 평가셋 전체에서 비교한다 — 읽기 전용.

55절 표준 지표(원시/최종 P·R·F1, 완전노출/부분노출/완전가림/전체노출률, 비PERSON 오탐(전체+중의성))
를 v12에 적용하고, 문형·단어유형·문체별로 쪼갠다. 완전노출 원인은 58절에서 정정한 대로
leaked>=n(완전노출) 기준에서만 4분류(후보 미생성/ORG·LOC 오분류/신뢰도 부족/1자 길이필터 탈락
+ 참고용 경계불일치)를 매긴다. 회귀 평가셋(r3~r6 heldout, 48절, draft56, v8, v9~v11)은 독립
근거가 아니라 회귀 확인용으로만 쓴다. ner.py·모델·DB·설정 미수정, 외부 LLM 미호출, 재학습 없음.
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

EVAL_ROOT = ROOT / "app" / "shared" / "privacy" / "eval"
REGRESSION_DATASETS = {
    "r3_heldout": Path(r"C:\Users\<user>\venvs\ner-runs\data_r3_phrase\r3_heldout_eval.jsonl"),
    "r4_heldout": Path(r"C:\Users\<user>\venvs\ner-runs\data_r4_phrase\r4_heldout_eval.jsonl"),
    "r5_heldout": Path(r"C:\Users\<user>\venvs\ner-runs\data_r5_phrase\r5_heldout_eval.jsonl"),
    "r6_heldout": Path(r"C:\Users\<user>\venvs\ner-runs\data_r6_phrase\r6_heldout_eval.jsonl"),
    "ambiguity_probe_48": EVAL_ROOT / "style_ambiguity_probe" / "ambiguity_probe.jsonl",
    "draft56": EVAL_ROOT / "step2_synth_draft" / "synth_draft.jsonl",
    "v8_independent": EVAL_ROOT / "style_v8_independent" / "v8_independent.jsonl",
    "v9_locked": EVAL_ROOT / "style_v9_locked" / "v9_locked.jsonl",
    "v10_locked": EVAL_ROOT / "style_v10_locked" / "v10_locked.jsonl",
    "v11_locked": EVAL_ROOT / "style_v11_locked" / "v11_locked.jsonl",
}
V12 = EVAL_ROOT / "style_v12_locked" / "v12_locked.jsonl"


def load_rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def gold_span(row: dict):
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


def diagnose_cause(text, raw_spans, g):
    overlapping = [s for s in raw_spans if max(s.start, g[0]) < min(s.end, g[1])]
    if not overlapping:
        return "후보 미생성"
    person_overlap = [s for s in overlapping if s.label in ("PERSON", "PERSON_ALIAS")]
    non_person = [s for s in overlapping if s.label not in ("PERSON", "PERSON_ALIAS")]
    if not person_overlap:
        return f"ORG/LOC 오분류({'/'.join(sorted({s.label for s in non_person}))})"
    short = [s for s in person_overlap if (s.end - s.start) < 2]
    long_ok = [s for s in person_overlap if (s.end - s.start) >= 2 and s.score >= ner.NER_MIN_SCORE]
    long_low = [s for s in person_overlap if (s.end - s.start) >= 2 and s.score < ner.NER_MIN_SCORE]
    if long_ok:
        return "경계 불일치/병합 충돌(참고)"
    if short and not long_low:
        return "1자 길이필터 탈락"
    if long_low:
        return "신뢰도 부족"
    return "기타"


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * p * r / (p + r)) if p and r and (p + r) else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def eval_rows(rows, collect_causes=False):
    raw_tp = raw_fp = raw_fn = fin_tp = fin_fp = fin_fn = 0
    full_exp = part_exp = full_mask = 0
    neg_total = neg_flagged = 0
    causes = Counter()
    full_cases = []

    for r in rows:
        text = r["text"]
        g = gold_span(r)
        raw = ner.spans_for(text) or []
        final = ner.detect_with_ner(text, ())
        raw_person = [(s.start, s.end) for s in raw if s.label in ("PERSON", "PERSON_ALIAS")]
        final_person = [(s.start, s.end) for s in final if s.kind is Kind.PERSON]

        if g is not None:
            n = g[1] - g[0]
            raw_tp += int(g in raw_person)
            fin_tp += int(g in final_person)
            raw_fn += int(g not in raw_person)
            fin_fn += int(g not in final_person)
            raw_fp += sum(1 for p in raw_person if p not in (g,))
            fin_fp += sum(1 for p in final_person if p not in (g,))
            leaked = n - len(covered(final, g))
            full = leaked >= n
            if full:
                full_exp += 1
                if collect_causes:
                    cause = diagnose_cause(text, raw, g)
                    causes[cause] += 1
                    full_cases.append({"id": r["id"], "word": r.get("word"), "category": r.get("category"),
                                        "style": r.get("style"), "text": text, "cause": cause})
            elif leaked > 0:
                part_exp += 1
            else:
                full_mask += 1
        else:
            neg_total += 1
            neg_flagged += int(bool(final_person))

    n_pos = full_exp + part_exp + full_mask
    out = {
        "pos_sentences": n_pos, "neg_sentences": neg_total,
        "raw_prf": prf(raw_tp, raw_fp, raw_fn), "final_prf": prf(fin_tp, fin_fp, fin_fn),
        "exposure": {
            "full_exposure": full_exp, "partial_exposure": part_exp, "fully_masked": full_mask,
            "full_exposure_rate": round(full_exp / n_pos, 4) if n_pos else None,
            "partial_exposure_rate": round(part_exp / n_pos, 4) if n_pos else None,
            "fully_masked_rate": round(full_mask / n_pos, 4) if n_pos else None,
            "total_exposure_rate": round((full_exp + part_exp) / n_pos, 4) if n_pos else None,
        },
        "nonperson_fpr": round(neg_flagged / neg_total, 4) if neg_total else None,
    }
    if collect_causes:
        out["cause_counts"] = dict(causes)
        out["full_exposure_cases"] = full_cases
    return out


def set_model(model_key, model_dir, expect_fp12):
    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = model_dir
    settings.privacy_ner_threads = 2
    status = ner.ner_status()
    fp12 = status["fingerprint"][:12]
    if expect_fp12 and fp12 != expect_fp12:
        raise SystemExit(f"[{model_key}] 지문 불일치: got {fp12}, expected {expect_fp12}")
    ner._cache.clear()
    return fp12


def run_model(model_key, model_dir, expect_fp12, v12_rows):
    fp12 = set_model(model_key, model_dir, expect_fp12)
    out = {"fingerprint12": fp12}
    out["v12_overall"] = eval_rows(v12_rows, collect_causes=True)
    out["v12_by_category"] = {c: eval_rows([r for r in v12_rows if r.get("category") == c])
                               for c in sorted({r.get("category") for r in v12_rows if r.get("category") not in (None, "nonperson", "background")})}
    out["v12_by_word_type"] = {wt: eval_rows([r for r in v12_rows if r.get("word_type") == wt])
                                for wt in sorted({r.get("word_type") for r in v12_rows if r.get("word_type")})}
    out["v12_by_style"] = {st: eval_rows([r for r in v12_rows if r.get("style") == st])
                            for st in sorted({r.get("style") for r in v12_rows if r.get("style")})}
    for name, path in REGRESSION_DATASETS.items():
        out[name] = eval_rows(load_rows(path)) if path.exists() else {"error": f"파일 없음: {path}"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--service", required=True)
    ap.add_argument("--r3-candidate", required=True)
    ap.add_argument("--r4-candidate", required=True)
    ap.add_argument("--r5-candidate", required=True)
    ap.add_argument("--r6-candidate", required=True)
    ap.add_argument("--service-fp12", default="46210943c45f")
    ap.add_argument("--r3-fp12", default="14f297c7d5f5")
    ap.add_argument("--r4-fp12", default="56249473d8d9")
    ap.add_argument("--r5-fp12", default="a270ea570419")
    ap.add_argument("--r6-fp12", default="c59b174ba126")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    v12_rows = load_rows(V12)
    out = {
        "service": run_model("service", a.service, a.service_fp12, v12_rows),
        "r3_candidate": run_model("r3_candidate", a.r3_candidate, a.r3_fp12, v12_rows),
        "r4_candidate": run_model("r4_candidate", a.r4_candidate, a.r4_fp12, v12_rows),
        "r5_candidate": run_model("r5_candidate", a.r5_candidate, a.r5_fp12, v12_rows),
        "r6_candidate": run_model("r6_candidate", a.r6_candidate, a.r6_fp12, v12_rows),
    }

    for key in out:
        v = out[key]["v12_overall"]
        print(f"{key} fp12={out[key]['fingerprint12']}")
        print(f"  raw P/R/F1={v['raw_prf']['precision']}/{v['raw_prf']['recall']}/{v['raw_prf']['f1']}")
        print(f"  final P/R/F1={v['final_prf']['precision']}/{v['final_prf']['recall']}/{v['final_prf']['f1']}")
        print(f"  exposure={v['exposure']}")
        print(f"  nonperson_fpr(v12)={v['nonperson_fpr']}  nonperson_fpr(48절 중의성)={out[key]['ambiguity_probe_48']['nonperson_fpr']}")
        print(f"  cause_counts={v['cause_counts']}")

    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
