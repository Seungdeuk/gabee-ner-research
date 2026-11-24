"""R6+B(최종 후보: R6 모델 가중치 그대로 + 실험 B 병합 완화) v13 최종 1회 검증 — 읽기 전용.

55절 표준 8개 지표(원시/최종 P·R·F1, 완전노출/부분노출/완전가림/전체노출률, 비PERSON 오탐
전체+중의성)를 v13에 적용하고, 문형·단어유형·문체별로 쪼갠다. 완전노출 원인 4분류(60절 기준,
leaked>=n 만). 서비스·R3·R4·R5·R6(그대로)와 비교. ner.py·merge_ner·모델·DB·설정 미수정(실험 B는
이 스크립트 안에서만 디코딩을 바꾼 사본), v13 미수정, 외부 LLM 미호출, 재학습 없음.
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
from app.shared.privacy import detectors, ner  # noqa: E402
from app.shared.privacy.types import Kind  # noqa: E402

import experiment_b_relaxed_merge as eb  # noqa: E402

EVAL_ROOT = ROOT / "app" / "shared" / "privacy" / "eval"
V13 = EVAL_ROOT / "style_v13_locked" / "v13_locked.jsonl"
REGRESSION_DATASETS = {
    "r3_heldout": Path(r"C:\Users\<user>\venvs\ner-runs\data_r3_phrase\r3_heldout_eval.jsonl"),
    "r6_heldout": Path(r"C:\Users\<user>\venvs\ner-runs\data_r6_phrase\r6_heldout_eval.jsonl"),
    "ambiguity_probe_48": EVAL_ROOT / "style_ambiguity_probe" / "ambiguity_probe.jsonl",
    "draft56": EVAL_ROOT / "step2_synth_draft" / "synth_draft.jsonl",
    "v8_independent": EVAL_ROOT / "style_v8_independent" / "v8_independent.jsonl",
    "v9_locked": EVAL_ROOT / "style_v9_locked" / "v9_locked.jsonl",
    "v10_locked": EVAL_ROOT / "style_v10_locked" / "v10_locked.jsonl",
    "v11_locked": EVAL_ROOT / "style_v11_locked" / "v11_locked.jsonl",
    "v12_locked": EVAL_ROOT / "style_v12_locked" / "v12_locked.jsonl",
}


def load_rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


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


def diagnose_cause(raw_spans, g):
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


def eval_rows(rows, insp, collect_causes=False):
    raw_tp = raw_fp = raw_fn = fin_tp = fin_fp = fin_fn = 0
    full_exp = part_exp = full_mask = 0
    neg_total = neg_flagged = 0
    causes = Counter()
    full_cases = []

    for r in rows:
        text = r["text"]
        g = gold_span(r)
        toks = insp.token_predictions(text)
        relaxed = eb.decode_relaxed(toks)
        ner_spans = eb.to_nerspans(relaxed, text)
        rule = detectors.detect(text, ())
        final = ner.merge_ner(text, rule, ner_spans)
        raw_person = [(s.start, s.end) for s in ner_spans if s.label in ("PERSON", "PERSON_ALIAS")]
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
                    cause = diagnose_cause(ner_spans, g)
                    causes[cause] += 1
                    full_cases.append({"id": r["id"], "word": r.get("word"), "category": r.get("category"), "style": r.get("style"), "text": text, "cause": cause})
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


def run_r6b(model_dir, expect_fp12, v13_rows):
    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = model_dir
    settings.privacy_ner_threads = 2
    status = ner.ner_status()
    assert status["fingerprint"][:12] == expect_fp12, status["fingerprint"][:12]
    ner._cache.clear()
    insp = eb.TokenInspector(model_dir)

    out = {"fingerprint12": status["fingerprint"][:12]}
    out["v13_overall"] = eval_rows(v13_rows, insp, collect_causes=True)
    out["v13_by_category"] = {c: eval_rows([r for r in v13_rows if r.get("category") == c], insp)
                               for c in sorted({r.get("category") for r in v13_rows if r.get("category") not in (None, "nonperson", "background")})}
    out["v13_by_word_type"] = {wt: eval_rows([r for r in v13_rows if r.get("word_type") == wt], insp)
                                for wt in sorted({r.get("word_type") for r in v13_rows if r.get("word_type")})}
    out["v13_by_style"] = {st: eval_rows([r for r in v13_rows if r.get("style") == st], insp)
                            for st in sorted({r.get("style") for r in v13_rows if r.get("style")})}
    for name, path in REGRESSION_DATASETS.items():
        out[name] = eval_rows(load_rows(path), insp) if path.exists() else {"error": "missing"}
    return out


def run_plain(model_dir, expect_fp12, v13_rows, key, using_evaluate_r6):
    return using_evaluate_r6.run_model(key, model_dir, expect_fp12, v13_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--service", required=True)
    ap.add_argument("--r3-candidate", required=True)
    ap.add_argument("--r4-candidate", required=True)
    ap.add_argument("--r5-candidate", required=True)
    ap.add_argument("--r6-candidate", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    v13_rows = load_rows(V13)
    import evaluate_r6_candidate as ev6

    ev6.V12 = V13  # 55절 표준 run_model 을 v13에 그대로 재사용(변수명만 다름, 로직 동일)
    ev6.REGRESSION_DATASETS = REGRESSION_DATASETS

    out = {
        "service": ev6.run_model("service", a.service, "46210943c45f", v13_rows),
        "r3_candidate": ev6.run_model("r3_candidate", a.r3_candidate, "14f297c7d5f5", v13_rows),
        "r4_candidate": ev6.run_model("r4_candidate", a.r4_candidate, "56249473d8d9", v13_rows),
        "r5_candidate": ev6.run_model("r5_candidate", a.r5_candidate, "a270ea570419", v13_rows),
        "r6_normal": ev6.run_model("r6_normal", a.r6_candidate, "c59b174ba126", v13_rows),
        "r6_plus_b": run_r6b(a.r6_candidate, "c59b174ba126", v13_rows),
    }

    for key in out:
        v = out[key].get("v12_overall") or out[key].get("v13_overall")
        print(f"{key}: final F1={v['final_prf']['f1']} exposure={v['exposure']} fpr(main)={v['nonperson_fpr']} fpr(48amb)={out[key]['ambiguity_probe_48']['nonperson_fpr']}")

    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
