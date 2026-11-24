"""R7 4-way 비교(R6/R7 × 기존 병합/실험 B) — 개발셋 전용, 읽기 전용. 55절 8개 표준 지표를
실사용·중의성·스트레스 세 영역으로 나눠 계산하고, 세 영역을 **동일 비중(각 1/3)**으로 둔 매크로
평균을 "통합" 성능으로 보고한다(표본수 가중 없음 — 결과를 보기 전에 고정한 기준). 1글자 이름은
항상 별도로만 보고하고 통합 평균에 포함하지 않는다. `ner.py`/`merge_ner`/모델/DB/설정 미수정
(실험 B는 이 스크립트 안에서만 디코딩을 바꾼 사본), v12/v13/ambiguity_probe 미수정, 재학습 없음,
외부 LLM 미호출.
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
BUCKETS = {
    "실사용": [EVAL_ROOT / "style_r7_dev" / "r7_dev_realistic.jsonl"],
    "중의성": [EVAL_ROOT / "style_ambiguity_probe" / "ambiguity_probe.jsonl"],
    "스트레스": [EVAL_ROOT / "style_v12_locked" / "v12_locked.jsonl", EVAL_ROOT / "style_v13_locked" / "v13_locked.jsonl"],
}
ONE_CHAR_BUCKET = EVAL_ROOT / "style_r7_dev" / "r7_dev_1char.jsonl"


def load_rows(paths) -> list[dict]:
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def gold_spans(row) -> list[tuple[int, int]]:
    return [(sp["start"], sp["end"]) for sp in row.get("spans", []) if sp.get("kind") in ("PERSON", "PERSON_ALIAS")]


def covered(preds, g) -> set:
    cov = set()
    for s in preds:
        if s.kind is Kind.PERSON:
            cov.update(range(max(s.start, g[0]), min(s.end, g[1])))
    return cov


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * p * r / (p + r)) if p and r and (p + r) else None
    return {"precision": p, "recall": r, "f1": f1}


def predict(text, mode, insp=None):
    """mode "normal": 운영 그대로 ner.detect_with_ner(). mode "b": 실험 B(디코딩만 사본, merge_ner 는 그대로)."""
    if mode == "normal":
        final = ner.detect_with_ner(text)
        raw = ner.spans_for(text) or []
        raw_person = [(s.start, s.end) for s in raw if s.label in ("PERSON", "PERSON_ALIAS")]
        return raw_person, final
    toks = insp.token_predictions(text)
    relaxed = eb.decode_relaxed(toks)
    ner_spans = eb.to_nerspans(relaxed, text)
    rule = detectors.detect(text, ())
    final = ner.merge_ner(text, rule, ner_spans)
    raw_person = [(s.start, s.end) for s in ner_spans if s.label in ("PERSON", "PERSON_ALIAS")]
    return raw_person, final


def eval_rows(rows, mode, insp=None):
    raw_tp = raw_fp = raw_fn = fin_tp = fin_fp = fin_fn = 0
    fin_fp_positions = Counter()
    full_exp = part_exp = full_mask = 0
    neg_total = neg_flagged = 0

    for r in rows:
        text = r["text"]
        golds = gold_spans(r)
        raw_person, final = predict(text, mode, insp)
        final_person = [(s.start, s.end) for s in final if s.kind is Kind.PERSON]

        if golds:
            for g in golds:
                n = g[1] - g[0]
                raw_tp += int(g in raw_person)
                fin_tp += int(g in final_person)
                raw_fn += int(g not in raw_person)
                fin_fn += int(g not in final_person)
                leaked = n - len(covered(final, g))
                if leaked >= n:
                    full_exp += 1
                elif leaked > 0:
                    part_exp += 1
                else:
                    full_mask += 1
            raw_fp += sum(1 for p in raw_person if p not in golds)
            fin_fp += sum(1 for p in final_person if p not in golds)
        else:
            neg_total += 1
            neg_flagged += int(bool(final_person))

    n_pos = full_exp + part_exp + full_mask
    return {
        "n_pos": n_pos, "n_neg": neg_total,
        "raw_prf": prf(raw_tp, raw_fp, raw_fn),
        "final_prf": prf(fin_tp, fin_fp, fin_fn),
        "exposure": {
            "full": full_exp, "partial": part_exp, "masked": full_mask,
            "full_rate": round(full_exp / n_pos, 4) if n_pos else None,
            "partial_rate": round(part_exp / n_pos, 4) if n_pos else None,
            "masked_rate": round(full_mask / n_pos, 4) if n_pos else None,
            "total_rate": round((full_exp + part_exp) / n_pos, 4) if n_pos else None,
        },
        "nonperson_fpr": round(neg_flagged / neg_total, 4) if neg_total else None,
    }


def eval_one_char(rows, mode, insp=None):
    """1글자 이름: merge_ner 의 len(value)<2 필터로 정책상 항상 완전노출된다 — 모델 품질이 아니라
    정책 확인용이다. 2글자 이상 지표와 절대 합치지 않는다."""
    full = 0
    for r in rows:
        g = gold_spans(r)[0]
        _, final = predict(r["text"], mode, insp)
        leaked = (g[1] - g[0]) - len(covered(final, g))
        if leaked >= (g[1] - g[0]):
            full += 1
    return {"n": len(rows), "full_exposure": full, "full_exposure_rate": round(full / len(rows), 4) if rows else None}


def macro_combined(bucket_results: dict) -> dict:
    """세 영역(실사용/중의성/스트레스)을 표본수 가중 없이 동일 비중(1/3)으로 평균한다."""
    keys_prf = ["precision", "recall", "f1"]
    out = {"final_prf": {}, "exposure": {}}
    for k in keys_prf:
        vals = [bucket_results[b]["final_prf"][k] for b in BUCKETS if bucket_results[b]["final_prf"][k] is not None]
        out["final_prf"][k] = round(sum(vals) / len(vals), 4) if vals else None
    for k in ["full_rate", "partial_rate", "masked_rate", "total_rate"]:
        vals = [bucket_results[b]["exposure"][k] for b in BUCKETS if bucket_results[b]["exposure"][k] is not None]
        out["exposure"][k] = round(sum(vals) / len(vals), 4) if vals else None
    fprs = [bucket_results[b]["nonperson_fpr"] for b in BUCKETS if bucket_results[b]["nonperson_fpr"] is not None]
    out["nonperson_fpr_macro"] = round(sum(fprs) / len(fprs), 4) if fprs else None
    return out


def run_combo(model_key, model_dir, expect_fp12, mode, bucket_rows, one_char_rows):
    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = model_dir
    settings.privacy_ner_threads = 2
    status = ner.ner_status()
    assert status["fingerprint"][:12] == expect_fp12, (model_key, status["fingerprint"][:12])
    insp = eb.TokenInspector(model_dir) if mode == "b" else None

    out = {"fingerprint12": status["fingerprint"][:12], "mode": mode}
    for bucket, rows in bucket_rows.items():
        out[bucket] = eval_rows(rows, mode, insp)
    out["통합(매크로평균)"] = macro_combined({b: out[b] for b in BUCKETS})
    out["1글자이름(정책, 통합 미포함)"] = eval_one_char(one_char_rows, mode, insp)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--r6-candidate", required=True)
    ap.add_argument("--r7-candidate", required=True)
    ap.add_argument("--r6-fp12", default="c59b174ba126")
    ap.add_argument("--r7-fp12", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    bucket_rows = {b: load_rows(paths) for b, paths in BUCKETS.items()}
    one_char_rows = load_rows([ONE_CHAR_BUCKET])
    for b, rows in bucket_rows.items():
        print(f"[{b}] {len(rows)}행 로드")
    print(f"[1글자] {len(one_char_rows)}행 로드")

    combos = [
        ("R6+기존병합", a.r6_candidate, a.r6_fp12, "normal"),
        ("R6+B", a.r6_candidate, a.r6_fp12, "b"),
        ("R7+기존병합", a.r7_candidate, a.r7_fp12, "normal"),
        ("R7+B", a.r7_candidate, a.r7_fp12, "b"),
    ]
    out = {}
    for key, model_dir, fp12, mode in combos:
        out[key] = run_combo(key, model_dir, fp12, mode, bucket_rows, one_char_rows)
        c = out[key]["통합(매크로평균)"]
        oc = out[key]["1글자이름(정책, 통합 미포함)"]
        print(f"{key}: 통합 F1={c['final_prf']['f1']} 전체노출={c['exposure']['total_rate']} "
              f"| 실사용 전체노출={out[key]['실사용']['exposure']['total_rate']} "
              f"중의성 FPR={out[key]['중의성']['nonperson_fpr']} "
              f"스트레스 전체노출={out[key]['스트레스']['exposure']['total_rate']} "
              f"| 1글자 완전노출={oc['full_exposure_rate']}(정책, 별도)")

    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
