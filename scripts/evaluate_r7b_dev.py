"""R7b 개발셋 평가 — P+Amb 다중 시드(7/13/21/42/99) 검증 + R6+B/R7+B 기준선 비교. 읽기 전용.

**모델 선정 기준(평가 전에 고정, 결과를 본 뒤 바꾸지 않는다)**:
  1순위: 비PERSON 오탐률과 인접 엔티티 과잉병합 회귀 방지(안전성 게이트 — 둘 중 하나라도 R6+B보다
         나쁘면 "안전성 미달"로 표시하고 그 후보는 노출률이 아무리 좋아도 R7b 후보에서 제외한다)
  2순위: 안전성 게이트를 통과한 모델 중 전체 이름 노출률(완전+부분) 최소화
  3순위: Recall, Exact F1
  4순위: 기존 평가셋 회귀(v9~v14)·ONNX 일치성·CPU 추론 성능(이번 스크립트 밖에서 별도 확인)

"안전성 게이트" 판정 기준(고정): 오탐은 R6+B의 매크로 오탐률 대비 **절대값 +2%p 이내**(그 이상이면
미달), 과잉병합은 `r7b_adjacent.jsonl`(9건)+`check_adjacent_entity_merge.py`(19건) 합산 28건 중
**R6+B보다 과잉병합 건수가 늘면** 미달로 본다. `ner.py`/`merge_ner`/운영 모델/DB/설정 미수정, v15는
이 결과를 본 뒤에만 만든다(이번 스크립트는 안 만든다), 외부 LLM 미호출.
"""

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
import check_adjacent_entity_merge as cae  # noqa: E402

DEV = ROOT / "app/shared/privacy/eval/style_r7b_dev"
BUCKETS = {
    "실사용": DEV / "r7b_realistic.jsonl",
    "무공백인접": DEV / "r7b_adjacent.jsonl",
    "중의성": DEV / "r7b_ambiguity.jsonl",
    "스트레스": DEV / "r7b_stress.jsonl",
}
ONE_CHAR = DEV / "r7b_one_char.jsonl"


def load_rows(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def gold_spans(row):
    return [(sp["start"], sp["end"]) for sp in row.get("spans", []) if sp.get("kind") in ("PERSON", "PERSON_ALIAS")]


def covered(preds, g):
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


def eval_bucket(rows, mode, insp=None, check_overmerge=False):
    raw_tp = raw_fp = raw_fn = fin_tp = fin_fp = fin_fn = 0
    full = partial = masked = 0
    neg_total = neg_flagged = 0
    overmerge = 0

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
                    full += 1
                elif leaked > 0:
                    partial += 1
                else:
                    masked += 1
            raw_fp += sum(1 for p in raw_person if p not in golds)
            fin_fp += sum(1 for p in final_person if p not in golds)
            if check_overmerge and len(golds) >= 2:
                entities = [text[g[0]:g[1]] for g in golds]
                over = any(
                    s.kind is Kind.PERSON and sum(1 for e in entities if text[s.start:s.end].find(e) >= 0) >= 2
                    for s in final
                )
                if over:
                    overmerge += 1
        else:
            neg_total += 1
            neg_flagged += int(bool(final_person))

    n_pos = full + partial + masked
    out = {
        "n_pos": n_pos, "n_neg": neg_total,
        "raw_prf": prf(raw_tp, raw_fp, raw_fn), "final_prf": prf(fin_tp, fin_fp, fin_fn),
        "exposure": {
            "full": full, "partial": partial, "masked": masked,
            "full_rate": round(full / n_pos, 4) if n_pos else None,
            "partial_rate": round(partial / n_pos, 4) if n_pos else None,
            "masked_rate": round(masked / n_pos, 4) if n_pos else None,
            "total_rate": round((full + partial) / n_pos, 4) if n_pos else None,
        },
        "nonperson_fpr": round(neg_flagged / neg_total, 4) if neg_total else None,
    }
    if check_overmerge:
        out["overmerge_count"] = overmerge
    return out


def eval_one_char(rows, mode, insp=None):
    full = 0
    for r in rows:
        g = gold_spans(r)[0]
        _, final = predict(r["text"], mode, insp)
        leaked = (g[1] - g[0]) - len(covered(final, g))
        if leaked >= (g[1] - g[0]):
            full += 1
    return {"n": len(rows), "full_exposure": full, "full_exposure_rate": round(full / len(rows), 4) if rows else None}


def run_cae_overmerge(mode, insp=None):
    """check_adjacent_entity_merge.py 19건 중 과잉병합 건수"""
    count = 0
    hits = []
    for name, text, entities in cae.CASES:
        _, final = predict(text, mode, insp)
        over = any(
            s.kind is Kind.PERSON and sum(1 for e in entities if text[s.start:s.end].find(e) >= 0) >= 2
            for s in final
        )
        if over:
            count += 1
            hits.append(name)
    return count, hits


def macro(bucket_results):
    out = {"final_prf": {}, "exposure": {}}
    for k in ["precision", "recall", "f1"]:
        vals = [bucket_results[b]["final_prf"][k] for b in BUCKETS if bucket_results[b]["final_prf"][k] is not None]
        out["final_prf"][k] = round(sum(vals) / len(vals), 4) if vals else None
    for k in ["full_rate", "partial_rate", "masked_rate", "total_rate"]:
        vals = [bucket_results[b]["exposure"][k] for b in BUCKETS if bucket_results[b]["exposure"][k] is not None]
        out["exposure"][k] = round(sum(vals) / len(vals), 4) if vals else None
    fprs = [bucket_results[b]["nonperson_fpr"] for b in BUCKETS if bucket_results[b]["nonperson_fpr"] is not None]
    out["nonperson_fpr_macro"] = round(sum(fprs) / len(fprs), 4) if fprs else None
    return out


def run_combo(key, model_dir, fp12, mode, bucket_rows, one_char_rows):
    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = model_dir
    settings.privacy_ner_threads = 2
    status = ner.ner_status()
    assert status["fingerprint"][:12] == fp12, (key, status["fingerprint"][:12])
    insp = eb.TokenInspector(model_dir) if mode == "b" else None

    out = {"fingerprint12": status["fingerprint"][:12], "mode": mode}
    for bucket, rows in bucket_rows.items():
        out[bucket] = eval_bucket(rows, mode, insp, check_overmerge=(bucket == "무공백인접"))
    out["통합(매크로평균)"] = macro({b: out[b] for b in BUCKETS})
    out["1글자이름(정책, 통합 미포함)"] = eval_one_char(one_char_rows, mode, insp)
    cae_count, cae_hits = run_cae_overmerge(mode, insp)
    out["adjacent_stress_overmerge"] = {"count": cae_count, "cases": cae_hits}
    out["total_overmerge"] = out["무공백인접"]["overmerge_count"] + cae_count
    return out


MODELS = [
    ("R6+기존병합", r"C:\Users\<user>\venvs\ner-runs\export\r6-candidate-s42", "c59b174ba126", "normal"),
    ("R6+B", r"C:\Users\<user>\venvs\ner-runs\export\r6-candidate-s42", "c59b174ba126", "b"),
    ("R7+기존병합(참고)", r"C:\Users\<user>\venvs\ner-runs\export\r7-candidate-s42", "098cb439c7e4", "normal"),
    ("R7+B(참고)", r"C:\Users\<user>\venvs\ner-runs\export\r7-candidate-s42", "098cb439c7e4", "b"),
]
PAMB_SEEDS = {
    7: "d38ba679c06d", 13: "91ee359461a5", 21: "b01dce5ebab0",
    42: "32b849f80307", 99: "f7b508aaf7eb",
}
for seed, fp in PAMB_SEEDS.items():
    MODELS.append((f"P+Amb-s{seed}+기존병합", rf"C:\Users\<user>\venvs\ner-runs\export\r7abl-pamb-s{seed}", fp, "normal"))
    MODELS.append((f"P+Amb-s{seed}+B", rf"C:\Users\<user>\venvs\ner-runs\export\r7abl-pamb-s{seed}", fp, "b"))


def main():
    bucket_rows = {b: load_rows(p) for b, p in BUCKETS.items()}
    one_char_rows = load_rows(ONE_CHAR)
    for b, rows in bucket_rows.items():
        print(f"[{b}] {len(rows)}행")

    out = {}
    for key, model_dir, fp12, mode in MODELS:
        out[key] = run_combo(key, model_dir, fp12, mode, bucket_rows, one_char_rows)
        c = out[key]["통합(매크로평균)"]
        print(f"{key}: 전체노출(macro)={c['exposure']['total_rate']} 오탐(macro)={c['nonperson_fpr_macro']} "
              f"과잉병합(무공백9+스트레스19)={out[key]['total_overmerge']}/28 1글자={out[key]['1글자이름(정책, 통합 미포함)']['full_exposure_rate']}")

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "r7b_dev_eval.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
