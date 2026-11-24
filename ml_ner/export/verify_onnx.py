"""ONNX NER 검증(프로젝트 가상환경 파이썬: tokenizers+onnxruntime+numpy 만 있으면 된다; torch 불필요).

사전 준비: 학습용 venv 에서 dump_torch_reference.py 로 PyTorch 참조값(reference.json·logits_*.npz)을 만든다.

    <REPO>\\apps\\api\\.venv\\Scripts\\python.exe ml/ner/export/verify_onnx.py \\
        --export-dir <산출 폴더> --ref-dir <참조 폴더> --model-file model.onnx --out <결과.json>

검증: (a) 같은 입력의 PyTorch·ONNX 로짓 최대 절대 차이 (b) PyTorch·ONNX 예측 span 일치율
(c) HF AutoTokenizer 와 tokenizers 단독 토큰화(input_ids·offset) 일치율 (d) ONNX 경로의 overlap/exact P/R/F1·라벨별·음성 문장 오탐률.
추론 참조 구현은 onnx_ner.py(OnnxNer).
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import ner_common as nc  # noqa: E402
import numpy as np  # noqa: E402
from onnx_ner import OnnxNer  # noqa: E402


def key(spans):
    """span 목록을 비교용 튜플 집합으로 바꾼다."""
    return sorted((s["label"], s["start"], s["end"]) for s in spans)


def metrics(rows, preds):
    """overlap/exact 지표와 음성 문장 오탐률.

    @param rows: [{gold(or spans)}]
    @param preds: 문장별 예측 span
    @returns: dict
    """
    golds = [r["gold"] for r in rows]
    neg = [i for i, g in enumerate(golds) if not g]
    out = {"sentences": len(rows), "gold_spans": sum(len(g) for g in golds)}
    for mode in ("overlap", "exact"):
        m = nc.span_metrics(golds, preds, mode)
        out[mode] = {"micro": {k: round(m["micro"][k], 4) for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
                     "per_label": {lab: {k: round(v[k], 4) for k in ("precision", "recall", "f1", "tp", "fp", "fn")}
                                   for lab, v in m["per_label"].items()}}
    out["negative_sentences"] = len(neg)
    out["false_positive_sentences"] = sum(1 for i in neg if preds[i])
    out["fpr"] = round(out["false_positive_sentences"] / len(neg), 4) if neg else 0.0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--model-file", default="model.onnx")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    ref = json.loads((Path(a.ref_dir) / "reference.json").read_text(encoding="utf-8"))
    ner = OnnxNer(a.export_dir, a.model_file, threads=a.threads)
    res = {"model_file": a.model_file, "run_id": ref["run_id"], "logits": {}, "span_agreement": {}, "tokenizer": {}, "metrics": {},
           "disagreements": []}

    # (c) 토크나이저: 자르지 않은 전체 인코딩의 id·offset 일치
    for name, recs in ref["sets"].items():
        ok_ids = ok_off = 0
        bad = []
        for r in recs:
            enc = ner.tk.encode(r["text"])
            i_ok = list(enc.ids) == r["hf_full_ids"]
            o_ok = [list(o) for o in enc.offsets] == r["hf_full_offsets"]
            ok_ids += i_ok
            ok_off += o_ok
            if not (i_ok and o_ok) and len(bad) < 5:
                bad.append({"text": r["text"][:60], "hf": r["hf_full_ids"][:12], "tk": list(enc.ids)[:12]})
        res["tokenizer"][name] = {"sentences": len(recs), "ids_match": ok_ids, "offsets_match": ok_off,
                                  "ids_rate": ok_ids / len(recs), "offsets_rate": ok_off / len(recs), "mismatch_examples": bad}

    # (a) 로짓 차이: HF 가 만든 id(특수 토큰 포함)로 ONNX 를 호출해 PyTorch 로짓과 비교(문장 1개씩, 패딩 없음)
    for name in ("baseline_val", "synth_val", "edge"):
        recs = ref["sets"][name]
        z = np.load(Path(a.ref_dir) / f"logits_{name}.npz")
        lens, flat = z["lens"], z["logits"]
        pos, mx, argmax_diff, nt = 0, 0.0, 0, 0
        for r, n in zip(recs, lens, strict=False):
            lg = ner.run_batch([r["hf_ids"]])[0]
            ref_lg = flat[pos : pos + n]
            pos += n
            mx = max(mx, float(np.abs(lg - ref_lg).max()))
            argmax_diff += int((lg.argmax(-1) != ref_lg.argmax(-1)).sum())
            nt += int(n)
        res["logits"][name] = {"max_abs_diff": mx, "tokens": nt, "argmax_mismatch_tokens": argmax_diff}
    # 배치(패딩 포함) 경로의 로짓 차이: 길이가 다른 문장을 한 배치에 넣는다
    recs = ref["sets"]["synth_val"][:64]
    z = np.load(Path(a.ref_dir) / "logits_synth_val.npz")
    offs = np.concatenate([[0], np.cumsum(z["lens"])])
    outs = ner.run_batch([r["hf_ids"] for r in recs])
    res["logits"]["synth_val_padded_batch64"] = {
        "max_abs_diff": max(float(np.abs(o - z["logits"][offs[i] : offs[i + 1]]).max()) for i, o in enumerate(outs))}

    # (b) span 일치율(런타임 전체 경로: tokenizers -> 윈도우 -> ONNX -> BIO)
    for name, recs in ref["sets"].items():
        texts = [r["text"] for r in recs]
        single = [ner.predict(t) for t in texts]
        batched = ner.predict_many(texts, batch_size=32)
        long_set = name == "long"
        base = [r["torch_windowed_spans"] if long_set else r["torch_spans"] for r in recs]
        same = sum(key(p) == key(b) for p, b in zip(single, base, strict=False))
        same_batch = sum(key(p) == key(b) for p, b in zip(batched, base, strict=False))
        trunc = sum(1 for r in recs if len(r["hf_full_ids"]) > 128)
        res["span_agreement"][name] = {"sentences": len(recs), "reference": "torch_windowed" if long_set else "torch(truncate128)",
                                       "same_single": same, "same_batched": same_batch, "rate_single": same / len(recs),
                                       "rate_batched": same_batch / len(recs), "over_128_tokens": trunc}
        for r, p, b in zip(recs, single, base, strict=False):
            if key(p) != key(b) and len(res["disagreements"]) < 20:
                res["disagreements"].append({"set": name, "text": r["text"][:80], "torch": key(b), "onnx": key(p)})
        # 윈도우 방식 PyTorch 와 truncate 방식 차이(참고): 128 토큰 넘는 문장이 없으면 항상 같다
        if name in ("baseline_val", "synth_val"):
            win_same = sum(key(r["torch_windowed_spans"]) == key(r["torch_spans"]) for r in recs)
            res["span_agreement"][name]["torch_windowed_vs_truncated_same"] = win_same

    # (d) ONNX 경로 지표
    for name in ("baseline_val", "synth_val", "long"):
        recs = ref["sets"][name]
        preds = ner.predict_many([r["text"] for r in recs], batch_size=32)
        res["metrics"][name] = metrics(recs, preds)
        tp = [r["torch_windowed_spans"] if name == "long" else r["torch_spans"] for r in recs]
        res["metrics"][name]["pytorch_same_set"] = metrics(recs, tp)
    out = json.dumps(res, ensure_ascii=False, indent=2)
    if a.out:
        Path(a.out).write_text(out, encoding="utf-8")
    print(f"[{a.model_file}] tokenizer", {k: (v["ids_rate"], v["offsets_rate"]) for k, v in res["tokenizer"].items()})
    print("logits max abs diff", {k: v["max_abs_diff"] for k, v in res["logits"].items()})
    print("span agreement", {k: (v["same_single"], v["same_batched"], v["sentences"]) for k, v in res["span_agreement"].items()})
    for k, v in res["metrics"].items():
        print(k, "overlap F1", v["overlap"]["micro"]["f1"], "exact F1", v["exact"]["micro"]["f1"], "FPR", v["fpr"],
              "| torch", v["pytorch_same_set"]["overlap"]["micro"]["f1"], v["pytorch_same_set"]["exact"]["micro"]["f1"], v["pytorch_same_set"]["fpr"])
    return res


if __name__ == "__main__":
    main()
