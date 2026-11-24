"""한 run 의 best 체크포인트를 val 로 종합 평가한다(test 는 읽지 않는다).

내용: Baseline val(주)·합성 val(보조)의 exact/overlap P/R/F1·라벨별 FNR, Baseline val max_len 256 보조 평가, 오류 목록,
지연(CPU 4스레드 문장 길이 30/100/300자 중앙값·P95, GPU 1개 값), 가중치 MB.

    python ml/ner/train/evaluate_full.py --run-dir <ner-runs 폴더>/<run_id>
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_ner as ev  # noqa: E402
import ner_common as nc  # noqa: E402
import ner_infer as ni  # noqa: E402
import tokenize_align as ta  # noqa: E402


def _ov(a, b):
    return min(a["end"], b["end"]) > max(a["start"], b["start"])


def error_cases(rows, preds):
    """문장별 오류를 분류한다: MISS(미탐), BOUNDARY(같은 라벨 겹침·경계 다름), CONFUSION(다른 라벨과 겹침), FALSE_POS(정답과 안 겹침).

    @param rows: 평가 행(id,text,spans)
    @param preds: 문장별 예측 span
    @returns: 오류 dict 목록
    """
    out = []
    for r, ps in zip(rows, preds, strict=False):
        text = r["text"]
        for g in r["spans"]:
            same = [p for p in ps if p["label"] == g["label"] and _ov(p, g)]
            if any((p["start"], p["end"]) == (g["start"], g["end"]) for p in same):
                continue
            other = [p for p in ps if p["label"] != g["label"] and _ov(p, g)]
            kind = "BOUNDARY" if same else "CONFUSION" if other else "MISS"
            pr = (same or other or [None])[0]
            out.append({"id": r["id"], "type": kind, "gold_label": g["label"], "gold": text[g["start"] : g["end"]],
                        "pred_label": pr["label"] if pr else None, "pred": text[pr["start"] : pr["end"]] if pr else None,
                        "text": text})
        for p in ps:
            if not any(_ov(p, g) for g in r["spans"]):
                out.append({"id": r["id"], "type": "FALSE_POS", "gold_label": None, "gold": None, "pred_label": p["label"],
                            "pred": text[p["start"] : p["end"]], "text": text})
    return out


def _sentences(rows, length, n):
    """실제 문장을 이어 붙여 정확히 length 자인 문장 n 개를 만든다."""
    texts = [r["text"] for r in rows]
    out = []
    for i in range(n):
        s = ""
        j = i
        while len(s) < length:
            s += texts[j % len(texts)] + " "
            j += 7
        out.append(s[:length])
    return out


def latency(model, tok, texts, device, n_threads=None, warm=5, amp=False):
    """문장 1개씩(batch 1) 예측 지연(ms) 목록. 토큰 수가 max_len 으로 잘리지 않게 512 까지 허용."""
    import torch

    if n_threads:
        torch.set_num_threads(n_threads)
    model.eval()
    ts = []
    with torch.no_grad():
        for i, t in enumerate(texts):
            enc = tok(t, return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                model(**enc).logits.argmax(-1).cpu()
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= warm:
                ts.append(1000 * (time.perf_counter() - t0))
    ts.sort()
    return {"median_ms": statistics.median(ts), "p95_ms": ts[min(len(ts) - 1, int(0.95 * len(ts)))], "n": len(ts)}


def main(argv=None):
    import torch
    from transformers import AutoModelForTokenClassification

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--data-dir", default=None, help="synth_val* 가 있는 폴더(기본 ner-runs/data)")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    cfg = json.loads((run / "train_log.json").read_text(encoding="utf-8"))["config"]
    best = run / "best"
    tok = ta.load_tokenizer(str(best))
    model = AutoModelForTokenClassification.from_pretrained(str(best))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    res = {"run_id": run.name, "model": cfg["model"], "sets": {}, "errors": {}}
    dd = Path(a.data_dir) if a.data_dir else nc.DATA_OUT
    sets = {"baseline_val": ev.load_eval_rows("baseline_val"), "synth_val": ni.load_rows("synth_val", dd)}
    for extra in ("synth_val_core", "synth_val_aug"):  # 보강 전/후 틀을 따로 본다(있을 때만)
        if (dd / f"{extra}.jsonl").exists() and (dd / f"{extra}.jsonl").stat().st_size > 0:
            sets[extra] = ni.load_rows(extra, dd)
    for name, rows in sets.items():
        res["sets"][name] = ev.evaluate_rows(model, tok, rows, cfg["max_len"], 64, dev, cfg["policy"], True)
    res["sets"]["baseline_val_maxlen256"] = ev.evaluate_rows(model, tok, sets["baseline_val"], 256, 64, dev, cfg["policy"], True)
    for name, rows in sets.items():
        preds, _ = ni.predict_spans(model, tok, rows, cfg["max_len"], 64, dev, cfg["policy"], True)
        res["errors"][name] = error_cases(rows, preds)
    res["weights_mb"] = sum(f.stat().st_size for f in best.glob("*.safetensors")) / 2**20
    res["params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
    if not a.skip_latency:
        lat = {}
        model.cpu()
        for length in (30, 100, 300):
            texts = _sentences(sets["baseline_val"] + sets["synth_val"], length, 105)
            ntok = statistics.mean(len(tok(t)["input_ids"]) for t in texts[:20])
            lat[f"cpu4_{length}chars"] = {**latency(model, tok, texts, torch.device("cpu"), 4), "mean_tokens": ntok}
        if dev.type == "cuda":
            model.to(dev)
            texts = _sentences(sets["baseline_val"] + sets["synth_val"], 100, 105)
            lat["gpu_100chars_fp16"] = latency(model, tok, texts, dev, amp=True)
        res["latency"] = lat
    out = run / "eval_full.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    b = res["sets"]["baseline_val"]
    print(run.name, "baseline_val overlap F1 {:.3f} exact F1 {:.3f}".format(b["overlap"]["micro"]["f1"], b["exact"]["micro"]["f1"]))
    return res


if __name__ == "__main__":
    main()
