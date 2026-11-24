"""NER 추론 속도·메모리 측정(CPU). 프로세스 하나가 (백엔드, 모델 파일, 스레드 수) 한 조합을 잰다.

    # ONNX: 프로젝트 venv      python bench_onnx.py --backend onnx --export-dir <폴더> --model-file model.onnx --threads 2 --ref <reference.json> --out r.json
    # PyTorch: 학습용 venv      python bench_onnx.py --backend torch --run-dir <run> --threads 4 --ref <reference.json> --out r.json

측정: 모델 로딩 시간, 프로세스 RSS(작업 집합) 증가, 첫 호출 지연, 문장당 지연 중앙값·P95(30/100/300자; 전체 경로 e2e = 토큰화+추론+디코딩),
배치 8·32 처리량(100자 문장; 문장/초). 문장은 val 문장을 이어 붙여 정확히 N 자로 만든다.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def rss_mb():
    """현재 프로세스의 메모리(작업 집합, MB). Windows 는 ctypes, 그 외는 /proc.

    @returns: float
    """
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.WorkingSetSize / 2**20
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def make_texts(pool, length, n):
    """pool 문장을 이어 붙여 정확히 length 자인 문장 n 개를 만든다.

    @param pool: 문장 목록
    @param length: 글자 수
    @param n: 개수
    @returns: list[str]
    """
    out = []
    for i in range(n):
        s, j = "", i
        while len(s) < length:
            s += pool[j % len(pool)] + " "
            j += 7
        out.append(s[:length])
    return out


def stats(ts):
    """지연 목록(ms)의 중앙값·P95.

    @param ts: list[float]
    @returns: dict
    """
    ts = sorted(ts)
    return {"median_ms": round(statistics.median(ts), 3), "p95_ms": round(ts[min(len(ts) - 1, int(0.95 * len(ts)))], 3), "n": len(ts)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=["onnx", "torch"], required=True)
    ap.add_argument("--export-dir")
    ap.add_argument("--model-file", default="model.onnx")
    ap.add_argument("--run-dir")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ref = json.loads(Path(a.ref).read_text(encoding="utf-8"))
    pool = [r["text"] for k in ("baseline_val", "synth_val") for r in ref["sets"][k]]
    res = {"backend": a.backend, "model_file": a.model_file if a.backend == "onnx" else "pytorch", "threads": a.threads}
    r0 = rss_mb()
    t0 = time.perf_counter()
    if a.backend == "onnx":
        from onnx_ner import OnnxNer

        ner = OnnxNer(a.export_dir, a.model_file, threads=a.threads)
        predict, predict_many = ner.predict, ner.predict_many
        pure = lambda t: ner.run_batch([[ner.cls_id, *ner.encode(t)[0], ner.sep_id]])  # noqa: E731 — 윈도우 없는 순수 모델 호출
    else:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        torch.set_num_threads(a.threads)
        best = Path(a.run_dir) / "best"
        tok = AutoTokenizer.from_pretrained(str(best), use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(str(best)).eval()

        def predict_many(texts, batch_size=32):
            outs = []
            for i in range(0, len(texts), batch_size):
                enc = tok(texts[i : i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=512,
                          return_offsets_mapping=True)
                enc.pop("offset_mapping")
                with torch.no_grad():
                    outs.append(model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}).logits.argmax(-1))
            return outs

        def predict(t):
            return predict_many([t], 1)

        pure = predict
    res["load_s"] = round(time.perf_counter() - t0, 3)
    r1 = rss_mb()
    res["rss_before_mb"], res["rss_after_load_mb"] = round(r0, 1), round(r1, 1)
    # 첫 호출(워밍업) 지연
    first = []
    for t in make_texts(pool, 100, 3):
        t1 = time.perf_counter()
        predict(t)
        first.append(1000 * (time.perf_counter() - t1))
    res["first_call_ms"], res["second_third_call_ms"] = round(first[0], 2), [round(x, 2) for x in first[1:]]
    for length in (30, 100, 300):
        texts = make_texts(pool, length, 105)
        ts = []
        for i, t in enumerate(texts):
            t1 = time.perf_counter()
            predict(t)
            if i >= 5:
                ts.append(1000 * (time.perf_counter() - t1))
        res[f"e2e_{length}chars"] = stats(ts)
        ts = []
        for i, t in enumerate(texts):
            t1 = time.perf_counter()
            pure(t)
            if i >= 5:
                ts.append(1000 * (time.perf_counter() - t1))
        res[f"model_only_{length}chars"] = stats(ts)
    for bs in (8, 32):
        texts = make_texts(pool, 100, bs)
        predict_many(texts, bs)
        reps = 20
        t1 = time.perf_counter()
        for _ in range(reps):
            predict_many(texts, bs)
        dt = time.perf_counter() - t1
        res[f"batch{bs}_sent_per_s"] = round(reps * bs / dt, 1)
    res["rss_after_bench_mb"] = round(rss_mb(), 1)
    res["rss_increase_mb"] = round(res["rss_after_bench_mb"] - res["rss_before_mb"], 1)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("backend", "model_file", "threads", "load_s", "rss_increase_mb", "first_call_ms", "e2e_100chars", "batch32_sent_per_s")}))


if __name__ == "__main__":
    main()
