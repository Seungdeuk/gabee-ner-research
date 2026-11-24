"""S2b 컨테이너 측정(이미지 안에서 실행): g12_latency.py 와 같은 방법(워밍업 30, 측정 300문장, 5회 반복, 시드 20261127, 문장 캐시 제거)을
운영 자원 제한(--cpus/--memory)·스레드·동시성 설정에서 측정한다. 사전 등록 코드(g12_latency.py)는 건드리지 않았다.
문장 파일은 개발 검증셋 r11_val(봉인 평가셋 아님). 문장 원문은 출력하지 않는다.
사용(컨테이너 안): python /check/s2b_container_latency.py --sentences /data/r11_val.jsonl --threads N --concurrency C --out /out/x.json
"""
import argparse
import json
import os
import platform
import random
import resource
import statistics as st
import sys
import threading
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--sentences", type=Path, required=True)
ap.add_argument("--threads", type=int, required=True)
ap.add_argument("--concurrency", type=int, default=1)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
os.environ["OMP_NUM_THREADS"] = str(a.threads)
os.environ["MKL_NUM_THREADS"] = str(a.threads)
os.environ["PRIVACY_NER_THREADS"] = str(a.threads)
os.environ["PRIVACY_NER_MAX_CONCURRENCY"] = str(max(1, a.concurrency))
os.environ.setdefault("PRIVACY_NER_MODE", "auto")  # S0 이미지는 auto, R11 이미지는 required(이미지 기본값)

import onnxruntime as ort  # noqa: E402
from app.shared.privacy import ner  # noqa: E402

N_WARM, N_MEASURE, N_REPEAT, SEED = 30, 300, 5, 20261127
texts = [json.loads(x)["text"] for x in a.sentences.read_text(encoding="utf-8").splitlines() if x.strip()]
pool = random.Random(SEED).sample(texts, N_WARM + N_MEASURE)
warm, meas = pool[:N_WARM], pool[N_WARM:]


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def rss_mb():
    with open("/proc/self/status") as f:
        for ln in f:
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) / 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


st0 = ner.ner_status()
if not st0.get("active"):
    sys.exit("NER 엔진이 active 가 아니다: " + json.dumps(st0))
for t in warm:
    ner._cache.clear()
    ner.detect_with_ner(t, ())


def one_pass():
    lat = []
    lock = threading.Lock()

    def worker(chunk):
        for t in chunk:
            t0 = time.perf_counter()
            ner.detect_with_ner(t, ())
            d = (time.perf_counter() - t0) * 1000
            with lock:
                lat.append(d)

    conc = max(1, a.concurrency)
    ner._cache.clear()
    chunks = [meas[i::conc] for i in range(conc)]
    ths = [threading.Thread(target=worker, args=(c,)) for c in chunks]
    w0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - w0
    return {"p50_ms": pct(lat, 0.5), "p95_ms": pct(lat, 0.95), "mean_ms": sum(lat) / len(lat), "throughput_per_s": len(lat) / wall}


passes = []
for _ in range(N_REPEAT):
    passes.append(one_pass())
res = {"status": st0, "rss_mb_after": round(rss_mb(), 1), "p50_ms": st.median(p["p50_ms"] for p in passes), "p95_ms": st.median(p["p95_ms"] for p in passes),
       "mean_ms": st.median(p["mean_ms"] for p in passes), "throughput_per_s": st.median(p["throughput_per_s"] for p in passes),
       "model_size_mb": round(os.path.getsize(os.path.join(os.environ["PRIVACY_NER_MODEL_DIR"], "model.onnx")) / 2**20, 3), "passes": passes,
       "conditions": {"threads": a.threads, "concurrency": a.concurrency, "warmup": N_WARM, "measured": N_MEASURE, "repeats": N_REPEAT, "seed": SEED,
                      "onnxruntime": ort.__version__, "python": platform.python_version(), "cpu_count_visible": os.cpu_count()}}
a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("rss_mb_after", "p50_ms", "p95_ms", "mean_ms", "throughput_per_s", "model_size_mb")}))
