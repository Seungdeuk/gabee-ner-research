"""R10 전체 학습 파이프라인(조건부 승인 2026-11-20): 6개 군 x 5개 시드 = 30런. 런마다 학습 -> evaluate_full -> ONNX 내보내기 -> 추론 일치 -> 개발셋 평가 -> 해시 기록.

- 시드 우선 순서(시드 7 의 8개 군 -> 시드 13 ...). 완료 표지(`pipeline_done.json`)가 있는 런은 건너뛴다. 실패하면 자동 재시작 없이 로그를 남기고 중단한다.
- 학습 설정은 사전 등록 값 고정(8에폭, patience 2, 배치 32, lr 5e-5, KLUE 469, `--klue-ratio 0`). 선택은 baseline_val 만 사용한다. 봉인 평가셋은 사용하지 않는다.
- 단계별 시간, 학습 중 RAM(작업 집합)·GPU 메모리 최대치를 런마다 기록한다.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_r10_arm as R  # noqa: E402

PY = r"C:\Users\<user>\venvs\gabee-ner\Scripts\python.exe"
API_PY = r"C:\Users\<user>\Documents\gabee\apps\api\.venv\Scripts\python.exe"
EXPORT = Path(r"C:\Users\<user>\venvs\ner-runs\r10\export")
DEV = Path(r"C:\Users\<user>\venvs\ner-runs\r10\dev")
ARMS = ["R10-full", "C10", "R10-D", "R10-BC", "R10-N", "C10s"]
MIN_AVAIL_MB = 4096  # 런 시작 전 최소 가용 RAM(실행 안전장치, 학습 조건 아님). 측정된 학습 최대 RSS 약 1.5GB, 2GB 기준은 보호 중단을 막지 못했다
ENV = {
    **os.environ,
    "HF_HOME": r"C:\Users\<user>\venvs\hf-cache",
    "HF_HUB_OFFLINE": "1",
    "PYTHONIOENCODING": "utf-8",
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gpu_mb():
    try:
        return int(
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.split()[0]
        )
    except Exception:
        return 0


def step(name, cmd, run_id, timing, monitor=False, cwd=None):
    log = R.RUNS / "logs"
    t0 = time.time()
    peak = {"rss": 0, "gpu": 0}
    with (
        open(log / f"{run_id}.{name}.out.log", "w", encoding="utf-8") as o,
        open(log / f"{run_id}.{name}.err.log", "w", encoding="utf-8") as e,
    ):
        p = subprocess.Popen(cmd, cwd=cwd or str(R.REPO), stdout=o, stderr=e, env=ENV)
        if monitor:
            ps = psutil.Process(p.pid)
            while p.poll() is None:
                try:
                    peak["rss"] = max(
                        peak["rss"],
                        ps.memory_info().rss
                        + sum(c.memory_info().rss for c in ps.children(recursive=True)),
                    )
                except psutil.Error:
                    pass
                peak["gpu"] = max(peak["gpu"], gpu_mb())
                time.sleep(2)
        else:
            p.wait()
    timing[name + "_s"] = round(time.time() - t0, 1)
    if monitor:
        timing[name + "_peak_rss_mb"] = round(peak["rss"] / 2**20)
        timing[name + "_peak_gpu_used_mb(system)"] = peak["gpu"]
    if p.returncode != 0:
        print(
            json.dumps(
                {
                    "failed": run_id,
                    "stage": name,
                    "returncode": p.returncode,
                    "log": str(log / f"{run_id}.{name}.err.log"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        sys.exit(p.returncode)


def one(arm, seed):
    run_id, train_cmd = R.command(arm, seed, False)
    run_dir = R.RUNS / run_id
    if (run_dir / "pipeline_done.json").exists():
        print("SKIP(완료)", run_id, flush=True)
        return
    (R.RUNS / "logs").mkdir(parents=True, exist_ok=True)
    # 실행 안전장치(학습 조건 아님, 2026-11-20 재시작 승인): 런 시작 전 가용 RAM 확인. 부족하면 재시도 없이 중단한다.
    avail_mb = round(psutil.virtual_memory().available / 2**20)
    print(f"PRE {run_id} avail_ram_mb={avail_mb} gpu_used_mb={gpu_mb()}", flush=True)
    if avail_mb < MIN_AVAIL_MB:
        print(json.dumps({"stopped_before": run_id, "reason": "available RAM below gate", "avail_mb": avail_mb, "gate_mb": MIN_AVAIL_MB}, ensure_ascii=False), flush=True)
        sys.exit(3)
    timing = {"avail_ram_mb_before": avail_mb}
    t_all = time.time()
    step("train", train_cmd, run_id, timing, monitor=True)
    step(
        "evalfull",
        [
            PY,
            "-X",
            "utf8",
            str(R.REPO / "ml/ner/train/evaluate_full.py"),
            "--run-dir",
            str(run_dir),
            "--data-dir",
            R.DATA_R9,
            "--skip-latency",
        ],
        run_id,
        timing,
    )
    out = EXPORT / run_id
    step(
        "export",
        [
            PY,
            "-X",
            "utf8",
            str(R.REPO / "ml/ner/export/export_onnx.py"),
            "--run-dir",
            str(run_dir),
            "--out",
            str(out),
            "--no-int8",
        ],
        run_id,
        timing,
    )
    parity = run_dir / "parity.json"
    step(
        "parity",
        [
            PY,
            "-X",
            "utf8",
            str(R.REPO / "ml/ner/train/pilot_checks_r9.py"),
            "--run-dir",
            str(run_dir),
            "--onnx-dir",
            str(out),
            "--out",
            str(parity),
            "--skip-batch64",
        ],
        run_id,
        timing,
    )
    DEV.mkdir(parents=True, exist_ok=True)
    step(
        "devEval",
        [
            API_PY,
            "-X",
            "utf8",
            str(R.REPO / "apps/api/scripts/evaluate_r10_dev.py"),
            "--model",
            run_id,
            str(out),
            "--out",
            str(DEV / f"{run_id}.json"),
        ],
        run_id,
        timing,
        cwd=str(R.REPO / "apps/api"),
    )
    tl = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))
    pj = json.loads(parity.read_text(encoding="utf-8"))
    rec = {
        "run_id": run_id,
        "arm": arm,
        "seed": seed,
        "stopped_epoch": tl["stopped_epoch"],
        "best_epoch": tl["best_epoch"],
        "best_val_baseline_overlap_f1": tl["best_val_score"],
        "epochs": [
            {
                "epoch": e["epoch"],
                "baseline_val_overlap_f1": e["val_score"],
                "elapsed_s": round(e["elapsed_s"], 1),
            }
            for e in tl["epochs"]
        ],
        "steps_per_epoch_x_epochs_run": (tl["step_time_s"]["steps"]),
        "step_time_ms": {
            k: round(v * 1000, 1) for k, v in tl["step_time_s"].items() if k != "steps"
        },
        "train_time_s": round(tl["total_train_time_s"], 1),
        "max_vram_mb": round(tl["max_vram_mb"]),
        "timing": timing,
        "total_pipeline_s": round(time.time() - t_all, 1),
        "parity": {
            "onnx_vs_torch_token_agreement_pct": pj[
                "onnx_vs_torch_token_agreement_pct"
            ],
            "identical_sentences": pj["onnx_vs_torch_identical_sentences"],
            "checkpoint_reload_identical": pj[
                "checkpoint_reload_identical_predictions"
            ],
        },
        "sha256": {
            "best/model.safetensors": sha(run_dir / "best/model.safetensors"),
            "export/model.onnx": sha(out / "model.onnx"),
            "train_log.json": sha(run_dir / "train_log.json"),
            "dev_result.json": sha(DEV / f"{run_id}.json"),
            "train_data_files": {
                f.split(":")[0]: sha(Path(R.DATA_R9) / (f.split(":")[0] + ".jsonl"))
                for f in R.BASE
                + ([R.EXTRA[arm].format(seed=seed)] if R.EXTRA[arm] else [])
            },
        },
    }
    (run_dir / "pipeline_done.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    time.sleep(3)
    print(f"POST {run_id} avail_ram_mb={round(psutil.virtual_memory().available / 2**20)} gpu_used_mb={gpu_mb()}", flush=True)
    print(
        f"DONE {run_id} stop_ep={rec['stopped_epoch']} best_ep={rec['best_epoch']} val={rec['best_val_baseline_overlap_f1']:.4f} train={rec['train_time_s']}s total={rec['total_pipeline_s']}s",
        flush=True,
    )


def main():
    t0 = time.time()
    for seed in R.SEEDS:
        for arm in ARMS:
            one(arm, seed)
    print("ALL_DONE", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
