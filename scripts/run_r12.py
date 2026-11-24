"""R12 학습 파이프라인(5시드, 단일 구성). 사전 등록 NER_R12_PREREGISTRATION.md §2·§4·§6.

학습 조건은 R11 B군과 같다(max_len 128, batch 32, lr 5e-5, 최대 8에폭, patience 2, warmup 0.1, weight decay 0.01, 시드 7·13·21·42·99).
데이터 = R11 B군 그대로 + 신규 `r12_extra`. 체크포인트 선택은 `r12_val` 겹침 F1 (baseline_val·synth_val 은 모니터링). SWA·가중치 평균·B-new 는 쓰지 않는다.
런마다: 학습 -> best ONNX 내보내기 -> 추론 일치 -> 개발 평가(`evaluate_r12_dev.py`) -> 해시 기록. 실패하면 자동 재시작 없이 중단한다.
독립 평가셋·R11 봉인셋 경로는 이 파일에 없다. 기본은 드라이런이고 실제 학습은 `--execute` 가 있어야 한다.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PY = r"C:\Users\<user>\venvs\gabee-ner\Scripts\python.exe"
API_PY = r"C:\Users\<user>\Documents\gabee\apps\api\.venv\Scripts\python.exe"
ROOT = Path(r"C:\Users\<user>\venvs\ner-runs")
DATA = ROOT / "data_r12"
RUNS = ROOT / "r12" / "runs"
EXPORT = ROOT / "r12" / "export"
DEVOUT = ROOT / "r12" / "dev"
SEEDS = [7, 13, 21, 42, 99]
BASE = ["baseline_train", "synth_train", "klue_train:469", "r9_Ffull_train", "r11_B_extra"]
EXTRA = "r12_extra"
VAL = ["r12_val", "baseline_val", "synth_val"]
MIN_AVAIL_MB = 4096
MIN_FREE_DISK_GB = 4.0  # 시드마다 가중치+ONNX 약 0.7GB 만 추가로 쓴다(R11 평가 폴더는 만들지 않는다)
ENV = {**os.environ, "HF_HOME": r"C:\Users\<user>\venvs\hf-cache", "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def train_cmd(seed):
    run_id = f"r12-s{seed}"
    cmd = [PY, str(REPO / "ml/ner/train/train_ner.py"), "--model", "monologg/koelectra-small-v3-discriminator", "--run-id", run_id,
           "--train-files", *BASE, EXTRA, "--val-files", *VAL, "--klue-ratio", "0", "--max-len", "128", "--batch-size", "32", "--lr", "5e-5",
           "--epochs", "8", "--patience", "2", "--warmup", "0.1", "--weight-decay", "0.01", "--seed", str(seed), "--select-on", "r12_val",
           "--select-metric", "overlap", "--data-dir", str(DATA), "--out-root", str(RUNS)]
    return run_id, cmd


def gpu_info():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False).stdout.strip()
        n, u, t = [x.strip() for x in q.split(",")]
        return {"name": n, "used_mb": int(u), "total_mb": int(t)}
    except Exception:  # noqa: BLE001
        return None


def preflight():
    import psutil

    chk = {"avail_ram_mb": round(psutil.virtual_memory().available / 2**20)}
    chk["avail_ram_ok"] = chk["avail_ram_mb"] >= MIN_AVAIL_MB
    free = shutil.disk_usage("C:/").free / 2**30
    chk["free_disk_gb"] = round(free, 1)
    chk["disk_ok"] = free >= MIN_FREE_DISK_GB
    chk["gpu"] = gpu_info()
    chk["gpu_ok"] = bool(chk["gpu"]) and chk["gpu"]["used_mb"] < 0.6 * chk["gpu"]["total_mb"]
    need = [f"{f.split(':')[0]}.jsonl" for f in BASE + [EXTRA] + VAL] + ["r12_dev.jsonl"]
    chk["data_files_present"] = all((DATA / n).exists() and (DATA / n).stat().st_size > 0 for n in need)
    chk["data_hashes_16"] = {n.split(".")[0]: sha(DATA / n)[:16] for n in need if (DATA / n).exists()}
    chk["prereg_tag_present"] = bool(subprocess.run(["git", "tag", "-l", "r12-prereg-1.0"], capture_output=True, text=True, cwd=str(REPO)).stdout.strip())
    must = ("avail_ram_ok", "disk_ok", "gpu_ok", "data_files_present", "prereg_tag_present")
    chk["failed_checks"] = [k for k in must if not chk[k]]
    chk["ok"] = not chk["failed_checks"]
    return chk


def step(name, cmd, run_id, timing, monitor=False, cwd=None):
    import psutil

    log = RUNS / "logs"
    log.mkdir(parents=True, exist_ok=True)
    t0, peak = time.time(), {"rss": 0}
    with open(log / f"{run_id}.{name}.out.log", "w", encoding="utf-8") as o, open(log / f"{run_id}.{name}.err.log", "w", encoding="utf-8") as e:
        p = subprocess.Popen(cmd, cwd=cwd or str(REPO), stdout=o, stderr=e, env=ENV)
        if monitor:
            ps = psutil.Process(p.pid)
            while p.poll() is None:
                try:
                    peak["rss"] = max(peak["rss"], ps.memory_info().rss + sum(c.memory_info().rss for c in ps.children(recursive=True)))
                except psutil.Error:
                    pass
                time.sleep(2)
        else:
            p.wait()
    timing[name + "_s"] = round(time.time() - t0, 1)
    if monitor:
        timing[name + "_peak_rss_mb"] = round(peak["rss"] / 2**20)
    if p.returncode != 0:
        print(json.dumps({"failed": run_id, "stage": name, "returncode": p.returncode, "log": str(log / f"{run_id}.{name}.err.log")}, ensure_ascii=False), flush=True)
        sys.exit(p.returncode)  # 자동 재시작 없음


def one(seed, resume=False):
    import psutil

    run_id, tcmd = train_cmd(seed)
    run = RUNS / run_id
    if (run / "pipeline_done.json").exists():
        print("SKIP(완료)", run_id, flush=True)
        return
    avail = round(psutil.virtual_memory().available / 2**20)
    free_gb = shutil.disk_usage("C:/").free / 2**30
    g = gpu_info()
    print(f"PRE {run_id} avail_ram_mb={avail} free_disk_gb={free_gb:.1f} gpu={g}", flush=True)
    if avail < MIN_AVAIL_MB or free_gb < MIN_FREE_DISK_GB or (g and g["used_mb"] >= 0.6 * g["total_mb"]):
        print(json.dumps({"stopped_before": run_id, "reason": "ram/disk/GPU gate", "avail_mb": avail, "free_disk_gb": round(free_gb, 1), "gpu": g}), flush=True)
        sys.exit(3)
    timing, t_all = {"avail_ram_mb_before": avail, "free_disk_gb_before": round(free_gb, 1)}, time.time()
    if run.exists():
        tlp = run / "train_log.json"
        done = tlp.exists() and json.loads(tlp.read_text(encoding="utf-8")).get("finished") and (run / "best" / "model.safetensors").exists()
        if not (resume and done):
            print(json.dumps({"stopped": "런 폴더가 이미 있음: 덮어쓰지 않는다(재학습 방지). 학습이 끝난 런만 --resume 으로 이어간다", "run": run_id}, ensure_ascii=False), flush=True)
            sys.exit(4)
        print("RESUME: 학습 완료 런 재사용(재학습 없음)", run_id, flush=True)
        timing["resumed_after_train"] = True
        if (EXPORT / run_id).exists():
            shutil.rmtree(EXPORT / run_id)  # 이 런의 부분 내보내기 잔여물만 지운다
    else:
        step("train", tcmd, run_id, timing, monitor=True)
    step("evalfull", [PY, "-X", "utf8", str(REPO / "ml/ner/train/evaluate_full.py"), "--run-dir", str(run), "--data-dir", str(DATA), "--skip-latency"], run_id, timing)
    out_best = EXPORT / run_id
    step("export_best", [PY, "-X", "utf8", str(REPO / "ml/ner/export/export_onnx.py"), "--run-dir", str(run), "--out", str(out_best), "--no-int8"], run_id, timing)
    step("parity", [PY, "-X", "utf8", str(REPO / "ml/ner/train/pilot_checks_r9.py"), "--run-dir", str(run), "--onnx-dir", str(out_best), "--out", str(run / "parity.json"), "--skip-batch64"], run_id, timing)
    DEVOUT.mkdir(parents=True, exist_ok=True)
    step("devEval", [API_PY, "-X", "utf8", str(REPO / "apps/api/scripts/evaluate_r12_dev.py"), "--model", run_id, str(out_best), "--out", str(DEVOUT / f"{run_id}.json")], run_id, timing, cwd=str(REPO / "apps/api"))
    tl = json.loads((run / "train_log.json").read_text(encoding="utf-8"))
    par = json.loads((run / "parity.json").read_text(encoding="utf-8"))
    rec = {"run_id": run_id, "seed": seed, "stopped_epoch": tl["stopped_epoch"], "best_epoch": tl["best_epoch"], "train_sentences": tl["train_sentences"],
           "parity": {k: par.get(k) for k in ("checkpoint_reload_identical_predictions", "onnx_vs_torch_token_agreement_pct")},
           "train_time_s": round(tl["total_train_time_s"], 1), "max_vram_mb": round(tl["max_vram_mb"]), "timing": timing, "total_pipeline_s": round(time.time() - t_all, 1),
           "sha256": {"best/model.safetensors": sha(run / "best/model.safetensors"), "train_log.json": sha(run / "train_log.json"), "parity.json": sha(run / "parity.json"),
                      "dev_result.json": sha(DEVOUT / f"{run_id}.json"), "onnx": sha(out_best / "model.onnx"),
                      "train_data_files": {f.split(":")[0]: sha(DATA / (f.split(":")[0] + ".jsonl")) for f in BASE + [EXTRA]}}}
    (run / "pipeline_done.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"DONE {run_id} stop_ep={rec['stopped_epoch']} best_ep={rec['best_epoch']} total={rec['total_pipeline_s']}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--execute", action="store_true", help="실제 학습")
    ap.add_argument("--resume", action="store_true", help="학습이 끝난 런 폴더를 재사용하고 이후 단계부터 이어간다(재학습 없음)")
    a = ap.parse_args()
    chk = preflight()
    print(json.dumps(chk, ensure_ascii=False, indent=1))
    for s in a.seeds:
        rid, cmd = train_cmd(s)
        print(("SKIP(완료) " if (RUNS / rid / "pipeline_done.json").exists() else "RUN ") + rid)
        if not a.execute:
            print("   ", " ".join(f'"{c}"' if " " in c else c for c in cmd[1:]))
    if not a.execute:
        print("드라이런: 학습을 실행하지 않았습니다.")
        return
    if not chk["ok"]:
        print("사전 점검 실패: 중단(재시도 없음)")
        sys.exit(2)
    for s in a.seeds:
        one(s, a.resume)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
