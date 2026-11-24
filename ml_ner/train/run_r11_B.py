"""R11 B군 학습 파이프라인(5시드). 사전 등록 NER_R11_PREREGISTRATION.md §3·§6·§8.

기본은 **드라이런**(명령 출력과 사전 점검만). 실제 학습은 `--execute` 가 있어야 하며 사용자의 별도 최종 승인(승인 D) 전에는 실행하지 않는다.
런마다: 학습(`--save-every-epoch`) -> evalfull -> best(=B-old) ONNX -> 추론 일치 -> 에폭 후보·가중치 평균 폴더(`r11_derive.py cands`) -> 후보 ONNX 내보내기
-> Validation 평가(`evaluate_r11_val.py`) -> B-new 선정(`r11_derive.py select`) -> 개발 평가(B-old, B-new, C-SWA; 같은 ONNX 는 한 번만) -> 해시 기록.
학습 조건은 R10 과 같다(max_len 128, batch 32, lr 5e-5, 최대 8에폭, patience 2, baseline_val 겹침 F1 선택, 시드 7·13·21·42·99). 실패하면 자동 재시작 없이 중단한다.
가용 RAM 4,096MB 미만이면 런을 시작하지 않는다(재시도 없음). 독립 평가셋·R9 봉인셋 경로는 이 파일에 없다.
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
DATA = ROOT / "data_r11"
RUNS = ROOT / "r11" / "runs"
EXPORT = ROOT / "r11" / "export"
DEVOUT = ROOT / "r11" / "dev"
SEEDS = [7, 13, 21, 42, 99]
BASE = ["baseline_train", "synth_train", "klue_train:469", "r9_Ffull_train"]
EXTRA = "r11_B_extra"
VAL = ["baseline_val", "synth_val"]  # 선택은 baseline_val(B-old 규칙), synth_val 은 모니터링
MIN_AVAIL_MB = 4096
MIN_FREE_DISK_GB = 6.0
EXPECT_ROWS, EXPECT_STEPS = 9915, 310
ENV = {**os.environ, "HF_HOME": r"C:\Users\<user>\venvs\hf-cache", "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def train_cmd(seed):
    run_id = f"r11-B-s{seed}"
    cmd = [PY, str(REPO / "ml/ner/train/train_ner.py"), "--model", "monologg/koelectra-small-v3-discriminator", "--run-id", run_id,
           "--train-files", *BASE, EXTRA, "--val-files", *VAL, "--klue-ratio", "0", "--max-len", "128", "--batch-size", "32", "--lr", "5e-5",
           "--epochs", "8", "--patience", "2", "--warmup", "0.1", "--weight-decay", "0.01", "--seed", str(seed), "--select-on", "baseline_val",
           "--select-metric", "overlap", "--data-dir", str(DATA), "--out-root", str(RUNS), "--save-every-epoch"]
    return run_id, cmd


def gpu_info():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False).stdout.strip()
        n, u, t = [x.strip() for x in q.split(",")]
        return {"name": n, "used_mb": int(u), "total_mb": int(t)}
    except Exception:  # noqa: BLE001
        return None


def preflight():
    """학습 전 점검(학습·평가 실행 없음). 결과 dict 와 통과 여부."""
    import psutil

    chk = {}
    chk["avail_ram_mb"] = round(psutil.virtual_memory().available / 2**20)
    chk["avail_ram_ok"] = chk["avail_ram_mb"] >= MIN_AVAIL_MB
    free = shutil.disk_usage("C:/").free / 2**30
    chk["free_disk_gb"] = round(free, 1)
    chk["disk_ok"] = free >= MIN_FREE_DISK_GB
    chk["gpu"] = gpu_info()
    chk["gpu_ok"] = bool(chk["gpu"]) and chk["gpu"]["used_mb"] < 0.6 * chk["gpu"]["total_mb"]
    rep = json.loads((DATA / "r11_B_assembly_report.json").read_text(encoding="utf-8"))
    chk["assembly_total_rows"] = rep["rows"]["total"]
    chk["assembly_ok"] = rep["rows"]["total"] == EXPECT_ROWS and rep["steps_per_epoch"] == EXPECT_STEPS and rep["klue_share_pct"] == 4.73 and rep["relabeled_rows_in_training"] == 292
    chk["extra_hash_matches_report"] = sha(DATA / f"{EXTRA}.jsonl") == rep["r11_B_extra_sha256"]
    chk["r9f_copies_identical"] = all(v["copy_identical"] for v in rep["r9f_files_copied"].values())
    chk["no_eval_overlap"] = rep["extra_sentences_equal_to_r11_val_dev"] == 0 and rep["base_sentences_equal_to_r11_val_dev"] == 0
    # 평가셋·A군 정합성(해시)
    chk["eval_set_hashes_recorded"] = {"r11_dev": sha(DATA / "r11_dev.jsonl")[:16], "r11_val": sha(DATA / "r11_val.jsonl")[:16]}
    chk["train_ner_has_save_every_epoch"] = "--save-every-epoch" in (REPO / "ml/ner/train/train_ner.py").read_text(encoding="utf-8")
    chk["steps_per_epoch_expected"] = EXPECT_STEPS
    must = ("avail_ram_ok", "disk_ok", "gpu_ok", "assembly_ok", "extra_hash_matches_report", "r9f_copies_identical", "no_eval_overlap", "train_ner_has_save_every_epoch")
    chk["failed_checks"] = [k for k in must if not chk[k]]
    chk["ok"] = not chk["failed_checks"]
    return chk


def step(name, cmd, run_id, timing, monitor=False, cwd=None):
    import psutil

    log = RUNS / "logs"
    log.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    peak = {"rss": 0}
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


def stage_marker(run, name):
    return run / "stages" / f"{name}.done.json"


def mark_stage(run, name, outputs):
    """단계가 정상 종료됐을 때만 출력 파일 해시를 남긴다(재개 시 해시 재확인용)."""
    d = {"stage": name, "outputs_sha256": {k: sha(v) for k, v in outputs.items()}}
    stage_marker(run, name).parent.mkdir(parents=True, exist_ok=True)
    stage_marker(run, name).write_text(json.dumps(d, indent=1), encoding="utf-8")


def stage_done(run, name, outputs):
    """표식이 있고 모든 출력이 표식의 해시와 같을 때만 완료로 본다."""
    m = stage_marker(run, name)
    if not m.exists():
        return False
    try:
        rec = json.loads(m.read_text(encoding="utf-8"))["outputs_sha256"]
    except (OSError, ValueError, KeyError):
        return False
    return set(rec) == set(outputs) and all(Path(v).exists() and sha(v) == rec[k] for k, v in outputs.items())


def fail(msg, code=5, **kw):
    print(json.dumps({"stopped": msg, **kw}, ensure_ascii=False), flush=True)
    sys.exit(code)  # 자동 재시도 없음


def verify_resume_base(run, run_id, seed):
    """재개 전 완료 단계(train, evalfull, export_best, parity, cands)의 산출물을 해시·내용으로 확인한다. 문제 목록(비면 정상)."""
    sys.path.insert(0, str(HERE))
    import r11_derive

    pr = []
    try:
        tl = json.loads((run / "train_log.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"train_log.json 읽기 실패: {e}"]
    cfg = tl.get("config", {})
    if cfg.get("seed") != seed or cfg.get("run_id") != run_id or not cfg.get("save_every_epoch"):
        pr.append("train_log 설정(시드·run_id·save_every_epoch)이 이 런과 다름")
    if tl.get("train_sentences") != EXPECT_ROWS or not tl.get("finished"):
        pr.append("학습 행 수가 9,915 가 아니거나 학습이 끝나지 않음")
    for k, v in tl.get("data_files", {}).items():
        f = DATA / (k.split(":")[0] + ".jsonl")
        if not f.exists() or sha(f)[:12] != v.get("sha256_12"):
            pr.append(f"학습 입력 {k} 해시가 train_log 기록과 다름(고정 데이터 불일치)")
    try:
        ev = json.loads((run / "eval_full.json").read_text(encoding="utf-8"))
        if ev.get("run_id") != run_id or "baseline_val" not in ev.get("sets", {}):
            pr.append("eval_full.json 의 run_id/세트가 맞지 않음")
    except (OSError, ValueError) as e:
        pr.append(f"eval_full.json 읽기 실패: {e}")
    try:
        par = json.loads((run / "parity.json").read_text(encoding="utf-8"))
        if not par.get("checkpoint_reload_identical_predictions") or par.get("onnx_vs_torch_token_agreement_pct", 0) < 99.9:
            pr.append("parity.json 이 기준(재로드 동일, 토큰 일치 99.9% 이상)에 미달")
    except (OSError, ValueError) as e:
        pr.append(f"parity.json 읽기 실패: {e}")
    ex = EXPORT / run_id
    for f in ("model.onnx", "model_card.json", "labels.json", "tokenizer.json"):
        if not (ex / f).exists():
            pr.append(f"export/{run_id}/{f} 누락(export_best 미완료)")
    pr += r11_derive.verify_cands(run)
    return pr


def check_cand_eval(run, name, weights_sha, best_epoch):
    """후보 eval_full.json 이 그 후보의 것인지 확인(복사 금지). best 에폭 후보는 런 전체 평가와 TP/FP/FN 이 같아야 한다."""
    cd = run / "cand" / name
    ev = json.loads((cd / "eval_full.json").read_text(encoding="utf-8"))
    if ev.get("run_id") != name:
        fail("후보 eval_full.json 의 run_id 가 후보 이름과 다름", stage=f"evalfull_{name}")
    if sha(cd / "best" / "model.safetensors") != weights_sha:
        fail("평가 중 후보 가중치가 바뀜", stage=f"evalfull_{name}")
    if name == f"ep{best_epoch}":
        top = json.loads((run / "eval_full.json").read_text(encoding="utf-8"))
        for mode in ("overlap", "exact"):
            x, y = ev["sets"]["baseline_val"][mode]["micro"], top["sets"]["baseline_val"][mode]["micro"]
            if any(x[k] != y[k] for k in ("tp", "fp", "fn")):
                fail("best 에폭 후보의 평가가 런 전체 평가와 다름(저장·평가 정합성 실패)", stage=f"evalfull_{name}", mode=mode)


def one(seed, resume=False):
    import psutil

    run_id, tcmd = train_cmd(seed)
    run = RUNS / run_id
    if (run / "pipeline_done.json").exists():
        print("SKIP(완료)", run_id, flush=True)
        return
    avail = round(psutil.virtual_memory().available / 2**20)
    print(f"PRE {run_id} avail_ram_mb={avail} gpu={gpu_info()} resume={resume}", flush=True)
    if avail < MIN_AVAIL_MB:
        print(json.dumps({"stopped_before": run_id, "reason": "available RAM below gate", "avail_mb": avail}), flush=True)
        sys.exit(3)
    free_gb = shutil.disk_usage("C:/").free / 2**30
    g = gpu_info()
    if free_gb < MIN_FREE_DISK_GB or (g and g["used_mb"] >= 0.6 * g["total_mb"]):  # 시드마다 디스크·GPU 게이트 재확인(재시도 없음)
        print(json.dumps({"stopped_before": run_id, "reason": "disk or GPU gate", "free_disk_gb": round(free_gb, 1), "gpu": g}), flush=True)
        sys.exit(3)
    timing, t_all = {"avail_ram_mb_before": avail, "free_disk_gb_before": round(free_gb, 1)}, time.time()
    reused = []
    if run.exists() and not resume:
        fail("런 폴더가 이미 있음: --resume 없이 덮어쓰지 않는다(재학습 방지)", code=4, run=run_id)
    if run.exists():
        problems = verify_resume_base(run, run_id, seed)
        if problems:
            fail("재개 전 검증 실패: 정상 완료로 처리하지 않음", run=run_id, problems=problems)
        reused = ["train", "evalfull", "export_best", "parity", "cands"]
        print("RESUME 검증 통과, 재사용:", reused, flush=True)
    else:
        step("train", tcmd, run_id, timing, monitor=True)
        step("evalfull", [PY, "-X", "utf8", str(REPO / "ml/ner/train/evaluate_full.py"), "--run-dir", str(run), "--data-dir", str(DATA), "--skip-latency"], run_id, timing)
        out_best = EXPORT / run_id
        step("export_best", [PY, "-X", "utf8", str(REPO / "ml/ner/export/export_onnx.py"), "--run-dir", str(run), "--out", str(out_best), "--no-int8"], run_id, timing)
        step("parity", [PY, "-X", "utf8", str(REPO / "ml/ner/train/pilot_checks_r9.py"), "--run-dir", str(run), "--onnx-dir", str(out_best), "--out", str(run / "parity.json"), "--skip-batch64"], run_id, timing)
        step("cands", [PY, "-X", "utf8", str(HERE / "r11_derive.py"), "cands", "--run-dir", str(run)], run_id, timing)
    info = json.loads((run / "cand_info.json").read_text(encoding="utf-8"))
    tl = json.loads((run / "train_log.json").read_text(encoding="utf-8"))

    def staged(name, cmd, outputs, cwd=None, pre=None):
        """재개 모드에서만 완료 단계를 건너뛴다. 표식은 단계가 정상 종료됐을 때만 만든다."""
        if resume and stage_done(run, name, outputs):
            reused.append(name)
            print("SKIP(재개, 해시 확인)", name, flush=True)
            return
        if pre:
            pre()
        step(name, cmd, run_id, timing, cwd=cwd)
        for v in outputs.values():
            if not Path(v).exists():
                fail("단계 출력 누락", stage=name, missing=str(v))
        mark_stage(run, name, outputs)

    for name, meta in info["candidates"].items():
        cd = run / "cand" / name
        exp = EXPORT / f"{run_id}-{name}"

        def clean(exp=exp):
            if exp.exists():  # 이 단계의 부분 산출물(예: export_ep1 실패 잔여물)만 지운다
                assert exp.parent == EXPORT and exp.name.startswith(run_id + "-")
                shutil.rmtree(exp)

        staged(f"evalfull_{name}", [PY, "-X", "utf8", str(REPO / "ml/ner/train/evaluate_full.py"), "--run-dir", str(cd), "--data-dir", str(DATA), "--skip-latency"], {"eval_full": cd / "eval_full.json"})
        check_cand_eval(run, name, meta["weights_sha256"], tl["best_epoch"])
        staged(f"export_{name}", [PY, "-X", "utf8", str(REPO / "ml/ner/export/export_onnx.py"), "--run-dir", str(cd), "--out", str(exp), "--no-int8"], {"onnx": exp / "model.onnx", "model_card": exp / "model_card.json"}, pre=clean)
    val_json = run / "val_by_epoch.json"
    margs = []
    for name in info["candidates"]:
        if name.startswith("ep"):
            margs += ["--model", name, str(EXPORT / f"{run_id}-{name}")]
    staged("valEval", [API_PY, "-X", "utf8", str(REPO / "apps/api/scripts/evaluate_r11_val.py"), *margs, "--out", str(val_json)], {"val": val_json}, cwd=str(REPO / "apps/api"))
    staged("select", [PY, "-X", "utf8", str(HERE / "r11_derive.py"), "select", "--run-dir", str(run), "--val-json", str(val_json)], {"selection": run / "selection.json"})
    sel = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    # 개발 평가: B-old(best 에폭), B-new, C-SWA. 같은 ONNX 는 한 번만 평가하고 변형 이름을 매핑해 둔다
    variants = {"B-old": f"{run_id}-ep{sel['B_old_epoch']}", "B-new": f"{run_id}-ep{sel['B_new_epoch']}", "C-SWA": f"{run_id}-swa"}
    uniq = sorted(set(variants.values()))
    dargs = []
    for u in uniq:
        dargs += ["--model", u, str(EXPORT / u)]
    DEVOUT.mkdir(parents=True, exist_ok=True)
    staged("devEval", [API_PY, "-X", "utf8", str(REPO / "apps/api/scripts/evaluate_r11_dev.py"), *dargs, "--out", str(DEVOUT / f"{run_id}.json")], {"dev": DEVOUT / f"{run_id}.json"}, cwd=str(REPO / "apps/api"))
    rec = {"run_id": run_id, "seed": seed, "stopped_epoch": tl["stopped_epoch"], "best_epoch": tl["best_epoch"], "epochs": [{"epoch": e["epoch"], "baseline_val_overlap_f1": e["val_score"], "checkpoint_sha256": e.get("checkpoint_sha256")} for e in tl["epochs"]],
           "selection": {k: v for k, v in sel.items() if k != "val_by_epoch"}, "variants": variants, "swa": info["candidates"]["swa"],
           "train_time_s": round(tl["total_train_time_s"], 1), "max_vram_mb": round(tl["max_vram_mb"]), "timing": timing, "total_pipeline_s": round(time.time() - t_all, 1),
           "resumed": resume and bool(reused), "reused_stages": reused,
           "sha256": {"best/model.safetensors": sha(run / "best/model.safetensors"), "train_log.json": sha(run / "train_log.json"), "eval_full.json": sha(run / "eval_full.json"), "parity.json": sha(run / "parity.json"),
                      "cand_info.json": sha(run / "cand_info.json"), "dev_result.json": sha(DEVOUT / f"{run_id}.json"),
                      "cand_eval_full": {n: sha(run / "cand" / n / "eval_full.json") for n in info["candidates"]},
                      "onnx": {u: sha(EXPORT / u / "model.onnx") for u in uniq}, "train_data_files": {f.split(":")[0]: sha(DATA / (f.split(":")[0] + ".jsonl")) for f in BASE + [EXTRA]}},
           "best_equals_epoch_best_checkpoint": sha(run / "best/model.safetensors") == sha(run / f"epoch_{tl['best_epoch']}/model.safetensors")}
    (run / "pipeline_done.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    time.sleep(3)
    print(f"DONE {run_id} stop_ep={rec['stopped_epoch']} best_ep={rec['best_epoch']} newsel={sel['B_new_epoch']} total={rec['total_pipeline_s']}s resumed={rec['resumed']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--execute", action="store_true", help="실제 학습(별도 최종 승인 후에만)")
    ap.add_argument("--resume", action="store_true", help="이미 학습된 런 폴더를 해시로 검증해 재사용하고 미완료 단계부터 이어간다(재학습 없음)")
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
