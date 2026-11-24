"""파일럿 RAM 측정용: 승인된 CP/시드7/2에폭 설정을 run-id 만 바꿔 한 번 더 돌리며 프로세스 작업 집합(RSS)·GPU 메모리를 샘플링하고,
같은 시드에서 결과가 재현되는지(개발 점수 비교)도 본다. 연구 경로 전용."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_r9_arm as R

run_id, cmd = R.command("CP", 7, True)
cmd = [c if c != run_id else run_id + "-ramprobe" for c in cmd]
env = {
    **os.environ,
    "HF_HOME": r"C:\Users\<user>\venvs\hf-cache",
    "HF_HUB_OFFLINE": "1",
    "PYTHONIOENCODING": "utf-8",
}
log = R.RUNS / "logs"
with (
    open(log / "ramprobe.out.log", "w", encoding="utf-8") as o,
    open(log / "ramprobe.err.log", "w", encoding="utf-8") as e,
):
    p = subprocess.Popen(cmd, cwd=str(R.REPO), stdout=o, stderr=e, env=env)
    ps = psutil.Process(p.pid)
    peak, t0 = 0, time.time()
    while p.poll() is None:
        try:
            peak = max(
                peak,
                ps.memory_info().rss
                + sum(c.memory_info().rss for c in ps.children(recursive=True)),
            )
        except psutil.Error:
            pass
        time.sleep(1)
a = json.loads(
    (R.RUNS / (run_id + "-ramprobe") / "train_log.json").read_text(encoding="utf-8")
)
b = json.loads((R.RUNS / run_id / "train_log.json").read_text(encoding="utf-8"))
print(
    json.dumps(
        {
            "returncode": p.returncode,
            "peak_rss_mb": round(peak / 2**20),
            "wall_s": round(time.time() - t0, 1),
            "val_scores_probe": [round(x["val_score"], 5) for x in a["epochs"]],
            "val_scores_pilot": [round(x["val_score"], 5) for x in b["epochs"]],
        }
    )
)
