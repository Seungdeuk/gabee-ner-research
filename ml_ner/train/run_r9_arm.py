"""R9 실험군 학습 실행기(런 단위 재개). `train_ner.py` 는 수정하지 않고 서브프로세스로 호출한다.

- 군·시드마다 `train_log.json` 이 있으면 끝난 런으로 보고 건너뛴다(재개). 실패하면 자동 재시작하지 않고 종료 코드와 로그 위치만 남긴다(사전 등록 §3).
- 기본은 **드라이런**(명령만 출력, 학습 안 함). 실제 실행은 `--execute` 가 있어야 하며, 이번 단계에서는 사용하지 않는다(파일럿·전체 학습은 별도 승인).
- KLUE 는 `klue_train:469` 고정, `--klue-ratio 0`. 체크포인트 선택은 `--select-on baseline_val`(개발셋)만 쓴다. 봉인 평가셋 경로는 이 파일에 없다.
- 파일럿: `--pilot` 이면 epochs=2, 체크포인트 저장, 런 폴더 `pilot-*`(전체 학습 결과와 섞이지 않음).

사용: python run_r9_arm.py --arms CP F-full --seeds 7 [--pilot] [--execute]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PY = r"C:\Users\<user>\venvs\gabee-ner\Scripts\python.exe"
REPO = Path(__file__).resolve().parents[3]
DATA = {"C1": r"C:\Users\<user>\venvs\ner-runs\data_r8"}
DATA_R9 = r"C:\Users\<user>\venvs\ner-runs\data_r9"
RUNS = Path(r"C:\Users\<user>\venvs\ner-runs\r9\runs")
SEEDS = [7, 13, 21, 42, 99]
BASE = ["baseline_train", "synth_train", "klue_train:469"]
EXTRA = {
    "C1": None,
    "CP": None,
    "F-full": "r9_Ffull_train",
    "E1": "r9_E1_train",
    "E2": "r9_E2_train",
    "E3": "r9_E3_train",
    "E4": "r9_E4_train",
    "C0": "c0_extra_s{seed}",
    "C0s": "c0s_extra_s{seed}",
}
VAL = [
    "baseline_val",
    "synth_val",
    "r9_val_extra",
]  # 선택은 baseline_val, 나머지는 기록용


def command(arm, seed, pilot):
    extra = EXTRA[arm]
    train = BASE + ([extra.format(seed=seed)] if extra else [])
    run_id = ("pilot-" if pilot else "") + f"r9-{arm}-s{seed}"
    cmd = [
        PY,
        str(REPO / "ml/ner/train/train_ner.py"),
        "--model",
        "monologg/koelectra-small-v3-discriminator",
        "--run-id",
        run_id,
        "--train-files",
        *train,
        "--val-files",
        *VAL,
        "--klue-ratio",
        "0",
        "--max-len",
        "128",
        "--batch-size",
        "32",
        "--lr",
        "5e-5",
        "--epochs",
        "2" if pilot else "8",
        "--patience",
        "2",
        "--warmup",
        "0.1",
        "--weight-decay",
        "0.01",
        "--seed",
        str(seed),
        "--select-on",
        "baseline_val",
        "--select-metric",
        "overlap",
        "--data-dir",
        DATA.get(arm, DATA_R9),
        "--out-root",
        str(RUNS),
    ]
    return run_id, cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", nargs="+", default=list(EXTRA))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument(
        "--execute",
        action="store_true",
        help="실제 학습(이번 단계 금지: 별도 승인 필요)",
    )
    a = ap.parse_args()
    for arm in a.arms:
        for seed in a.seeds:
            run_id, cmd = command(arm, seed, a.pilot)
            done = (RUNS / run_id / "train_log.json").exists()
            print(("SKIP(완료) " if done else "RUN ") + run_id)
            if not a.execute:
                print("   ", " ".join(f'"{c}"' if " " in c else c for c in cmd[1:]))
                continue
            if done:
                continue
            log = RUNS / "logs"
            log.mkdir(parents=True, exist_ok=True)
            with (
                open(log / f"{run_id}.out.log", "w", encoding="utf-8") as o,
                open(log / f"{run_id}.err.log", "w", encoding="utf-8") as e,
            ):
                rc = subprocess.run(
                    cmd,
                    check=False,
                    cwd=str(REPO),
                    stdout=o,
                    stderr=e,
                    env={
                        **__import__("os").environ,
                        "HF_HOME": r"C:\Users\<user>\venvs\hf-cache",
                        "HF_HUB_OFFLINE": "1",
                        "PYTHONIOENCODING": "utf-8",
                    },
                ).returncode
            if rc != 0:
                print(
                    json.dumps(
                        {
                            "failed": run_id,
                            "returncode": rc,
                            "log": str(log / f"{run_id}.err.log"),
                        },
                        ensure_ascii=False,
                    )
                )
                sys.exit(rc)  # 자동 재시작 없음


if __name__ == "__main__":
    main()
