"""KLUE 혼합 비율 x 모델 비교 실험을 순차 실행한다(GPU 1개). 각 run 은 train_ner.py 를 서브프로세스로 호출한다.

예::

    python ml/ner/train/run_matrix.py --seeds 42            # 6 run
    python ml/ner/train/run_matrix.py --seeds 7 --only koelectra-small:0.1 roberta-small:0.0
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = {"koelectra-small": "monologg/koelectra-small-v3-discriminator", "roberta-small": "klue/roberta-small"}
RATIOS = [0.0, 0.1, 0.3]


def run_id(model, ratio, seed, extra="", prefix=""):
    """run 이름: [prefix-]<모델>-klue<퍼센트>-s<시드>[-extra]."""
    return f"{prefix}{model}-klue{round(ratio * 100)}-s{seed}{extra}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--only", nargs="*", help="모델:비율 목록(없으면 전체)")
    ap.add_argument("--extra", default="", help="run id 접미사")
    ap.add_argument("--prefix", default="", help="run id 접두사(예: v2b-)")
    ap.add_argument("--data-dir", default=None, help="build_ner_data 산출 폴더(기본 ner-runs/data)")
    ap.add_argument("--ratios", nargs="+", type=float, default=RATIOS)
    ap.add_argument("rest", nargs="*", help="train_ner.py 에 그대로 넘길 추가 인자(-- 뒤)")
    a = ap.parse_args()
    combos = [(m, r) for m in MODELS for r in a.ratios]
    if a.only:
        want = {(x.split(":")[0], float(x.split(":")[1])) for x in a.only}
        combos = [c for c in combos if c in want]
    for seed in a.seeds:
        for m, r in combos:
            cmd = [sys.executable, str(HERE / "train_ner.py"), "--model", MODELS[m], "--run-id", run_id(m, r, seed, a.extra, a.prefix),
                   "--train-files", "baseline_train", "synth_train", "--val-files", "baseline_val", "synth_val",
                   "--klue-ratio", str(r), "--max-len", "128", "--batch-size", "32", "--lr", "5e-5", "--epochs", "8",
                   "--patience", "2", "--warmup", "0.1", "--weight-decay", "0.01", "--seed", str(seed),
                   "--select-on", "baseline_val", "--select-metric", "overlap",
                   *(["--data-dir", a.data_dir] if a.data_dir else []), *a.rest]
            print(">>", " ".join(cmd), flush=True)
            rc = subprocess.run(cmd, check=False).returncode
            print(f"<< rc={rc}", flush=True)


if __name__ == "__main__":
    main()
