"""R12 개발 평가. 사전 등록 NER_R12_PREREGISTRATION.md §4. 독립 평가셋·R11 봉인셋 경로는 이 파일에 없다.

모델 하나마다 R11 개발 평가(`evaluate_r11_dev.run_model`: R9 회귀·r11_dev·r11_val·KLUE 검증, 같은 코드)에 더해 R12 개발(`r12_dev`)·검증(`r12_val`)을 `score_dataset` 로 채점한다.
R12 음성 유형 `FAM_NEG`(가족 채팅 음성)의 오탐률은 `r12_neg_fpr` 로 따로 낸다. 결과 JSON 에는 문장 원문을 싣지 않는다.

사용: python scripts/evaluate_r12_dev.py --model LABEL DIR [FP12] ... --out result.json
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate_r11_dev as E11  # noqa: E402
import r11_eval_lib as r11  # noqa: E402

D12 = Path(r"C:\Users\<user>\venvs\ner-runs\data_r12")
R12_DEV = D12 / "r12_dev.jsonl"
R12_VAL = D12 / "r12_val.jsonl"
MIN_AVAIL_MB = 3072


def run_model(label, model_dir, fp12):
    out = E11.run_model(label, model_dir, fp12)
    for name, path in (("r12dev", R12_DEV), ("r12val", R12_VAL)):
        rows = E11.rows_of(path)
        sc = r11.score_dataset(rows)
        neg = sc["by_type"].get("FAM_NEG")
        sc["r12_neg_fpr"] = neg["nonperson_fpr"] if neg else None
        out[name] = sc
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", nargs="+", action="append", default=[], metavar=("LABEL", "DIR"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import psutil

    res = json.loads(a.out.read_text(encoding="utf-8")) if a.out.exists() else {}
    res.setdefault("_meta", {"eval_sets": {"r12_dev": hashlib.sha256(R12_DEV.read_bytes()).hexdigest(), "r12_val": hashlib.sha256(R12_VAL.read_bytes()).hexdigest(),
                                           "r11_dev": E11.sha256(E11.R11_DEV), "r11_val": E11.sha256(E11.R11_VAL), "klue_val": E11.sha256(E11.KLUE_VAL)},
                             "git_head": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()})
    for spec in a.model:
        label, d = spec[0], spec[1]
        fp = spec[2] if len(spec) > 2 else None
        if label in res:
            print("SKIP(이미 있음)", label, flush=True)
            continue
        avail = psutil.virtual_memory().available / 2**20
        if avail < MIN_AVAIL_MB:
            print(f"STOP: 가용 RAM {avail:.0f}MB < {MIN_AVAIL_MB}MB, 중단(재시도 없음)", flush=True)
            sys.exit(3)
        res[label] = run_model(label, d, fp)
        m, m11 = res[label]["r12dev"], res[label]["r11dev"]
        print(f"[{label}] r12dev 노출 {m['ALL']['total_exposure_rate']:.4f} D1P노출 {m['by_type']['D1P']['total_exposure_rate']:.3f} Recall {m['ALL']['exact_recall']:.4f} "
              f"pair {m['pairs']['pair_context_accuracy']:.3f} D1오탐 {m['D1_fpr']:.3f} NEG {m['r12_neg_fpr']:.3f} | r11dev 노출 {m11['ALL']['total_exposure_rate']:.4f} "
              f"오탐매크로 {m11['neg_fpr_macro']:.3f} | KLUE F1 {res[label]['klue_val']['prf']['f1']:.4f}", flush=True)
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
