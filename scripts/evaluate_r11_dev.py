"""R11 개발 평가(기준선 재추론 포함). 사전 등록 NER_R11_PREREGISTRATION.md §5-5, §7. 독립 평가셋·R9 봉인셋은 열지 않는다(경로 없음).

모델 하나마다: R9 회귀 5세트·오탐 4집합(`evaluate_r8_dev.run_model`, 1글자 이름 행 제외 패치 포함), R11 Development(`r11_dev`, 유형별·쌍 지표),
R11 Validation(`r11_val`, 쌍 지표, 참고), KLUE 검증(공식 채점)을 같은 코드로 채점한다. 결과 JSON 에는 문장 원문을 싣지 않는다.

사용: python scripts/evaluate_r11_dev.py --model LABEL DIR [FP12] ... --out result.json
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate_r9_dev  # noqa: E402,F401  (임포트만으로 1글자 이름 제외 패치가 적용된다)
import evaluate_r8_dev as D  # noqa: E402
import r8_eval_lib as lib  # noqa: E402
import r11_eval_lib as r11  # noqa: E402

DATA = Path(r"C:\Users\<user>\venvs\ner-runs\data_r11")
R11_DEV = DATA / "r11_dev.jsonl"
R11_VAL = DATA / "r11_val.jsonl"
KLUE_VAL = Path(r"C:\Users\<user>\venvs\ner-runs\data_r9\klue_val.jsonl")
MIN_AVAIL_MB = 3072


def rows_of(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_model(label, model_dir, fp12):
    out = D.run_model(label, model_dir, fp12, False)  # 모델 지문 확인 + R9 회귀 세트
    dev = rows_of(R11_DEV)
    out["r11dev"] = r11.score_dataset(dev)
    val = rows_of(R11_VAL)
    for r in val:
        r["type"] = "D1P" if r["pair_role"] == "person" else "D1"
    out["r11val"] = r11.score_dataset(val)
    klue = rows_of(KLUE_VAL)
    out["klue_val"] = lib.score_rows(klue)  # 공식 채점(정답 없는 문장의 FP 제외)
    out["klue_val_r11"] = r11.score_rows_r11(klue)
    out["targets_macro_exposure"] = lib.macro(
        [out["r11dev"]["by_type"][t]["total_exposure_rate"] for t in ("D1P", "B1", "C2", "NEWNAME", "LONGNAME")])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", nargs="+", action="append", default=[], metavar=("LABEL", "DIR"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import psutil

    res = json.loads(a.out.read_text(encoding="utf-8")) if a.out.exists() else {}
    res.setdefault("_meta", {"eval_sets": {"r11_dev": sha256(R11_DEV), "r11_val": sha256(R11_VAL), "klue_val": sha256(KLUE_VAL)},
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
        m = res[label]["r11dev"]
        print(f"[{label}] Recall {m['ALL']['exact_recall']:.4f} 가림 {m['ALL']['mask_rate']:.4f} 노출 {m['ALL']['total_exposure_rate']:.4f} "
              f"pair {m['pairs']['pair_context_accuracy']:.3f} D1P노출 {m['by_type']['D1P']['total_exposure_rate']:.3f} 오탐매크로 {m['neg_fpr_macro']:.3f} "
              f"identity {m['identity_ok_all']} | R9dev {res[label]['macro']['total_exposure']:.4f} | KLUE F1 {res[label]['klue_val']['prf']['f1']:.4f}", flush=True)
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
