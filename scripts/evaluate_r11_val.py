"""R11 Validation(`r11_val`) 평가: B 군 에폭 후보 선정 전용(사전 등록 NER_R11_PREREGISTRATION.md §6). 개발셋·독립 평가셋은 열지 않는다.

사용: python scripts/evaluate_r11_val.py --model ep1 DIR [FP12] ... --out val_by_epoch.json   (라벨 ep<k> 는 r11_derive.py select 가 읽는다)
결과: 후보별 {pair_context_accuracy, exact_recall, person_total_exposure, n_pairs} (문장 원문 없음)
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r8_eval_lib as lib  # noqa: E402
import r11_eval_lib as r11  # noqa: E402

R11_VAL = Path(r"C:\Users\<user>\venvs\ner-runs\data_r11\r11_val.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", nargs="+", action="append", default=[], metavar=("LABEL", "DIR"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(x) for x in R11_VAL.read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        r["type"] = "D1P" if r["pair_role"] == "person" else "D1"
    res = json.loads(a.out.read_text(encoding="utf-8")) if a.out.exists() else {}
    res.setdefault("_meta", {"r11_val_sha256": hashlib.sha256(R11_VAL.read_bytes()).hexdigest()})
    for spec in a.model:
        label, d = spec[0], spec[1]
        if label in res:
            continue
        lib.set_model(d, spec[2] if len(spec) > 2 else None)
        s = r11.score_dataset(rows)
        res[label] = {"pair_context_accuracy": s["pairs"]["pair_context_accuracy"], "exact_recall": s["ALL"]["exact_recall"],
                      "person_total_exposure": s["by_type"]["D1P"]["total_exposure_rate"], "n_pairs": s["pairs"]["n_pairs"],
                      "d1_fpr": s["D1_fpr"]}
        print(label, json.dumps(res[label]), flush=True)
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
