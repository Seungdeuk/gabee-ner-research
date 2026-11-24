"""R11 B군 5시드 결과 요약(CSV/JSON). 평가셋: r11_dev(Development, 1,920행). 자동 검증 기반 연구용 · 사람 검수 미완료.

기존 select_r11.py·aggregate_r11_baselines.py 의 지표 함수를 그대로 쓰고 새 지표 정의는 만들지 않는다. 실제 대화 문장은 출력하지 않는다.
사용: python scripts/r11_B_report.py --out-dir <폴더>
"""

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aggregate_r11_baselines import metrics  # noqa: E402

V = Path(r"C:\Users\<user>\Documents\gabee-ner\docs\security\r11_validation")
R = Path(r"C:\Users\<user>\venvs\ner-runs\r11")
TYPES = ("D1P", "B1", "C2", "NEWNAME", "LONGNAME")
SCALAR = ("exact_recall", "precision_official", "f1_official", "precision_all", "total_exposure", "partial_rate", "full_rate", "mask_rate", "targets_macro_exposure", "neg_fpr_macro", "pair_context_accuracy", "klue_f1")


def pct(x):
    return None if x is None else round(100 * x, 2)


def stat(vals):
    return {"mean": pct(st.mean(vals)), "sd_pop": pct(st.pstdev(vals)) if len(vals) > 1 else 0.0, "min": pct(min(vals)), "max": pct(max(vals)), "values": [pct(v) for v in vals]}


def row_of(m):
    r = {k: m[k] for k in SCALAR}
    for t in TYPES:
        r[f"{t}_exposure"] = m[f"type.{t}"]["total_exposure_rate"]
        r[f"{t}_recall"] = m[f"type.{t}"]["exact_recall"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    ref = json.loads((V / "r11_reference_baselines.json").read_text(encoding="utf-8"))
    sel = json.loads((V / "r11_B_results" / "selection_final.json").read_text(encoding="utf-8"))
    sens = json.loads((V / "r11_B_results" / "flagged_sensitivity_B.json").read_text(encoding="utf-8"))
    models = {}  # arm -> seed -> metrics row
    base = ref["per_model"]
    models["S0"] = {0: row_of(base["service"])}
    for arm, pre in (("R9F", "R9F"), ("C10", "C10"), ("R10-full", "R10full")):
        models[arm] = {int(k.split("-s")[1]): row_of(v) for k, v in base.items() if k.startswith(pre + "-s")}
    for arm in ("B-old", "B-new", "C-SWA"):
        models[arm] = {}
    for p in sorted((R / "runs").glob("r11-B-s*/pipeline_done.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        dev = json.loads((R / "dev" / f"{rec['run_id']}.json").read_text(encoding="utf-8"))
        for var, label in rec["variants"].items():
            models[var][rec["seed"]] = row_of(metrics(dev[label]))
    summary = {}
    for arm, seeds in models.items():
        vals = {k: [s[k] for s in seeds.values()] for k in next(iter(seeds.values()))}
        summary[arm] = {k: (stat(v) if len(v) > 1 else {"mean": pct(v[0]), "sd_pop": 0.0, "min": pct(v[0]), "max": pct(v[0]), "values": [pct(v[0])]}) for k, v in vals.items()}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = {"label": "자동 검증 기반 연구용 · 사람 검수 미완료", "evalset": "r11_dev(Development 1,920행)", "summary_pct": summary,
           "per_seed_pct": {arm: {str(s): {k: pct(v) for k, v in r.items()} for s, r in seeds.items()} for arm, seeds in models.items()},
           "gates": {v: {"group_S": x["group_S"], "per_seed": {str(s): {"label": y["label"], "failed_gates": [g for g, ok in y["gates"].items() if not ok], "G12": y["G12"], "eligible": y["eligible"]} for s, y in x["per_seed"].items()}} for v, x in sel["variants"].items()},
           "selection": sel["selection"], "sensitivity_quality_sensitive_models": {k: v for k, v in sens["quality_sensitive_models"].items() if k.split("-s")[0] in ("B-old", "B-new", "C-SWA")}}
    (a.out_dir / "r11_B_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(a.out_dir / "r11_B_per_seed.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        cols = list(next(iter(next(iter(models.values())).values())))
        w.writerow(["arm", "seed", *cols])
        for arm, seeds in models.items():
            for s, r in sorted(seeds.items()):
                w.writerow([arm, s, *[pct(r[c]) for c in cols]])
    with open(a.out_dir / "r11_B_arm_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["arm", "metric", "mean_pct", "sd_pop_pp", "min_pct", "max_pct"])
        for arm, d in summary.items():
            for k, v in d.items():
                w.writerow([arm, k, v["mean"], v["sd_pop"], v["min"], v["max"]])
    for arm, d in summary.items():
        print(f"{arm:9s} n={len(models[arm])} R {d['exact_recall']['mean']:.2f}±{d['exact_recall']['sd_pop']:.2f} (worst {d['exact_recall']['min']:.2f}) P_off {d['precision_official']['mean']:.2f} F1 {d['f1_official']['mean']:.2f} P_all {d['precision_all']['mean']:.2f} exp {d['total_exposure']['mean']:.2f} (partial {d['partial_rate']['mean']:.2f} full {d['full_rate']['mean']:.2f}) pair {d['pair_context_accuracy']['mean']:.2f} negFPR {d['neg_fpr_macro']['mean']:.2f}")
        print("          types exposure:", {t: d[f"{t}_exposure"]["mean"] for t in TYPES}, "sd", {t: d[f"{t}_exposure"]["sd_pop"] for t in TYPES})
    print("quality_sensitive(B):", out["sensitivity_quality_sensitive_models"])


if __name__ == "__main__":
    main()
