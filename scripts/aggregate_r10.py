"""R10 개발 평가 집계: 런별 결과(r10/dev/*.json)와 기준선(baselines.json)을 합쳐 군별 5시드 평균·표준편차·최악값 표와 G1~G11 판정을 만든다.

사전 등록 NER_R10_PREREGISTRATION.md §6·§11. 독립 최종 평가셋·R9 봉인셋은 열지 않는다.
사용: python scripts/aggregate_r10.py --dev-dir <r10/dev> --out-dir <r10/summary>
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_r10_dev as E  # noqa: E402

SEEDS = [7, 13, 21, 42, 99]
ARMS = ["C10", "R10-full", "R10-D", "R10-BC", "R10-N", "C10s"]


def load(dev_dir):
    res = json.loads((dev_dir / "baselines.json").read_text(encoding="utf-8"))
    arms = {a: {} for a in ARMS}
    incomplete = []
    for a in ARMS:
        for s in SEEDS:
            p = dev_dir / f"r10-{a}-s{s}.json"
            if not p.exists():
                incomplete.append(f"{a}-s{s}")
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            (label, v), = d.items()
            arms[a][s] = v
            res[("R10full" if a == "R10-full" else a) + f"-s{s}"] = v
    return res, arms, incomplete


def metrics(v):
    r = v["r10dev"]
    k = v["klue_val"]
    row = {
        "R9dev macro exposure(total)": v["macro"]["total_exposure"],
        "R9dev macro exposure(full)": v["macro"]["full_exposure"],
        "R9dev macro exposure(partial)": v["macro"]["partial_exposure"],
        "R9dev macro exact F1": v["macro"]["exact_f1"],
        "R9dev macro precision": v["macro"]["precision"],
        "R9dev macro recall": v["macro"]["recall"],
        "R9dev macro nonPERSON FPR": v["macro"]["nonperson_fpr"],
        "R9dev overmerge": v["overmerge"]["count"],
        "r10dev ALL exposure(total)": r["ALL"]["total_exposure_rate"],
        "r10dev ALL exact F1": r["ALL"]["prf"]["f1"],
        "r10dev ALL precision": r["ALL"]["prf"]["precision"],
        "r10dev ALL recall": r["ALL"]["prf"]["recall"],
        "KLUE val PERSON F1": k["prf"]["f1"],
        "KLUE val nonPERSON FPR": k["nonperson_fpr"],
        "targets macro exposure(D1P,B1,C2,NEWNAME)": v["targets_macro_exposure"],
        "D1 negative FPR": r["D1_all_negative"]["nonperson_fpr"],
    }
    for t in ("D1P", "B1", "C2", "NEWNAME", "NEWNAME_unseen_only"):
        s = r[t]
        row[f"{t} exposure(total)"] = s["total_exposure_rate"]
        row[f"{t} exposure(full)"] = s["full_rate"]
        row[f"{t} exposure(partial)"] = s["partial_rate"]
        row[f"{t} exact F1"] = s["prf"]["f1"]
        row[f"{t} exact P"] = s["prf"]["precision"]
        row[f"{t} exact R"] = s["prf"]["recall"]
    return row


def agg(models):
    rows = [metrics(m) for m in models]
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows if r[k] is not None]
        if not vals:
            out[k] = None
            continue
        out[k] = {"mean": st.mean(vals), "sd": st.pstdev(vals) if len(vals) > 1 else 0.0, "min": min(vals), "max": max(vals), "n": len(vals)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    res, arms, incomplete = load(a.dev_dir)
    groups = {
        "S0(service v2b)": [res["service"]],
        "R9F": [v for k, v in res.items() if k.startswith("R9F-")],
        **{x: [arms[x][s] for s in SEEDS if s in arms[x]] for x in ARMS},
    }
    table = {g: agg(ms) for g, ms in groups.items() if ms}
    sel = E.select10({k: v for k, v in res.items() if k == "service" or k.startswith(("R9F-", "R10full-"))}) if len(arms["R10-full"]) == 5 else None
    # C10 대비 R10-full(신규 데이터 효과) 및 귀속
    eff = None
    if arms["C10"] and arms["R10-full"]:
        t = lambda m: m["targets_macro_exposure"]  # noqa: E731
        eff = {
            "targets_macro_exposure_C10_mean": st.mean(t(v) for v in arms["C10"].values()),
            "targets_macro_exposure_R10full_mean": st.mean(t(v) for v in arms["R10-full"].values()),
        }
        eff["delta_pp(R10full-C10)"] = round(100 * (eff["targets_macro_exposure_R10full_mean"] - eff["targets_macro_exposure_C10_mean"]), 2)
        eff["new_data_effect_claimable(<= -1.0pp)"] = eff["delta_pp(R10full-C10)"] <= -1.0
        kc = st.mean(v["klue_val"]["prf"]["f1"] for v in arms["C10"].values())
        kr = st.mean(v["klue_val"]["prf"]["f1"] for v in arms["R10-full"].values())
        fc = st.mean(v["klue_val"]["nonperson_fpr"] for v in arms["C10"].values())
        fr = st.mean(v["klue_val"]["nonperson_fpr"] for v in arms["R10-full"].values())
        eff["KLUE_PERSON_F1_delta_pp(R10full-C10)"] = round(100 * (kr - kc), 2)
        eff["KLUE_nonPERSON_FPR_delta_pp(R10full-C10)"] = round(100 * (fr - fc), 2)
        eff["KLUE_degradation_warning"] = eff["KLUE_PERSON_F1_delta_pp(R10full-C10)"] < -2.0 or eff["KLUE_nonPERSON_FPR_delta_pp(R10full-C10)"] > 2.0
    out = {"incomplete_runs": incomplete, "table": table, "selection": sel, "new_data_effect": eff}
    (a.out_dir / "r10_dev_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"incomplete": incomplete, "eligible": sel and sel["eligible"], "chosen": sel and sel["chosen"], "effect": eff}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
