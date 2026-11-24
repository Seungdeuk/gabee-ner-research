"""R11 기준선(S0·R9F·C10·R10-full) 집계와 Recall 게이트 수치 계산(사전 등록 NER_R11_PREREGISTRATION.md §5-5, §7-2, §7-2b).

입력: evaluate_r11_dev.py 결과 JSON 여러 개(--in). 출력: r11_reference_baselines.json(군별 평균·표준편차·최악값과 상대 게이트 임계값),
r11_recall_gate_fixed.json(공식으로 계산한 δ_R9F, σ_R, ε_P 와 입력값). 학습한 B 군 결과는 이 단계에서 보지 않는다.
"""

import argparse
import hashlib
import json
import statistics as st
from pathlib import Path

GROUPS = {"S0": ["service"], "R9F": [f"R9F-s{s}" for s in (7, 13, 21, 42, 99)], "C10": [f"C10-s{s}" for s in (7, 13, 21, 42, 99)],
          "R10full": [f"R10full-s{s}" for s in (7, 13, 21, 42, 99)]}
TYPES = ("D1P", "B1", "C2", "NEWNAME", "LONGNAME", "EXISTING")
TARGET5 = ("D1P", "B1", "C2", "NEWNAME", "LONGNAME")
NEG_SETS = ("NEG_ORGLOC", "NEG_FAMILY", "NEG_D2", "NEG_GENERAL")
MAIN5 = ("v7_namefreq", "realistic_family", "v14_realistic", "nemotron_full", "nemotron_given")


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def metrics(m):
    d = m["r11dev"]
    a = d["ALL"]
    out = {
        "exact_recall": a["exact_recall"], "mask_rate": a["mask_rate"], "boundary_only_rate": a["boundary_only_rate"],
        "partial_rate": a["partial_rate"], "full_rate": a["full_rate"], "total_exposure": a["total_exposure_rate"],
        "precision_official": a["precision_official"], "precision_all": a["precision_all"], "f1_official": a["f1_official"], "f1_all": a["f1_all"],
        "neg_fpr_macro": d["neg_fpr_macro"], "D1_fpr": d["D1_fpr"], "pair_context_accuracy": d["pairs"]["pair_context_accuracy"],
        "targets_macro_exposure": m["targets_macro_exposure"], "r9dev_macro_exposure": m["macro"]["total_exposure"],
        "klue_f1": m["klue_val"]["prf"]["f1"], "klue_recall": m["klue_val"]["prf"]["recall"], "overmerge": m["overmerge"]["count"],
        "identity_ok": d["identity_ok_all"],
        "r9dev_set_exposure": {k: m["main5"][k]["total_exposure_rate"] for k in MAIN5},
        "neg_fpr_by_set": d["neg_fpr_by_set"],
    }
    for t in TYPES + ("NEWNAME_unseen_only",):
        if t in d["by_type"]:
            b = d["by_type"][t]
            out[f"type.{t}"] = {k: b[k] for k in ("n_gold_spans", "exact_recall", "mask_rate", "boundary_only_rate", "partial_rate", "full_rate",
                                                  "total_exposure_rate", "precision_official", "f1_official")}
    return out


def stats(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return {"mean": sum(v) / len(v), "sd_pop": st.pstdev(v) if len(v) > 1 else 0.0, "sd_sample": st.stdev(v) if len(v) > 1 else 0.0,
            "min": min(v), "max": max(v), "values": v}


def flat(m, path):
    cur = m
    for p in path.split("."):
        cur = cur[p]
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    res, meta = {}, None
    for p in a.inputs:
        d = json.loads(p.read_text(encoding="utf-8"))
        meta = meta or d.get("_meta")
        for k, v in d.items():
            if k != "_meta":
                res.setdefault(k, v)
    missing = [m for g in GROUPS.values() for m in g if m not in res]
    if missing:
        raise SystemExit(f"결과 없음: {missing}")
    per_model = {k: metrics(v) for k, v in res.items()}
    groups = {}
    keys = [k for k, v in per_model["service"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for g, ms in GROUPS.items():
        gs = {k: stats([per_model[m][k] for m in ms]) for k in keys}
        for t in TYPES:
            for k in ("exact_recall", "total_exposure_rate", "mask_rate", "boundary_only_rate", "partial_rate", "full_rate", "f1_official", "precision_official"):
                gs[f"type.{t}.{k}"] = stats([per_model[m][f"type.{t}"][k] for m in ms])
        gs["r9dev_set_exposure"] = {s: stats([per_model[m]["r9dev_set_exposure"][s] for m in ms]) for s in MAIN5}
        gs["neg_fpr_by_set"] = {s: stats([per_model[m]["neg_fpr_by_set"][s] for m in ms]) for s in NEG_SETS}
        groups[g] = gs
    S0, R9 = per_model["service"], groups["R9F"]

    # ---- Recall 게이트 수치(공식, §7-2b) ----
    sd_r9_recall = R9["exact_recall"]["sd_pop"]
    delta = clamp(sd_r9_recall, 0.005, 0.015)
    sigma_r = clamp(2 * sd_r9_recall, 0.010, 0.030)
    eps_p = 0.030
    r9_mean_rec = R9["exact_recall"]["mean"]
    gate_fixed = {
        "formula": {"delta_R9F": "clamp(SD_pop(R9F 5시드 전체 Exact Recall), 0.5%p, 1.5%p)", "sigma_R": "clamp(2*SD_pop(R9F Recall), 1.0%p, 3.0%p)", "eps_P": "3.0%p(고정)"},
        "inputs": {"R9F_recall_values": R9["exact_recall"]["values"], "R9F_recall_mean": r9_mean_rec, "R9F_recall_sd_pop": sd_r9_recall,
                   "S0_recall": S0["exact_recall"], "S0_precision_all": S0["precision_all"], "S0_neg_fpr_macro": S0["neg_fpr_macro"]},
        "values": {"delta_R9F": delta, "sigma_R": sigma_r, "eps_P": eps_p},
        "thresholds": {
            "R11-R1_recall_min": r9_mean_rec - delta,
            "R11-R2_recall_must_exceed_S0": S0["exact_recall"],
            "R11-R3_type_recall_min": {t: max(S0[f"type.{t}"]["exact_recall"], R9[f"type.{t}.exact_recall"]["mean"] - delta) for t in TARGET5},
            "R11-R4_precision_all_min": S0["precision_all"] - eps_p,
            "R11-R4_dFP_max_formula": "dFP <= 0.5*dRecall + 0.01 (dRecall, dFP: 모델 - S0, 오탐 음성 4집합 매크로 오탐률)",
            "R11-S_recall_sd_pop_max": sigma_r,
            "R11-S_worst_seed_recall_min": r9_mean_rec - 2 * delta,
        },
        "rule": "수치는 B 학습 전에 확정, 이후 변경 금지. 공식은 사전 등록 §7-2b 에서 이미 고정.",
    }
    # ---- 상대 게이트 임계값(G1~G11), 기준값은 같은 r11_dev ----
    thr = {
        "G1_macro_fpr_max": S0["neg_fpr_macro"] + 0.02,
        "G1_set_fpr_max": {s: per_model["service"]["neg_fpr_by_set"][s] + 0.05 for s in NEG_SETS},
        "G2_overmerge_max": S0["overmerge"],
        "G3_total_exposure_max_vs_R9F": R9["total_exposure"]["mean"] + 0.01,
        "G3_total_exposure_must_be_below_S0": S0["total_exposure"],
        "G3_f1_official_min": S0["f1_official"] - 0.01,
        "G4_set_exposure_max": {s: S0["r9dev_set_exposure"][s] + 0.01 for s in MAIN5},
        "G4_macro_exposure_max": R9["r9dev_macro_exposure"]["mean"] + 0.005,
        "G5_D1P_exposure_max": min(S0["type.D1P"]["total_exposure_rate"], R9["type.D1P.total_exposure_rate"]["mean"] + 0.01),
        "G6_D1_fpr_max": S0["D1_fpr"] + 0.03,
        "G7_pair_accuracy_min": R9["pair_context_accuracy"]["mean"] + 0.05,
        "G8_B1_exposure_max": S0["type.B1"]["total_exposure_rate"], "G8_C2_exposure_max": S0["type.C2"]["total_exposure_rate"],
        "G8_C2_f1_official_min": S0["type.C2"]["f1_official"] - 0.01,
        "G9_NEWNAME_exposure_max": S0["type.NEWNAME"]["total_exposure_rate"], "G9_LONGNAME_exposure_max": min(S0["type.LONGNAME"]["total_exposure_rate"], R9["type.LONGNAME.total_exposure_rate"]["mean"] + 0.01),
        "G9_EXISTING_exposure_max(R9F+1pp)": R9["type.EXISTING.total_exposure_rate"]["mean"] + 0.01,
        "G10_partial_rate_max": S0["partial_rate"] + 0.01,
        "G11_klue_f1_min": R9["klue_f1"]["mean"] - 0.01,
        "R11-S_D1P_exposure_sd_pop_max": 0.05, "R11-S_targets_macro_exposure_sd_pop_max": 0.02,
        "G12": "ONNX 토큰 일치 >=99.9%, CPU 문장당 지연·파일 크기·RSS <= S0 +30% (후보 단계에서 측정)",
    }
    ref = {"meta": meta, "per_model": per_model, "groups": groups, "relative_gate_thresholds": thr,
           "r10_gate_history_note": "R10-full 시드 13 은 R10 개발 평가에서 G1·G6 등 실패, G11 실패(연구 비교용, 교체 자격 없음). 이력은 그대로 유지."}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in (("r11_reference_baselines.json", ref), ("r11_recall_gate_fixed.json", gate_fixed)):
        p = a.out_dir / name
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
        print(name, hashlib.sha256(p.read_bytes()).hexdigest())
    f = lambda x: "-" if x is None else f"{100 * x:.1f}"  # noqa: E731
    print("\n그룹        Recall  가림   경계만 부분   완전   노출   Prec(공식) Prec(전체) F1공식  쌍정확  D1P노출 D1오탐 오탐매크로 R9dev  KLUE")
    for g, gs in groups.items():
        print(f"{g:8s} " + " ".join(f"{f(gs[k]['mean']):>6s}" for k in ("exact_recall", "mask_rate", "boundary_only_rate", "partial_rate", "full_rate", "total_exposure",
              "precision_official", "precision_all", "f1_official", "pair_context_accuracy")) + " " + f(gs["type.D1P.total_exposure_rate"]["mean"]).rjust(6)
              + " " + f(gs["D1_fpr"]["mean"]).rjust(6) + " " + f(gs["neg_fpr_macro"]["mean"]).rjust(6) + " " + f(gs["r9dev_macro_exposure"]["mean"]).rjust(6) + " " + f(gs["klue_f1"]["mean"]).rjust(6))
    print("Recall SD(pop) %p:", {g: round(100 * gs["exact_recall"]["sd_pop"], 2) for g, gs in groups.items()}, "delta", delta, "sigma_R", sigma_r)


if __name__ == "__main__":
    main()
