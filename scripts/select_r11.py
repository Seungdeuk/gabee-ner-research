"""R11 게이트 판정과 후보 선정(사전 등록 NER_R11_PREREGISTRATION.md §7, §7-4 r11-1.3). B 학습 결과를 보기 **전에** 작성·시험한다.

- 입력: `--runs-dir`(r11 런 폴더들의 pipeline_done.json), `--dev-dir`(런별 개발 평가 JSON), 고정된 `r11_reference_baselines.json`·`r11_recall_gate_fixed.json`,
  선택 입력 `--g12`(G12 ONNX·성능 판정 {라벨: true/false}, 없으면 pending 으로 표시하고 후보 선정에서 보류).
- 후보 변형: `B-new`, `C-SWA`(후보 군), `B-old`(분석 전용, 후보 자격 없음). 같은 모델을 두 변형이 가리키면 같은 개발 평가 결과를 쓴다.
- 비교는 소수 둘째 자리 %p 로 반올림한 값에서 한다(경계값은 통과). 임계값은 고정 파일에서만 읽고 이 코드에서 바꾸지 않는다.
"""

import argparse
import json
from decimal import ROUND_HALF_UP, Decimal
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aggregate_r11_baselines import MAIN5, NEG_SETS, metrics  # noqa: E402

CAND_VARIANTS = ("B-new", "C-SWA")
TARGET5 = ("D1P", "B1", "C2", "NEWNAME", "LONGNAME")


def rnd(x, nd):
    """비율 x 를 %로 바꿔 소수 nd 자리에서 반올림(0.5 는 올림). 부동소수 오차(76.05 -> 76.0)를 막으려고 먼저 9자리에서 정리한다."""
    return float(Decimal(repr(round(100 * x, 9))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def r2(x):
    return rnd(x, 2)


def le(a, b):
    return r2(a) <= r2(b)


def ge(a, b):
    return r2(a) >= r2(b)


def gates(m, ref, rg):
    """한 모델의 게이트 판정(G12 제외). m: aggregate.metrics 결과, ref: relative_gate_thresholds, rg: recall_gate_fixed.thresholds."""
    S0 = ref["_S0"]
    t = ref
    g = {}
    g["G1"] = le(m["neg_fpr_macro"], t["G1_macro_fpr_max"]) and all(le(m["neg_fpr_by_set"][s], t["G1_set_fpr_max"][s]) for s in NEG_SETS)
    g["G2"] = m["overmerge"] <= t["G2_overmerge_max"]
    g["G3"] = le(m["total_exposure"], t["G3_total_exposure_max_vs_R9F"]) and r2(m["total_exposure"]) < r2(t["G3_total_exposure_must_be_below_S0"]) and ge(m["f1_official"], t["G3_f1_official_min"])
    g["G4"] = all(le(m["r9dev_set_exposure"][s], t["G4_set_exposure_max"][s]) for s in MAIN5) and le(m["r9dev_macro_exposure"], t["G4_macro_exposure_max"])
    g["G5"] = le(m["type.D1P"]["total_exposure_rate"], t["G5_D1P_exposure_max"])
    g["G6"] = le(m["D1_fpr"], t["G6_D1_fpr_max"])
    g["G7"] = ge(m["pair_context_accuracy"], t["G7_pair_accuracy_min"])
    g["G8"] = le(m["type.B1"]["total_exposure_rate"], t["G8_B1_exposure_max"]) and le(m["type.C2"]["total_exposure_rate"], t["G8_C2_exposure_max"]) and ge(m["type.C2"]["f1_official"], t["G8_C2_f1_official_min"])
    g["G9"] = (le(m["type.NEWNAME"]["total_exposure_rate"], t["G9_NEWNAME_exposure_max"]) and le(m["type.LONGNAME"]["total_exposure_rate"], t["G9_LONGNAME_exposure_max"])
               and le(m["type.EXISTING"]["total_exposure_rate"], t["G9_EXISTING_exposure_max(R9F+1pp)"]))
    g["G10"] = le(m["partial_rate"], t["G10_partial_rate_max"])
    g["G11"] = ge(m["klue_f1"], t["G11_klue_f1_min"])
    g["R1"] = ge(m["exact_recall"], rg["R11-R1_recall_min"])
    g["R2"] = r2(m["exact_recall"]) > r2(rg["R11-R2_recall_must_exceed_S0"])
    g["R3"] = all(ge(m[f"type.{x}"]["exact_recall"], v) for x, v in rg["R11-R3_type_recall_min"].items())
    d_rec = m["exact_recall"] - S0["exact_recall"]
    d_fp = m["neg_fpr_macro"] - S0["neg_fpr_macro"]
    g["R4"] = ge(m["precision_all"], rg["R11-R4_precision_all_min"]) and le(d_fp, 0.5 * d_rec + 0.01)
    return g


def group_s(ms, eligible_n, rg, ref):
    """군 수준 R11-S: eligible>=4, D1P 노출 SD<=5.0%p, 표적 매크로 SD<=2.0%p, Recall SD<=sigma_R, 최악 Recall>=하한(탈락 시드 포함 5개)."""
    sd = lambda xs: st.pstdev(xs) if len(xs) > 1 else 0.0  # noqa: E731
    d1p = [m["type.D1P"]["total_exposure_rate"] for m in ms]
    tm = [m["targets_macro_exposure"] for m in ms]
    rec = [m["exact_recall"] for m in ms]
    out = {"eligible_seeds": eligible_n, "D1P_sd_pp": r2(sd(d1p)), "targets_macro_sd_pp": r2(sd(tm)), "recall_sd_pp": r2(sd(rec)), "worst_recall_pct": r2(min(rec))}
    out["pass"] = bool(eligible_n >= 4 and le(sd(d1p), ref["R11-S_D1P_exposure_sd_pop_max"]) and le(sd(tm), ref["R11-S_targets_macro_exposure_sd_pop_max"])
                       and le(sd(rec), rg["R11-S_recall_sd_pop_max"]) and ge(min(rec), rg["R11-S_worst_seed_recall_min"]))
    return out


def select_candidate(cands):
    """cands: [{label, seed, recall, f1, exposure, targets_macro, neg_fpr_macro, precision, group_sd_sum, latency}] -> (선정 label|None, 근거 dict). §7-4."""
    if not cands:
        return None, {"reason": "후보 없음"}
    v = lambda c: (rnd(c["recall"], 1), rnd(c["f1"], 1), rnd(c["exposure"], 1))  # noqa: E731

    def dominates(a, b):
        ra, rb = v(a), v(b)
        not_worse = ra[0] >= rb[0] and ra[1] >= rb[1] and ra[2] <= rb[2]
        better = ra[0] > rb[0] or ra[1] > rb[1] or ra[2] < rb[2]
        return not_worse and better

    dominated = {c["label"] for c in cands for o in cands if o is not c and dominates(o, c)}
    left = [c for c in cands if c["label"] not in dominated] or list(cands)
    info = {"dominated_removed": sorted(dominated) if len(left) != len(cands) or dominated else []}

    def ranks(key, reverse):
        vals = sorted({key(c) for c in left}, reverse=reverse)
        return {c["label"]: 1 + sum(1 for x in vals if (x > key(c) if reverse else x < key(c))) for c in left}

    r_rec = ranks(lambda c: v(c)[0], True)
    r_f1 = ranks(lambda c: v(c)[1], True)
    r_exp = ranks(lambda c: v(c)[2], False)
    total = {c["label"]: r_rec[c["label"]] + r_f1[c["label"]] + r_exp[c["label"]] for c in left}
    info["rank_sum"] = total
    best = min(total.values())
    tied = [c for c in left if total[c["label"]] == best]
    steps = [
        ("minimax_rank", lambda c: max(r_rec[c["label"]], r_f1[c["label"]], r_exp[c["label"]])),
        ("targets_macro_exposure", lambda c: rnd(c["targets_macro"], 1)),
        ("neg_fpr_macro", lambda c: rnd(c["neg_fpr_macro"], 1)),
        ("precision_official(desc)", lambda c: -rnd(c["precision"], 1)),
        ("group_sd_sum", lambda c: rnd(c["group_sd_sum"], 2)),
        ("latency", lambda c: c.get("latency") if c.get("latency") is not None else 0.0),
        ("seed", lambda c: c["seed"]),
    ]
    used = []
    for name, f in steps:
        if len(tied) == 1:
            break
        m_ = min(f(c) for c in tied)
        tied = [c for c in tied if f(c) == m_]
        used.append(name)
    info["tiebreak_steps_used"] = used
    return tied[0]["label"], info


RESEARCH_CLASSIFICATION = "연구상 최우수 후보(research-best candidate only)"
REPLACEMENT_STATUS = "운영 교체 후보 자격 없음: 사람 의미 검수 519건 미완료, 독립 테스트 미실행(사전 등록 r11-2.0). 교체 권고 불가, 결론은 S0 유지 + 연구 결과 보고"


def classify(chosen, sensitivity=None):
    """선정 결과에 연구 후보 지위와 교체 자격 없음을 항상 붙이고, 의심 행 제외 민감도에서 판정이 달라지는 모델이면 quality_sensitive 로 표시한다(대체 기준 아님)."""
    if chosen is None:
        return {"classification": None, "replacement_eligibility": REPLACEMENT_STATUS, "quality_sensitive": None}
    sens = (sensitivity or {}).get("quality_sensitive_models", {})
    return {"classification": RESEARCH_CLASSIFICATION, "replacement_eligibility": REPLACEMENT_STATUS, "quality_sensitive": chosen in sens,
            "quality_sensitive_scenarios": sens.get(chosen, []), "note": "공식 게이트 판정은 원본 Development 기준. 민감도 분석은 보조이며 유리한 평가셋 선택에 쓰지 않는다"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--dev-dir", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--recall-gate", type=Path, required=True)
    ap.add_argument("--g12", type=Path)
    ap.add_argument("--sensitivity", type=Path, help="r11_flagged_sensitivity_B.json(보조 민감도 분석 결과, 선택)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    refj = json.loads(a.reference.read_text(encoding="utf-8"))
    ref = dict(refj["relative_gate_thresholds"])
    ref["_S0"] = refj["per_model"]["service"]
    rgj = json.loads(a.recall_gate.read_text(encoding="utf-8"))
    rg = rgj["thresholds"]
    g12 = json.loads(a.g12.read_text(encoding="utf-8")) if a.g12 and a.g12.exists() else {}
    table = {v: {} for v in ("B-old", "B-new", "C-SWA")}
    for p in sorted(a.runs_dir.glob("r11-B-s*/pipeline_done.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        dev = json.loads((a.dev_dir / f"{rec['run_id']}.json").read_text(encoding="utf-8"))
        for var, label in rec["variants"].items():
            table[var][rec["seed"]] = (label, metrics(dev[label]))
    out = {"variants": {}, "selection": None}
    for var, seeds in table.items():
        if not seeds:
            continue
        ms = [x[1] for x in seeds.values()]
        per = {}
        for seed, (label, m) in sorted(seeds.items()):
            g = gates(m, ref, rg)
            g12v = g12.get(f"{var}-s{seed}")
            per[seed] = {"label": label, "gates": g, "all_gates_G1_G11_R1_R4": all(g.values()), "G12": g12v if g12v is not None else "pending",
                         "eligible": bool(all(g.values()) and g12v is True),
                         "metrics": {k: m[k] for k in ("exact_recall", "f1_official", "total_exposure", "precision_official", "precision_all", "targets_macro_exposure", "neg_fpr_macro", "pair_context_accuracy", "klue_f1")}}
        elig = [s for s, x in per.items() if x["eligible"]]
        gs = group_s(ms, len(elig), rg, ref)
        out["variants"][var] = {"per_seed": per, "group_S": gs, "candidate_arm": var in CAND_VARIANTS}
    cands = []
    for var in CAND_VARIANTS:
        v = out["variants"].get(var)
        if not v or not v["group_S"]["pass"]:
            continue
        rec_sd = v["group_S"]["recall_sd_pp"] / 100
        d1p_sd = v["group_S"]["D1P_sd_pp"] / 100
        for seed, x in v["per_seed"].items():
            if x["eligible"]:
                mm = x["metrics"]
                cands.append({"label": f"{var}-s{seed}", "seed": seed, "recall": mm["exact_recall"], "f1": mm["f1_official"], "exposure": mm["total_exposure"], "targets_macro": mm["targets_macro_exposure"],
                              "neg_fpr_macro": mm["neg_fpr_macro"], "precision": mm["precision_official"], "group_sd_sum": rec_sd + d1p_sd, "latency": None})
    chosen, info = select_candidate(cands)
    sens = json.loads(a.sensitivity.read_text(encoding="utf-8")) if a.sensitivity and a.sensitivity.exists() else None
    out["selection"] = {"candidates": [c["label"] for c in cands], "chosen": chosen, "detail": info, **classify(chosen, sens),
                        "note": "G12 가 pending 이면 eligible 이 아니므로 후보 없음. 후보 없으면 독립 테스트를 열지 않는다."}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({v: {"group_S": x["group_S"], "eligible": [s for s, y in x["per_seed"].items() if y["eligible"]]} for v, x in out["variants"].items()}, ensure_ascii=False, indent=1))
    print(json.dumps(out["selection"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
