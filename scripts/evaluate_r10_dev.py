"""R10 개발 평가와 후보 선정(사전 등록 NER_R10_PREREGISTRATION.md §6, §11). 독립 최종 평가셋·R9 봉인셋은 열지 않는다(경로 없음).

- R9 회귀 확인용 5세트(1글자 이름 행 제외, `evaluate_r9_dev` 와 같은 코드)와 R10 개발셋 `r10_dev`(유형별), KLUE 검증(보고 전용)을 평가한다.
- 게이트: G1~G4·개선 요건(R9 와 동일, 회귀 5세트), G5 D1P, G6 D1 오탐, G7 B1·C2 노출, G8 R9F 대비 회귀, G9 신규 이름, G10 C2 Exact F1, G11 5시드 재현성.
- 모델 라벨 규칙: 서비스 `service`, R9F `R9F-s<seed>`, 후보 `R10full-s<seed>`. 그 밖 라벨(C10 등)은 보고만 하고 선정에는 쓰지 않는다.

사용: python scripts/evaluate_r10_dev.py --model LABEL DIR [FP12] ... --out result.json   /   --select result.json [--out-select sel.json]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate_r9_dev  # noqa: E402,F401  (임포트만으로 1글자 이름 제외 패치가 적용된다)
import evaluate_r8_dev as D  # noqa: E402
import r8_eval_lib as lib  # noqa: E402

R10_DEV = Path(r"C:\Users\<user>\venvs\ner-runs\data_r10\r10_dev.jsonl")
KLUE_VAL = Path(r"C:\Users\<user>\venvs\ner-runs\data_r9\klue_val.jsonl")
TARGETS = ("D1P", "B1", "C2", "NEWNAME")


def rows_of(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def run_model(label, model_dir, fp12):
    out = D.run_model(label, model_dir, fp12, False)  # 모델 지문 확인과 R9 회귀 세트
    dev = rows_of(R10_DEV)
    by = {}
    for r in dev:
        by.setdefault(r["stress_type"], []).append(r)
    out["r10dev"] = {t: lib.score_rows(rs) for t, rs in by.items()}
    unseen = [r for r in by.get("NEWNAME", []) if r["name_seen_in_reference"] and not any(r["name_seen_in_reference"])]
    out["r10dev"]["NEWNAME_unseen_only"] = lib.score_rows(unseen)
    d1_rows = by.get("D1", []) + by.get("D1_STANDALONE", [])
    out["r10dev"]["D1_all_negative"] = lib.score_rows(d1_rows)
    out["r10dev"]["ALL"] = lib.score_rows(dev)
    out["targets_macro_exposure"] = lib.macro([out["r10dev"][t]["total_exposure_rate"] for t in TARGETS])
    out["klue_val"] = lib.score_rows(rows_of(KLUE_VAL))
    return out


def pp(a, b):
    return round(100 * (a - b), 2)


def per_model_gates(c, svc, r9f_mean):
    """c: 후보 결과, svc: 서비스 결과, r9f_mean: R9F 5시드 평균 지표(dict). 반환: 게이트별 값과 통과 여부."""
    base = D.select({"service": svc, "x": c})["verdict"]["x"]  # G1~G4, 개선 요건
    g = dict(base)
    s, r = svc["r10dev"], c["r10dev"]
    g["G5_D1P_delta_vs_service_pp"] = pp(r["D1P"]["total_exposure_rate"], s["D1P"]["total_exposure_rate"])
    g["G5_D1P_delta_vs_R9Fmean_pp"] = pp(r["D1P"]["total_exposure_rate"], r9f_mean["D1P"])
    g["G5"] = g["G5_D1P_delta_vs_service_pp"] <= 0.0 and g["G5_D1P_delta_vs_R9Fmean_pp"] <= 1.0
    g["G6_D1_fp_delta_pp"] = pp(r["D1_all_negative"]["nonperson_fpr"], s["D1_all_negative"]["nonperson_fpr"])
    g["G6"] = g["G6_D1_fp_delta_pp"] <= 3.0
    g["G7_B1_delta_pp"] = pp(r["B1"]["total_exposure_rate"], s["B1"]["total_exposure_rate"])
    g["G7_C2_delta_pp"] = pp(r["C2"]["total_exposure_rate"], s["C2"]["total_exposure_rate"])
    g["G7"] = g["G7_B1_delta_pp"] <= 0.0 and g["G7_C2_delta_pp"] <= 0.0
    g["G8_regress_macro_delta_vs_R9Fmean_pp"] = pp(c["macro"]["total_exposure"], r9f_mean["main5_macro"])
    g["G8"] = g["G8_regress_macro_delta_vs_R9Fmean_pp"] <= 0.5
    g["G9_NEWNAME_delta_pp"] = pp(r["NEWNAME"]["total_exposure_rate"], s["NEWNAME"]["total_exposure_rate"])
    g["G9"] = g["G9_NEWNAME_delta_pp"] <= 0.0
    g["G9_report_unseen_only_delta_pp"] = pp(
        r["NEWNAME_unseen_only"]["total_exposure_rate"], s["NEWNAME_unseen_only"]["total_exposure_rate"]
    ) if r["NEWNAME_unseen_only"]["total_exposure_rate"] is not None else None
    g["G10_C2_exactF1_delta_pp"] = pp(r["C2"]["prf"]["f1"], s["C2"]["prf"]["f1"])
    g["G10"] = g["G10_C2_exactF1_delta_pp"] >= -1.0
    g["all_gates_G1_G10"] = bool(g["passes_gates"] and g["G5"] and g["G6"] and g["G7"] and g["G8"] and g["G9"] and g["G10"])
    g["eligible"] = bool(g["all_gates_G1_G10"] and g["meets_improvement"])
    g["targets_macro_exposure"] = c["targets_macro_exposure"]
    return g


def r9f_means(res):
    ms = [v for k, v in res.items() if k.startswith("R9F-")]
    if not ms:
        raise SystemExit("R9F 시드 결과가 없다(G5·G8 의 기준)")
    return {
        "D1P": sum(m["r10dev"]["D1P"]["total_exposure_rate"] for m in ms) / len(ms),
        "main5_macro": sum(m["macro"]["total_exposure"] for m in ms) / len(ms),
        "targets_macro": sum(m["targets_macro_exposure"] for m in ms) / len(ms),
        "n": len(ms),
    }


def select10(res):
    svc = res["service"]
    mean9 = r9f_means(res)
    cands = {k: v for k, v in res.items() if k.startswith("R10full-")}
    verdict = {k: per_model_gates(v, svc, mean9) for k, v in cands.items()}
    elig = [k for k, g in verdict.items() if g["eligible"]]
    # G11: 3개 이상 자격 + 5시드 평균 비악화(G5 서비스 대비, G8, G9)
    n = len(cands)
    mean = lambda f: sum(f(v) for v in cands.values()) / n  # noqa: E731
    g11 = {
        "eligible_seed_count": len(elig),
        "mean_D1P_delta_vs_service_pp": pp(mean(lambda v: v["r10dev"]["D1P"]["total_exposure_rate"]), svc["r10dev"]["D1P"]["total_exposure_rate"]),
        "mean_regress_macro_delta_vs_R9Fmean_pp": pp(mean(lambda v: v["macro"]["total_exposure"]), mean9["main5_macro"]),
        "mean_NEWNAME_delta_vs_service_pp": pp(mean(lambda v: v["r10dev"]["NEWNAME"]["total_exposure_rate"]), svc["r10dev"]["NEWNAME"]["total_exposure_rate"]),
    }
    g11["pass"] = bool(
        len(elig) >= 3
        and g11["mean_D1P_delta_vs_service_pp"] <= 0.0
        and g11["mean_regress_macro_delta_vs_R9Fmean_pp"] <= 0.5
        and g11["mean_NEWNAME_delta_vs_service_pp"] <= 0.0
    )
    chosen = None
    if g11["pass"] and elig:
        best = min(cands[k]["targets_macro_exposure"] for k in elig)
        tied = [k for k in elig if cands[k]["targets_macro_exposure"] - best <= 0.001]
        seed = lambda k: int(k.split("-s")[-1])  # noqa: E731
        chosen = sorted(
            tied,
            key=lambda k: (cands[k]["macro"]["total_exposure"], cands[k]["macro"]["nonperson_fpr"], -cands[k]["macro"]["exact_f1"], seed(k)),
        )[0]
    worst = {
        "max_D1P_exposure": max(v["r10dev"]["D1P"]["total_exposure_rate"] for v in cands.values()),
        "max_D1P_delta_vs_service_pp": max(pp(v["r10dev"]["D1P"]["total_exposure_rate"], svc["r10dev"]["D1P"]["total_exposure_rate"]) for v in cands.values()),
        "min_C2_exactF1": min(v["r10dev"]["C2"]["prf"]["f1"] for v in cands.values()),
    }
    worst["high_variance_flag(D1P +4.0pp)"] = worst["max_D1P_delta_vs_service_pp"] > 4.0
    only_g10 = [k for k, g in verdict.items() if (not g["eligible"]) and not g["G10"] and g["passes_gates"] and g["G5"] and g["G6"] and g["G7"] and g["G8"] and g["G9"] and g["meets_improvement"]]
    return {"verdict": verdict, "eligible": elig, "G11": g11, "chosen": chosen, "worst_seed": worst, "G10_only_failures": only_g10, "R9F_reference": mean9}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", nargs="+", action="append", default=[], metavar=("LABEL", "DIR"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--select", type=Path)
    ap.add_argument("--out-select", type=Path)
    a = ap.parse_args()
    if a.select:
        res = json.loads(a.select.read_text(encoding="utf-8"))
        sel = select10(res)
        if a.out_select:
            a.out_select.write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: sel[k] for k in ("eligible", "G11", "chosen", "worst_seed", "G10_only_failures")}, ensure_ascii=False, indent=1))
        return
    res = json.loads(a.out.read_text(encoding="utf-8")) if a.out and a.out.exists() else {}
    for spec in a.model:
        label, d = spec[0], spec[1]
        fp = spec[2] if len(spec) > 2 else None
        if label in res:
            print("SKIP(이미 있음)", label, flush=True)
            continue
        res[label] = run_model(label, d, fp)
        m = res[label]
        r = m["r10dev"]
        print(
            f"[{label}] R9dev 노출 {m['macro']['total_exposure']:.4f} | D1P {r['D1P']['total_exposure_rate']:.3f} B1 {r['B1']['total_exposure_rate']:.3f} C2 {r['C2']['total_exposure_rate']:.3f} NEW {r['NEWNAME']['total_exposure_rate']:.3f} D1fp {r['D1_all_negative']['nonperson_fpr']:.3f}",
            flush=True,
        )
        if a.out:
            a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
