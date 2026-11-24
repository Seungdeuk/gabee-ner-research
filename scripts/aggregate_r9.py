"""R9 1차 학습 결과 종합: 군·시드별 표, 평균·표준편차, 사전 등록 효과(정책 CP-C1, 보강 F-full-C0, 버킷 Ek-C0s), 게이트 판정(참고), 서비스 모델 대비 노출 변화.

입력: ner-runs/r9/dev/<run_id>.json (evaluate_r9_dev), 서비스 기준선 dev_baseline_r9.json, C1 = R8 대조군(r9 개발 평가로 재산출한 dev/c1-s*.json),
전체 파이프라인 기록 runs/*/pipeline_done.json, 선택적으로 strata/<run_id>.json. 봉인 평가셋은 읽지 않는다.
사용: python aggregate_r9.py --out summary.json
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_r8_dev as D  # noqa: E402

R9 = Path(r"C:\Users\<user>\venvs\ner-runs\r9")
SEEDS = [7, 13, 21, 42, 99]
ARMS = ["C1", "CP", "C0", "C0s", "F-full", "E1", "E2", "E3", "E4"]
METRICS = [
    "total_exposure",
    "full_exposure",
    "partial_exposure",
    "exact_f1",
    "precision",
    "recall",
    "nonperson_fpr",
]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def rid(arm, seed):
    return f"c1-s{seed}" if arm == "C1" else f"r9-{arm}-s{seed}"


def one(arm, seed):
    p = R9 / "dev" / f"{rid(arm, seed)}.json"
    return list(load(p).values())[0] if p.exists() else None


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {
        "mean": round(st.mean(vals), 5),
        "sd": round(st.stdev(vals), 5) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 5),
        "max": round(max(vals), 5),
        "n": len(vals),
        "values": [round(v, 5) for v in vals],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    svc = list(load(R9 / "dev_baseline_r9.json").values())[0]
    res = {
        "service": {m: round(svc["macro"][m], 5) for m in METRICS}
        | {"overmerge": svc["overmerge"]["count"]},
        "arms": {},
        "effects": {},
        "gates_reference_only": {},
    }
    runs = {arm: {s: one(arm, s) for s in SEEDS} for arm in ARMS}
    for arm in ARMS:
        r = runs[arm]
        d = {m: ms([x["macro"][m] if x else None for x in r.values()]) for m in METRICS}
        d["overmerge"] = ms([x["overmerge"]["count"] if x else None for x in r.values()])
        d["per_set_total_exposure"] = {
            k: ms([x["main5"][k]["total_exposure_rate"] if x else None for x in r.values()])
            for k in D.MAIN5
        }
        d["per_set_exact_f1"] = {
            k: ms([x["main5"][k]["prf"]["f1"] if x else None for x in r.values()]) for k in D.MAIN5
        }
        d["per_set_nonperson_fpr"] = {
            k: ms(
                [
                    ({**x["main5"], **x["safety"]}[k])["nonperson_fpr"] if x else None
                    for x in r.values()
                ]
            )
            for k in D.FP_SETS
        }
        done = [
            load(R9 / "runs" / rid(arm, s) / "pipeline_done.json")
            for s in SEEDS
            if (R9 / "runs" / rid(arm, s) / "pipeline_done.json").exists()
        ]
        d["stopped_epoch"] = [x["stopped_epoch"] for x in done]
        d["best_epoch"] = [x["best_epoch"] for x in done]
        d["baseline_val_best_overlap_f1"] = ms([x["best_val_baseline_overlap_f1"] for x in done])
        res["arms"][arm] = d

    # 짝지은(시드별) 효과
    def eff(name, a_arm, b_arm):
        out = {}
        for m in METRICS:
            diffs = [
                runs[a_arm][s]["macro"][m] - runs[b_arm][s]["macro"][m]
                for s in SEEDS
                if runs[a_arm][s] and runs[b_arm][s]
            ]
            out[m] = (
                {
                    "mean_pp": round(100 * st.mean(diffs), 3),
                    "sd_pp": round(100 * st.stdev(diffs), 3) if len(diffs) > 1 else 0.0,
                    "seeds_better": sum(
                        1
                        for x in diffs
                        if (
                            x < 0
                            if m
                            in (
                                "total_exposure",
                                "full_exposure",
                                "partial_exposure",
                                "nonperson_fpr",
                            )
                            else x > 0
                        )
                    ),
                    "n": len(diffs),
                    "per_seed_pp": [round(100 * x, 3) for x in diffs],
                }
                if diffs
                else None
            )
        res["effects"][name] = out

    eff("policy = CP - C1", "CP", "C1")
    eff("enrichment = F-full - C0", "F-full", "C0")
    for k in (1, 2, 3, 4):
        eff(f"bucket E{k} = E{k} - C0s", f"E{k}", "C0s")
    # 게이트(참고: 선정 자격은 F-full 과 E5 만)
    cand = {"service": svc}
    for arm in ARMS:
        for s in SEEDS:
            if runs[arm][s]:
                cand[f"{arm}|s{s}"] = runs[arm][s]
    sel = D.select({**cand})
    res["gates_reference_only"] = {
        k: {
            x: v[x]
            for x in (
                "G1",
                "G2",
                "G3",
                "G4",
                "passes_gates",
                "meets_improvement",
                "eligible",
                "macro_exposure_delta_pp",
                "G1_macro_fp_delta_pp",
                "G3_macro_f1_delta_pp",
            )
        }
        for k, v in sel["verdict"].items()
    }
    res["note"] = (
        "선정 자격은 F-full 과 E5 에만 있다(사전 등록). 이 표의 eligible 은 참고용이며 후보 선정에 쓰지 않는다."
    )
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {
                "service": res["service"],
                "arms": {
                    k: {
                        m: v[m]["mean"] if v[m] else None
                        for m in ("total_exposure", "exact_f1", "nonperson_fpr")
                    }
                    | {
                        "overmerge": v["overmerge"]["mean"] if v["overmerge"] else None,
                        "n": v["total_exposure"]["n"] if v["total_exposure"] else 0,
                    }
                    for k, v in res["arms"].items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
