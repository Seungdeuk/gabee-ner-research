"""R12b 개발 게이트 채점·시드 선택. 사전 등록 NER_R12B_PREREGISTRATION.md §4 규칙 그대로(결과를 본 뒤 바꾸지 않는다).
입력: ner-runs/r12b/dev/{R11_baseline,r12b-s*}.json, 런별 parity.json, ONNX 크기. 출력: r12b/gate_report.json 과 표.
게이트 1~4 는 r11_dev·r12_dev·r12b_dev 세 세트 모두에서 평가한다(게이트 6 의 p50 지연·RSS 는 확정 시드만 따로 잰다).
시드 선택: 적격 시드 중 세 세트 PERSON 전체 노출률 평균 최저(동률이면 r12b_dev F1 높은 쪽). 독립 평가셋은 이 선택에 쓰지 않는다.
"""

import json
import statistics as st
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\venvs\ner-runs")
DEV = ROOT / "r12b" / "dev"
RUNS = ROOT / "r12b" / "runs"
SEEDS = [7, 13, 21, 42, 99]
R11_LABEL = "R11-B-s21-ep5"
SETS = ("r11dev", "r12dev", "r12bdev")


def metrics(m):
    out = {}
    for name in SETS:
        s = m[name]
        d1p = s["by_type"].get("D1P", {})
        out[name] = {"exposure": s["ALL"]["total_exposure_rate"], "full_exposure": s["ALL"].get("full_exposure_rate"), "partial_exposure": s["ALL"].get("partial_exposure_rate"),
                     "precision": s["ALL"]["precision_official"], "recall": s["ALL"]["exact_recall"], "f1": s["ALL"].get("f1_official"),
                     "D1P_exposure": d1p.get("total_exposure_rate"), "D1P_recall": d1p.get("exact_recall"),
                     "D1_fpr": s["D1_fpr"], "neg_fpr": s["neg_fpr_macro"] if name == "r11dev" else s["r12_neg_fpr"], "pair_acc": s["pairs"]["pair_context_accuracy"]}
    out["klue_f1"] = m["klue_val"]["prf"]["f1"]
    return out


def main():
    base = metrics(json.loads((DEV / "R11_baseline.json").read_text(encoding="utf-8"))[R11_LABEL])
    r11_size = (ROOT / "r11" / "export" / R11_LABEL / "model.onnx").stat().st_size
    rep = {"R11": base, "seeds": {}}
    for s in SEEDS:
        rid = f"r12b-s{s}"
        p = DEV / f"{rid}.json"
        if not p.exists():
            rep["seeds"][rid] = {"missing": True}
            continue
        m = metrics(json.loads(p.read_text(encoding="utf-8"))[rid])
        par = json.loads((RUNS / rid / "parity.json").read_text(encoding="utf-8"))
        size = (ROOT / "r12b" / "export" / rid / "model.onnx").stat().st_size
        g = {}
        for name in SETS:
            b, x = base[name], m[name]
            g[f"1_precision_{name}"] = x["precision"] >= b["precision"] - 0.020
            g[f"2_D1fpr_{name}"] = x["D1_fpr"] <= b["D1_fpr"] + 0.030
            g[f"3_negfpr_{name}"] = x["neg_fpr"] <= b["neg_fpr"] + 0.015
            g[f"4_exposure_{name}"] = x["exposure"] < b["exposure"]
        g["5_klue_f1"] = m["klue_f1"] >= base["klue_f1"] - 0.010
        g["6_onnx_token_agreement"] = (par.get("onnx_vs_torch_token_agreement_pct") or 0) >= 99.9
        g["6_size"] = size <= 1.30 * r11_size
        mean_exp = st.mean(m[n]["exposure"] for n in SETS)
        rep["seeds"][rid] = {"metrics": m, "gates": g, "eligible": all(g.values()), "failed": [k for k, v in g.items() if not v], "onnx_size": size,
                             "token_agreement_pct": par.get("onnx_vs_torch_token_agreement_pct"), "mean_exposure_3sets": mean_exp}
    el = [(rid, v) for rid, v in rep["seeds"].items() if v.get("eligible")]
    el.sort(key=lambda t: (t[1]["mean_exposure_3sets"], -(t[1]["metrics"]["r12bdev"]["f1"] or 0)))
    rep["selected_seed"] = el[0][0] if el else None
    (ROOT / "r12b" / "gate_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    f = lambda v: "-" if v is None else f"{v * 100:.2f}"

    def line(tag, m, extra=""):
        return f"{tag} " + " | ".join(f"{n} 노출 {f(m[n]['exposure'])} 정밀 {f(m[n]['precision'])} Recall {f(m[n]['recall'])} D1P노출 {f(m[n]['D1P_exposure'])} D1오탐 {f(m[n]['D1_fpr'])} NEG {f(m[n]['neg_fpr'])}" for n in SETS) + f" | KLUE {f(m['klue_f1'])}{extra}"

    print(line("R11 ", base))
    for rid, v in rep["seeds"].items():
        if v.get("missing"):
            print(rid, "결과 없음")
            continue
        print(line(rid, v["metrics"], f" | 적격 {v['eligible']} {v['failed']}"))
    print("선택 시드:", rep["selected_seed"])


if __name__ == "__main__":
    main()
