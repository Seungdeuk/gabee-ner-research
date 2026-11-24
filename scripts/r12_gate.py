"""R12 개발 게이트 채점·시드 선택. 사전 등록 NER_R12_PREREGISTRATION.md §4 규칙 그대로(결과를 본 뒤 바꾸지 않는다).
입력: ner-runs/r12/dev/{R11_baseline,r12-s*}.json, 런별 parity.json, ONNX 크기. 출력: r12/gate_report.json 과 표.
게이트 1~5 는 r12_dev·r11_dev 두 세트 모두에서 평가한다. 게이트 6 의 p50 지연·RSS 는 확정 시드만 따로 잰다(구조 동일, 크기는 여기서 비교).
시드 선택: 적격 시드 중 r12_dev PERSON 전체 노출률 최저(동률이면 F1 높은 쪽). 독립 평가셋은 이 선택에 쓰지 않는다.
"""

import json
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\venvs\ner-runs")
DEV = ROOT / "r12" / "dev"
RUNS = ROOT / "r12" / "runs"
SEEDS = [7, 13, 21, 42, 99]
R11_LABEL = "R11-B-s21-ep5"


def metrics(m):
    out = {}
    for name in ("r12dev", "r11dev"):
        s = m[name]
        out[name] = {"exposure": s["ALL"]["total_exposure_rate"], "precision": s["ALL"]["precision_official"], "recall": s["ALL"]["exact_recall"],
                     "f1": s["ALL"].get("f1_official"), "D1P_exposure": s["by_type"]["D1P"]["total_exposure_rate"] if "D1P" in s["by_type"] else None,
                     "D1_fpr": s["D1_fpr"], "neg_fpr": s["neg_fpr_macro"] if name == "r11dev" else s["r12_neg_fpr"], "pair_acc": s["pairs"]["pair_context_accuracy"]}
    out["klue_f1"] = m["klue_val"]["prf"]["f1"]
    return out


def main():
    base = metrics(json.loads((DEV / "R11_baseline.json").read_text(encoding="utf-8"))[R11_LABEL])
    r11_size = (ROOT / "r11" / "export" / R11_LABEL / "model.onnx").stat().st_size
    rep = {"R11": base, "seeds": {}}
    for s in SEEDS:
        rid = f"r12-s{s}"
        m = metrics(json.loads((DEV / f"{rid}.json").read_text(encoding="utf-8"))[rid])
        par = json.loads((RUNS / rid / "parity.json").read_text(encoding="utf-8"))
        size = (ROOT / "r12" / "export" / rid / "model.onnx").stat().st_size
        g = {}
        for name in ("r12dev", "r11dev"):
            b, x = base[name], m[name]
            g[f"1_precision_{name}"] = x["precision"] >= b["precision"] - 0.020
            g[f"2_D1fpr_{name}"] = x["D1_fpr"] <= b["D1_fpr"] + 0.030
            g[f"3_negfpr_{name}"] = x["neg_fpr"] <= b["neg_fpr"] + 0.015
            g[f"4_exposure_{name}"] = x["exposure"] < b["exposure"]
        g["5_klue_f1"] = m["klue_f1"] >= base["klue_f1"] - 0.010
        g["6_onnx_token_agreement"] = (par.get("onnx_vs_torch_token_agreement_pct") or 0) >= 99.9
        g["6_size"] = size <= 1.30 * r11_size
        rep["seeds"][rid] = {"metrics": m, "gates": g, "eligible": all(g.values()), "failed": [k for k, v in g.items() if not v], "onnx_size": size,
                             "token_agreement_pct": par.get("onnx_vs_torch_token_agreement_pct")}
    el = [(rid, v) for rid, v in rep["seeds"].items() if v["eligible"]]
    el.sort(key=lambda t: (t[1]["metrics"]["r12dev"]["exposure"], -(t[1]["metrics"]["r12dev"]["f1"] or 0)))
    rep["selected_seed"] = el[0][0] if el else None
    (ROOT / "r12" / "gate_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    f = lambda v: "-" if v is None else f"{v * 100:.2f}"
    print("R11  r12dev 노출 %s 정밀 %s Recall %s D1P노출 %s D1오탐 %s NEG %s pair %s | r11dev 노출 %s 정밀 %s D1오탐 %s NEG %s | KLUE %s" % (
        f(base["r12dev"]["exposure"]), f(base["r12dev"]["precision"]), f(base["r12dev"]["recall"]), f(base["r12dev"]["D1P_exposure"]), f(base["r12dev"]["D1_fpr"]),
        f(base["r12dev"]["neg_fpr"]), f(base["r12dev"]["pair_acc"]), f(base["r11dev"]["exposure"]), f(base["r11dev"]["precision"]), f(base["r11dev"]["D1_fpr"]),
        f(base["r11dev"]["neg_fpr"]), f(base["klue_f1"])))
    for rid, v in rep["seeds"].items():
        m = v["metrics"]
        print("%s r12dev 노출 %s 정밀 %s Recall %s D1P노출 %s D1오탐 %s NEG %s pair %s | r11dev 노출 %s 정밀 %s D1오탐 %s NEG %s | KLUE %s | 적격 %s %s" % (
            rid, f(m["r12dev"]["exposure"]), f(m["r12dev"]["precision"]), f(m["r12dev"]["recall"]), f(m["r12dev"]["D1P_exposure"]), f(m["r12dev"]["D1_fpr"]),
            f(m["r12dev"]["neg_fpr"]), f(m["r12dev"]["pair_acc"]), f(m["r11dev"]["exposure"]), f(m["r11dev"]["precision"]), f(m["r11dev"]["D1_fpr"]), f(m["r11dev"]["neg_fpr"]),
            f(m["klue_f1"]), v["eligible"], v["failed"]))
    print("선택 시드:", rep["selected_seed"])


if __name__ == "__main__":
    main()
