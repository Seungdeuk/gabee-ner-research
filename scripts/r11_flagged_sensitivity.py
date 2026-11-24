"""[보조 민감도 분석] 의심 행(Development)을 제외했을 때 모델 지표와 게이트 판정이 달라지는지 정확히 계산한다(데이터 수정 없음, 문장 원문 미출력).

**공식 평가의 대체 기준이 아니다.** 공식 게이트 판정은 고정된 원본 Development 기준이며, 이 분석은 결과에 병기하는 보조 자료다(사전 등록 r11-2.0). 판정이 달라지는 모델은 `quality_sensitive` 로 표시한다.
`--b-runs-dir` 를 주면 B군 파생 모델(B-old·B-new·C-SWA)도 같은 방식으로 분석한다.

방법: 기준선 15개(S0·R9F·C10·R10-full) 모델을 의심 행에만 다시 추론해 유형별 개수(맞음·경계만 틀림·부분·완전·TP·FP 등)를 얻고, 전체 결과(`baselines_*.json`)의 개수에서 빼서
파생 지표(Recall·노출·Precision·F1·쌍 정확률·D1 오탐)를 다시 계산한다. 고정 임계값(`r11_reference_baselines.json`)은 바꾸지 않고 같은 값으로 `select_r11.gates` 를 다시 적용해
시드별 게이트 판정이 뒤집히는지 본다. 어떤 행도 삭제·수정하지 않는다.
사용: python scripts/r11_flagged_sensitivity.py --ids suspects.json --out out.json
"""

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aggregate_r11_baselines as agg  # noqa: E402
import r8_eval_lib as lib  # noqa: E402
import r11_eval_lib as r11  # noqa: E402
import select_r11 as sel  # noqa: E402

DATA = Path(r"C:\Users\<user>\venvs\ner-runs\data_r11")
V = Path(r"C:\Users\<user>\Documents\gabee-ner\docs\security\r11_validation")
MODELS = {"service": r"C:\Users\<user>\venvs\ner-runs\service_v2b"}
for s in (7, 13, 21, 42, 99):
    MODELS[f"R9F-s{s}"] = rf"C:\Users\<user>\venvs\ner-runs\r9\export\r9-F-full-s{s}"
    MODELS[f"C10-s{s}"] = rf"C:\Users\<user>\venvs\ner-runs\r10\export\r10-C10-s{s}"
    MODELS[f"R10full-s{s}"] = rf"C:\Users\<user>\venvs\ner-runs\r10\export\r10-R10-full-s{s}"
CNT = ("n_gold_spans", "exact", "boundary_only", "partial", "full", "tp", "fn", "fp_in_gold_rows", "fp_in_neg_rows", "n_neg_sentences", "neg_flagged", "n_sentences")


def derive(c):
    n = c["n_gold_spans"]
    d = dict(c)
    d["exact_recall"] = c["exact"] / n if n else None
    d["mask_rate"] = (c["exact"] + c["boundary_only"]) / n if n else None
    d["boundary_only_rate"] = c["boundary_only"] / n if n else None
    d["partial_rate"] = c["partial"] / n if n else None
    d["full_rate"] = c["full"] / n if n else None
    d["total_exposure_rate"] = (c["partial"] + c["full"]) / n if n else None
    tp, fg, fn_ = c["tp"], c["fp_in_gold_rows"], c["fp_in_neg_rows"]
    po = tp / (tp + fg) if tp + fg else None
    pa = tp / (tp + fg + fn_) if tp + fg + fn_ else None
    rec = d["exact_recall"]
    d["precision_official"], d["precision_all"] = po, pa
    d["f1_official"] = 2 * po * rec / (po + rec) if po and rec else 0.0
    d["f1_all"] = 2 * pa * rec / (pa + rec) if pa and rec else 0.0
    d["nonperson_fpr"] = c["neg_flagged"] / c["n_neg_sentences"] if c["n_neg_sentences"] else None
    return d


def sub_counts(full, part):
    return {k: full[k] - (part.get(k, 0) if part else 0) for k in CNT}


def exclude(raw, sub):
    """raw: 모델 전체 결과(r11dev 포함), sub: 의심 행만의 score_dataset 결과. 제외 후 r11dev 구조를 돌려준다."""
    out = copy.deepcopy(raw)
    d = out["r11dev"]
    d["ALL"] = derive(sub_counts(d["ALL"], sub["ALL"]))
    for t, b in d["by_type"].items():
        d["by_type"][t] = derive(sub_counts(b, sub["by_type"].get(t)))
    p, sp = d["pairs"], sub["pairs"]
    d["pairs"]["n_pairs"] = p["n_pairs"] - sp["n_pairs"]
    d["pairs"]["context_correct"] = p["context_correct"] - sp["context_correct"]
    d["pairs"]["pair_context_accuracy"] = d["pairs"]["context_correct"] / d["pairs"]["n_pairs"]
    d["D1_fpr"] = d["by_type"]["D1"]["nonperson_fpr"]
    out["targets_macro_exposure"] = lib.macro([d["by_type"][t]["total_exposure_rate"] for t in ("D1P", "B1", "C2", "NEWNAME", "LONGNAME")])
    return out


def load_models(a):
    """기준선 모델과(선택) B군 파생 모델의 {라벨: ONNX 폴더}, {라벨: 원 결과}."""
    raw = {}
    for f in ("baselines_a.json", "baselines_b.json", "baselines_try.json"):
        for k, v in json.loads((DATA / f).read_text(encoding="utf-8")).items():
            if k != "_meta":
                raw.setdefault(k, v)
    models = dict(MODELS)
    if a.b_runs_dir:
        for pd in sorted(a.b_runs_dir.glob("r11-B-s*/pipeline_done.json")):
            rec = json.loads(pd.read_text(encoding="utf-8"))
            dev = json.loads((a.b_dev_dir / f"{rec['run_id']}.json").read_text(encoding="utf-8"))
            for var, label in rec["variants"].items():
                key = f"{var}-s{rec['seed']}"
                models[key] = str(a.b_export_dir / label)
                raw[key] = dev[label]
    return models, raw


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--b-runs-dir", type=Path)
    ap.add_argument("--b-dev-dir", type=Path)
    ap.add_argument("--b-export-dir", type=Path)
    a = ap.parse_args()
    sus = json.loads(a.ids.read_text(encoding="utf-8"))["rows"]
    dev_rows = {json.loads(x)["id"]: json.loads(x) for x in (DATA / "r11_dev.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    sets = {
        "all_12_dev_flagged": [x["id"] for x in sus if x["set"] == "dev"],
        "confirmed_surface_F_only(3)": [x["id"] for x in sus if x["set"] == "dev" and any(r.startswith(("S:", "F:")) for r in x["reasons"])],
    }
    models, raw = load_models(a)
    ref = json.loads((V / "r11_reference_baselines.json").read_text(encoding="utf-8"))
    thr = dict(ref["relative_gate_thresholds"])
    thr["_S0"] = ref["per_model"]["service"]
    rg = json.loads((V / "r11_recall_gate_fixed.json").read_text(encoding="utf-8"))["thresholds"]
    subs = {name: {} for name in sets}
    for label, d in models.items():
        lib.set_model(d, None)
        for name, ids in sets.items():
            subs[name][label] = r11.score_dataset([dev_rows[i] for i in ids])
    out = {"role": "보조 민감도 분석(공식 평가의 대체 기준 아님, 공식 게이트는 원본 Development 기준)",
           "flagged_rows": {k: len(v) for k, v in sets.items()}, "denominators": {"dev_gold_spans": raw["service"]["r11dev"]["ALL"]["n_gold_spans"], "dev_pairs": raw["service"]["r11dev"]["pairs"]["n_pairs"]}, "scenarios": {}}
    base_gates = {k: sel.gates(agg.metrics(raw[k]), thr, rg) for k in models}
    sensitive = {}
    for name in sets:
        flips, deltas = {}, {}
        for label in models:
            m2 = agg.metrics(exclude(raw[label], subs[name][label]))
            g2 = sel.gates(m2, thr, rg)
            ch = {g: [base_gates[label][g], g2[g]] for g in g2 if base_gates[label][g] != g2[g]}
            if ch:
                flips[label] = ch
                sensitive.setdefault(label, []).append(name)
            m1 = agg.metrics(raw[label])
            deltas[label] = {"recall_pp": round(100 * (m2["exact_recall"] - m1["exact_recall"]), 3), "total_exposure_pp": round(100 * (m2["total_exposure"] - m1["total_exposure"]), 3),
                             "pair_acc_pp": round(100 * (m2["pair_context_accuracy"] - m1["pair_context_accuracy"]), 3), "targets_macro_pp": round(100 * (m2["targets_macro_exposure"] - m1["targets_macro_exposure"]), 3),
                             "f1_pp": round(100 * (m2["f1_official"] - m1["f1_official"]), 3)}
        mx = {k: max(abs(d[k]) for d in deltas.values()) for k in ("recall_pp", "total_exposure_pp", "pair_acc_pp", "targets_macro_pp", "f1_pp")}
        out["scenarios"][name] = {"gate_status_flips": flips, "n_models_with_flip": len(flips), "max_abs_metric_change_pp": mx, "per_model_delta_pp": deltas}
    out["quality_sensitive_models"] = {k: v for k, v in sorted(sensitive.items())}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"quality_sensitive_models": out["quality_sensitive_models"], **{k: v["max_abs_metric_change_pp"] for k, v in out["scenarios"].items()}}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
