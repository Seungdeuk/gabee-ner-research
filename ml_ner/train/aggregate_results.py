"""run 들의 train_log.json·eval_full.json 과 규칙 탐지기 예측을 모아 결과 JSON 과 표(마크다운 조각)를 만든다(val 만).

    python ml/ner/train/aggregate_results.py --runs-dir <ner-runs 폴더> --out-json <결과.json> --out-tables <표.md>
"""

import argparse
import json
import re
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

import ner_common as nc  # noqa: E402

RUN_RE = re.compile(r"^(?P<model>[a-z\-]+)-klue(?P<pct>\d+)-s(?P<seed>\d+)(?P<extra>.*)$")


def load_runs(runs_dir):
    """eval_full.json 이 있는 run 폴더를 모아 요약 dict 목록으로 만든다."""
    runs = []
    for d in sorted(Path(runs_dir).iterdir()):
        m = RUN_RE.match(d.name)
        if not m or not (d / "eval_full.json").exists():
            continue
        tl = json.loads((d / "train_log.json").read_text(encoding="utf-8"))
        ef = json.loads((d / "eval_full.json").read_text(encoding="utf-8"))
        epochs = [{"epoch": e["epoch"], "train_loss": e["train_loss"], "val_score": e["val_score"], "elapsed_s": e["elapsed_s"],
                   "val": {k: {"char_exact": v["char_exact"]["micro"], "char_overlap": v["char_overlap"]["micro"],
                               "seqeval_overall": v["seqeval"]["overall"]} for k, v in e["val"].items()}} for e in tl["epochs"]]
        runs.append({
            "run_id": d.name, "model": tl["config"]["model"], "klue_ratio": int(m["pct"]) / 100, "seed": int(m["seed"]),
            "extra": m["extra"], "config": tl["config"], "started": tl["started"], "finished": tl["finished"],
            "git_head": tl["git_head"], "data_files": tl["data_files"], "data_build": tl.get("data_build"),
            "train_sentences": tl["train_sentences"], "train_sentences_klue": tl.get("train_sentences_klue"),
            "klue_share_actual": tl.get("klue_share_actual"), "class_weights": tl["class_weights"],
            "tag_token_counts": tl["tag_token_counts"], "gpu": tl["gpu"],
            "total_train_time_s": tl["total_train_time_s"], "step_time_s": tl["step_time_s"], "max_vram_mb": tl["max_vram_mb"],
            "best_epoch": tl["best_epoch"], "stopped_epoch": tl["stopped_epoch"], "checkpoint_mb": tl["checkpoint_mb"],
            "weights_mb": ef["weights_mb"], "params_m": ef["params_m"], "epochs": epochs,
            "eval": {k: {"sentences": v["sentences"], "gold_spans": v["gold_spans"], "exact": v["exact"], "overlap": v["overlap"],
                         "fpr": v["fpr"], "false_positive_sentences": v["false_positive_sentences"],
                         "negative_sentences": v["negative_sentences"]} for k, v in ef["sets"].items()},
            "latency": ef.get("latency"), "error_counts": {k: _count(v) for k, v in ef["errors"].items()},
        })
    return runs


def _count(errs):
    out = {}
    for e in errs:
        key = f"{e['type']}:{e['gold_label'] or e['pred_label']}"
        out[key] = out.get(key, 0) + 1
    return out


def fold(spans):
    """PERSON_ALIAS 를 PERSON 으로 접는다(규칙 탐지기 비교용)."""
    return [{**s, "label": "PERSON" if s["label"] == "PERSON_ALIAS" else s["label"]} for s in spans]


def rule_comparison(rule_path, runs, rows_loader):
    """같은 val·같은 라벨 정의로 규칙 단독과 각 NER 의 비교(탐지기가 낼 수 있는 PERSON/ADDRESS 와 접은 PERSON 중심).

    @param rule_path: rule_baseline_preds.py 산출 JSON
    @param runs: load_runs 결과(예측은 이 함수가 다시 만들지 않으므로 eval 요약만 사용)
    @param rows_loader: baseline_val 행을 돌려주는 함수
    @returns: dict
    """
    rows = rows_loader()
    rp = json.loads(Path(rule_path).read_text(encoding="utf-8"))
    out = {}
    for mode in ("with_known_names", "no_known_names"):
        golds = [r["spans"] for r in rows]
        preds = [rp[r["id"]][mode] for r in rows]
        fine = {m: nc.span_metrics(golds, preds, m) for m in ("exact", "overlap")}
        # 접은 PERSON: 정답의 PERSON_ALIAS 도 PERSON 으로 본다(규칙은 PERSON 으로만 낼 수 있다)
        folded = {m: nc.span_metrics([fold(g) for g in golds], preds, m) for m in ("exact", "overlap")}
        out[mode] = {"fine": fine, "person_folded": folded}
    return out


def pct(x):
    return f"{x:.3f}"


def tables(runs, rule, bba):
    """마크다운 표 조각."""
    L = []
    def key(r):
        return (r["model"], r["klue_ratio"], r["seed"], r["extra"])
    rs = sorted(runs, key=key)
    L.append("### 핵심 표 (Baseline val 주 / 합성 val 보조, char span micro)\n")
    L.append("| run | KLUE | seed | BL overlap P/R/F1 | BL exact P/R/F1 | BL 이름(PERSON) 재현율 ov | 합성 val overlap F1 | CPU100자 중앙 ms | VRAM MB | 학습 s | best ep |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        b, s = r["eval"]["baseline_val"], r["eval"]["synth_val"]
        o, e = b["overlap"]["micro"], b["exact"]["micro"]
        lat = (r["latency"] or {}).get("cpu4_100chars", {}).get("median_ms")
        L.append(f"| {r['run_id']} | {int(r['klue_ratio']*100)}% | {r['seed']} | {pct(o['precision'])}/{pct(o['recall'])}/{pct(o['f1'])} | "
                 f"{pct(e['precision'])}/{pct(e['recall'])}/{pct(e['f1'])} | {pct(b['overlap']['per_label']['PERSON']['recall'])} | "
                 f"{pct(s['overlap']['micro']['f1'])} | {f'{lat:.1f}' if lat else '-'} | {r['max_vram_mb']:.0f} | {r['total_train_time_s']:.0f} | {r['best_epoch']}/{r['stopped_epoch']} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-tables", required=True)
    ap.add_argument("--rule-preds", required=True)
    ap.add_argument("--baseline-before-after", default=None)
    a = ap.parse_args()
    import evaluate_ner as ev  # noqa: E402

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    runs = load_runs(a.runs_dir)
    rule = rule_comparison(a.rule_preds, runs, lambda: ev.load_eval_rows("baseline_val"))
    bba = json.loads(Path(a.baseline_before_after).read_text(encoding="utf-8")) if a.baseline_before_after else None
    res = {"note": "모든 수치는 val 에서의 실측. Baseline test 는 사용하지 않음.", "runs": runs, "rule_detector_on_baseline_val": rule,
           "baseline_rule_label_revision_impact": bba}
    Path(a.out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(a.out_tables).write_text(tables(runs, rule, bba), encoding="utf-8")
    print(f"{len(runs)} runs -> {a.out_json}")


if __name__ == "__main__":
    main()
