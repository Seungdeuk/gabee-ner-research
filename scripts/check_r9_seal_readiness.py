"""R9 평가셋 봉인 준비 점검(봉인은 하지 않는다): 사람 검수 기록이 사전 등록 §10-7 요건을 충족하는지, R8 대조군(C1) 재사용 동등성.
평가셋 원문·정답은 출력하지 않고 id·건수만 낸다.

사용: python check_r9_seal_readiness.py --rep R.jsonl --stress S.jsonl --prev-rep V11R.jsonl --prev-stress V11S.jsonl --ledger L.csv --compliance C.json --samples-dir D --out report.json
"""

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9_review_ledger as L  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RUNS = Path(r"C:\Users\<user>\venvs\ner-runs")


def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def wilson_upper(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return min(1.0, c + h)


def ledger_checks(a):
    rep, st = rows(a.rep), rows(a.stress)
    prev = {r["id"]: r for r in rows(a.prev_rep) + rows(a.prev_stress)}
    cur = {r["id"]: r for r in rep + st}
    meta = {r["id"]: ("representative" if r in rep else "stress") for r in rep + st}
    raw = Path(a.ledger).read_text(encoding="utf-8-sig").splitlines()
    led = {}
    for r0 in csv.DictReader(raw):
        r = {k.strip().lower(): v for k, v in r0.items()}
        led[r["id"].strip()] = r
    ids = set(led)
    out = {"records": len(raw) - 1, "unique_ids": len(ids), "invalid_ids": sorted(i for i in ids if i not in cur), "verdicts": dict(Counter(r["review_status"] for r in led.values()))}
    # 1) 층화·표본 요건
    out["total_reviewed"] = len(ids)
    out["meets_min_240"], out["meets_target_300"] = len(ids) >= 240, len(ids) >= 300
    tot_cat = Counter(r["category"] for r in rep)
    rev_cat = Counter(cur[i]["category"] for i in ids if meta[i] == "representative")
    out["representative_category_coverage_pct"] = {c: round(100 * rev_cat[c] / tot_cat[c], 1) for c in sorted(tot_cat)}
    tot_t = Counter(r.get("primary_stress_type") for r in st)
    rev_t = Counter(cur[i].get("primary_stress_type") for i in ids if meta[i] == "stress")
    out["stress_type_coverage_pct"] = {t: round(100 * rev_t[t] / tot_t[t], 1) for t in sorted(tot_t)}
    out["min_stratum_coverage_pct"] = min(list(out["representative_category_coverage_pct"].values()) + list(out["stress_type_coverage_pct"].values()))
    # 제안 표본과의 겹침(참고: 사용자 표본이 무작위인지 직접 증명하지는 못한다)
    samp = Path(a.samples_dir)
    ov = {}
    for k in ("representative", "stress"):
        s = json.loads((samp / f"review_sample_{k}.json").read_text(encoding="utf-8"))
        rs = set(s["random_stratified_ids"])
        ov[k] = {"proposed_random_ids": len(rs), "in_ledger": len(rs & ids), "share_pct": round(100 * len(rs & ids) / len(rs), 1)}
        hr = set(s["high_risk_extra_ids"])
        ov[k]["proposed_high_risk_extra_ids"] = len(hr)
        ov[k]["high_risk_in_ledger"] = len(hr & ids)
    out["overlap_with_proposed_sample"] = ov
    # 2) 고위험 요건(§10-7 (2))
    comp = json.loads(Path(a.compliance).read_text(encoding="utf-8"))
    groups = L.high_risk_groups(rep, st, comp)
    g = {k: {"size": len(v), "reviewed": len(v & ids)} for k, v in groups.items()}
    req = {
        "unlabeled_name_suspect_all(110)": g["unlabeled_name_suspect_rep"]["reviewed"] + g["unlabeled_name_suspect_stress"]["reviewed"] == g["unlabeled_name_suspect_rep"]["size"] + g["unlabeled_name_suspect_stress"]["size"] == 110,
        "one_char_suspect_all(5)": g["one_char_name_suspect_rep"]["reviewed"] + g["one_char_name_suspect_stress"]["reviewed"] == 5,
        "alias>=48of95": g["alias_sentences"]["reviewed"] >= 48,
        "kin_only_negatives_rep>=20of40": g["kin_only_negatives_rep"]["reviewed"] >= 20,
        **{f"stress_{t}>=20": g[f"stress_{t}"]["reviewed"] >= 20 for t in ("A3", "A3N", "D1", "D1P", "D2", "C2", "D3")},
    }
    out["high_risk"] = {"groups": g, "requirements": req, "all_met": all(req.values())}
    # 3) 오류율(구간 단위)
    spans = sum(len(cur[i]["spans"]) for i in ids)
    fix = sum(1 for r in led.values() if r["review_status"] in ("fix", "exclude"))
    out["error_rate"] = {"reviewed_sentences": len(ids), "reviewed_spans": spans, "sentences_fix_or_exclude": fix, "span_error_upper95_pct_if_zero_errors": round(100 * wilson_upper(0, spans), 2),
                         "sentence_error_upper95_pct_if_zero_errors": round(100 * wilson_upper(0, len(ids)), 2), "meets_2pct_threshold_on_point_estimate": fix == 0,
                         "note": "수정 0건이고 오류는 구간이 아니라 문장 단위로 판정됐다. 상한은 오류 0건 가정의 Wilson 95% 상한이다."}
    # 4) 변경된 문장은 v1.2 에서 모두 재검수됐는가, v1.1 승계 검수는 본문·라벨이 같은 문장에만 적용됐는가
    changed = [i for i in cur if i in prev and (cur[i]["text"] != prev[i]["text"] or cur[i]["spans"] != prev[i]["spans"])]
    out["changed_vs_v1.1"] = {"changed_rows": len(changed), "changed_not_in_ledger": sorted(i for i in changed if i not in ids)}
    carry = [i for i, r in led.items() if "v1.1 변경문장 사용자 확인: v1.2 본문·라벨 동일" in r["basis"]]
    mism = [i for i in carry if i in prev and (cur[i]["text"] != prev[i]["text"] or cur[i]["spans"] != prev[i]["spans"])]
    only_carry = [i for i in carry if "v1.2 변경문장/층화표본/고위험" not in led[i]["basis"]]
    out["v1_1_carry_over"] = {"ids_with_carry_over_basis": len(carry), "carry_over_but_changed_in_v1_2": sorted(mism), "carry_over_only_basis_ids": len(only_carry),
                              "carry_over_only_basis_changed_in_v1_2": sorted(i for i in only_carry if i in mism)}
    # 5) 미검수 문장은 정상 처리하지 않음
    out["unreviewed_rows"] = len(cur) - len(ids)
    return out


def c1_equivalence():
    sys.path.insert(0, str(REPO / "ml" / "ner"))
    sys.path.insert(0, str(REPO / "ml" / "ner" / "train"))
    import ner_infer as ni
    exp = {"model": "monologg/koelectra-small-v3-discriminator", "train_files": ["baseline_train", "synth_train"], "val_files": ["baseline_val", "synth_val"], "klue_ratio": 0.1, "klue_file": "klue_train",
           "select_on": "baseline_val", "select_metric": "overlap", "max_len": 128, "batch_size": 32, "lr": 5e-5, "epochs": 8, "patience": 2, "class_weights": "none", "policy": "all", "no_fp16": False,
           "warmup": 0.1, "weight_decay": 0.01}
    res = {"seeds": {}, "problems": []}
    data_dir = RUNS / "data_r8"
    heads = set()
    for seed in (7, 13, 21, 42, 99):
        run = f"r8ctl-koelectra-small-klue10-s{seed}"
        log = json.loads((RUNS / run / "train_log.json").read_text(encoding="utf-8"))
        c = log["config"]
        diffs = {k: [c.get(k), v] for k, v in exp.items() if c.get(k) != v}
        if c.get("seed") != seed:
            diffs["seed"] = [c.get("seed"), seed]
        sha_now = {}
        for spec, info in log["data_files"].items():
            name = spec.partition(":")[0]
            h = hashlib.sha256((data_dir / f"{name}.jsonl").read_bytes()).hexdigest()[:12]
            sha_now[name] = (info["sha256_12"] == h)
        # 데이터 구성과 KLUE 표본·순서 동일성(비율 경로 vs 고정 경로)
        base = [r for f in ("baseline_train", "synth_train") for r in ni.load_rows(f, str(data_dir), seed)]
        k_ratio = ni.load_rows(f"klue_train:{round(len(base) * 0.1 / 0.9)}", str(data_dir), seed)
        k_fixed = ni.load_rows("klue_train:469", str(data_dir), seed)
        order_ratio = hashlib.sha256("|".join(r["id"] for r in base + k_ratio).encode()).hexdigest()[:12]
        order_fixed = hashlib.sha256("|".join(r["id"] for r in base + k_fixed).encode()).hexdigest()[:12]
        heads.add(log.get("git_head"))
        onnx = RUNS / "r8" / "export" / run / "model.onnx"
        res["seeds"][seed] = {"train_sentences": [log["train_sentences"], log["train_sentences_non_klue"], log["train_sentences_klue"]], "config_diffs_vs_planned_C1": diffs,
                              "data_file_hashes_unchanged": sha_now, "klue_ids_same_ratio_vs_fixed": [r["id"] for r in k_ratio] == [r["id"] for r in k_fixed],
                              "training_order_hash_ratio_vs_fixed": [order_ratio, order_fixed], "best_epoch": log["best_epoch"], "stopped_epoch": log["stopped_epoch"],
                              "onnx_exists": onnx.exists(), "onnx_sha12": hashlib.sha256(onnx.read_bytes()).hexdigest()[:12] if onnx.exists() else None}
        if diffs or not all(sha_now.values()) or order_ratio != order_fixed or log["train_sentences"] != 4689:
            res["problems"].append(run)
    res["train_git_heads"] = sorted(h for h in heads if h)
    code_diff = {}
    for h in res["train_git_heads"]:
        r = subprocess.run(["git", "-C", str(REPO), "diff", "--stat", h, "HEAD", "--", "ml/ner/train/train_ner.py", "ml/ner/train/ner_infer.py", "ml/ner/train/tokenize_align.py", "ml/ner/ner_common.py",
                            "ml/ner/data/synth.py", "ml/ner/data/build_ner_data.py", "ml/ner/data/leak_guard.py"], capture_output=True, text=True, encoding="utf-8")
        code_diff[h] = r.stdout.strip() or "(변경 없음)"
    res["training_code_diff_since_run"] = code_diff
    res["equivalent"] = not res["problems"] and all(v.startswith("(변경 없음") for v in code_diff.values())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True, type=Path)
    ap.add_argument("--stress", required=True, type=Path)
    ap.add_argument("--prev-rep", required=True, type=Path)
    ap.add_argument("--prev-stress", required=True, type=Path)
    ap.add_argument("--ledger", required=True, type=Path)
    ap.add_argument("--compliance", required=True, type=Path)
    ap.add_argument("--samples-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    out = {"ledger": ledger_checks(a), "c1_reuse": c1_equivalence()}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
