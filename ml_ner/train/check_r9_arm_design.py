"""R9 실험군(대조군 포함) 설계 점검: 학습은 하지 않고 데이터·스텝·KLUE 표본·라벨 정책이 군 사이에서 의도한 것만 다른지 확인한다.

확인 항목:
1. C1(구정책 data_r8)과 CP(새 정책 data_r9)가 라벨 정책 외에 같은지: 문장·id·행 수·KLUE 파일 해시 동일, 달라진 것은 재라벨 구간뿐.
2. 모든 군이 같은 KLUE 표본(klue_train:469, 시드별)을 쓰는지, `--klue-ratio` 산식에 의존하지 않는지.
3. F-full 과 C0(C0s 와 E1~E4)가 같은 라벨 정책, 같은 행 수, 같은 에폭당 스텝·LR 스케줄을 갖는지. C0 구성(기존 비KLUE 전부 1회 + 추가 행을 비복원 추출) 시연.
사용: python check_r9_arm_design.py --old-dir <data_r8> --new-dir <data_r9> --out report.json
"""

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import ner_infer as ni  # noqa: E402

BATCH, EPOCHS, KLUE_N = 32, 8, 469
NEW_ROWS = {"F-full": 3126, "E1": 782, "E2": 782, "E3": 782, "E4": 782}
SEEDS = [7, 13, 21, 42, 99]


def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def lab(s):
    return s.get("label") or s.get("kind")


def make_c0_rows(non_klue, extra, seed):
    """기존 비KLUE 행 전부 1회 + extra 행을 비복원 추출(행당 최대 2회). 시드 고정."""
    idx = random.Random(seed * 1000 + 17).sample(range(len(non_klue)), extra)
    return list(non_klue) + [non_klue[i] for i in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old-dir", required=True, type=Path)
    ap.add_argument("--new-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    out = {"checks": {}, "arms": {}}

    # 1) C1 vs CP: 정책 외 동일
    diff = {}
    for f in sorted(p.name for p in a.old_dir.glob("*.jsonl")):
        po, pn = a.old_dir / f, a.new_dir / f
        if sha(po) == sha(pn):
            diff[f] = {"identical_sha": True}
            continue
        ro, rn = rows(po), rows(pn)
        same_ids = [x["id"] for x in ro] == [x["id"] for x in rn] and [x["text"] for x in ro] == [x["text"] for x in rn]
        changed = Counter()
        for x, y in zip(ro, rn):
            for sx, sy in zip(x["spans"], y["spans"]):
                if sx != sy:
                    changed[f"{lab(sx)}->{lab(sy)}"] += 1
        other_fields = sum(1 for x, y in zip(ro, rn) if {k for k in y} - {k for k in x} - {"relabel_r9p"})
        diff[f] = {"identical_sha": False, "rows": len(ro), "same_ids_and_texts_in_order": same_ids, "changed_spans": dict(changed), "unexpected_extra_fields": other_fields}
    out["checks"]["c1_vs_cp_file_diff"] = diff
    train_files = ["baseline_train", "synth_train"]
    n_old = sum(len(rows(a.old_dir / f"{f}.jsonl")) for f in train_files)
    n_new = sum(len(rows(a.new_dir / f"{f}.jsonl")) for f in train_files)
    out["checks"]["existing_non_klue_rows"] = {"old": n_old, "new": n_new, "equal": n_old == n_new}
    out["checks"]["val_files_same_rows"] = {f: len(rows(a.old_dir / f"{f}.jsonl")) == len(rows(a.new_dir / f"{f}.jsonl")) for f in ("baseline_val", "synth_val")}

    # 2) KLUE 표본: 모든 군이 klue_train:469 사용, 시드별 표본이 군 사이에서 같음
    klue = {}
    for s in SEEDS:
        ids = [r["id"] for r in ni.load_rows(f"klue_train:{KLUE_N}", str(a.new_dir), s)]
        ids_old = [r["id"] for r in ni.load_rows(f"klue_train:{KLUE_N}", str(a.old_dir), s)]
        ratio_n = round(n_old * 0.1 / (1 - 0.1))
        klue[s] = {"rows": len(ids), "same_in_old_and_new_dir": ids == ids_old, "ids_sha": hashlib.sha256("|".join(ids).encode()).hexdigest()[:12], "ratio_formula_gives_n_for_R8_control": ratio_n}
    out["checks"]["klue_sample"] = klue
    out["checks"]["klue_ratio_pitfall"] = {"R8_control_non_klue": n_old, "ratio_0.1_klue_for_control": round(n_old * 0.1 / 0.9),
                                           "ratio_0.1_klue_for_F-full_if_ratio_used": round((n_old + NEW_ROWS["F-full"]) * 0.1 / 0.9),
                                           "note": "R9 는 --klue-ratio 0 과 klue_train:469 로 KLUE 행 수를 고정한다(비율 산식을 쓰면 신규 데이터가 있는 군만 KLUE 가 늘어난다)"}

    # 3) 군별 크기·스텝
    def arm(name, non_klue, policy):
        total = non_klue + KLUE_N
        spe = math.ceil(total / BATCH)
        return {"label_policy": policy, "non_klue_rows": non_klue, "klue_rows": KLUE_N, "total_rows": total, "klue_share_pct": round(100 * KLUE_N / total, 1),
                "steps_per_epoch": spe, "max_total_steps": spe * EPOCHS, "warmup_steps": int(0.1 * spe * EPOCHS)}
    arms = {"C1": arm("C1", n_old, "r9-1.0(구)"), "CP": arm("CP", n_new, "r9-1.1")}
    arms["C0"] = arm("C0", n_new + NEW_ROWS["F-full"], "r9-1.1")
    arms["F-full"] = arm("F-full", n_new + NEW_ROWS["F-full"], "r9-1.1")
    arms["C0s"] = arm("C0s", n_new + NEW_ROWS["E1"], "r9-1.1")
    for k in ("E1", "E2", "E3", "E4"):
        arms[k] = arm(k, n_new + NEW_ROWS[k], "r9-1.1")
    out["arms"] = arms
    eq = {
        "C1_vs_CP_same_rows_and_steps": arms["C1"]["total_rows"] == arms["CP"]["total_rows"] and arms["C1"]["max_total_steps"] == arms["CP"]["max_total_steps"],
        "F-full_vs_C0_same_rows_steps_klue_share": all(arms["F-full"][k] == arms["C0"][k] for k in ("total_rows", "max_total_steps", "klue_share_pct", "warmup_steps")),
        "E_vs_C0s_same_rows_steps_klue_share": all(arms[f"E{i}"][k] == arms["C0s"][k] for i in range(1, 5) for k in ("total_rows", "max_total_steps", "klue_share_pct", "warmup_steps")),
        "F-full_vs_C0_same_label_policy": arms["F-full"]["label_policy"] == arms["C0"]["label_policy"],
    }
    out["checks"]["arm_equalities"] = eq
    # C0 구성 시연(시드 7)
    nk = []
    for f in train_files:
        nk += rows(a.new_dir / f"{f}.jsonl")
    c0 = make_c0_rows(nk, NEW_ROWS["F-full"], 7)
    cnt = Counter(r["id"] for r in c0)
    out["checks"]["c0_construction_seed7"] = {"rows": len(c0), "unique_existing_rows": len(cnt), "max_copies_of_a_row": max(cnt.values()), "rows_present_twice": sum(1 for v in cnt.values() if v == 2),
                                              "all_existing_present": len(cnt) == len(nk), "policy_changed_rows_in_c0": sum(1 for r in c0 if r.get("relabel_r9p"))}
    out["checks"]["early_stopping_note"] = "에폭당 스텝 수·LR 스케줄(총 스텝, 워밍업)은 군 사이에서 같게 맞춘다. 조기 종료(patience 2, baseline_val 겹침 F1)로 실제 학습 스텝은 시드·군마다 다를 수 있어 실제 종료 에폭과 스텝을 기록·보고한다."
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"arm_equalities": eq, "existing_non_klue_rows": out["checks"]["existing_non_klue_rows"], "klue_ratio_pitfall": out["checks"]["klue_ratio_pitfall"],
                      "klue_same_in_both_dirs": all(v["same_in_old_and_new_dir"] for v in klue.values()), "c0": out["checks"]["c0_construction_seed7"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
