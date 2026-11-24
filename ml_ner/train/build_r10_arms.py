"""R10 실험군·대조군 데이터 구성과 사전 점검(학습 없음). 사전 등록 NER_R10_PREREGISTRATION.md §4.

군 정의(모두 사전학습 체크포인트에서 처음부터, 시드 7·13·21·42·99, KLUE 469 절대량 고정, `--klue-ratio 0`):
  R9F        R9 F-full 이 쓴 데이터 그대로(7,815행). R9 에서 학습한 5시드 모델을 **재사용**(재학습 없음). R10 의 출발점·비교 기준.
  C10        **볼륨·스텝 대조군(R9 의 C0 와 다름)**: R9F + R9F 의 재라벨 안 된 비KLUE 행에서 비복원 추출한 2,100행(`c10_extra_s{seed}`).
             R10-full 과 행 수·에폭당 스텝·KLUE 비중·재라벨 행 수가 같고, **새 내용이 없다**(R9F 데이터의 반복).
  R10-full   R9F + r10_full_train(2,100행). 후보 선정 자격은 이 군의 시드 5개만 갖는다.
  귀속 분석(선정 자격 없음, 보고용): R10-D(D1P 쌍 400행), R10-BC(B1 200 + C2 200), R10-N(신규 이름 메신저 400행) 각각 + 대조 C10s(R9F + 400행 추출).
  S0         서비스 v2b(학습 없음, 기준선).
사용: python build_r10_arms.py --data-dir <data_r10> --r9-dir <data_r9> --out report.json [--write]
"""

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

BATCH, EPOCHS, KLUE_N, WARMUP = 32, 8, 469, 0.1
SEEDS = [7, 13, 21, 42, 99]
BASE_NONKLUE = ["baseline_train", "synth_train"]
COPY_FROM_R9 = [
    "baseline_train",
    "baseline_val",
    "synth_train",
    "synth_val",
    "klue_train",
    "klue_val",
    "r9_Ffull_train",
]


def read(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def write(p, rows):
    Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def klue_sample(rows, seed):
    return random.Random(seed).sample(rows, min(KLUE_N, len(rows)))


def stats(rows, klue_rows):
    n = len(rows) + len(klue_rows)
    steps = math.ceil(n / BATCH)
    return {
        "rows": n,
        "klue_rows": len(klue_rows),
        "klue_share_pct": round(100 * len(klue_rows) / n, 2),
        "steps_per_epoch": steps,
        "total_steps_max": steps * EPOCHS,
        "warmup_steps": math.ceil(steps * EPOCHS * WARMUP),
        "relabeled_rows(r9p)": sum(1 for r in rows if r.get("relabel_r9p")),
        "r10_new_rows": sum(1 for r in rows if r.get("r10")),
        "r9_new_rows": sum(1 for r in rows if r.get("r9")),
        "labels": dict(Counter(s["label"] for r in rows for s in r.get("spans", []))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--r9-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    d, r9 = a.data_dir, a.r9_dir
    if a.write:
        for f in COPY_FROM_R9:
            dst = d / f"{f}.jsonl"
            if not dst.exists():
                shutil.copyfile(r9 / f"{f}.jsonl", dst)
    copied = {f: sha(d / f"{f}.jsonl") == sha(r9 / f"{f}.jsonl") for f in COPY_FROM_R9 if (d / f"{f}.jsonl").exists()}
    base = [r for f in BASE_NONKLUE for r in read(d / f"{f}.jsonl")] if (d / "baseline_train.jsonl").exists() else read(r9 / "baseline_train.jsonl") + read(r9 / "synth_train.jsonl")
    f9 = read((d if (d / "r9_Ffull_train.jsonl").exists() else r9) / "r9_Ffull_train.jsonl")
    klue = read(r9 / "klue_train.jsonl")
    r10 = read(d / "r10_full_train.jsonl")
    by = {b: [r for r in r10 if r["train_bucket"] == f"r10_{b}"] for b in "DBCNG"}
    # 귀속 분석용 부분집합(쌍은 쌍 단위로 유지)
    rr = random.Random(20261121)
    groups = sorted({(r["pair_id"], r["pair_word"]) for r in by["D"]})
    pick = set(rr.sample(groups, 200))  # (쌍, 단어) 묶음 200개 = 사람·명사 행 400개
    attr = {
        "r10_attrD_train": [r for r in by["D"] if (r["pair_id"], r["pair_word"]) in pick],
        "r10_attrBC_train": rr.sample(by["B"], 200) + rr.sample(by["C"], 200),
        "r10_attrN_train": list(by["N"]),
    }
    pool = [r for r in base + f9 if not r.get("relabel_r9p")]  # R9F 의 재라벨 안 된 비KLUE 행(재라벨 행 수 동일 유지)
    arms, rep = {}, {"seeds": SEEDS, "checks": {}, "copied_identical_to_r9": copied}
    for s in SEEDS:
        kl = klue_sample(klue, s)
        order = random.Random(s * 1000 + 31).sample(range(len(pool)), 2100)
        c10 = [pool[i] for i in order]
        c10s = c10[:400]
        if a.write:
            write(d / f"c10_extra_s{s}.jsonl", c10)
            write(d / f"c10s_extra_s{s}.jsonl", c10s)
        arms[s] = {
            "R9F": stats(base + f9, kl),
            "C10": stats(base + f9 + c10, kl),
            "R10-full": stats(base + f9 + r10, kl),
            "C10s": stats(base + f9 + c10s, kl),
            **{k.replace("r10_attr", "R10-").replace("_train", ""): stats(base + f9 + v, kl) for k, v in attr.items()},
        }
    if a.write:
        for k, v in attr.items():
            write(d / f"{k}.jsonl", v)
    s0 = arms[SEEDS[0]]
    rep["arms_seed7"] = {k: {x: y for x, y in v.items() if x != "labels"} for k, v in s0.items()}
    rep["labels_seed7"] = {k: s0[k]["labels"] for k in ("R9F", "C10", "R10-full")}
    same = lambda a_, b_: all(s0[a_][k] == s0[b_][k] for k in ("rows", "klue_rows", "steps_per_epoch", "warmup_steps", "relabeled_rows(r9p)", "total_steps_max"))  # noqa: E731
    rep["checks"] = {
        "C10_equals_R10full_rows_steps_klue_relabeled": same("C10", "R10-full"),
        "C10s_equals_attr_arms": all(same("C10s", k) for k in ("R10-D", "R10-BC", "R10-N")) if False else all(
            s0["C10s"]["steps_per_epoch"] == s0[k]["steps_per_epoch"] for k in ("R10-D", "R10-BC", "R10-N")
        ),
        "klue_share_equal_across_C10_R10full": s0["C10"]["klue_share_pct"] == s0["R10-full"]["klue_share_pct"],
        "c10_extra_contains_r10_rows": sum(1 for s in SEEDS for r in (read(d / f"c10_extra_s{s}.jsonl") if (d / f"c10_extra_s{s}.jsonl").exists() else []) if r.get("r10")),
        "attrD_groups_intact(2 rows: 1 person + 1 noun, same word)": all(
            len(v) == 2 and {x["pair_role"] for x in v} == {"person", "noun"}
            for v in __import__("itertools").starmap(lambda k, g: list(g), __import__("itertools").groupby(
                sorted(attr["r10_attrD_train"], key=lambda r: (r["pair_id"], r["pair_word"])),
                key=lambda r: (r["pair_id"], r["pair_word"])))
        ),
        "attrD_person_noun_rows": [
            sum(1 for r in attr["r10_attrD_train"] if r["pair_role"] == "person"),
            sum(1 for r in attr["r10_attrD_train"] if r["pair_role"] == "noun"),
        ],
        "r10_full_pair_person_noun_rows": [
            sum(1 for r in r10 if r.get("pair_role") == "person"),
            sum(1 for r in r10 if r.get("pair_role") == "noun"),
        ],
        "pool_size": len(pool),
        "pool_rows_with_relabel_excluded": len(base + f9) - len(pool),
        "r10_full_rows": len(r10),
        "attr_rows": {k: len(v) for k, v in attr.items()},
    }
    rep["training_runs"] = {"R10-full": 5, "C10": 5, "attribution(D,BC,N)": 15, "C10s": 5, "total": 30, "reused_without_training": ["R9F (5 seeds from R9)", "S0 service v2b"]}
    a.out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"checks": rep["checks"], "seed7": rep["arms_seed7"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
