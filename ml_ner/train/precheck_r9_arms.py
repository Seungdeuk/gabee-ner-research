"""R9 실험군 사전 점검(학습 없음): 실제 데이터 파일로 군별 행 수·라벨 분포·스텝·KLUE 표본을 계산하고, C0/C0s 추가 행 파일을 시드별로 만든다.

학습기(`train_ner.py`)와 똑같이 `ni.load_rows(spec, data_dir, seed)` 로 행을 모은 뒤 집계하므로, 여기서 본 행 수가 곧 실제 학습 행 수다.
군 구성(사전 등록 §3, KLUE 469 고정, `--klue-ratio 0`):
  C1 = data_r8 의 baseline_train+synth_train + klue_train:469        (구 정책, R8 대조군 체크포인트 재사용)
  CP = data_r9 의 같은 파일들 + klue_train:469                         (새 정책만 다름)
  F-full = CP + r9_Ffull_train(3,126)        C0 = CP + c0_extra_s{seed}(기존 비KLUE 3,126행 비복원 재추출)
  E1~E4 = CP + r9_E{k}_train(782)             C0s = CP + c0s_extra_s{seed}(782행 재추출)
사용: python precheck_r9_arms.py --old-dir data_r8 --new-dir data_r9 --out report.json [--write-c0]
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
import ner_infer as ni

BATCH, EPOCHS, KLUE_N, WARMUP = 32, 8, 469, 0.1
SEEDS = [7, 13, 21, 42, 99]
BASE_FILES = ["baseline_train", "synth_train"]
EXTRA = {
    "F-full": "r9_Ffull_train",
    "E1": "r9_E1_train",
    "E2": "r9_E2_train",
    "E3": "r9_E3_train",
    "E4": "r9_E4_train",
}


def read(p):
    return [
        json.loads(x)
        for x in Path(p).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def labels(rows):
    c = Counter()
    for r in rows:
        for s in r.get("spans", []):
            c[s["label"]] += 1
    return dict(c)


def arm_rows(data_dir, extra_spec, seed):
    specs = BASE_FILES + [f"klue_train:{KLUE_N}"] + ([extra_spec] if extra_spec else [])
    rows = [r for spec in specs for r in ni.load_rows(spec, str(data_dir), seed)]
    return rows


def summary(rows):
    n = len(rows)
    spe = math.ceil(n / BATCH)
    total = spe * EPOCHS
    c = labels(rows)
    new = [r for r in rows if r.get("r9")]
    pers = c.get("PERSON", 0) + c.get("PERSON_ALIAS", 0)
    return {
        "rows": n,
        "klue_rows": sum(
            1
            for r in rows
            if str(r.get("id", "")).startswith("klue") or r.get("source") == "klue"
        ),
        "steps_per_epoch": spe,
        "max_total_steps": total,
        "warmup_steps": int(WARMUP * total),
        "label_spans": c,
        "person_family_spans": pers,
        "negative_rows(no NER span)": sum(1 for r in rows if not r.get("spans")),
        "r9_new_rows": len(new),
        "relabeled_rows(r9p)": sum(1 for r in rows if r.get("relabel_r9p")),
        "long_text_rows(>120 chars)": sum(1 for r in rows if len(r["text"]) > 120),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old-dir", required=True, type=Path)
    ap.add_argument("--new-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--write-c0",
        action="store_true",
        help="C0/C0s 추가 행 파일(시드별)을 new-dir 에 쓴다",
    )
    a = ap.parse_args()
    rep = {
        "seeds": SEEDS,
        "batch": BATCH,
        "epochs": EPOCHS,
        "klue_rows_fixed": KLUE_N,
        "arms": {},
        "checks": {},
    }

    existing = [r for f in BASE_FILES for r in read(a.new_dir / f"{f}.jsonl")]
    # C0/C0s 추가 행: 기존 비KLUE(재라벨 반영된 새 정책 행)에서 비복원 추출, 시드별
    c0_info = {}
    pool = [
        r for r in existing if not r.get("relabel_r9p")
    ]  # 보정(2026-11-19 승인): 재라벨되지 않은 기존 행에서만 추출 -> 정책 재라벨 행은 F-full 과 같은 292
    for s in SEEDS:
        order = random.Random(s * 1000 + 17).sample(
            range(len(pool)), 3126
        )  # 시드별 고정 순서
        c0 = [pool[i] for i in order]
        c0s = c0[:782]  # C0s 는 C0 추가 표본의 부분집합
        for name, rows in (("c0_extra", c0), ("c0s_extra", c0s)):
            p = a.new_dir / f"{name}_s{s}.jsonl"
            if a.write_c0:
                p.write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8",
                )
            c0_info[f"{name}_s{s}"] = {
                "rows": len(rows),
                "unique": len({r["id"] for r in rows}),
                "relabeled_rows": sum(1 for r in rows if r.get("relabel_r9p")),
                "labels": labels(rows),
                "sha16": hashlib.sha256(
                    ("|".join(r["id"] for r in rows)).encode()
                ).hexdigest()[:16],
            }
        c0_info[f"c0s_subset_of_c0_s{s}"] = {r["id"] for r in c0s} <= {
            r["id"] for r in c0
        }
    rep["checks"]["c0_extra_files"] = c0_info

    for s in SEEDS:
        arms = {"C1": arm_rows(a.old_dir, None, s), "CP": arm_rows(a.new_dir, None, s)}
        for k, f in EXTRA.items():
            arms[k] = arm_rows(a.new_dir, f, s)
        if a.write_c0 or (a.new_dir / f"c0_extra_s{s}.jsonl").exists():
            arms["C0"] = arm_rows(a.new_dir, f"c0_extra_s{s}", s)
            arms["C0s"] = arm_rows(a.new_dir, f"c0s_extra_s{s}", s)
        rep["arms"][s] = {k: summary(v) for k, v in arms.items()}
        # KLUE 표본이 모든 군에서 같은지(시드별)
        kl = {
            k: hashlib.sha256(
                "|".join(
                    sorted(
                        r["id"]
                        for r in v
                        if r.get("source") == "klue" or str(r["id"]).startswith("klue")
                    )
                ).encode()
            ).hexdigest()[:12]
            for k, v in arms.items()
        }
        rep["checks"].setdefault("klue_sample_hash_by_seed", {})[s] = {
            "hashes": sorted(set(kl.values())),
            "same_across_all_arms": len(set(kl.values())) == 1,
        }

    s0 = rep["arms"][7]
    eq = {
        "C1_vs_CP_rows_steps_equal": s0["C1"]["rows"] == s0["CP"]["rows"]
        and s0["C1"]["max_total_steps"] == s0["CP"]["max_total_steps"],
        "F-full_vs_C0_rows_steps_equal": s0["F-full"]["rows"] == s0["C0"]["rows"]
        and s0["F-full"]["max_total_steps"] == s0["C0"]["max_total_steps"]
        and s0["F-full"]["warmup_steps"] == s0["C0"]["warmup_steps"],
        "E_vs_C0s_rows_steps_equal": all(
            s0[f"E{i}"]["rows"] == s0["C0s"]["rows"]
            and s0[f"E{i}"]["max_total_steps"] == s0["C0s"]["max_total_steps"]
            for i in range(1, 5)
        ),
        "klue_rows_equal_all_arms": len({v["klue_rows"] for v in s0.values()}) == 1
        and next(iter({v["klue_rows"] for v in s0.values()})) == KLUE_N,
        "C0_has_no_r9_new_rows": s0["C0"]["r9_new_rows"] == 0
        and s0["C0s"]["r9_new_rows"] == 0,
        "relabeled_share_F-full_vs_C0_within_0.5pp": s0["F-full"]["relabeled_rows(r9p)"]
        == s0["C0"]["relabeled_rows(r9p)"],
        "relabeled_rows_equal_CP_F-full_C0_C0s_E": len(
            {
                s0[k]["relabeled_rows(r9p)"]
                for k in ("CP", "F-full", "C0", "C0s", "E1", "E2", "E3", "E4")
            }
        )
        == 1,
        "c0s_subset_of_c0_all_seeds": all(
            v
            for k, v in rep["checks"]["c0_extra_files"].items()
            if k.startswith("c0s_subset")
        ),
        "C1_has_zero_relabeled": s0["C1"]["relabeled_rows(r9p)"] == 0,
        "klue_same_in_all_arms_all_seeds": all(
            v["same_across_all_arms"]
            for v in rep["checks"]["klue_sample_hash_by_seed"].values()
        ),
    }
    # 효과 분리: policy = CP-C1 (같은 문장, 정책만), enrichment = F-full - C0 (같은 정책, 같은 행 수), bucket = Ek - C0s
    rep["checks"]["arm_equalities"] = eq
    rep["checks"]["effect_decomposition"] = {
        "policy_effect = CP - C1": "같은 행·같은 KLUE·같은 스텝, 라벨 정책만 다름 → 정책 효과만",
        "enrichment_effect = F-full - C0": "같은 정책·행 수·스텝, 새 합성 행 vs 기존 행 재추출 → 보강 효과만",
        "bucket_effect = Ek - C0s": "같은 정책·행 수·스텝(782), 버킷 하나 vs 기존 행 재추출",
        "confound_check": "C0/C0s 는 새 R9 행 0개, E/F-full 은 기존 행 그대로 + 새 행. 신규 행이 정책 재라벨 행과 겹치지 않음",
    }
    # 신규 행과 기존 행의 문장 겹침(학습 안에서)
    ex_txt = {r["text"] for r in existing}
    new_all = read(a.new_dir / "r9_Ffull_train.jsonl")
    rep["checks"]["new_vs_existing_training_sentence_overlap"] = sum(
        1 for r in new_all if r["text"] in ex_txt
    )
    a.out.write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "arm_equalities": eq,
                "seed7": {
                    k: {
                        x: v[x]
                        for x in (
                            "rows",
                            "klue_rows",
                            "steps_per_epoch",
                            "max_total_steps",
                            "r9_new_rows",
                            "relabeled_rows(r9p)",
                        )
                    }
                    for k, v in s0.items()
                },
                "new_vs_existing_overlap": rep["checks"][
                    "new_vs_existing_training_sentence_overlap"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
