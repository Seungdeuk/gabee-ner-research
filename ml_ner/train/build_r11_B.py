"""R11 B군 학습 데이터 조립(학습·신규 생성 없음). 사전 등록 NER_R11_PREREGISTRATION.md §3·§4-2.

B 군 = R9F 데이터(baseline_train + synth_train + klue_train:469 + r9_Ffull_train = 7,815행) + 신규 구성 2,100행(`r11_B_extra`):
  - 신규 대조쌍 D' 800행(`r11_Dprime_train`)
  - R10 기존 데이터 900행: r10_B 290 · r10_C 220 · r10_N 280 · r10_G 110 (시드 고정 무작위 추출, 대조쌍 r10_D 는 쓰지 않음)
  - R9F 보호 재사용 400행: R9F 의 재라벨 안 된 비KLUE 행(baseline/synth/r9_Ffull) 중 PERSON 4글자 이상 행 200 + ORG/LOC 행 200 우선(부족하면 다른 쪽에서 채움)
출력(--data-dir = data_r11): R9F 원본 복사본(해시 동일 확인), r11_B_extra.jsonl, r11_B_assembly_report.json(건수·해시만, 원문 없음).
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "data"))
import build_r8_data as r8  # noqa: E402

R9 = Path(r"C:\Users\<user>\venvs\ner-runs\data_r9")
R10 = Path(r"C:\Users\<user>\venvs\ner-runs\data_r10")
R11 = Path(r"C:\Users\<user>\venvs\ner-runs\data_r11")
COPY = ["baseline_train", "baseline_val", "synth_train", "synth_val", "klue_train", "klue_val", "r9_Ffull_train"]
R10_PICK = {"r10_B_train": 290, "r10_C_train": 220, "r10_N_train": 280, "r10_G_train": 110}
SEED = 20261125
KLUE_N, BATCH = 469, 32


def read(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=R11)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    d = a.data_dir
    rng = random.Random(SEED)
    rep = {"seed": SEED}
    # 1) R9F 원본 복사(해시 동일)
    copied = {}
    for f in COPY:
        dst, src = d / f"{f}.jsonl", R9 / f"{f}.jsonl"
        if a.write and not dst.exists():
            shutil.copyfile(src, dst)
        copied[f] = {"sha256": sha(src), "copy_identical": (dst.exists() and sha(dst) == sha(src))}
    rep["r9f_files_copied"] = copied
    base = {f: read(R9 / f"{f}.jsonl") for f in ("baseline_train", "synth_train", "r9_Ffull_train")}
    # 2) 구성 요소
    dprime = read(d / "r11_Dprime_train.jsonl")
    r10_rows = []
    for f, n in R10_PICK.items():
        rows = read(R10 / f"{f}.jsonl")
        r10_rows += rng.sample(rows, n)
    cand = [r for f in base for r in base[f] if not r.get("relabel_r9p")]

    def long_name(r):
        return any(s["label"] == "PERSON" and s["end"] - s["start"] >= 4 for s in r["spans"])

    def orgloc(r):
        return any(s["label"] in ("ORGANIZATION", "LOCATION") for s in r["spans"])

    cp = [r for r in cand if long_name(r)]
    co = [r for r in cand if orgloc(r) and not long_name(r)]
    pick_p = rng.sample(cp, min(200, len(cp)))
    pick_o = rng.sample(co, min(200, len(co)))
    short = 400 - len(pick_p) - len(pick_o)
    if short > 0:  # 한쪽이 모자라면 다른 쪽 후보에서 채움
        pool = [r for r in cp + co if r not in pick_p and r not in pick_o]
        extra = rng.sample(pool, short)
    else:
        extra = []
    protect = pick_p + pick_o + extra
    rep["protect_candidates"] = {"person_ge4_chars": len(cp), "orgloc_only": len(co), "picked_person_ge4": len(pick_p), "picked_orgloc": len(pick_o), "filled_from_other": len(extra)}
    extra_rows = []
    for tag, rows in (("r11_Dprime", dprime), ("r10_reuse", r10_rows), ("r9f_protect_replay", protect)):
        for r in rows:
            extra_rows.append({**r, "r11_B_component": tag})
    # 3) 검증
    allb = [r for f in base for r in base[f]]
    base_norm = {r8.norm(r["text"]) for r in allb}
    klue_n = KLUE_N
    total = len(allb) + klue_n + len(extra_rows)
    comp = Counter(r["r11_B_component"] for r in extra_rows)
    norm_extra = Counter(r8.norm(r["text"]) for r in extra_rows if r["r11_B_component"] != "r9f_protect_replay")
    new_vs_base = sum(1 for r in extra_rows if r["r11_B_component"] != "r9f_protect_replay" and r8.norm(r["text"]) in base_norm)
    rep["composition"] = dict(comp)
    rep["rows"] = {"base_nonklue": len(allb), "klue": klue_n, "extra": len(extra_rows), "total": total}
    rep["steps_per_epoch"] = math.ceil(total / BATCH)
    rep["total_steps_max"] = rep["steps_per_epoch"] * 8
    rep["klue_share_pct"] = round(100 * klue_n / total, 2)
    rep["relabeled_rows_in_training"] = sum(1 for r in allb if r.get("relabel_r9p")) + sum(1 for r in extra_rows if r.get("relabel_r9p"))
    rep["dup_within_new_components"] = sum(c - 1 for c in norm_extra.values())
    rep["new_components_sentence_equal_to_R9F_base"] = new_vs_base
    rep["protect_replay_rows_are_intended_duplicates_of_R9F"] = len(protect)
    rep["labels_in_extra"] = dict(Counter(s["label"] for r in extra_rows for s in r["spans"]))
    rep["person_len_in_extra"] = dict(Counter(min(s["end"] - s["start"], 5) for r in extra_rows for s in r["spans"] if s["label"] == "PERSON"))
    rep["one_char_person_in_extra"] = sum(1 for r in extra_rows for s in r["spans"] if s["label"] == "PERSON" and s["end"] - s["start"] < 2)
    rep["pairs_in_extra"] = len({r["pair_id"] for r in extra_rows if r.get("pair_id")})
    rep["sealed_or_eval_files_used"] = "없음(R9 봉인셋·R10 dev·R11 val/dev·독립 평가셋 미사용)"
    # 5) 학습 데이터와 R11 Validation/Development 겹침(문장)
    ev = set()
    for f in ("r11_val", "r11_dev"):
        ev |= {r8.norm(r["text"]) for r in read(d / f"{f}.jsonl")}
    rep["extra_sentences_equal_to_r11_val_dev"] = sum(1 for r in extra_rows if r8.norm(r["text"]) in ev)
    rep["base_sentences_equal_to_r11_val_dev"] = sum(1 for r in allb if r8.norm(r["text"]) in ev)
    if a.write:
        (d / "r11_B_extra.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in extra_rows), encoding="utf-8")
        rep["r11_B_extra_sha256"] = sha(d / "r11_B_extra.jsonl")
        (d / "r11_B_assembly_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
