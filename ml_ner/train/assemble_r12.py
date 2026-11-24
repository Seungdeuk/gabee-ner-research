"""R12 데이터 조립·독립성 점검(사전 등록 NER_R12_PREREGISTRATION.md §3).
작성된 원고(data_r12/raw/*.jsonl)를 변환기로 바꿔 r12_extra(train)·r12_val·r12_dev 를 만들고, 세트 사이·R11 데이터와의 문장·이름·단어·틀 겹침을 점검한다.
학습에 필요한 R11 데이터 파일은 data_r12 로 하드링크한다(디스크 사용 없음). 겹침이 있으면 학습 쪽에서 제거하고 보고한다.
사용: python assemble_r12.py [--check-only]
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\venvs\ner-runs")
D11, D12 = ROOT / "data_r11", ROOT / "data_r12"
RAW = D12 / "raw"
REPO = Path(__file__).resolve().parents[3]
CONVERTER = REPO / "apps/api/scripts/tag_to_r11_jsonl.py"
PY = r"C:\Users\<user>\venvs\ner-runs\verify-py312\Scripts\python.exe"
GROUP_FILE = {"train": "r12_extra", "val": "r12_val", "dev": "r12_dev"}
LINK = ["baseline_train", "baseline_val", "synth_train", "synth_val", "klue_train", "r9_Ffull_train", "r11_B_extra", "r11_dev", "r11_val", "klue_val"]


def rows_of(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def frame(t):
    w = t.split()
    return " ".join(w[:3]) + " … " + (w[-1] if w else "")


def persons(r):
    return [r["text"][s["start"]:s["end"]] for s in r["spans"] if (s.get("label") or s.get("mapped_kind")) in ("PERSON", "PERSON_ALIAS")]


def convert(group):
    index = json.loads((D12 / "batches" / "index.json").read_text(encoding="utf-8"))
    marked = []
    for b in index:
        spec = json.loads((D12 / "batches" / f"{b}.spec.json").read_text(encoding="utf-8"))
        if spec["group"] != group:
            continue
        p = RAW / f"{b}.jsonl"
        if not p.exists():
            raise SystemExit(f"원고 없음: {b}")
        marked += rows_of(p)
    tmp = D12 / f"_{group}_marked.jsonl"
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in marked), encoding="utf-8")
    out = D12 / f"_{group}_conv.jsonl"
    r = subprocess.run([PY, "-X", "utf8", str(CONVERTER), "--in", str(tmp), "--out", str(out), "--source", "r12_written"], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit("변환 실패: " + (r.stderr or r.stdout)[-600:])
    conv = rows_of(out)
    by_id = {m["id"]: m for m in marked}
    name = GROUP_FILE[group]
    res = []
    for c in conv:
        m = by_id[c["id"]]
        c["type"] = m.get("primary_stress_type")
        if m.get("role"):
            c["pair_id"], c["pair_role"], c["pair_word"] = m["pair_id"], m["role"], m["pair_word"]
        c["split"] = name
        c["train_bucket"] = name
        c["r12"] = True
        c["frame_tag"] = "r12_written"
        c["template_id"] = f"r12w-{c['id']}"  # 문장마다 따로(템플릿 치환이 아님)
        res.append(c)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    data = {g: convert(g) for g in ("train", "val", "dev")}
    # 기준 R11 자료
    ref_texts = set()
    ref_names = set()
    for f in ("r11_dev", "r11_val", "r11_B_extra", "baseline_train", "baseline_val", "synth_val"):
        for r in rows_of(D11 / f"{f}.jsonl"):
            ref_texts.add(r["text"])
            if f in ("r11_dev", "r11_val", "baseline_val", "synth_val"):
                ref_names.update(persons(r))
    rep = {"counts": {g: len(v) for g, v in data.items()}}
    # 1. 문장 중복(세트 간, R11 평가·검증 자료와)
    texts = {g: collections.Counter(r["text"] for r in v) for g, v in data.items()}
    rep["dup_within"] = {g: sum(1 for c in cnt.values() if c > 1) for g, cnt in texts.items()}
    rep["dup_between"] = {f"{a}-{b}": len(set(texts[a]) & set(texts[b])) for a, b in (("train", "val"), ("train", "dev"), ("val", "dev"))}
    rep["dup_with_r11_eval_or_val"] = {g: len(set(texts[g]) & ref_texts) for g in data}
    # 2. 이름·단어 겹침
    nm = {g: set(n for r in v for n in persons(r)) for g, v in data.items()}
    rep["name_overlap_between"] = {f"{a}-{b}": len(nm[a] & nm[b]) for a, b in (("train", "val"), ("train", "dev"), ("val", "dev"))}
    rep["name_overlap_with_r11_dev_val_gold"] = {g: len(nm[g] & ref_names) for g in data}
    pw = {g: set(r["pair_word"] for r in v if r.get("pair_word")) for g, v in data.items()}
    rep["pair_word_overlap_between"] = {f"{a}-{b}": len(pw[a] & pw[b]) for a, b in (("train", "val"), ("train", "dev"), ("val", "dev"))}
    # 3. 틀 겹침(앞 3어절 + 끝 어절)
    fr = {g: collections.Counter(frame(r["text"]) for r in v) for g, v in data.items()}
    rep["frame_overlap_between"] = {f"{a}-{b}": sum(1 for f in fr[a] if f in fr[b]) for a, b in (("train", "val"), ("train", "dev"), ("val", "dev"))}
    rep["frame_distinct"] = {g: len(c) for g, c in fr.items()}
    rep["frame_max_repeat"] = {g: max(c.values()) for g, c in fr.items()}
    # 4. 사람 문장 직접 단서 없는 비율, 유형 분포, PERSON 값 길이
    cue = re.compile(r"이름표|별명|이름이|이름은|불러|부르|애칭|닉네임|이름으로")
    for g, v in data.items():
        pr = [r for r in v if r.get("pair_role") == "person"]
        rep.setdefault("weak_cue_share_person_sentences", {})[g] = round(sum(1 for r in pr if not cue.search(r["text"])) / len(pr), 3) if pr else None
        rep.setdefault("types", {})[g] = dict(collections.Counter(r["type"] for r in v))
        rep.setdefault("person_len", {})[g] = dict(collections.Counter(min(len(n), 5) for r in v for n in persons(r)))
    # 5. 학습 쪽 겹침 제거(평가·검증 쪽은 건드리지 않는다)
    r11_extra_texts = set(x["text"] for x in rows_of(D11 / "r11_B_extra.jsonl"))

    def prune(rows, banned_names, banned_texts):
        """겹치는 이름·문장을 가진 행을 제거한다. 대조쌍은 한쪽이 걸리면 쌍 전체를 뺀다(쌍 균형 유지)."""
        bad_pairs = {r["pair_id"] for r in rows if r.get("pair_id") and (set(persons(r)) & banned_names or r["text"] in banned_texts)}
        keep = [r for r in rows if r.get("pair_id") not in bad_pairs and not (set(persons(r)) & banned_names) and r["text"] not in banned_texts]
        return keep, len(rows) - len(keep)

    ban_train = nm["val"] | nm["dev"] | ref_names
    data["train"], n_tr = prune(data["train"], set(ban_train), set(texts["val"]) | set(texts["dev"]) | (ref_texts - r11_extra_texts))
    data["dev"], n_dv = prune(data["dev"], set(nm["val"]) | ref_names, set(texts["val"]))
    rep["train_rows_dropped_for_overlap"] = n_tr
    rep["dev_rows_dropped_for_overlap"] = n_dv
    nm2 = {g: set(n for r in v for n in persons(r)) for g, v in data.items()}
    rep["name_overlap_after_prune"] = {f"{a}-{b}": len(nm2[a] & nm2[b]) for a, b in (("train", "val"), ("train", "dev"), ("val", "dev"))}
    rep["name_overlap_with_r11_gold_after_prune"] = {g: len(nm2[g] & ref_names) for g in ("train", "dev")}
    rep["counts_after_prune"] = {g: len(v) for g, v in data.items()}
    rep["types_after_prune"] = {g: dict(collections.Counter(r["type"] for r in v)) for g, v in data.items()}
    (D12 / "assemble_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if a.check_only:
        return
    for g, v in data.items():
        (D12 / f"{GROUP_FILE[g]}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in v), encoding="utf-8")
    for f in LINK:
        src, dst = D11 / f"{f}.jsonl", D12 / f"{f}.jsonl"
        if src.exists() and not dst.exists():
            os.link(src, dst)
    print("조립 완료:", {GROUP_FILE[g]: len(v) for g, v in data.items()})


if __name__ == "__main__":
    sys.exit(main())
