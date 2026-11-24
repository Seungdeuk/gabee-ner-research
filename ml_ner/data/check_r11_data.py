"""R11 데이터 품질·격리 검사(원문 미출력, 건수만). 사전 등록 NER_R11_PREREGISTRATION.md §4-3, §5-2.

대상: data_r11 의 r11_Dprime_train / r11_val / r11_dev. 비교: R9 학습·검증(baseline/synth/r9_Ffull/r9_val_extra), R10 학습·모니터링·개발, 세트 사이.
R9 봉인 평가셋은 읽지 않는다. 출력: 보고서 JSON(--out).
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_r8_data as r8
import build_r10_data as b10
import build_r11_data as b11
import check_r10_train_quality as q10

PERSON = ("PERSON", "PERSON_ALIAS")
ROLES = {"train": "r11_Dprime_train", "val": "r11_val", "dev": "r11_dev"}


def rd(p):
    return r8.read_jsonl(p)


def pvals(rows, alias=True, names_only=False):
    """PERSON 값 목록. names_only 이면 대조쌍의 명사형 이름 단어(pair_word)로 쓰인 사람 구간은 뺀다(이름 풀이 아니라 단어 목록이 정함)."""
    out = []
    for r in rows:
        for s in r["spans"]:
            if s.get("label") in (PERSON if alias else ("PERSON",)):
                v = r["text"][s["start"] : s["end"]]
                if names_only and r.get("pair_role") == "person" and v == r.get("pair_word"):
                    continue
                out.append(v)
    return out


def skel(r):
    return r8.skeleton_of(r["text"], [{"start": s["start"], "end": s["end"]} for s in r["spans"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=b11.OUT_DEFAULT)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    D = {k: rd(a.data_dir / f"{v}.jsonl") for k, v in ROLES.items()}
    r9 = {f: rd(b10.R9_DIR / f"{f}.jsonl") for f in ("baseline_train", "baseline_val", "synth_train", "synth_val", "r9_Ffull_train", "r9_val_extra")}
    r10 = {f: rd(b10.OUT_DEFAULT / f"{f}.jsonl") for f in ("r10_full_train", "r10_val_extra", "r10_dev")}
    existing = [r for v in r9.values() for r in v] + [r for v in r10.values() for r in v]
    rep = {"rows": {k: len(v) for k, v in D.items()}}

    # 1) 중복: 세트 내부·세트 사이 정확/정규화, 틀
    norm = {k: [r8.norm(r["text"]) for r in v] for k, v in D.items()}
    sk = {k: [skel(r) for r in v] for k, v in D.items()}
    rep["dup_within"] = {k: len(v) - len(set(v)) for k, v in norm.items()}
    rep["skeleton_reuse_within(max count)"] = {k: max(Counter(v).values()) for k, v in sk.items()}
    rep["dup_between_roles(sentences)"] = {f"{x}-{y}": len(set(norm[x]) & set(norm[y])) for x, y in (("train", "val"), ("train", "dev"), ("val", "dev"))}
    rep["skeleton_overlap_between_roles"] = {f"{x}-{y}": len(set(sk[x]) & set(sk[y])) for x, y in (("train", "val"), ("train", "dev"), ("val", "dev"))}
    ex_norm = {r8.norm(r["text"]) for r in existing}
    ex_sk = {skel(r) for r in existing}
    rep["sentence_in_existing(R9,R10)"] = {k: len(set(v) & ex_norm) for k, v in norm.items()}
    rep["skeleton_in_existing(R9,R10)"] = {k: len(set(v) & ex_sk) for k, v in sk.items()}

    # 2) 4-gram 유사(>=0.7): 기존 학습, 세트 사이
    thr = 0.7
    ex_train = [r for f in ("baseline_train", "synth_train", "r9_Ffull_train") for r in r9[f]] + r10["r10_full_train"]
    rep["near_dup_rows_vs_R9R10_train(count,>=0.7)"] = {k: q10.near_dup_share(v, ex_train, thr) for k, v in D.items()}
    rep["near_dup_rows_between_roles(count,>=0.7)"] = {
        f"{x}->{y}": q10.near_dup_share(D[x], D[y], thr) for x, y in (("val", "train"), ("dev", "train"), ("dev", "val"), ("train", "val"))
    }

    # 3) 이름 역할 분리와 기존 이름 겹침
    ref_given = set()
    for r in existing:
        for v in pvals([r]):
            ref_given |= b10.given_parts(v)
    name_role_viol, name_overlap = {}, {}
    names_by_role = {}
    for k, v in D.items():
        rows = [r for r in v if r.get("type") != "EXISTING"]  # 기존 이름 층은 학습 이름에서 고름(의도)
        vals = pvals(rows, alias=False, names_only=True)
        names_by_role[k] = vals
        name_role_viol[k] = sum(1 for x in vals if b11.role_of(x) != k)
    rep["name_role_violations(wrong hash role)"] = name_role_viol
    gp = {k: set().union(*(b10.given_parts(x) for x in vals)) if vals else set() for k, vals in names_by_role.items()}
    rep["given_part_overlap_between_roles"] = {f"{x}-{y}": len(gp[x] & gp[y]) for x, y in (("train", "val"), ("train", "dev"), ("val", "dev"))}
    seen_dev_names = [b10.given_parts(x) & ref_given != set() for x in names_by_role["dev"]]
    rep["dev_new_name_spans(unseen vs R9,R10 names)"] = {
        "unseen": sum(1 for x in seen_dev_names if not x), "seen": sum(1 for x in seen_dev_names if x)}
    unseen_by_type = {}
    for t in ("NEWNAME", "LONGNAME", "B1", "C2", "D1P"):
        sp = [(b10.given_parts(r["text"][s["start"] : s["end"]]) & ref_given) != set() for r in D["dev"] if r["type"] == t for s in r["spans"] if s["label"] in PERSON]
        unseen_by_type[t] = {"spans": len(sp), "seen_in_reference": sum(sp)}
    rep["dev_name_seen_by_type"] = unseen_by_type
    ex_vals = pvals(existing)
    rep["train_names_in_R9R10(exact values)"] = sum(1 for x in set(names_by_role["train"]) if x in set(ex_vals))
    rep["person_value_reuse_max(names only, label separate)"] = {k: max(Counter(pvals(v, names_only=True)).values()) for k, v in D.items()}
    rep["alias_distinct_and_spans"] = {k: [len({r["text"][s["start"]:s["end"]] for r in v for s in r["spans"] if s["label"] == "PERSON_ALIAS"}), sum(1 for r in v for s in r["spans"] if s["label"] == "PERSON_ALIAS")] for k, v in D.items()}
    rep["person_spans"] = {k: len(pvals(v)) for k, v in D.items()}
    rep["distinct_person_values"] = {k: len(set(pvals(v))) for k, v in D.items()}
    rep["one_char_person"] = {k: sum(1 for x in pvals(v) if len(x) < 2) for k, v in D.items()}
    rep["family_term_inside_person"] = {k: sum(1 for x in pvals(v) if re.search(r"(엄마|아빠|할머니|할아버지|이모|삼촌|누나|형|언니|오빠)$", x)) for k, v in D.items()}

    # 4) 대조쌍 단어: 역할 간 분리, 기존 학습과 분리(val·dev), 쌍 무결성
    pw = {k: [r["pair_word"] for r in v if r.get("pair_word") and r.get("pair_role") == "person"] for k, v in D.items()}
    sets = {k: set(v) for k, v in pw.items()}
    rep["pair_words"] = {k: {"pairs": len(v), "distinct": len(sets[k]), "max_pairs_per_word": max(Counter(v).values())} for k, v in pw.items()}
    rep["pair_word_overlap_between_roles(exact)"] = {f"{x}-{y}": len(sets[x] & sets[y]) for x, y in (("train", "val"), ("train", "dev"), ("val", "dev"))}
    rep["pair_word_containment_between_roles"] = {
        f"{x}-{y}": sum(1 for a in sets[x] for b in sets[y] if a != b and (a in b or b in a)) for x, y in (("train", "val"), ("train", "dev"), ("val", "dev"))}
    seen_words = set(b10.NAMEWORDS_R10) | set(b11.b9.NAMEWORDS_NOUN) | {r["pair_word"] for r in r10["r10_full_train"] + r10["r10_val_extra"] + r10["r10_dev"] if r.get("pair_word")}
    seen_words |= set(ex_vals)
    rep["pair_words_seen_in_R9R10_training(val,dev must be 0)"] = {k: len(sets[k] & seen_words) for k in ("train", "val", "dev")}
    integ = {}
    for k, v in D.items():
        pairs = {}
        for r in v:
            if r.get("pair_id"):
                pairs.setdefault(r["pair_id"], {})[r["pair_role"]] = r
        bad = 0
        n2_pairs = 0
        for p in pairs.values():
            if not ({"person", "noun"} <= set(p)):
                bad += 1
                continue
            w = p["person"]["pair_word"]
            pr, nr = p["person"], p["noun"]
            ok = any(pr["text"][s["start"] : s["end"]] == w for s in pr["spans"]) and not any(nr["text"][s["start"] : s["end"]] == w for s in nr["spans"]) and w in nr["text"]
            bad += not ok
            n2_pairs += any(s["label"] in PERSON for s in nr["spans"])
        integ[k] = {"pairs": len(pairs), "invalid": bad, "pairs_with_regular_name_in_noun_sentence": n2_pairs}
    rep["pair_integrity"] = integ

    # 5) 길이·유형·틀 작성자
    rep["length_chars"] = {k: {"min": min(len(r["text"]) for r in v), "mean": round(sum(len(r["text"]) for r in v) / len(v), 1), "max": max(len(r["text"]) for r in v)} for k, v in D.items()}
    rep["dev_by_type"] = dict(Counter(r["type"] for r in D["dev"]))
    rep["dev_person_spans_by_type"] = dict(Counter(r["type"] for r in D["dev"] for s in r["spans"] if s["label"] in PERSON))
    rep["frame_authors"] = {k: dict(Counter(r["frame_file"] for r in v)) for k, v in D.items()}
    rep["author_overlap_between_roles"] = sorted(set(r["frame_file"].split("/")[0] for r in D["train"]) & set(r["frame_file"].split("/")[0] for r in D["dev"]))
    rep["frame_reuse_max"] = {k: max(Counter(r["template_id"] for r in v).values()) for k, v in D.items()}
    rep["pii_pattern_rows(digits>=7 or @)"] = {k: sum(1 for r in v if re.search(r"\d{7,}|@", r["text"])) for k, v in D.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
