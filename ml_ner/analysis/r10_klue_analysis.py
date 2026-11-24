"""KLUE 검증 PERSON F1 하락(C10 -> R10-full) 원인 분석. 학습 없음, KLUE 본문·실명은 출력하지 않는다(길이·유형·건수만).

(1) 학습 분포 비교: C10 추가분 / R10 신규분 / KLUE 학습 469행 / KLUE 검증의 통계
(2) 모델 오류 분해: FN(경계 오류 vs 완전 미탐), FP(경계 vs 순수, 다른 KLUE 라벨과의 혼동), 정답 길이별·조사 유무별
(3) 같은 정답을 C10 은 잡고 R10-full 은 놓친 구간의 특성
"""

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

D10 = Path(r"C:\Users\<user>\venvs\ner-runs\data_r10")
D9 = Path(r"C:\Users\<user>\venvs\ner-runs\data_r9")
SEEDS = [7, 13, 21, 42, 99]
PARTICLES = ("이", "가", "은", "는", "을", "를", "도", "와", "과", "의", "에", "한", "께", "야", "아", "랑", "만", "로", "으")


def rows(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def sd(v):
    return st.pstdev(v) if len(v) > 1 else 0.0


def person_spans(r):
    return [(s["start"], s["end"]) for s in r["spans"] if s.get("label") in ("PERSON", "PERSON_ALIAS")]


def followed(text, e):
    return "particle" if e < len(text) and text[e] in PARTICLES else ("end/punct/space" if e >= len(text) or not text[e].isalnum() else "other")


def dist_stats(rs, name):
    n = len(rs)
    pl, par, nospace = Counter(), Counter(), 0
    n_person = 0
    rows_person = rows_other = rows_empty = rows_noun = 0
    for r in rs:
        ps = person_spans(r)
        oth = [s for s in r["spans"] if s.get("label") not in ("PERSON", "PERSON_ALIAS")]
        rows_person += bool(ps)
        rows_other += bool(oth)
        rows_empty += not r["spans"]
        rows_noun += r.get("pair_role") == "noun"
        for a, b in ps:
            n_person += 1
            pl[min(b - a, 5)] += 1
            par[followed(r["text"], b)] += 1
    return {
        "name": name, "rows": n, "mean_chars": round(mean([len(r["text"]) for r in rs]), 1), "person_spans": n_person, "person_per_row": round(n_person / n, 3),
        "person_len_share(1,2,3,4,5+)": {k: round(v / n_person, 3) for k, v in sorted(pl.items())} if n_person else {},
        "person_followed_by": {k: round(v / n_person, 3) for k, v in par.items()} if n_person else {},
        "rows_with_person": round(rows_person / n, 3), "rows_with_ORG_LOC": round(rows_other / n, 3), "rows_no_entity": round(rows_empty / n, 3), "rows_person_lookalike_noun": round(rows_noun / n, 3),
    }


def load_dump(d, label):
    p = Path(d) / f"klue_rowpreds_{label}.jsonl"
    if not p.exists():
        return None
    lines = p.read_text(encoding="utf-8").splitlines()
    return {x["id"]: x for x in map(json.loads, lines[1:])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {}
    # (1) 학습 분포
    c10 = [r for s in SEEDS for r in rows(D10 / f"c10_extra_s{s}.jsonl")]
    r10 = rows(D10 / "r10_full_train.jsonl")
    ktr = rows(D9 / "klue_train.jsonl")[:469]
    kva = rows(D9 / "klue_val.jsonl")
    res["dist"] = [dist_stats(c10, "C10 추가분(5시드 합, 중복 포함)"), dist_stats(r10, "R10 신규 추가분"), dist_stats(ktr, "KLUE 학습(앞 469행 근사)"), dist_stats(kva, "KLUE 검증 5000행")]
    # (2) 오류 분해
    kmap = {r["id"]: r for r in kva}
    groups = {"R9F": [f"R9F-s{s}" for s in SEEDS], "C10": [f"C10-s{s}" for s in SEEDS], "R10full": [f"R10-full-s{s}" for s in SEEDS], "S0": ["S0"]}
    dumps = {m: load_dump(a.dump_dir, m) for g in groups.values() for m in g}
    out = {}
    per_gold = {}  # (g, id, gi) -> detected seeds
    for g, ms in groups.items():
        per_model = []
        for m in ms:
            d = dumps.get(m)
            if not d:
                continue
            tp = fn_b = fn_m = fp_b = fp_conf = fp_pure = fp_negrows = 0
            neg_n = neg_f = 0
            fn_len, fn_part, gold_len = Counter(), Counter(), Counter()
            fp_pos_rows = 0
            for k, x in d.items():
                golds = [tuple(t) for t in x["gold"]]
                final = [tuple(t) for t in x["final"]]
                row = kmap[k]
                others = [(s["start"], s["end"]) for s in row["spans"] if s.get("label") not in ("PERSON", "PERSON_ALIAS")]
                if not golds:
                    neg_n += 1
                    neg_f += int(bool(final))
                for gi, gsp in enumerate(golds):
                    gold_len[min(gsp[1] - gsp[0], 5)] += 1
                    ok = gsp in final
                    per_gold.setdefault((g, k, gi), []).append(ok)
                    if ok:
                        tp += 1
                    else:
                        ov = any(max(p[0], gsp[0]) < min(p[1], gsp[1]) for p in final)
                        fn_b += ov
                        fn_m += (not ov)
                        fn_len[min(gsp[1] - gsp[0], 5)] += 1
                        fn_part[followed(row["text"], gsp[1])] += 1
                if not golds:
                    fp_negrows += len(final)  # 공식 F1 은 정답 없는 문장의 FP 를 세지 않는다(오탐률로 따로 보고)
                for p in final:
                    if p in golds:
                        continue
                    if any(max(p[0], q[0]) < min(p[1], q[1]) for q in golds):
                        fp_b += 1
                    elif any(max(p[0], q[0]) < min(p[1], q[1]) for q in others):
                        fp_conf += 1
                    else:
                        fp_pure += 1
            fp_all = fp_b + fp_conf + fp_pure
            fp = fp_all - fp_negrows  # 공식 채점과 같은 기준(정답 있는 문장의 FP 만)
            fn = fn_b + fn_m
            per_model.append({"tp": tp, "fn": fn, "fn_boundary": fn_b, "fn_miss": fn_m, "fp": fp, "fp_in_no_gold_rows": fp_negrows, "fp_boundary": fp_b, "fp_ORG_LOC_confusion": fp_conf, "fp_pure": fp_pure,
                              "f1": 2 * tp / (2 * tp + fp + fn), "precision": tp / (tp + fp), "recall": tp / (tp + fn), "neg_fpr": neg_f / neg_n,
                              "fn_by_len": dict(sorted(fn_len.items())), "fn_followed": dict(fn_part), "gold_by_len": dict(sorted(gold_len.items()))})
        if per_model:
            keys = ["tp", "fn", "fn_boundary", "fn_miss", "fp", "fp_in_no_gold_rows", "fp_boundary", "fp_ORG_LOC_confusion", "fp_pure", "f1", "precision", "recall", "neg_fpr"]
            out[g] = {k: {"mean": round(mean([x[k] for x in per_model]), 4), "sd": round(sd([x[k] for x in per_model]), 4)} for k in keys}
            lens = sorted({l for x in per_model for l in x["fn_by_len"]})
            out[g]["fn_by_len_mean"] = {str(l): round(mean([x["fn_by_len"].get(l, 0) for x in per_model]), 1) for l in lens}
            out[g]["gold_by_len"] = {str(l): v for l, v in per_model[0]["gold_by_len"].items()}
            parts = sorted({l for x in per_model for l in x["fn_followed"]})
            out[g]["fn_followed_mean"] = {l: round(mean([x["fn_followed"].get(l, 0) for x in per_model]), 1) for l in parts}
    res["errors"] = out
    # (3) C10 은 잡고 R10-full 은 놓친 구간 / 반대
    lost = Counter()
    gained = Counter()
    n = 0
    lost_len, lost_f = Counter(), Counter()
    for (g, k, gi), v in per_gold.items():
        if g != "C10":
            continue
        w = per_gold.get(("R10full", k, gi))
        if not w:
            continue
        n += 1
        pc, pr = sum(v) / len(v), sum(w) / len(w)
        gsp = kmap[k]
        gold = [(s["start"], s["end"]) for s in gsp["spans"] if s.get("label") in ("PERSON", "PERSON_ALIAS")]
        gsp0 = gold[gi] if gi < len(gold) else None
        if pc >= 0.8 and pr <= 0.2:
            lost["C10>=0.8 & R10<=0.2"] += 1
            if gsp0:
                lost_len[min(gsp0[1] - gsp0[0], 5)] += 1
                lost_f[followed(gsp["text"], gsp0[1])] += 1
        if pr >= 0.8 and pc <= 0.2:
            gained["R10>=0.8 & C10<=0.2"] += 1
    res["flip"] = {"n_gold": n, "lost": dict(lost), "gained": dict(gained), "lost_len": dict(sorted(lost_len.items())), "lost_followed": dict(lost_f)}
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("written", a.out)


if __name__ == "__main__":
    main()
