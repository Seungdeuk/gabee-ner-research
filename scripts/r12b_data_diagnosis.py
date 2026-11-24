"""R12b 1단계: R12 일반화 실패 원인 분석(데이터 쪽 진단). 학습·평가를 하지 않고 데이터 파일만 읽는다.
봉인 평가셋(r11_stress_v1)·독립 평가셋 경로는 이 파일에 없다. 결과 JSON 에는 원문 문장을 싣지 않고 id 와 집계만 둔다.

검사 항목
 A. 구성: 세트별 문장 수·양성/음성·유형·쌍 비율·길이·문체
 B. 단서 비대칭: 쌍 단어 뒤 1~2글자(조사)·앞 단어가 사람/일반명사 어느 한쪽에만 나오는 정도
 C. 단서만 쓰는 분류기: 학습(r12_extra 또는 r11_B_extra)에서 단어 자체를 가린 채 앞뒤 문맥 n-gram 만으로 사람/명사를 학습 → 각 개발셋 정확도.
    "단어 정체"가 아니라 "작성 방식 단서"로 풀리는 정도를 보는 근거(분포 안 vs 밖 차이)
 D. 어휘 겹침: 쌍 단어·PERSON 값이 학습/개발 사이에 겹치는 정도
 E. 문장 유사도: 개발 문장별 학습셋 최근접(문자 3-gram TF-IDF 코사인) 평균·상위 분포
"""

import collections
import json
import math
import re
import statistics as st
from pathlib import Path

D = Path(r"C:\Users\<user>\venvs\ner-runs\data_r12")
OUT = Path(r"C:\Users\<user>\venvs\ner-runs\r12b\analysis")
SETS = {"r12_extra": "train", "r11_B_extra": "train", "r12_val": "val", "r12_dev": "dev", "r11_dev": "dev"}


def load(n):
    return [json.loads(x) for x in (D / f"{n}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def pair_rows(rows):
    """대조쌍 문장만: (word, label_person?, text, span)"""
    out = []
    for r in rows:
        w, role = r.get("pair_word"), r.get("pair_role") or r.get("role")
        if not w or role not in ("person", "noun"):
            continue
        t = r["text"]
        if role == "person" and r["spans"]:
            s = next((x for x in r["spans"] if x["label"] == "PERSON"), None)
            if not s:
                continue
            i, j = s["start"], s["end"]
        else:
            i = t.find(w)
            if i < 0:
                continue
            j = i + len(w)
        out.append({"id": r["id"], "word": w, "person": role == "person", "text": t, "i": i, "j": j})
    return out


def ctx_feats(p, mask_word=True):
    t, i, j = p["text"], p["i"], p["j"]
    f = []
    after, before = t[j:j + 3], t[max(0, i - 3):i]
    f += [f"A1:{after[:1]}", f"A2:{after[:2]}", f"A3:{after}", f"B1:{before[-1:]}", f"B2:{before[-2:]}"]
    toks = t.split()
    # 단어 단위 앞뒤 토큰(쌍 단어 포함 어절 제외)
    pos = 0
    for k, tok in enumerate(toks):
        s = t.find(tok, pos)
        e = s + len(tok)
        pos = e
        if s <= i < e or s < j <= e:
            if k > 0:
                f.append(f"PW:{toks[k - 1]}")
            if k + 1 < len(toks):
                f.append(f"NW:{toks[k + 1]}")
            f.append(f"POS:{'first' if k == 0 else 'last' if k == len(toks) - 1 else 'mid'}")
            break
    f.append(f"END:{t.rstrip()[-3:]}")
    return f


def nb_train(rows):
    cnt = {True: collections.Counter(), False: collections.Counter()}
    prior = collections.Counter()
    for p in rows:
        prior[p["person"]] += 1
        cnt[p["person"]].update(set(ctx_feats(p)))
    vocab = set(cnt[True]) | set(cnt[False])
    return cnt, prior, len(vocab)


def nb_pred(model, p):
    cnt, prior, V = model
    sc = {}
    for y in (True, False):
        n = prior[y] + 1e-9
        s = math.log((prior[y] + 1) / (sum(prior.values()) + 2))
        for f in set(ctx_feats(p)):
            s += math.log((cnt[y][f] + 0.5) / (n + 0.5 * V))
        sc[y] = s
    return sc[True] > sc[False]


def acc(model, rows):
    if not rows:
        return None
    return round(sum(nb_pred(model, p) == p["person"] for p in rows) / len(rows), 4)


def particle_table(rows, top=12):
    c = {True: collections.Counter(), False: collections.Counter()}
    for p in rows:
        c[p["person"]][p["text"][p["j"]:p["j"] + 1] or "∅"] += 1
    tot = {k: sum(v.values()) for k, v in c.items()}
    keys = sorted(set(c[True]) | set(c[False]), key=lambda k: -(c[True][k] + c[False][k]))[:top]
    return {k: {"person%": round(100 * c[True][k] / max(1, tot[True]), 1), "noun%": round(100 * c[False][k] / max(1, tot[False]), 1)} for k in keys}


def asym(rows, key_fn, min_n=8):
    """한쪽 라벨에만 거의 나오는 단서(순도>=0.9, 빈도>=min_n)의 개수와 그 단서가 덮는 문장 비율"""
    c = collections.defaultdict(lambda: [0, 0])
    for p in rows:
        for k in set(key_fn(p)):
            c[k][p["person"]] += 1
    pure = {k: v for k, v in c.items() if sum(v) >= min_n and max(v) / sum(v) >= 0.9}
    covered = sum(1 for p in rows if any(k in pure for k in set(key_fn(p))))
    top = sorted(pure.items(), key=lambda kv: -sum(kv[1]))[:8]
    return {"pure_cues": len(pure), "covered_rows%": round(100 * covered / max(1, len(rows)), 1),
            "top": [(k, {"person": v[1] and v[1] or 0, "noun": v[0]}) for k, v in top]}


def ngrams(t, n=3):
    t = re.sub(r"\s+", " ", t)
    return collections.Counter(t[i:i + n] for i in range(len(t) - n + 1))


def tfidf_nn(train, dev):
    df = collections.Counter()
    tv = [ngrams(r["text"]) for r in train]
    for g in tv:
        df.update(g.keys())
    N = len(tv)
    idf = {k: math.log((1 + N) / (1 + v)) + 1 for k, v in df.items()}

    def vec(g):
        v = {k: c * idf.get(k, math.log(1 + N) + 1) for k, c in g.items()}
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1
        return {k: x / nrm for k, x in v.items()}

    tvec = [vec(g) for g in tv]
    inv = collections.defaultdict(list)
    for idx, v in enumerate(tvec):
        for k in v:
            inv[k].append(idx)
    sims = []
    for r in dev:
        v = vec(ngrams(r["text"]))
        sc = collections.Counter()
        for k, x in v.items():
            for idx in inv.get(k, ()):
                sc[idx] += x * tvec[idx][k]
        sims.append(max(sc.values()) if sc else 0.0)
    q = lambda p: round(sorted(sims)[int(p * (len(sims) - 1))], 3)
    return {"mean": round(st.mean(sims), 3), "p50": q(0.5), "p90": q(0.9), "share>=0.6%": round(100 * sum(s >= 0.6 for s in sims) / len(sims), 1)}


def comp(rows):
    L = [len(r["text"]) for r in rows]
    neg = sum(1 for r in rows if not r["spans"])
    types = collections.Counter(r.get("type") or r.get("primary_stress_type") or "?" for r in rows)
    ends = collections.Counter(r["text"].rstrip()[-1:] for r in rows)
    return {"n": len(rows), "neg%": round(100 * neg / len(rows), 1), "len_mean": round(st.mean(L), 1), "len_p10_p90": [sorted(L)[len(L) // 10], sorted(L)[9 * len(L) // 10]],
            "types": dict(types.most_common(8)), "end_punct": dict(ends.most_common(5)),
            "pair_rows": sum(1 for r in rows if r.get("pair_word"))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {n: load(n) for n in SETS}
    pairs = {n: pair_rows(r) for n, r in data.items()}
    res = {"A_composition": {n: comp(r) for n, r in data.items()}}
    res["B_cue_asymmetry"] = {}
    for n, p in pairs.items():
        res["B_cue_asymmetry"][n] = {"pairs": len(p), "person": sum(x["person"] for x in p), "noun": sum(not x["person"] for x in p),
                                      "next_char": particle_table(p), "pure_next2": asym(p, lambda q: [f for f in ctx_feats(q) if f.startswith("A2:")]),
                                      "pure_any_cue": asym(p, ctx_feats)}
    res["C_cue_only_classifier"] = {}
    for tr in ("r12_extra", "r11_B_extra"):
        # 학습셋 두 개를 단독으로 학습 → 각 개발셋 정확도(단어 정체는 쓰지 않는 앞뒤 문맥 특징만)
        m = nb_train(pairs[tr] if tr in pairs else pair_rows(load(tr)))
        res["C_cue_only_classifier"][tr] = {n: acc(m, pairs[n]) for n in ("r12_val", "r12_dev", "r11_dev")}
        res["C_cue_only_classifier"][tr]["train_self"] = acc(m, pairs[tr] if tr in pairs else pair_rows(load(tr)))
    # r12_extra 는 SETS 에 없을 수 있으므로 별도 로드
    D_ = {}
    for n in ("r12_extra", "r11_B_extra"):
        tw = {p["word"] for p in pairs[n]}
        tn = {p["text"][p["i"]:p["j"]] for p in pairs[n] if p["person"]}
        for d in ("r12_val", "r12_dev", "r11_dev"):
            dw = {p["word"] for p in pairs[d]}
            dn = {p["text"][p["i"]:p["j"]] for p in pairs[d] if p["person"]}
            D_[f"{n}->{d}"] = {"dev_pair_words": len(dw), "words_seen_in_train%": round(100 * len(dw & tw) / max(1, len(dw)), 1),
                               "dev_person_values": len(dn), "values_seen_in_train%": round(100 * len(dn & tn) / max(1, len(dn)), 1)}
    # 전체 PERSON 값 겹침(쌍 아님 포함)
    pv = lambda rows: {r["text"][s["start"]:s["end"]] for r in rows for s in r["spans"] if s["label"] == "PERSON"}
    for n in ("r12_extra", "r11_B_extra"):
        tv_ = pv(data[n])
        for d in ("r12_val", "r12_dev", "r11_dev"):
            dv = pv(data[d])
            D_[f"{n}->{d}"]["all_person_values_seen%"] = round(100 * len(dv & tv_) / max(1, len(dv)), 1)
    res["D_lexical_overlap"] = D_
    res["E_nearest_train_similarity"] = {}
    for tr in ("r12_extra", "r11_B_extra"):
        res["E_nearest_train_similarity"][tr] = {d: tfidf_nn(data[tr], data[d]) for d in ("r12_val", "r12_dev", "r11_dev")}
    # 대표 사례(원문은 합성 데이터이므로 id+문장 일부만)
    samp = {}
    for n in ("r12_extra", "r12_dev", "r11_dev"):
        pp = pairs[n]
        samp[n] = {"person": [(p["id"], p["text"][:60]) for p in pp if p["person"]][:3], "noun": [(p["id"], p["text"][:60]) for p in pp if not p["person"]][:3]}
    res["F_samples"] = samp
    (OUT / "diagnosis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("C_cue_only_classifier", "D_lexical_overlap", "E_nearest_train_similarity")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
