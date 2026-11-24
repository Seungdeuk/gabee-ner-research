"""R10 개발 평가 결과 기반 오류 분석(학습·신규 데이터 없음). r10_rowpreds 덤프를 읽어 집계한다.

- 개발 데이터(R10 dev + R9 회귀 5세트)만 사용한다. R9 봉인셋·독립 평가셋 경로는 없다.
- 출력은 집계 수치와 단어(합성 개발 어휘) 수준이다. 문장 원문은 출력·저장하지 않는다.
사용: python r10_error_analysis.py --dump-dir <analysis> --out <json>
"""

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

DEV = Path(r"C:\Users\<user>\venvs\ner-runs\data_r10\r10_dev.jsonl")
RUNS = Path(r"C:\Users\<user>\venvs\ner-runs\r10\runs")
DEVRES = Path(r"C:\Users\<user>\venvs\ner-runs\r10\dev")
SEEDS = [7, 13, 21, 42, 99]
GROUPS = {
    "R9F": [f"R9F-s{s}" for s in SEEDS],
    "C10": [f"C10-s{s}" for s in SEEDS],
    "R10full": [f"R10-full-s{s}" for s in SEEDS],
}
R9SETS = ["v7_namefreq", "realistic_family", "v14_realistic", "nemotron_full", "nemotron_given"]
NER_MIN_SCORE = 0.5  # app/shared/privacy/ner.py 의 값과 같다(merge_ner 의 최소 점수)
PARTICLES = ("이", "가", "은", "는", "을", "를", "도", "와", "과", "의", "에", "한테", "께", "야", "아", "랑", "이랑", "만", "로", "으로")


def load_dump(d):
    out = {}
    for f in Path(d).glob("rowpreds_*.jsonl"):
        lines = f.read_text(encoding="utf-8").splitlines()
        head = json.loads(lines[0])
        out[head["label"]] = {x["id"] if x["set"] == "r10_dev" else f'{x["set"]}:{x["id"]}': x for x in map(json.loads, lines[1:])}
    return out


def rate(c, keys):
    n = sum(c[k] for k in keys)
    return None if n == 0 else (c["full"] + c["partial"]) / n


def exposed(cls):
    return cls in ("full", "partial")


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def sd(v):
    v = [x for x in v if x is not None]
    return st.pstdev(v) if len(v) > 1 else 0.0


def spearman(a, b):
    def rk(x):
        s = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and x[s[j + 1]] == x[s[i]]:
                j += 1
            for k in range(i, j + 1):
                r[s[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    ra, rb = rk(a), rk(b)
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def stage(item, gi):
    """노출된 정답 구간이 어느 단계에서 새는지: 원시 span/라벨/확률 기준 분류."""
    a, b = item["gold"][gi]
    ov = [r for r in item["raw"] if max(r[1], a) < min(r[2], b)]
    person_ov = [r for r in ov if str(r[0]).startswith("PERSON")]
    if person_ov and all(r[3] < NER_MIN_SCORE for r in person_ov):
        return "점수탈락(원시 PERSON 예측은 있으나 전부 점수<0.5, merge_ner 가 버림)"
    if any(r[1] <= a and r[2] >= b and r[3] >= NER_MIN_SCORE for r in person_ov):
        return "raw_PERSON_covers(점수>=0.5인데 최종 미반영: 병합·트림·친족어)"
    if person_ov:
        return "raw_PERSON_partial(점수>=0.5 경계 오류)"
    if ov:
        return "raw_other_label(다른 라벨로 예측)"
    p = (item.get("p_gold") or [None])[gi]
    if p is None:
        return "no_raw_p_unknown"
    return "no_raw_p<0.10" if p < 0.10 else ("no_raw_p0.10-0.50" if p < 0.50 else "no_raw_p>=0.50(토큰만 약하게)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    D = load_dump(a.dump_dir)
    dev = {r["id"]: r for r in map(json.loads, DEV.read_text(encoding="utf-8").splitlines()) if r}
    res = {"models_loaded": sorted(D)}
    labels = {g: [l for l in ls if l in D] for g, ls in GROUPS.items()}
    labels["S0"] = ["S0"] if "S0" in D else []
    allg = list(labels)

    # ---------- A1: 모델별 D1P 노출 / D1 오탐
    def dev_items(m, stress):
        return [x for k, x in D[m].items() if x["set"] == "r10_dev" and x["stress"] in stress]

    per_model = {}
    for g in allg:
        for m in labels[g]:
            p = dev_items(m, ("D1P",))
            n = [x for x in dev_items(m, ("D1", "D1_STANDALONE"))]
            ne = sum(1 for x in p for c in x["cls"] if exposed(c))
            ng = sum(len(x["cls"]) for x in p)
            fl = sum(1 for x in n if x["cls"] == ["neg_flagged"])
            per_model[m] = {"group": g, "D1P_exposure": ne / ng if ng else None, "D1_fpr": fl / len(n) if n else None}
    res["A1_per_model"] = per_model
    res["A1_group"] = {g: {"D1P_mean": mean([per_model[m]["D1P_exposure"] for m in labels[g]]), "D1P_sd": sd([per_model[m]["D1P_exposure"] for m in labels[g]]),
                          "D1fpr_mean": mean([per_model[m]["D1_fpr"] for m in labels[g]]), "D1fpr_sd": sd([per_model[m]["D1_fpr"] for m in labels[g]])} for g in allg if labels[g]}
    tr = [m for g in ("R9F", "C10", "R10full") for m in labels[g]]
    if len(tr) >= 5:
        xs = [per_model[m]["D1P_exposure"] for m in tr]
        ys = [per_model[m]["D1_fpr"] for m in tr]
        res["A1_spearman_D1P_vs_D1fpr(all trained models)"] = spearman(xs, ys)
        for g in ("R9F", "C10", "R10full"):
            if len(labels[g]) >= 4:
                res[f"A1_spearman_{g}"] = spearman([per_model[m]["D1P_exposure"] for m in labels[g]], [per_model[m]["D1_fpr"] for m in labels[g]])

    # ---------- A2: 쌍 단위 2x2 (같은 단어의 사람문장 미탐 / 명사문장 오탐)
    pairs = defaultdict(dict)
    for r in dev.values():
        if r.get("pair_id") and r.get("pair_role"):
            pairs[(r["pair_id"], r["pair_word"])][r["pair_role"]] = r["id"]
    quad = {}
    for g in allg:
        for m in labels[g]:
            c = Counter()
            for _, rr in pairs.items():
                if "person" in rr and "noun" in rr and rr["person"] in D[m] and rr["noun"] in D[m]:
                    pe = any(exposed(x) for x in D[m][rr["person"]]["cls"])
                    nf = D[m][rr["noun"]]["cls"] == ["neg_flagged"]
                    c[("사람 미탐" if pe else "사람 가림") + "+" + ("명사 오탐" if nf else "명사 정상")] += 1
            quad[m] = dict(c)
    res["A2_pair_quadrants_per_model"] = quad
    qg = {}
    for g in allg:
        if not labels[g]:
            continue
        keys = sorted({k for m in labels[g] for k in quad[m]})
        qg[g] = {k: round(mean([quad[m].get(k, 0) for m in labels[g]]), 1) for k in keys}
    res["A2_pair_quadrants_group_mean"] = qg
    res["A2_n_pairs"] = sum(1 for rr in pairs.values() if "person" in rr and "noun" in rr)

    # ---------- A3: 단어별 (시드 간 불일치)
    word_rows = defaultdict(lambda: {"person": None, "noun": None})
    for (pid, w), rr in pairs.items():
        if "person" in rr and "noun" in rr:
            word_rows[(pid, w)] = rr
    word = {}
    for g in allg:
        if not labels[g]:
            continue
        per = {}
        for (pid, w), rr in word_rows.items():
            pe = [any(exposed(x) for x in D[m][rr["person"]]["cls"]) for m in labels[g] if rr["person"] in D[m]]
            nf = [D[m][rr["noun"]]["cls"] == ["neg_flagged"] for m in labels[g] if rr["noun"] in D[m]]
            per[(pid, w)] = (sum(pe), len(pe), sum(nf), len(nf))
        word[g] = per
    hist = {}
    for g, per in word.items():
        hist[g] = {"person_exposed_seeds_hist": dict(sorted(Counter(v[0] for v in per.values()).items())), "noun_flagged_seeds_hist": dict(sorted(Counter(v[2] for v in per.values()).items())),
                   "n_words": len(per), "n_seeds": len(labels[g])}
        unstable = sum(1 for v in per.values() if 0 < v[0] < v[1])
        hist[g]["person_unstable_words"] = unstable
        hist[g]["person_unstable_share"] = unstable / len(per)
        hist[g]["noun_unstable_words"] = sum(1 for v in per.values() if 0 < v[2] < v[3])
    res["A3_seed_agreement_hist"] = hist
    if "R10full" in word and "R9F" in word:
        diffs = []
        for k in word["R10full"]:
            r10 = word["R10full"][k][0] / word["R10full"][k][1]
            r9 = word["R9F"][k][0] / word["R9F"][k][1]
            diffs.append((round(r10 - r9, 2), k[1], round(r9, 2), round(r10, 2), round(word["R10full"][k][2] / word["R10full"][k][3], 2), round(word["R9F"][k][2] / word["R9F"][k][3], 2)))
        diffs.sort()
        res["A3_words_R10_better(person_exposure_delta,word,R9F,R10,R10_noun_fpr,R9F_noun_fpr)"] = diffs[:12]
        res["A3_words_R10_worse"] = diffs[-12:][::-1]
    if "S0" in word:
        always_s0 = sum(1 for v in word["S0"].values() if v[0] == v[1])
        res["A3_words_exposed_by_S0"] = always_s0
    # 모든 모델이 노출한 단어 / 하나도 노출하지 않은 단어
    allm = [m for g in ("R9F", "C10", "R10full") for m in labels[g]]
    if allm:
        cnt = {}
        for (pid, w), rr in word_rows.items():
            cnt[(pid, w)] = (sum(any(exposed(x) for x in D[m][rr["person"]]["cls"]) for m in allm), sum(D[m][rr["noun"]]["cls"] == ["neg_flagged"] for m in allm))
        res["A3_words_exposed_by_all_models"] = sum(1 for v in cnt.values() if v[0] == len(allm))
        res["A3_words_exposed_by_no_model"] = sum(1 for v in cnt.values() if v[0] == 0)
        res["A3_words_noun_flagged_by_all_models"] = sum(1 for v in cnt.values() if v[1] == len(allm))
        res["A3_words_noun_flagged_by_no_model"] = sum(1 for v in cnt.values() if v[1] == 0)
        both = [k[1] for k, v in cnt.items() if v[0] >= len(allm) * 0.6 and v[1] >= len(allm) * 0.6]
        res["A3_words_both_errors_gt60pct_models"] = both

    # ---------- A4: 문장 구조(frame_tag/category)별
    st_rows = {}
    for g in allg:
        if not labels[g]:
            continue
        agg = defaultdict(lambda: [0, 0, 0, 0])
        for m in labels[g]:
            for k, x in D[m].items():
                if x["set"] != "r10_dev" or x["stress"] not in ("D1P", "D1"):
                    continue
                r = dev[k]
                key = (x["stress"], r.get("frame_tag") or "-", r.get("category") or "-")
                if x["stress"] == "D1P":
                    agg[key][0] += sum(1 for c in x["cls"] if exposed(c))
                    agg[key][1] += len(x["cls"])
                else:
                    agg[key][2] += int(x["cls"] == ["neg_flagged"])
                    agg[key][3] += 1
        st_rows[g] = {f"{k[0]}|{k[1]}|{k[2]}": {"exposure": (v[0] / v[1]) if v[1] else None, "fpr": (v[2] / v[3]) if v[3] else None, "n": v[1] or v[3]} for k, v in sorted(agg.items())}
    res["A4_structure"] = st_rows

    # ---------- A5: 단계 분해
    stg = {}
    fp_stage = {}
    for g in allg:
        if not labels[g]:
            continue
        c = Counter()
        f = Counter()
        for m in labels[g]:
            for k, x in D[m].items():
                if x["set"] != "r10_dev":
                    continue
                if x["stress"] == "D1P":
                    for gi, cl in enumerate(x["cls"]):
                        if exposed(cl):
                            c[stage(x, gi)] += 1
                elif x["stress"] in ("D1", "D1_STANDALONE") and x["cls"] == ["neg_flagged"]:
                    covered = any(any(str(rw[0]).startswith("PERSON") and max(rw[1], fa) < min(rw[2], fb) for rw in x["raw"]) for fa, fb in x["final"])
                    f["raw_PERSON_span_존재(모델 예측 때문)" if covered else "raw에 없음(규칙 병합이 만든 PERSON)"] += 1
        n = len(labels[g])
        stg[g] = {k: round(v / n, 1) for k, v in sorted(c.items())}
        fp_stage[g] = {k: round(v / n, 1) for k, v in sorted(f.items())}
    res["A5_D1P_miss_stage_per_model_mean"] = stg
    res["A5_D1_fp_stage_per_model_mean"] = fp_stage

    # ---------- A6: 확률 분포 (D1P 정답 구간의 PERSON 확률 / D1 명사 단어의 PERSON 확률)
    prob = {}
    for g in allg:
        if not labels[g]:
            continue
        pg, pw = [], []
        for m in labels[g]:
            for k, x in D[m].items():
                if x["set"] != "r10_dev":
                    continue
                if x["stress"] == "D1P" and x.get("p_gold"):
                    pg += [p for p in x["p_gold"] if p is not None]
                if x["stress"] == "D1" and x.get("p_word") is not None:
                    pw.append(x["p_word"])
        q = lambda v: {"mean": round(mean(v), 3), "median": round(st.median(v), 3), "share_ge_0.5": round(sum(1 for p in v if p >= 0.5) / len(v), 3), "share_0.1_0.9": round(sum(1 for p in v if 0.1 <= p <= 0.9) / len(v), 3)} if v else None
        prob[g] = {"D1P_person_prob(gold)": q(pg), "D1_noun_word_person_prob": q(pw)}
    res["A6_prob"] = prob

    # ---------- B: R9 회귀 5세트
    setres = {}
    for g in allg:
        if not labels[g]:
            continue
        setres[g] = {}
        for s in R9SETS:
            vals = []
            for m in labels[g]:
                c = Counter()
                for k, x in D[m].items():
                    if x["set"] == s:
                        for cl in x["cls"]:
                            c[cl] += 1
                ng = c["full"] + c["partial"] + c["masked"]
                vals.append((c["full"] + c["partial"]) / ng if ng else None)
            setres[g][s] = {"mean": mean(vals), "min": min(vals), "max": max(vals), "per_seed": [round(v, 4) for v in vals]}
        setres[g]["macro"] = mean([setres[g][s]["mean"] for s in R9SETS])
    res["B_set_exposure"] = setres
    # 회귀 사례: R9F 5시드 모두 가렸는데 해당 모델이 노출한 정답 구간
    reg = {}
    if labels["R9F"]:
        keys = defaultdict(list)
        for k, x in D[labels["R9F"][0]].items():
            if x["set"] in R9SETS:
                keys[x["set"]].append(k)
        for g in ("C10", "R10full"):
            for m in labels[g]:
                c = Counter()
                lens = Counter()
                nxt = Counter()
                stg2 = Counter()
                for s in R9SETS:
                    for k in keys[s]:
                        x = D[m].get(k)
                        if not x:
                            continue
                        for gi, cl in enumerate(x["cls"]):
                            if exposed(cl) and all(not exposed(D[r][k]["cls"][gi]) for r in labels["R9F"] if k in D[r]):
                                c[s] += 1
                                aa, bb = x["gold"][gi]
                                lens[bb - aa if bb - aa <= 4 else 5] += 1
                                stg2[stage(x, gi)] += 1
                reg[m] = {"newly_exposed_vs_all_R9F(by set)": dict(c), "gold_len": dict(sorted(lens.items())), "stage": dict(stg2)}
                # 반대 방향: R9F 5시드 모두 노출했는데 이 모델은 가린 것
                c2 = Counter()
                for s in R9SETS:
                    for k in keys[s]:
                        x = D[m].get(k)
                        if not x:
                            continue
                        for gi, cl in enumerate(x["cls"]):
                            if (not exposed(cl)) and all(exposed(D[r][k]["cls"][gi]) for r in labels["R9F"] if k in D[r]):
                                c2[s] += 1
                reg[m]["newly_masked_vs_all_R9F(by set)"] = dict(c2)
    res["B_regression_cases"] = reg

    # 시드 7 v14_realistic: 노출된 구간 중 조사/길이/단계
    if "R10-full-s7" in D:
        m = "R10-full-s7"
        detail = Counter()
        tot = 0
        by_seed = {}
        for mm in labels["R10full"] + labels["C10"] + labels["R9F"] + (["S0"] if "S0" in D else []):
            n_exp = 0
            n_g = 0
            for k, x in D[mm].items():
                if x["set"] == "v14_realistic":
                    for cl in x["cls"]:
                        if cl in ("full", "partial", "masked"):
                            n_g += 1
                            n_exp += int(exposed(cl))
            by_seed[mm] = {"exposed": n_exp, "gold": n_g, "rate": round(n_exp / n_g, 4) if n_g else None}
        res["B_v14_realistic_per_model"] = by_seed
        ids_new = [k for k, x in D[m].items() if x["set"] == "v14_realistic" and any(exposed(c) and all(not exposed(D[r][k]["cls"][gi]) for r in labels["R9F"]) for gi, c in enumerate(x["cls"]))]
        res["B_v14_s7_newly_exposed_sentences"] = len(ids_new)

    # ---------- E: 사후 민감도(탐색용, 사전 등록 평가·후보 선정과 무관): merge_ner 최소 점수를 낮추면?
    # 근사: 최종 span 에 점수>=t 인 원시 PERSON span(길이>=2)을 합친다(규칙 병합 세부는 재현하지 않음).
    def final_t(x, t):
        fs = [tuple(s) for s in x["final"]]
        for lab, a_, b_, sc in x["raw"]:
            if str(lab).startswith("PERSON") and sc >= t and b_ - a_ >= 2:
                fs.append((a_, b_))
        return fs

    def cls_of(golds, final):
        out = []
        for g0, g1 in golds:
            cov = set()
            for p0, p1 in final:
                cov.update(range(max(p0, g0), min(p1, g1)))
            leaked = (g1 - g0) - len(cov)
            out.append("full" if leaked >= g1 - g0 else ("partial" if leaked > 0 else "masked"))
        return out

    whatif = {}
    for g in allg:
        if not labels[g]:
            continue
        whatif[g] = {}
        for t in (0.5, 0.45, 0.4, 0.35, 0.3, 0.25):
            r9e, r9n, d1p_e, d1p_n, fp_f, fp_n = [], [], 0, 0, 0, 0
            for m in labels[g]:
                e = n_ = 0
                for k, x in D[m].items():
                    ft = final_t(x, t)
                    cl = cls_of([tuple(q) for q in x["gold"]], ft)
                    if x["set"] in R9SETS and cl:
                        e += sum(1 for c in cl if exposed(c))
                        n_ += len(cl)
                    if x["set"] == "r10_dev" and x["stress"] == "D1P":
                        d1p_e += sum(1 for c in cl if exposed(c))
                        d1p_n += len(cl)
                    if x["set"] == "r10_dev" and x["stress"] in ("D1", "D1_STANDALONE"):
                        fp_n += 1
                        fp_f += int(bool(ft))
                r9e.append(e / n_ if n_ else None)
            whatif[g][str(t)] = {"R9dev_pooled_exposure": round(mean(r9e), 4), "D1P_exposure": round(d1p_e / d1p_n, 4), "D1_fpr": round(fp_f / fp_n, 4)}
    res["E_threshold_whatif(approx, exploratory)"] = whatif

    # 단어 수준 상관: (R10full - R9F) 사람 미탐 변화 vs 명사 오탐 변화
    if "R10full" in word and "R9F" in word:
        dx = [word["R10full"][k][0] / word["R10full"][k][1] - word["R9F"][k][0] / word["R9F"][k][1] for k in word["R10full"]]
        dy = [word["R10full"][k][2] / word["R10full"][k][3] - word["R9F"][k][2] / word["R9F"][k][3] for k in word["R10full"]]
        res["A3_spearman_word_delta_person_exposure_vs_delta_noun_fpr(R10full-R9F)"] = spearman(dx, dy)
        dx = [word["C10"][k][0] / word["C10"][k][1] - word["R9F"][k][0] / word["R9F"][k][1] for k in word["C10"]]
        dy = [word["C10"][k][2] / word["C10"][k][3] - word["R9F"][k][2] / word["R9F"][k][3] for k in word["C10"]]
        res["A3_spearman_word_delta_person_exposure_vs_delta_noun_fpr(C10-R9F)"] = spearman(dx, dy)

    # ---------- C: baseline_val 포화
    vals = {}
    for d in sorted(RUNS.glob("r10-*")):
        tl = d / "train_log.json"
        pj = d / "pipeline_done.json"
        if not tl.exists() or not pj.exists():
            continue
        t = json.loads(tl.read_text(encoding="utf-8"))
        eps = [round(e["val_score"], 6) for e in t["epochs"]]
        best = max(eps)
        dj = json.loads((DEVRES / f"{d.name}.json").read_text(encoding="utf-8"))
        v = list(dj.values())[0]
        vals[d.name] = {"best_val": best, "epochs_val": eps, "n_tied_best": sum(1 for e in eps if e == best), "best_epoch": t["best_epoch"], "stopped_epoch": t["stopped_epoch"],
                        "D1P": v["r10dev"]["D1P"]["total_exposure_rate"], "targets_macro": v["targets_macro_exposure"], "R9dev_macro": v["macro"]["total_exposure"]}
    res["C_baseline_val"] = vals
    if vals:
        bv = [x["best_val"] for x in vals.values()]
        res["C_baseline_val_summary"] = {
            "distinct_best_val_values": sorted(set(bv)),
            "share_runs_with_tie_at_best": sum(1 for x in vals.values() if x["n_tied_best"] > 1) / len(vals),
            "spearman_bestval_vs_D1P": spearman(bv, [x["D1P"] for x in vals.values()]),
            "spearman_bestval_vs_targets_macro": spearman(bv, [x["targets_macro"] for x in vals.values()]),
            "spearman_bestval_vs_R9dev_macro": spearman(bv, [x["R9dev_macro"] for x in vals.values()]),
            "mean_range_of_epoch_val_within_run": mean([max(x["epochs_val"]) - min(x["epochs_val"]) for x in vals.values()]),
        }
        for arm in ("R10-full", "C10"):
            sub = {k: x for k, x in vals.items() if k.startswith(f"r10-{arm}-")}
            if len(sub) >= 4:
                res[f"C_baseline_val_{arm}"] = {
                    "spearman_bestval_vs_D1P": spearman([x["best_val"] for x in sub.values()], [x["D1P"] for x in sub.values()]),
                    "spearman_bestval_vs_targets": spearman([x["best_val"] for x in sub.values()], [x["targets_macro"] for x in sub.values()]),
                    "per_seed": {k.split("-s")[-1]: {"val": x["best_val"], "best_ep": x["best_epoch"], "D1P": round(x["D1P"], 3), "targets": round(x["targets_macro"], 3), "R9dev": round(x["R9dev_macro"], 4)} for k, x in sorted(sub.items())},
                }
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("written", a.out, "models", len(D))


if __name__ == "__main__":
    main()
