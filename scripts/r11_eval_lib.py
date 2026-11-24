"""R11 평가 라이브러리(사전 등록 NER_R11_PREREGISTRATION.md §7-1). `r8_eval_lib` 의 판정(최종 예측 = detect_with_ner 의 PERSON 구간)을 그대로 쓰고
Exact Span Recall·가림률·경계만 틀림·전체/공식 Precision·쌍 문맥 정확률·유형별 지표를 더한다. 모델·DB·설정 미수정, 문장 원문을 결과에 싣지 않는다.

정답 구간 하나는 서로 배타적인 4가지 중 하나다.
 exact        : 예측 구간과 (start, end) 가 정확히 같음 (Recall 에 포함)
 boundary_only: 정답 글자가 모두 가려졌지만 경계가 다름(과대 마스킹 등) (가림률에는 포함, Recall 에는 불포함)
 partial      : 일부만 가려짐(부분 노출)
 full         : 전혀 안 가려짐(완전 노출)
가림률 = exact + boundary_only, Recall = exact, 전체 노출률 = partial + full = 1 - 가림률.  항등식: boundary_only == (가림 구간 수) - exact (독립 집계로 검증).
"""

import math
from collections import defaultdict

import r8_eval_lib as lib


def classify_row(golds, pred):
    """한 문장의 정답 구간 분류. 반환: (exact, boundary_only, partial, full, masked_all)."""
    pset = set(pred)
    ex = bo = pa = fu = 0
    for g in golds:
        cov = set()
        for a, b in pred:
            cov.update(range(max(a, g[0]), min(b, g[1])))
        leaked = (g[1] - g[0]) - len(cov)
        if g in pset:
            ex += 1
        elif leaked == 0:
            bo += 1
        elif leaked >= g[1] - g[0]:
            fu += 1
        else:
            pa += 1
    return ex, bo, pa, fu, (pa + fu) == 0


def wilson(k, n):
    return lib.wilson(k, n)


def score_rows_r11(rows, pred_fn=None, keep_rows=False):
    """rows 전체를 R11 지표로 채점한다(한 유형 안 전체)."""
    ex = bo = pa = fu = 0
    tp = fp_gold = fp_neg = 0
    neg_n = neg_flag = 0
    per_row = []
    for r in rows:
        golds = lib.gold_person(r)
        text = r["text"]
        pred = pred_fn(r) if pred_fn else lib.final_person_spans(text)
        if golds:
            gset = set(golds)
            e, b, p, f, m = classify_row(golds, pred)
            ex += e
            bo += b
            pa += p
            fu += f
            tp += sum(1 for g in golds if g in pred)
            fp_gold += sum(1 for x in pred if x not in gset)
            per_row.append({"id": r.get("id"), "masked_all": m, "pred_any": bool(pred), "n_gold": len(golds)})
        else:
            neg_n += 1
            neg_flag += int(bool(pred))
            fp_neg += len(pred)
            per_row.append({"id": r.get("id"), "masked_all": None, "pred_any": bool(pred), "n_gold": 0})
    n = ex + bo + pa + fu
    masked = ex + bo
    # 항등식 검증(독립 집계): 가림 구간 수 - exact == boundary_only
    identity_ok = (masked - ex) == bo and (pa + fu) == n - masked
    fn = n - ex
    prec_off = tp / (tp + fp_gold) if (tp + fp_gold) else None
    prec_all = tp / (tp + fp_gold + fp_neg) if (tp + fp_gold + fp_neg) else None
    rec = ex / n if n else None

    def f1(p):
        return (2 * p * rec / (p + rec)) if (p and rec and (p + rec)) else (0.0 if (p is not None and rec is not None) else None)

    out = {
        "n_sentences": len(rows), "n_gold_spans": n, "n_neg_sentences": neg_n,
        "exact": ex, "boundary_only": bo, "partial": pa, "full": fu,
        "exact_recall": rec, "exact_recall_ci": wilson(ex, n),
        "mask_rate": masked / n if n else None,
        "boundary_only_rate": bo / n if n else None,
        "partial_rate": pa / n if n else None, "full_rate": fu / n if n else None,
        "total_exposure_rate": (pa + fu) / n if n else None, "total_exposure_ci": wilson(pa + fu, n),
        "precision_official": prec_off, "precision_all": prec_all,
        "f1_official": f1(prec_off), "f1_all": f1(prec_all),
        "tp": tp, "fn": fn, "fp_in_gold_rows": fp_gold, "fp_in_neg_rows": fp_neg,
        "neg_flagged": neg_flag, "nonperson_fpr": neg_flag / neg_n if neg_n else None, "nonperson_fpr_ci": wilson(neg_flag, neg_n),
        "identity_ok": identity_ok,
    }
    if keep_rows:
        out["_rows"] = per_row
    return out


def pair_accuracy(rows, per_row):
    """쌍 문맥 정확률: 사람 문장 정답 구간 전부 가림 + 명사 문장 PERSON 예측 없음."""
    byid = {x["id"]: x for x in per_row}
    pairs = defaultdict(dict)
    for r in rows:
        if r.get("pair_id"):
            pairs[r["pair_id"]][r["pair_role"]] = byid[r["id"]]
    ok = person_ok = noun_ok = n = 0
    for p in pairs.values():
        if {"person", "noun"} <= set(p):
            n += 1
            a = bool(p["person"]["masked_all"])
            b = not p["noun"]["pred_any"]
            person_ok += a
            noun_ok += b
            ok += a and b
    return {"n_pairs": n, "context_correct": ok, "pair_context_accuracy": ok / n if n else None,
            "pair_context_ci": wilson(ok, n), "person_masked_rate": person_ok / n if n else None,
            "noun_clean_rate": noun_ok / n if n else None}


def score_dataset(rows, pred_fn=None):
    """유형(type)별과 전체, 쌍 지표를 한 번에. rows 의 type 필드 사용."""
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    res = {"by_type": {}}
    for t, rs in by.items():
        res["by_type"][t] = score_rows_r11(rs, pred_fn)
    allr = score_rows_r11(rows, pred_fn, keep_rows=True)
    pr = allr.pop("_rows")
    res["ALL"] = allr
    res["pairs"] = pair_accuracy(rows, pr)
    # NEWNAME 중 이름 부분이 참조 집합에 없는 구간만(unseen only)
    unseen = [r for r in by.get("NEWNAME", []) if r.get("name_seen_in_reference") and not any(r["name_seen_in_reference"])]
    if unseen:
        res["by_type"]["NEWNAME_unseen_only"] = score_rows_r11(unseen, pred_fn)
    # 오탐 음성 4집합 매크로, D1 오탐(D1 명사 문장 + 명사만)
    neg_sets = [t for t in ("NEG_ORGLOC", "NEG_FAMILY", "NEG_D2", "NEG_GENERAL") if t in res["by_type"]]
    res["neg_fpr_by_set"] = {t: res["by_type"][t]["nonperson_fpr"] for t in neg_sets}
    res["neg_fpr_macro"] = sum(res["neg_fpr_by_set"].values()) / len(neg_sets) if neg_sets else None
    d1 = by.get("D1", [])
    res["D1_fpr"] = score_rows_r11(d1, pred_fn)["nonperson_fpr"] if d1 else None
    res["identity_ok_all"] = all(v["identity_ok"] for v in res["by_type"].values()) and res["ALL"]["identity_ok"]
    return res
