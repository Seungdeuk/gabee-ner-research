"""운영 교체 평가 실행기(S0 vs R11 B-new seed 21). 사전 판정 기준: docs/security/r11_validation/ops_replacement/ops_decision_criteria.json

- 두 모델에 같은 입력·정답·평가 코드(r11_eval_lib 의 판정, 최종 예측 = detect_with_ner 의 PERSON 구간)를 적용하고, 문장(또는 쌍) 단위 쌍체 군집 부트스트랩으로 차이의 95% 구간을 낸다.
- 실제 모드(`--sealed-dir`): MANIFEST.sha256 으로 봉인 무결성 확인 → 모델 식별(ops_model_lock.json) 해시 확인 → 판정 기준 해시 확인 → `EVALUATED.lock` 생성(없을 때만) → 평가. 한 번 실행하면 다시 실행되지 않는다.
- 리허설 모드(`--rehearsal-data X.jsonl`): 봉인 세트가 아닌 데이터(예: Development)로 실행기만 시험한다. 결과에 REHEARSAL 로 표시하고 잠금 파일은 만들지 않으며 판정 문구도 교체 판정이 아니다.
- 문장 원문은 출력하지 않는다(건수·지표·id 없음).
사용(API 가상환경, cwd = apps/api): PYTHONPATH=. python scripts/evaluate_ops_final.py --rehearsal-data <jsonl> --out <json>
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OPS = Path(r"C:\Users\<user>\Documents\gabee-ner\docs\security\r11_validation\ops_replacement")
COLS = ("n_gold", "ex", "bo", "pa", "fu", "tp", "fpg", "fpn", "neg_n", "neg_flag")
TARGET_TYPES = ("D1P", "B1", "C2", "NEWNAME", "LONGNAME", "ORGLOC_MIX")
NEG_ONLY_EXCLUDED = ("D1",)  # 리허설에서만 쓰는 규칙: D1 은 따로 보고(P8)하고 매크로에서 뺀다
P5_NEG_TYPES = ("A3N", "D2", "GENERIC", "OTHER")  # 독립 평가셋 P5 음성 유형(사양 §3): 유형별 오탐률의 단순 평균
MIN_GOLD_P6, MIN_D1_SENT = 100, 250  # 이보다 작으면 평가셋 불충분(INVALID_DATASET), 판정하지 않는다


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_lf(p):
    """텍스트 파일 해시(CRLF/LF 차이로 해시가 달라지지 않도록 줄바꿈을 LF 로 정규화한 뒤 계산)."""
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def per_row_counts(rows, pred_fn):
    """모델 하나의 문장별 집계. r11_eval_lib 의 판정을 한 문장씩 적용한다(합치면 score_rows_r11 과 같다)."""
    import r11_eval_lib as ev

    recs = []
    for r in rows:
        s = ev.score_rows_r11([r], pred_fn, keep_rows=True)
        pr = s["_rows"][0]
        recs.append({"id": r["id"], "type": r["type"], "pair_id": r.get("pair_id"), "role": r.get("pair_role") or r.get("role"),
                     "c": [s["n_gold_spans"], s["exact"], s["boundary_only"], s["partial"], s["full"], s["tp"], s["fp_in_gold_rows"], s["fp_in_neg_rows"], s["n_neg_sentences"], s["neg_flagged"]],
                     "masked_all": pr["masked_all"], "pred_any": pr["pred_any"]})
    return recs


def build_units(recs_by_model, rows):
    """두 모델 공통 군집(쌍이면 pair_id, 아니면 문장 id) 목록과 유형 색인."""
    types = sorted({r["type"] for r in rows})
    tix = {t: i for i, t in enumerate(types)}
    keys = {}
    for r in rows:
        keys.setdefault(r.get("pair_id") or r["id"], []).append(r["id"])
    ukeys = sorted(keys)
    uix = {rid: i for i, k in enumerate(ukeys) for rid in keys[k]}
    out = {}
    for m, recs in recs_by_model.items():
        arr = np.zeros((len(ukeys), len(types), len(COLS)), dtype=np.int64)
        pairs = np.zeros((len(ukeys), 2), dtype=np.int64)  # [n_pairs, ok]
        byuid = defaultdict(dict)
        for x in recs:
            arr[uix[x["id"]], tix[x["type"]]] += np.array(x["c"])
            if x["pair_id"]:
                byuid[x["pair_id"]][x["role"]] = x
        for k, d in byuid.items():
            if {"person", "noun"} <= set(d):
                pairs[ukeys.index(k), 0] = 1
                pairs[ukeys.index(k), 1] = int(bool(d["person"]["masked_all"]) and not d["noun"]["pred_any"])
        out[m] = (arr, pairs)
    return types, out


def _div(a, b):
    return float(a) / float(b) if b else float("nan")


def metrics_from(arr, pairs, types, neg_types=None):
    """arr: (types, cols) 합계, pairs: [n_pairs, ok]. 지표 사전."""
    t = {ty: dict(zip(COLS, arr[i].tolist())) for i, ty in enumerate(types)}
    tot = {c: sum(v[c] for v in t.values()) for c in COLS}
    rec = _div(tot["ex"], tot["n_gold"])
    p_off = _div(tot["tp"], tot["tp"] + tot["fpg"])
    p_all = _div(tot["tp"], tot["tp"] + tot["fpg"] + tot["fpn"])
    f1 = _div(2 * p_off * rec, p_off + rec) if p_off == p_off and rec == rec and (p_off + rec) else float("nan")
    m = {"exact_recall": rec, "precision_official": p_off, "precision_all": p_all, "f1_official": f1,
         "total_exposure_rate": _div(tot["pa"] + tot["fu"], tot["n_gold"]), "partial_rate": _div(tot["pa"], tot["n_gold"]), "full_rate": _div(tot["fu"], tot["n_gold"]),
         "mask_rate": _div(tot["ex"] + tot["bo"], tot["n_gold"]), "pair_context_accuracy": _div(pairs[1], pairs[0]), "n_gold": tot["n_gold"]}
    negs = [ty for ty in neg_types if ty in t and t[ty]["neg_n"] > 0] if neg_types else [ty for ty, v in t.items() if v["n_gold"] == 0 and v["neg_n"] > 0 and ty not in NEG_ONLY_EXCLUDED]
    for ty, v in t.items():
        if v["n_gold"]:
            m[f"type.{ty}.exposure"] = _div(v["pa"] + v["fu"], v["n_gold"])
            m[f"type.{ty}.recall"] = _div(v["ex"], v["n_gold"])
            m[f"type.{ty}.n_gold"] = v["n_gold"]
        if v["neg_n"]:
            m[f"type.{ty}.fpr"] = _div(v["neg_flag"], v["neg_n"])
    m["neg_fpr_macro"] = float(np.mean([_div(t[ty]["neg_flag"], t[ty]["neg_n"]) for ty in negs])) if negs else float("nan")
    m["D1_fpr"] = m.get("type.D1.fpr", float("nan"))
    return m


def bootstrap(units, types, resamples, seed, neg_types=None):
    """B-A 차이의 percentile 구간(두 모델이 같은 재표본). 반환: {지표: [lo, hi]}."""
    (arrA, prA), (arrB, prB) = units["A"], units["B"]
    n = arrA.shape[0]
    rng = np.random.default_rng(seed)
    diffs = defaultdict(list)
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        mA = metrics_from(arrA[idx].sum(0), prA[idx].sum(0), types, neg_types)
        mB = metrics_from(arrB[idx].sum(0), prB[idx].sum(0), types, neg_types)
        for k in mA:
            if k.endswith("n_gold") or k == "n_gold":
                continue
            diffs[k].append(mB[k] - mA[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in diffs.items()}


def ge(a, b):
    return a >= b - 1e-12


def le(a, b):
    return a <= b + 1e-12


def decide(crit, mA, mB, ci, types, rehearsal=False):
    """사전 판정 기준 적용. 반환: (verdict, 검사별 결과). 구간 기준이 하나라도 안 맞으면 KEEP_S0."""
    pc = crit["primary_checks"]
    d = lambda k: 100 * (mB[k] - mA[k])  # noqa: E731
    lo = lambda k: 100 * ci[k][0]  # noqa: E731
    hi = lambda k: 100 * ci[k][1]  # noqa: E731
    r = {}
    c = pc["P1_recall"]
    r["P1_recall"] = ge(d(c["metric"]), c["delta_pp_min"]) and lo(c["metric"]) > c["ci_lower_pp_gt"]
    c = pc["P2_total_exposure"]
    r["P2_total_exposure"] = le(d(c["metric"]), c["delta_pp_max"]) and hi(c["metric"]) < c["ci_upper_pp_lt"]
    c = pc["P3_f1_official"]
    r["P3_f1_official"] = ge(d(c["metric"]), c["delta_pp_min"]) and lo(c["metric"]) > c["ci_lower_pp_gt"]
    c = pc["P4_precision_all_noninferior"]
    r["P4_precision_all"] = ge(d(c["metric"]), c["delta_pp_min"]) and ge(lo(c["metric"]), c["ci_lower_pp_ge"])
    c = pc["P5_negative_fpr"]
    r["P5_negative_fpr"] = le(mB["neg_fpr_macro"], c["absolute_max"]) and le(d("neg_fpr_macro"), c["delta_pp_max"]) and le(hi("neg_fpr_macro"), c["ci_upper_pp_le"])
    c = pc["P6_target_types_exposure"]
    ok = True
    for ty in c["types"]:
        k = f"type.{ty}.exposure"
        ok &= k in mA and le(d(k), c["delta_pp_max"]) and le(hi(k), c["ci_upper_pp_le"])
    r["P6_target_types_exposure"] = bool(ok)
    c = pc["P7_regression_types_exposure"]
    excl = set(pc["P6_target_types_exposure"]["types"]) | {"D1", "D2", "A3N", "GENERIC", "OTHER"}
    ok, detail = True, {}
    # 최종 평가: 사전 등록된 고정 목록(A1·A3·B2·C1·D3). 리허설(Development)은 유형이 달라 있는 유형 전체를 본다.
    p7_types = [ty for ty in types if ty not in excl] if rehearsal else list(c["types"])
    for ty in p7_types:
        k = f"type.{ty}.exposure"
        if k not in mA or mA.get(f"type.{ty}.n_gold", 0) < 1:
            if rehearsal:
                continue
            detail[ty] = False  # precheck 가 막지만 방어적으로 미충족 처리(침묵 통과 금지)
            ok = False
            continue
        if mA[f"type.{ty}.n_gold"] >= c["min_gold_spans_for_ci"]:
            good = le(d(k), c["delta_pp_max"]) and le(hi(k), c["ci_upper_pp_le"])
        else:
            good = le(d(k), c["small_type_point_delta_pp_max"])
        detail[ty] = good
        ok &= good
    r["P7_regression_types_exposure"] = bool(ok)
    c = pc["P8_D1_noun_fpr"]
    r["P8_D1_noun_fpr"] = le(mB["D1_fpr"], c["absolute_max"]) and le(d("D1_fpr"), c["delta_pp_max"]) and le(hi("D1_fpr"), c["ci_upper_pp_le"])
    c = pc["P9_pair_context_accuracy"]
    r["P9_pair_context_accuracy"] = ge(d("pair_context_accuracy"), c["delta_pp_min"]) and ge(mB["pair_context_accuracy"], c["absolute_min"])
    return ("PRIMARY_ALL_PASS" if all(r.values()) else "KEEP_S0"), r, detail


def verify_seal(sealed_dir):
    bad = []
    for ln in (sealed_dir / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        h, n = ln.split(" *", 1)
        if not (sealed_dir / n).exists() or sha(sealed_dir / n) != h:
            bad.append(n)
    if bad:
        raise SystemExit(f"봉인 무결성 실패: {bad}")


def verify_models(lock):
    for who in ("A", "B"):
        d = Path(lock["models"][who]["onnx_dir"])
        for f, h in lock["models"][who]["sha256"].items():
            if sha(d / f) != h:
                raise SystemExit(f"모델 {who} 해시 불일치: {f}")


def load_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def normalize_rows(rows):
    """봉인 평가셋 행(primary_stress_type, role, exclude)을 평가 필드(type, pair_role)로 맞춘다.
    exclude 표시 행과, 쌍의 한쪽이 제외되면 그 쌍 전체를 두 모델 모두에서 같이 뺀다. 반환: (행 목록, 제외 건수)."""
    bad_pairs = {r["pair_id"] for r in rows if r.get("exclude") and r.get("pair_id")}
    out, dropped = [], 0
    for r in rows:
        if r.get("exclude") or (r.get("pair_id") in bad_pairs):
            dropped += 1
            continue
        r = dict(r)
        r["type"] = r.get("type") or r.get("primary_stress_type")
        if not r["type"]:
            raise SystemExit("행에 type/primary_stress_type 이 없다")
        r["pair_role"] = r.get("pair_role") or r.get("role")
        out.append(r)
    return out, dropped


P7_TYPES = ("A1", "A3", "B2", "C1", "D3")
ALLOWED_TYPES = ("A1", "A3", "B1", "B2", "C1", "C2", "D1", "D1P", "D2", "D3", "A3N", "GENERIC", "OTHER", "NEWNAME", "LONGNAME", "ORGLOC_MIX")


def dataset_valid(arr_by_type_total, types, d1_sentences, rows_by_type=None):
    """평가셋 충분성(사전 등록): P6 유형의 정답 span 100 이상, D1 문장 250 이상. 부족하면 판정하지 않는다.
    rows_by_type 를 주면(최종 평가) 추가로: P7 유형 정답 span 0 → 미달, 허용 목록 밖 유형 → 미달, 허용 유형이 비어 있으면 미달."""
    miss = [ty for ty in TARGET_TYPES if arr_by_type_total.get(ty, 0) < MIN_GOLD_P6]
    if d1_sentences < MIN_D1_SENT:
        miss.append("D1_sentences")
    if rows_by_type is not None:
        miss += [f"{ty}(P7 정답 구간 0)" for ty in P7_TYPES if arr_by_type_total.get(ty, 0) < 1]
        miss += [f"{ty}(허용 목록 밖 유형)" for ty in sorted(rows_by_type) if ty not in ALLOWED_TYPES]
        miss += [f"{ty}(유형 없음)" for ty in ALLOWED_TYPES if rows_by_type.get(ty, 0) < 1]
    return miss


def _h(prefix, key):
    return hashlib.sha256((prefix + str(key)).encode("utf-8")).hexdigest()


def select_subset(rows, spec):
    """보조 분석 S1600(회귀 유형 중심) 선택: 평가셋 도착 전에 고정한 규칙. 유형별 행(쌍 유형은 쌍)을 해시 오름차순으로 quota 만큼.
    어느 유형이 quota 보다 적으면 None(생략)과 사유를 돌려준다. 평가 결과를 보지 않으며 재선정하지 않는다."""
    prefix = "r11-ops-subset-1.0|"
    q = spec["quota_rows"]
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    ids, short = set(), []
    for ty, n in q.items():
        if ty == "D1P_D1_pair_rows":
            continue
        pool = sorted(by.get(ty, []), key=lambda r: _h(prefix, r["id"]))
        if len(pool) < n:
            short.append(f"{ty}:{len(pool)}<{n}")
        ids |= {r["id"] for r in pool[:n]}
    pairs = defaultdict(set)
    for r in by.get("D1P", []) + by.get("D1", []):
        if r.get("pair_id"):
            pairs[r["pair_id"]].add(r.get("pair_role"))
    full = sorted((k for k, v in pairs.items() if {"person", "noun"} <= v), key=lambda k: _h(prefix, k))
    npairs = q["D1P_D1_pair_rows"] // 2
    if len(full) < npairs:
        short.append(f"pairs:{len(full)}<{npairs}")
    keep = set(full[:npairs])
    ids |= {r["id"] for r in by.get("D1P", []) + by.get("D1", []) if r.get("pair_id") in keep}
    return (None, short) if short else (ids, [])


def precheck(rows, rehearsal):
    """모델을 돌리기 전 평가셋 충분성 확인(부족하면 잠금 파일도 만들지 않고 중단). 반환: 분모 요약."""
    import r8_eval_lib as lib

    gold, nrows = defaultdict(int), defaultdict(int)
    for r in rows:
        gold[r["type"]] += len(lib.gold_person(r))
        nrows[r["type"]] += 1
    d1 = nrows.get("D1", 0)
    pairs = len({r["pair_id"] for r in rows if r.get("pair_id")})
    den = {"gold_spans_by_type": dict(sorted(gold.items())), "rows_by_type": dict(sorted(nrows.items())), "D1_sentences": d1, "pairs": pairs, "gold_spans_total": sum(gold.values())}
    if not rehearsal:
        miss = dataset_valid(gold, None, d1, dict(nrows))
        if miss:
            raise SystemExit(f"INVALID_DATASET: 사전 등록 최소 분모 미달 {miss} (평가하지 않음, 잠금 파일 미생성)")
    return den


def run(rows, lock, crit, args, rehearsal, dropped=0, den=None):
    import r8_eval_lib as lib

    recs = {}
    for who in ("A", "B"):
        lib.set_model(lock["models"][who]["onnx_dir"], None)
        recs[who] = per_row_counts(rows, None)
    types, units = build_units(recs, rows)
    neg = None if rehearsal else P5_NEG_TYPES
    mA = metrics_from(units["A"][0].sum(0), units["A"][1].sum(0), types, neg)
    mB = metrics_from(units["B"][0].sum(0), units["B"][1].sum(0), types, neg)
    ci = bootstrap(units, types, crit["statistics"]["resamples"], crit["statistics"]["seed"], neg)
    verdict, checks, p7 = decide(crit, mA, mB, ci, types, rehearsal)
    supp = None
    spec = crit.get("supplementary_analysis", {}).get("S1600_regression_weighted")
    if spec:
        ids, why = select_subset(rows, spec)
        if ids is None:
            supp = {"available": False, "reason": f"quota 미달로 생략: {why}"}
        else:
            rs = [r for r in rows if r["id"] in ids]
            ts, us = build_units({w: [x for x in recs[w] if x["id"] in ids] for w in ("A", "B")}, rs)
            sA, sB = metrics_from(us["A"][0].sum(0), us["A"][1].sum(0), ts, neg), metrics_from(us["B"][0].sum(0), us["B"][1].sum(0), ts, neg)
            sci = bootstrap(us, ts, crit["statistics"]["resamples"], crit["statistics"]["seed"], neg)
            supp = {"available": True, "label": spec["label"], "n_rows": len(rs), "judgement": "없음(보고만)",
                    "metrics": {k: {"A_S0": sA[k], "B_B-new-s21": sB[k], "delta_pp": 100 * (sB[k] - sA[k]), "ci95_delta_pp": [100 * sci[k][0], 100 * sci[k][1]]} for k in sA if k in sci}}
    table = {k: {"A_S0": mA[k], "B_B-new-s21": mB[k], "delta_pp": 100 * (mB[k] - mA[k]), "relative_change_pct": (100 * (mB[k] - mA[k]) / mA[k]) if mA[k] else None, "ci95_delta_pp": [100 * ci[k][0], 100 * ci[k][1]]}
             for k in mA if k in ci}
    return {"mode": "REHEARSAL(교체 판정 아님)" if rehearsal else "FINAL_INDEPENDENT", "label": "자동 검증 기반 연구용 · 사람 검수 미완료",
            "n_sentences": len(rows), "rows_excluded_identically_for_both_models": dropped, "denominators": den, "n_clusters": int(units["A"][0].shape[0]),
            "p5_negative_types": list(neg) if neg else "리허설: 사람 없는 유형 전체(D1 제외)", "metrics": table, "checks": checks, "p7_by_type": p7,
            "verdict": ("교체 판정 아님(리허설)" if rehearsal else ("KEEP_S0" if verdict == "KEEP_S0" else "PRIMARY_PASS_PENDING_SECONDARY(S1·S2a·S2b·S3·S4 확인 전, 교체 승인 아님)")), "primary_verdict_if_final": verdict,
            "secondary_status": {"S1": "별도 측정 필요", "S2a": "연구 단계 통과(재측정 대상)", "S2b": "UNVERIFIED(운영 환경 확인 전)", "S3": "별도 실행 필요", "S4": "운영 담당자 확인 필요"},
            "supplementary_S1600": supp,
            "criteria_sha256": lock.get("criteria_sha256"), "models": {w: lock["models"][w]["sha256"]["model.onnx"] for w in ("A", "B")},
            "note": "R11-S 미충족(B-new 1/5시드 적격)·사람 검수 519건 미완료 위험은 판정과 별개로 계속 표기. 공식 연구 게이트 통과가 아님."}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rehearsal-data", type=Path)
    ap.add_argument("--sealed-dir", type=Path)
    ap.add_argument("--dataset-name", default="r11_stress_v1.jsonl")
    ap.add_argument("--lock", type=Path, default=OPS / "ops_model_lock.json")
    ap.add_argument("--criteria", type=Path, default=OPS / "ops_decision_criteria.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    lock = json.loads(a.lock.read_text(encoding="utf-8"))
    crit = json.loads(a.criteria.read_text(encoding="utf-8"))
    if lock.get("criteria_sha256") != sha_lf(a.criteria):
        raise SystemExit("판정 기준 파일 해시가 모델 고정 파일의 기록과 다름(기준 변경 금지)")
    verify_models(lock)
    if a.rehearsal_data:
        rows, dropped = normalize_rows(load_rows(a.rehearsal_data))
        res = run(rows, lock, crit, a, rehearsal=True, dropped=dropped, den=precheck(rows, True))
    else:
        if not a.sealed_dir:
            raise SystemExit("--sealed-dir 또는 --rehearsal-data 가 필요하다")
        verify_seal(a.sealed_dir)
        rows, dropped = normalize_rows(load_rows(a.sealed_dir / a.dataset_name))
        den = precheck(rows, False)  # 평가셋이 불충분하면 여기서 중단(잠금 파일 생성 전)
        lk = a.sealed_dir.parent / "EVALUATED.lock"
        try:
            with open(lk, "x", encoding="utf-8") as f:  # 없을 때만 생성(재실행 방지)
                f.write(json.dumps({"criteria_sha256": sha_lf(a.criteria), "models": lock["models"]["A"]["sha256"]["model.onnx"] + "|" + lock["models"]["B"]["sha256"]["model.onnx"]}))
        except FileExistsError:
            raise SystemExit("EVALUATED.lock 이 이미 있다: 독립 평가는 한 번만 실행한다")
        res = run(rows, lock, crit, a, rehearsal=False, dropped=dropped, den=den)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"mode": res["mode"], "verdict": res["verdict"], "checks": res["checks"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
