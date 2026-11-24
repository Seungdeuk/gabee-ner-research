"""R9 최종 독립 평가(봉인 평가셋 2개, 서비스 모델 vs 선정 후보 1개, 1회). 사전 등록 NER_R9_PREREGISTRATION.md §5, §7.

안전장치:
- 실제 봉인 평가셋은 ``--execute-final`` 이 있어야만 연다(별도 최종 승인 뒤). 없으면 거부한다. ``--selftest`` 는 합성 가짜 데이터만 쓰며 봉인 폴더를 열지 않는다.
- 봉인 해시(MANIFEST.sha256) 불일치, 모델 해시(``--expect-onnx-sha``) 불일치면 중단한다. 결과를 본 뒤 후보 교체·재학습·후처리 조정은 하지 않는다(§7).
- 대표(representative)와 스트레스(stress)는 따로 보고하고 평균 내지 않는다. 정답이 1글자인 문장(exclude)은 집계에서 빼고 사유별로 센다.

보고: 세트·모델별 Exact P/R/F1, 보호 대상 Recall, 완전·부분·전체 노출률, 비PERSON 오탐률, 과잉 병합, 문장당 지연, 층별(전체 / kin_adjacent / 정책 중립 / 이름 길이 2·3·4+ /
name_seen_in_reference / 스트레스 유형), 서비스→후보 짝지은 개선·악화, 이름·틀 단위 군집 부트스트랩 95% 구간(1,000회, 시드 고정), 교체 권고 조건 §7 판정.
"""

import argparse
import hashlib
import json
import random
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate_r8_new180 as E  # noqa: E402
import r8_eval_lib as lib  # noqa: E402

SEAL = Path(r"C:\Users\<user>\Documents\gabee-ner-sealed-r9\sealed")
FILES = {"representative": "r9_representative_v1.2.jsonl", "stress": "r9_stress_v1.2.jsonl"}
ANNOT = Path(r"C:\Users\<user>\venvs\ner-runs\r9\seal_run")  # id 만 든 주석(name_seen_in_reference, kin_adjacent 문장 id)
SEED, BOOT = 20261119, 1000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_sealed(set_name):
    man = {}
    for line in (SEAL / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if " *" in line:
            h, n = line.split(" *", 1)
            man[n.strip()] = h.strip()
    p = SEAL / FILES[set_name]
    if man.get(p.name) != sha(p):
        raise SystemExit(f"봉인 해시 불일치: {p.name}")
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    ann = json.loads((ANNOT / f"annot_{set_name}.json").read_text(encoding="utf-8"))
    return rows, set(ann.get("name_seen_in_reference", [])), set(ann.get("kin_adjacent_person_sentences", [])), sha(p)


def gold_person_spans(r):
    return [(s["start"], s["end"]) for s in r["spans"] if s["mapped_kind"] == "PERSON"]


def run(det_fn, rows):
    """det_fn(row) -> [(kind, start, end, score)]. 문장별 기록과 지연(문장당 초)을 돌려준다."""
    times = []

    def timed(r):
        t0 = time.perf_counter()
        out = det_fn(r)
        times.append(time.perf_counter() - t0)
        return out

    parts = E.analyze(timed, rows)
    return parts, times


def cluster_key(r):
    if r.get("template_id"):
        return "T:" + str(r["template_id"])
    g = gold_person_spans(r)
    return "N:" + (r["text"][g[0][0]:g[0][1]] if g else r["id"])


def counts(recs, rows, pred=lambda r: True):
    c = Counter()
    for rec, r in zip(recs, rows, strict=True):
        if not pred(r):
            continue
        gp = {(g["kind"], g["span"]) for g in rec["gold"]}
        for g in rec["gold"]:
            if g["kind"] != "PERSON":
                continue
            c["n"] += 1
            c["full"] += g["cls"] == "full"
            c["partial"] += g["cls"] == "partial"
            c["tp"] += g["exact"]
            c["fn"] += not g["exact"]
        c["fp"] += sum(1 for k, a, b in rec["pred"] if k == "PERSON" and ("PERSON", (a, b)) not in gp)
        if r["is_negative"]:
            c["neg"] += 1
            c["neg_flag"] += any(k == "PERSON" for k, _, _ in rec["pred"])
    return c


def rates(c):
    n, neg = c["n"], c["neg"]
    f1d = 2 * c["tp"] + c["fp"] + c["fn"]
    return {"n_gold_person_spans": n, "full_exposure": c["full"] / n if n else None, "partial_exposure": c["partial"] / n if n else None, "total_exposure": (c["full"] + c["partial"]) / n if n else None,
            "exact_f1": 2 * c["tp"] / f1d if f1d else None, "precision": c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None, "recall": c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None,
            "n_neg_sentences": neg, "nonperson_fpr": c["neg_flag"] / neg if neg else None, "counts": dict(c)}


def strata(recs, rows, seen, kin):
    defs = {"ALL": lambda r: True, "kin_adjacent": lambda r: r["id"] in kin, "policy_neutral": lambda r: r["id"] not in kin,
            "name_seen_in_reference": lambda r: r["id"] in seen, "name_unseen": lambda r: r["id"] not in seen}
    out = {k: rates(counts(recs, rows, f)) for k, f in defs.items()}
    # 이름 길이별: 정답 구간 단위(문장 단위가 아님). 길이별 미탐·부분은 해당 길이의 정답 구간만 센다.
    for label, ok in (("len2", lambda n: n == 2), ("len3", lambda n: n == 3), ("len4plus", lambda n: n >= 4)):
        c = Counter()
        for rec in recs:
            for g in rec["gold"]:
                if g["kind"] == "PERSON" and ok(g["span"][1] - g["span"][0]):
                    c["n"] += 1
                    c["full"] += g["cls"] == "full"
                    c["partial"] += g["cls"] == "partial"
                    c["tp"] += g["exact"]
                    c["fn"] += not g["exact"]
        out["name_" + label] = {"n_gold_person_spans": c["n"], "total_exposure": (c["full"] + c["partial"]) / c["n"] if c["n"] else None, "exact_rate": c["tp"] / c["n"] if c["n"] else None, "counts": dict(c)}
    return out


def stress_types(recs, rows):
    out = {}
    for t in sorted({r.get("primary_stress_type") for r in rows if r.get("primary_stress_type")}):
        out[t] = rates(counts(recs, rows, lambda r, t=t: r.get("primary_stress_type") == t))
    return out


def bootstrap_diff(recs_a, recs_b, rows):
    """군집(이름·틀) 부트스트랩: 후보(b)-서비스(a) 차이의 95% 구간. 군집 단위로 문장을 다시 뽑는다."""
    cl = defaultdict(list)
    for i, r in enumerate(rows):
        cl[cluster_key(r)].append(i)
    keys = sorted(cl)
    ca = {k: counts([recs_a[i] for i in cl[k]], [rows[i] for i in cl[k]]) for k in keys}
    cb = {k: counts([recs_b[i] for i in cl[k]], [rows[i] for i in cl[k]]) for k in keys}
    rng = random.Random(SEED)
    d_exp, d_fp, d_f1 = [], [], []
    for _ in range(BOOT):
        ta, tb = Counter(), Counter()
        for k in (rng.choice(keys) for _ in keys):
            ta.update(ca[k])
            tb.update(cb[k])
        ra, rb = rates(ta), rates(tb)
        if None in (ra["total_exposure"], rb["total_exposure"]):
            continue
        d_exp.append(rb["total_exposure"] - ra["total_exposure"])
        d_f1.append((rb["exact_f1"] or 0) - (ra["exact_f1"] or 0))
        if ra["nonperson_fpr"] is not None:
            d_fp.append(rb["nonperson_fpr"] - ra["nonperson_fpr"])

    def ci(v):
        v = sorted(v)
        return [round(v[int(0.025 * len(v))], 5), round(v[min(len(v) - 1, int(0.975 * len(v)))], 5)] if v else None
    return {"exposure_diff_ci95": ci(d_exp), "nonperson_fpr_diff_ci95": ci(d_fp), "exact_f1_diff_ci95": ci(d_f1), "clusters": len(keys), "resamples": BOOT}


def recommendation(rep_res, stress_res, om_service, om_cand):
    s, c = rep_res["service"]["strata"]["ALL"], rep_res["candidate"]["strata"]["ALL"]
    b = rep_res["bootstrap"]
    d_exp, d_fp, d_f1 = c["total_exposure"] - s["total_exposure"], c["nonperson_fpr"] - s["nonperson_fpr"], c["exact_f1"] - s["exact_f1"]
    worst = None
    if stress_res:
        diffs = {t: stress_res["candidate"]["stress_types"][t]["total_exposure"] - stress_res["service"]["stress_types"][t]["total_exposure"] for t in stress_res["service"]["stress_types"]
                 if stress_res["service"]["stress_types"][t]["total_exposure"] is not None and stress_res["candidate"]["stress_types"][t]["total_exposure"] is not None}
        worst = max(diffs.items(), key=lambda kv: kv[1]) if diffs else None
    conds = {"1_exposure_upper_ci<0_and_point<=-1.0pp": b["exposure_diff_ci95"][1] < 0 and d_exp <= -0.01,
             "2_fpr_diff_upper_ci<=+2.0pp": b["nonperson_fpr_diff_ci95"] is not None and b["nonperson_fpr_diff_ci95"][1] <= 0.02,
             "3_f1_diff_lower_ci>=-1.0pp": b["exact_f1_diff_ci95"][0] >= -0.01,
             "4_overmerge<=service": om_cand <= om_service,
             "5_no_stress_type_exposure_worse_than_+2.0pp": worst is None or worst[1] <= 0.02}
    return {"conditions": conds, "recommend_replacement": all(conds.values()), "worst_stress_type": worst, "point_diffs": {"exposure": d_exp, "fpr": d_fp, "f1": d_f1},
            "note": "사용자가 최종 결정한다. 하나라도 어기면 '교체 근거 불충분'. 유의하지 않은 차이는 개선이라고 쓰지 않는다."}


def evaluate_models(models, sets, make_det):
    """models: [(label, dir, fp12)], sets: {name: (rows, seen, kin)}. make_det(label) -> det_fn 을 쓰기 전에 모델을 설정한다."""
    out = {}
    recs_all = {}
    for label, d, fp12 in models:
        make_det(label, d, fp12)
        out[label] = {}
        for name, (rows, seen, kin) in sets.items():
            parts, times = run(lambda r: E.detect_real(r["text"]), rows)
            recs = parts[0]
            recs_all[(label, name)] = recs
            summ = E.summarize(rows, parts)
            entry = {"summary": summ, "strata": strata(recs, rows, seen, kin), "latency_ms_per_sentence": {"median": round(1000 * st.median(times), 3), "p95": round(1000 * sorted(times)[int(0.95 * len(times))], 3)}}
            if name == "stress":
                entry["stress_types"] = stress_types(recs, rows)
            out[label][name] = entry
    return out, recs_all


def selftest():
    """합성 가짜 데이터로 지표·층별·부트스트랩·권고 조건 계산을 검증한다(봉인 폴더·실제 모델 미사용)."""
    rows = []
    for i in range(40):
        name = ["가온", "나단", "다율", "라온솔"][i % 4]
        t = f"{name}이는 학교에 갔다 {i}" if i % 2 == 0 else f"{name} 엄마가 전화했다 {i}"
        a = t.index(name)
        rows.append({"id": f"F{i}", "text": t, "spans": [{"start": a, "end": a + len(name), "label": "PERSON", "mapped_kind": "PERSON", "value": name}], "is_negative": False, "template_id": f"t{i % 8}",
                     "primary_stress_type": "A1" if i < 20 else "B1", "category": "x", "source": "x"})
    for i in range(10):
        rows.append({"id": f"N{i}", "text": f"오늘 날씨가 좋다 {i}", "spans": [], "is_negative": True, "category": "x", "source": "x", "template_id": f"n{i % 5}"})
    kin = {r["id"] for r in rows if "엄마" in r["text"]}
    seen = {f"F{i}" for i in range(0, 40, 5)}

    def oracle(r):
        return [(s["mapped_kind"], s["start"], s["end"], 0.95) for s in r["spans"]]
    sets = {"representative": (rows, seen, kin), "stress": (rows, seen, kin)}

    def ev(det):
        parts, _ = run(det, rows)
        return parts[0], E.summarize(rows, parts)
    ro, so = ev(oracle)
    c = rates(counts(ro, rows))
    assert c["total_exposure"] == 0 and c["exact_f1"] == 1.0 and c["nonperson_fpr"] == 0
    rn, _ = ev(lambda r: [])
    cn = rates(counts(rn, rows))
    assert cn["total_exposure"] == 1.0 and cn["exact_f1"] == 0.0
    # 끝 글자 누락: 부분 노출 40/40, 정확 일치 0
    rc, _ = ev(lambda r: [(k, a, b - 1, s) for k, a, b, s in oracle(r)])
    cc = rates(counts(rc, rows))
    assert cc["partial_exposure"] == 1.0 and cc["full_exposure"] == 0.0 and cc["exact_f1"] == 0.0
    # 음성 문장 3개에 가짜 PERSON -> 오탐 3/10
    ids = {f"N{i}" for i in range(3)}
    rf, _ = ev(lambda r: oracle(r) + ([("PERSON", 0, 2, 0.6)] if r["id"] in ids else []))
    assert rates(counts(rf, rows))["nonperson_fpr"] == 0.3
    # 층별: kin_adjacent 20개 문장, 정책 중립 20 + 음성 10
    s = strata(rn, rows, seen, kin)
    assert s["kin_adjacent"]["n_gold_person_spans"] == 20 and s["policy_neutral"]["n_gold_person_spans"] == 20 and s["policy_neutral"]["n_neg_sentences"] == 10
    assert s["name_len2"]["n_gold_person_spans"] == 30 and s["name_len3"]["n_gold_person_spans"] == 10 and s["name_len4plus"]["n_gold_person_spans"] == 0
    assert s["name_seen_in_reference"]["n_gold_person_spans"] == 8
    # 부트스트랩: 같은 모델끼리는 차이 0, 서비스(무예측) vs 후보(정답)는 노출 -100%p 이고 구간 상한 < 0, 같은 시드라 재현
    b0 = bootstrap_diff(rn, rn, rows)
    assert b0["exposure_diff_ci95"] == [0.0, 0.0] and b0["exact_f1_diff_ci95"] == [0.0, 0.0]
    b1 = bootstrap_diff(rn, ro, rows)
    assert b1["exposure_diff_ci95"] == [-1.0, -1.0] and b1 == bootstrap_diff(rn, ro, rows)
    # 권고 조건: 무예측(서비스) vs 정답(후보)은 충족, 후보=서비스면 1번 조건 불충족
    sr = {"strata": strata(rn, rows, seen, kin), "stress_types": stress_types(rn, rows)}
    cr = {"strata": strata(ro, rows, seen, kin), "stress_types": stress_types(ro, rows)}
    rec = recommendation({"service": sr, "candidate": cr, "bootstrap": b1}, {"service": sr, "candidate": cr}, 9, 9)
    assert rec["recommend_replacement"] and all(rec["conditions"].values()), rec
    rec2 = recommendation({"service": sr, "candidate": sr, "bootstrap": b0}, {"service": sr, "candidate": sr}, 9, 9)
    assert not rec2["recommend_replacement"] and not rec2["conditions"]["1_exposure_upper_ci<0_and_point<=-1.0pp"]
    rec3 = recommendation({"service": sr, "candidate": cr, "bootstrap": b1}, {"service": sr, "candidate": cr}, 9, 10)
    assert not rec3["recommend_replacement"] and not rec3["conditions"]["4_overmerge<=service"]
    # 스트레스 유형 악화 조건: 후보가 B1 에서 서비스보다 5%p 나쁘면 5번 조건 불충족
    bad = {"service": cr, "candidate": {"strata": cr["strata"], "stress_types": {**cr["stress_types"], "B1": {**cr["stress_types"]["B1"], "total_exposure": 0.05}}}}
    rec4 = recommendation({"service": cr, "candidate": cr, "bootstrap": {"exposure_diff_ci95": [-0.2, -0.1], "nonperson_fpr_diff_ci95": [0, 0], "exact_f1_diff_ci95": [0, 0]}}, bad, 9, 9)
    assert not rec4["conditions"]["5_no_stress_type_exposure_worse_than_+2.0pp"]
    assert sets  # evaluate_models 의 입력 형식 확인용
    print("selftest 통과: 정답=예측 만점, 무예측·끝글자 누락·가짜 오탐 수치, 층별 건수, 부트스트랩 재현성, 권고 조건 5개 경계가 기대와 일치")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--execute-final", action="store_true", help="봉인 평가셋을 연다(최종 승인 뒤 1회만)")
    ap.add_argument("--model", nargs="+", action="append", default=[], metavar=("LABEL", "DIR"), help="LABEL DIR [FP12] (서비스, 후보 순)")
    ap.add_argument("--expect-onnx-sha", help="후보 model.onnx SHA256(선정 기록)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.execute_final:
        raise SystemExit("거부: 봉인 평가셋 실행은 별도 최종 승인 뒤 --execute-final 로만 가능하다(--selftest 는 합성 데이터만 쓴다)")
    if len(a.model) != 2:
        raise SystemExit("--model 은 서비스, 후보 순으로 정확히 2개")
    cand_dir = Path(a.model[1][1])
    if a.expect_onnx_sha and sha(cand_dir / "model.onnx") != a.expect_onnx_sha:
        raise SystemExit("후보 model.onnx 해시가 선정 기록과 다르다")
    sets, hashes = {}, {}
    for name in FILES:
        rows, seen, kin, h = load_sealed(name)
        hashes[name] = h
        n_all = len(rows)
        rows = [r for r in rows if not r.get("exclude")]
        sets[name] = (rows, seen, kin)
        print(f"{name}: {len(rows)}/{n_all} 문장(제외 {n_all - len(rows)})", flush=True)
    models = [(m[0], m[1], m[2] if len(m) > 2 else None) for m in a.model]

    def setm(label, d, fp12):
        lib.set_model(d, fp12)
    res, recs = evaluate_models(models, sets, setm)
    # 과잉 병합은 정책과 무관한 개발 인접 문항(28건)으로 모델마다 측정
    om = {}
    for label, d, fp12 in models:
        lib.set_model(d, fp12)
        om[label] = __import__("evaluate_r8_dev").overmerge_count()["count"]
    svc, cand = models[0][0], models[1][0]
    final = {"hashes": hashes, "models": {lb: {"dir": d, "onnx_sha256": sha(Path(d) / "model.onnx")} for lb, d, _ in models}, "overmerge": om, "results": res, "paired": {}, "bootstrap": {}, "targets": {}}
    for name, (rows, _, _) in sets.items():
        final["paired"][name] = E.paired(recs[(svc, name)], recs[(cand, name)])
        final["bootstrap"][name] = bootstrap_diff(recs[(svc, name)], recs[(cand, name)], rows)
        for label in (svc, cand):
            r = res[label][name]["strata"]["ALL"]
            final["targets"][f"{label}|{name}"] = {"exposure<5%": r["total_exposure"] < 0.05, "fpr<=5%": r["nonperson_fpr"] <= 0.05, "f1>=90%": r["exact_f1"] >= 0.90}
    rep = {"service": res[svc]["representative"], "candidate": res[cand]["representative"], "bootstrap": final["bootstrap"]["representative"]}
    st_ = {"service": res[svc]["stress"], "candidate": res[cand]["stress"]}
    final["recommendation"] = recommendation(rep, st_, om[svc], om[cand])
    if a.out:
        a.out.write_text(json.dumps(final, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print("저장:", a.out)


if __name__ == "__main__":
    main()
