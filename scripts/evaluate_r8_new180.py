"""R8 최종 평가: 봉인 평가셋 new180 에서 서비스 모델과 선정된 후보 1개를 같은 코드·같은 설정으로 1회 측정한다.

사전 등록: docs/security/NER_R8_PREREGISTRATION.md §6. 이 스크립트를 실제 모델에 돌리는 것은 후보 선정이 끝난 뒤 한 번뿐이다.
평가셋 내용으로 모델·데이터·후처리를 고치지 않는다. 읽기 전용(ner.py·모델·DB·설정 미수정), 외부 호출 없음.

``--selftest`` 는 모델을 쓰지 않고 정답을 그대로 예측값으로 넣어 만점이 나오는지, 일부러 망가뜨린 예측에서 기대한 수치가 나오는지만 확인한다
(평가셋의 스키마·라벨·정책·채점 코드 호환성 사전 검증).

정책(app/shared/privacy/policy.ANALYSIS 와 같은 표): PHONE·EMAIL·ADDRESS·PERSON = MASK, ACCOUNT·SECRET·RRN·CARD = BLOCK(점수 >= block_min_score
이면 메시지 통째 차단, 미만이면 값만 가림). 채점의 '보호됨' = 메시지가 차단되었거나 정답 구간 글자가 전부 가려진 경우.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r8_eval_lib as lib  # noqa: E402
from r8_eval_lib import ner  # noqa: E402
from app.shared.privacy import policy as pol  # noqa: E402
from app.shared.privacy.types import Kind  # noqa: E402

SEALED_DIR = Path(r"C:\Users\<user>\Documents\gabee-ner-sealed-new180")
DATA = SEALED_DIR / "gabee_ner_new180_v1.jsonl"
POLICY = pol.ANALYSIS
KINDS = ["PERSON", "ADDRESS", "PHONE", "EMAIL", "ACCOUNT"]


def load_checked():
    man = {}
    for line in (SEALED_DIR / "MANIFEST.sha256").read_text().splitlines():
        h, n = line.split(" *")
        man[n] = h
    got = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if man.get("gabee_ner_new180_v1.jsonl") != got:
        raise SystemExit("봉인 해시 불일치: 평가셋이 바뀌었다")
    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    validate_schema(rows)
    return rows


def validate_schema(rows):
    assert len(rows) == 180 and len({r["text"] for r in rows}) == 180
    kinds = Counter()
    for r in rows:
        assert set(r) >= {"id", "category", "text", "spans", "is_negative", "source"}
        assert bool(r["spans"]) != r["is_negative"]
        sp = sorted(r["spans"], key=lambda s: s["start"])
        for a, b in zip(sp, sp[1:]):
            assert b["start"] >= a["end"], "구간 겹침"
        for s in sp:
            assert r["text"][s["start"]:s["end"]] == s["value"]
            assert s["mapped_kind"] in KINDS and s["label"] in ("PERSON", "PERSON_ALIAS", "ADDRESS", "PHONE", "EMAIL", "ACCOUNT")
            assert (s["mapped_kind"] == "PERSON") == (s["label"] in ("PERSON", "PERSON_ALIAS"))
            kinds[s["mapped_kind"]] += 1
    assert kinds == Counter({"PERSON": 110, "ADDRESS": 28, "PHONE": 18, "EMAIL": 12, "ACCOUNT": 8}), kinds
    assert sum(1 for r in rows if r["is_negative"]) == 28


def detect_real(text):
    return [(s.kind.value, s.start, s.end, s.score) for s in ner.detect_with_ner(text, ())]


def analyze(det_fn, rows):
    """det_fn(row) -> [(kind, start, end, score)]. 문장별 기록과 집계를 돌려준다."""
    recs = []
    per_kind = {k: {"tp": 0, "fp": 0, "fn": 0} for k in KINDS}
    prot = Counter()
    person_cls = Counter()
    neg_person = neg_any = 0
    for r in rows:
        text = r["text"]
        dets = det_fn(r)
        gold = [(s["mapped_kind"], s["start"], s["end"]) for s in r["spans"]]
        blocked = any(POLICY.action_for(Kind(k)) is pol.Action.BLOCK and sc >= POLICY.block_min_score for k, _, _, sc in dets)
        masked = [(a, b) for k, a, b, sc in dets
                  if POLICY.action_for(Kind(k)) is pol.Action.MASK or (POLICY.action_for(Kind(k)) is pol.Action.BLOCK and sc < POLICY.block_min_score)]
        cov = set()
        for a, b in masked:
            cov.update(range(a, b))
        pred_person = [(a, b) for k, a, b, _ in dets if k == "PERSON"]
        rec = {"id": r["id"], "gold": [], "blocked": blocked, "pred": [(k, a, b) for k, a, b, _ in dets]}
        for kind, a, b in gold:
            exact = any(k == kind and (s, e) == (a, b) for k, s, e, _ in dets)
            per_kind[kind]["tp" if exact else "fn"] += 1
            covered_all = all(i in cov for i in range(a, b))
            protected = blocked or covered_all
            prot[(kind, "protected" if protected else "exposed")] += 1
            if kind == "ACCOUNT":
                prot[("ACCOUNT_policy", "blocked" if blocked else ("value_masked" if covered_all else "exposed"))] += 1
            entry = {"kind": kind, "span": (a, b), "exact": exact, "protected": protected}
            if kind == "PERSON":
                pc = set()
                for s, e in pred_person:
                    pc.update(range(max(s, a), min(e, b)))
                leaked = (b - a) - len(pc)
                c = "full" if leaked >= b - a else ("partial" if leaked > 0 else "masked")
                person_cls[c] += 1
                entry["cls"] = c
            rec["gold"].append(entry)
        gset = {(k, a, b) for k, a, b in gold}
        for k, a, b, _ in dets:
            if k in KINDS and (k, a, b) not in gset:
                per_kind[k]["fp"] += 1
        if r["is_negative"]:
            neg_person += int(bool(pred_person))
            neg_any += int(bool(dets))
        recs.append(rec)
    return recs, per_kind, prot, person_cls, neg_person, neg_any


def summarize(rows, parts):
    recs, per_kind, prot, person_cls, neg_person, neg_any = parts
    n_person = sum(person_cls.values())
    n_neg = sum(1 for r in rows if r["is_negative"])
    kinds = {k: lib.prf(v["tp"], v["fp"], v["fn"]) for k, v in per_kind.items()}
    tot = lib.prf(sum(v["tp"] for v in per_kind.values()), sum(v["fp"] for v in per_kind.values()), sum(v["fn"] for v in per_kind.values()))
    p_tp, p_fp, p_fn = per_kind["PERSON"]["tp"], per_kind["PERSON"]["fp"], per_kind["PERSON"]["fn"]
    person = lib.prf(p_tp, p_fp, p_fn)
    person["precision_ci"] = lib.wilson(p_tp, p_tp + p_fp)
    person["recall_ci"] = lib.wilson(p_tp, p_tp + p_fn)
    mask_kinds = ("PERSON", "ADDRESS", "PHONE", "EMAIL")
    mask_prot = sum(prot[(k, "protected")] for k in mask_kinds)
    mask_n = sum(prot[(k, "protected")] + prot[(k, "exposed")] for k in mask_kinds)
    all_prot = sum(prot[(k, "protected")] for k in KINDS)
    all_n = sum(prot[(k, "protected")] + prot[(k, "exposed")] for k in KINDS)
    acc = {s: prot[("ACCOUNT_policy", s)] for s in ("blocked", "value_masked", "exposed")}
    assert sum(acc.values()) == prot[("ACCOUNT", "protected")] + prot[("ACCOUNT", "exposed")]
    out = {
        "person_family": {
            "n_gold_spans": n_person, "exact": person,
            "full": person_cls["full"], "partial": person_cls["partial"], "masked": person_cls["masked"],
            "total_exposure_rate": (person_cls["full"] + person_cls["partial"]) / n_person,
            "total_exposure_ci": lib.wilson(person_cls["full"] + person_cls["partial"], n_person),
            "full_rate": person_cls["full"] / n_person, "partial_rate": person_cls["partial"] / n_person,
            "nonperson_fp_sentences": neg_person, "n_neg_sentences": n_neg,
            "nonperson_fpr": neg_person / n_neg, "nonperson_fpr_ci": lib.wilson(neg_person, n_neg),
        },
        "per_kind_exact": kinds, "all_kinds_exact_micro": tot,
        "protected_target_recall_mask_kinds": {"protected": mask_prot, "n": mask_n, "rate": mask_prot / mask_n, "ci": lib.wilson(mask_prot, mask_n)},
        "protected_all_kinds": {"protected": all_prot, "n": all_n, "rate": all_prot / all_n, "ci": lib.wilson(all_prot, all_n)},
        "account_block_policy": {**acc, "n": sum(acc.values())},
        "negatives_any_kind_flagged": {"count": neg_any, "n": n_neg, "rate": neg_any / n_neg, "ci": lib.wilson(neg_any, n_neg)},
    }
    # F1 신뢰구간(문장 단위 부트스트랩, 시드 고정)
    rng = random.Random(20261119)
    idx = list(range(len(recs)))
    f1s = []
    for _ in range(1000):
        tp = fp = fn = 0
        for i in (rng.choice(idx) for _ in idx):
            for g in recs[i]["gold"]:
                if g["kind"] == "PERSON":
                    tp += int(g["exact"]); fn += int(not g["exact"])
            gp = {(g["kind"], g["span"]) for g in recs[i]["gold"]}
            fp += sum(1 for k, a, b in recs[i]["pred"] if k == "PERSON" and ("PERSON", (a, b)) not in gp)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    f1s.sort()
    out["person_family"]["exact"]["f1_bootstrap95"] = [round(f1s[24], 4), round(f1s[974], 4)]
    return out


def paired(rec_a, rec_b):
    """서비스(a) → 후보(b) PERSON 정답 구간별 대응 비교."""
    imp = wor = same = 0
    order = {"masked": 0, "partial": 1, "full": 2}
    detail = Counter()
    for ra, rb in zip(rec_a, rec_b):
        for ga, gb in zip(ra["gold"], rb["gold"]):
            if ga["kind"] != "PERSON":
                continue
            d = order[gb["cls"]] - order[ga["cls"]]
            detail[(ga["cls"], gb["cls"])] += 1
            if d < 0:
                imp += 1
            elif d > 0:
                wor += 1
            else:
                same += 1
    fp_new = sum(1 for ra, rb in zip(rec_a, rec_b) if not ra["gold"] and rb["pred"] and not ra["pred"])
    fp_gone = sum(1 for ra, rb in zip(rec_a, rec_b) if not ra["gold"] and ra["pred"] and not rb["pred"])
    return {"person_spans_improved": imp, "person_spans_worsened": wor, "unchanged": same,
            "transitions_service_to_candidate": {f"{a}->{b}": n for (a, b), n in detail.items()},
            "negatives_newly_flagged": fp_new, "negatives_no_longer_flagged": fp_gone}


def selftest(rows):
    """모델 없이 채점 코드를 검증한다. 정답=예측이면 만점, 망가뜨린 예측은 기대 수치와 정확히 일치해야 한다."""
    def oracle(r):
        return [(s["mapped_kind"], s["start"], s["end"], 0.95) for s in r["spans"]]

    s = summarize(rows, analyze(oracle, rows))
    pf = s["person_family"]
    assert pf["exact"]["f1"] == 1.0 and pf["total_exposure_rate"] == 0 and pf["nonperson_fpr"] == 0, pf
    assert s["protected_all_kinds"]["rate"] == 1.0 and s["account_block_policy"]["blocked"] == 8, s["account_block_policy"]
    assert all(v["f1"] == 1.0 for v in s["per_kind_exact"].values())
    # 1) PERSON 구간 5개를 통째로 놓침 -> 완전 노출 5/110, 부분 0
    drop = {"n": 0}
    pers = [(r["id"], i) for r in rows for i, sp in enumerate(r["spans"]) if sp["mapped_kind"] == "PERSON"]
    dropset = set(pers[:5])

    def miss5(r):
        return [(k, a, b, sc) for i, (k, a, b, sc) in enumerate(oracle(r)) if (r["id"], i) not in dropset]
    s1 = summarize(rows, analyze(miss5, rows))["person_family"]
    assert (s1["full"], s1["partial"], s1["masked"]) == (5, 0, 105), s1
    assert abs(s1["total_exposure_rate"] - 5 / 110) < 1e-12
    # 2) PERSON 구간 3개의 끝 글자 누락 -> 부분 노출 3
    cut = set(pers[:3])

    def cut3(r):
        return [(k, a, b - 1 if (r["id"], i) in cut else b, sc) for i, (k, a, b, sc) in enumerate(oracle(r))]
    s2 = summarize(rows, analyze(cut3, rows))["person_family"]
    assert (s2["full"], s2["partial"]) == (0, 3), s2
    assert s2["exact"]["fn"] == 3 and s2["exact"]["fp"] == 3
    # 3) 이름 없는 문장 2개에 가짜 PERSON -> 비PERSON 오탐 2/28
    negs = [r["id"] for r in rows if r["is_negative"]][:2]

    def fp2(r):
        d = oracle(r)
        return d + [("PERSON", 0, 2, 0.6)] if r["id"] in negs else d
    s3 = summarize(rows, analyze(fp2, rows))["person_family"]
    assert (s3["nonperson_fp_sentences"], s3["n_neg_sentences"]) == (2, 28)
    # 4) ACCOUNT 을 낮은 점수로 예측 -> 차단되지 않지만 값은 가려짐(정책: 점수 < block_min_score)
    def lowacc(r):
        return [(k, a, b, 0.5 if k == "ACCOUNT" else sc) for k, a, b, sc in oracle(r)]
    s4 = summarize(rows, analyze(lowacc, rows))["account_block_policy"]
    assert (s4["blocked"], s4["value_masked"], s4["exposed"]) == (0, 8, 0), s4
    # 5) 전혀 예측하지 않음 -> 모든 정답 노출
    s5 = summarize(rows, analyze(lambda r: [], rows))
    assert s5["protected_all_kinds"]["protected"] == 0 and s5["person_family"]["full"] == 110
    print("selftest 통과: 정답=예측 만점, 놓침·끝글자 누락·오탐·ACCOUNT 정책·무예측 시나리오가 기대 수치와 정확히 일치")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", nargs="+", action="append", default=[], metavar=("LABEL", "DIR"), help="LABEL DIR [FP12] (서비스, 후보 순)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rows = load_checked()
    if a.selftest:
        selftest(rows)
        return
    res, records = {}, {}
    for spec in a.model:
        label, d = spec[0], spec[1]
        fp = lib.set_model(d, spec[2] if len(spec) > 2 else None)
        parts = analyze(lambda r: detect_real(r["text"]), rows)
        res[label] = {"fingerprint12": fp, "model_dir": d, **summarize(rows, parts)}
        records[label] = parts[0]
        print(f"[{label}] fp {fp}: 전체노출 {res[label]['person_family']['total_exposure_rate']:.4f} F1 {res[label]['person_family']['exact']['f1']:.4f} 오탐 {res[label]['person_family']['nonperson_fpr']:.4f}", flush=True)
    labels = list(records)
    if len(labels) == 2:
        res["paired_service_to_candidate"] = paired(records[labels[0]], records[labels[1]])
    if a.out:
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print("저장:", a.out)


if __name__ == "__main__":
    main()
