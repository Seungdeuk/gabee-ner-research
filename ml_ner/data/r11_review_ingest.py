"""사람 검수 결과(JSON, `review_packet.html` 에서 저장)를 집계하고 사전 등록 기준(NER_R11_PREREGISTRATION.md §4-4)으로 판정한다. 문장 원문은 출력하지 않는다.

기준(데이터 수정·학습 시작 판단): 세트별 라벨 오류 행 ≤2%, 경계 오류 행 ≤2%, 유형 D1P·D1·B1·C2 는 각 ≤3%, 대조쌍 구분 실패 쌍 ≤3%, 자연스러움 3점(부자연) 행 ≤10%.
하나라도 넘거나 미판정 항목이 5% 이상이면 `needs_action` 이며 임의로 학습을 시작하지 않고 수정 범위·영향을 먼저 보고한다.
사용: python r11_review_ingest.py --result r11_review_result.json --out summary.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(r"C:\Users\<user>\venvs\ner-runs\data_r11")
SETS = {"train": "r11_Dprime_train", "val": "r11_val", "dev": "r11_dev"}


def rd(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, nargs="+", required=True, help="결과 JSON 여러 개(우선 파일과 전체 파일). 같은 id 는 뒤 파일이 앞 파일을 덮는다")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    meta = {}
    for s, f in SETS.items():
        for r in rd(DATA / f"{f}.jsonl"):
            meta[r["id"]] = {"set": s, "type": r.get("type", "TRAIN_PAIR"), "pair_id": r.get("pair_id"), "pair_role": r.get("pair_role")}
    merged = {}
    for rp in a.result:
        for x in json.loads(rp.read_text(encoding="utf-8")):
            merged[x["id"]] = x
    res = list(merged.values())
    sample_ids = set(json.loads((DATA / "review" / "review_sample_ids.json").read_text(encoding="utf-8")))
    missing = sorted(sample_ids - set(merged))  # 표본에 있는데 결과가 없는 항목은 미판정으로 센다(자동 통과 처리 금지)
    for i in missing:
        res.append({"id": i, "natural": None, "label_ok": None, "boundary_ok": None, "pair_ok": None})
    by_set, by_type = defaultdict(Counter), defaultdict(Counter)
    pair_fail, pair_seen, fail_ids, undecided = defaultdict(set), defaultdict(set), [], Counter()
    for x in res:
        m = meta.get(x["id"])
        if not m:
            undecided["unknown_id"] += 1
            continue
        for key in (by_set[m["set"]], by_type[(m["set"], m["type"])]):
            key["n"] += 1
            if x.get("label_ok") is None or x.get("boundary_ok") is None or x.get("natural") is None:
                key["undecided"] += 1
                continue
            key["label_err"] += x["label_ok"] == "N"
            key["boundary_err"] += x["boundary_ok"] == "N"
            key["unnatural3"] += str(x["natural"]) == "3"
            key["any_err"] += (x["label_ok"] == "N") or (x["boundary_ok"] == "N")
        if m["pair_id"]:
            pair_seen[m["set"]].add(m["pair_id"])
            if x.get("pair_ok") == "N":
                pair_fail[m["set"]].add(m["pair_id"])
        if x.get("label_ok") == "N" or x.get("boundary_ok") == "N" or x.get("pair_ok") == "N":
            fail_ids.append(x["id"])
    pct = lambda a_, b_: round(100 * a_ / b_, 2) if b_ else None  # noqa: E731
    out = {"sets": {}, "types": {}, "fail_ids": fail_ids, "undecided_global": dict(undecided)}
    needs = False
    for s, c in by_set.items():
        dec = c["n"] - c["undecided"]
        row = {"n": c["n"], "undecided": c["undecided"], "label_err_pct": pct(c["label_err"], dec), "boundary_err_pct": pct(c["boundary_err"], dec),
               "unnatural_pct": pct(c["unnatural3"], dec), "pairs_reviewed": len(pair_seen[s]), "pair_fail_pct": pct(len(pair_fail[s]), len(pair_seen[s]))}
        row["pass"] = bool((row["label_err_pct"] or 0) <= 2 and (row["boundary_err_pct"] or 0) <= 2 and (row["unnatural_pct"] or 0) <= 10 and (row["pair_fail_pct"] or 0) <= 3 and c["undecided"] <= 0.05 * c["n"])
        needs |= not row["pass"]
        out["sets"][s] = row
    for (s, t), c in by_type.items():
        dec = c["n"] - c["undecided"]
        e = pct(c["any_err"], dec)
        lim = 3 if t in ("D1P", "D1", "B1", "C2") else 2
        out["types"][f"{s}/{t}"] = {"n": c["n"], "label_or_boundary_err_pct": e, "limit_pct": lim, "pass": (e or 0) <= lim}
        needs |= (e or 0) > lim
    out["verdict"] = "needs_action" if needs else "pass"
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("sets", "verdict")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
