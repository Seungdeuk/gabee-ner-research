"""R9 사람 검수 기록 집계기. **실제로 검수했다고 기록된 문장 id 만** 완료로 센다. 미검수 문장은 자동으로 정상 처리하지 않는다.

원문은 읽거나 출력하지 않는다(id 와 메타데이터만). 검수 기록 형식(아무 것이나 가능):
- JSON: {"reviews": [{"id": "R9R-0001", "verdict": "ok|fix|exclude", "reviewer": "...", "date": "2026-11-19", "note": "..."}]} 또는 위 객체 목록
- CSV: id,verdict,reviewer,date,note 열(머리글 필수)
- 텍스트: 한 줄에 id 하나(또는 id 가 섞인 줄). 이 경우 verdict 는 `unspecified` 로 기록한다(정상으로 간주하지 않고 '검수함'으로만 센다).
같은 id 가 여러 번 있으면 마지막 기록만 센다. 평가셋에 없는 id 는 무효로 보고한다.

사용: python r9_review_ledger.py --rep R.jsonl --stress S.jsonl --ledger ledger.json --out coverage.json [--min-total 240] [--target-total 300]
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ID_RX = re.compile(r"R9[RS]-\d{4}")
VERDICTS = {"ok", "fix", "exclude", "unspecified"}


def load_rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def read_ledger(path):
    p = Path(path)
    raw = p.read_text(encoding="utf-8-sig")
    recs = []
    if p.suffix.lower() == ".json":
        o = json.loads(raw)
        items = o.get("reviews", o) if isinstance(o, dict) else o
        for it in items:
            if isinstance(it, str):
                recs.append({"id": it, "verdict": "unspecified"})
            else:
                recs.append({"id": it.get("id"), "verdict": (it.get("verdict") or "unspecified"), "reviewer": it.get("reviewer"), "date": it.get("date")})
    elif p.suffix.lower() == ".csv":
        for r0 in csv.DictReader(raw.splitlines()):
            r = {(k or "").strip().lower(): v for k, v in r0.items()}  # 헤더 대소문자·BOM 무시
            verdict = (r.get("verdict") or r.get("review_status") or "unspecified").strip() or "unspecified"
            recs.append({"id": (r.get("id") or "").strip(), "verdict": verdict, "reviewer": r.get("reviewer") or r.get("reviewed_by"), "date": r.get("date") or r.get("review_date")})
    else:
        for line in raw.splitlines():
            for m in ID_RX.findall(line):
                recs.append({"id": m, "verdict": "unspecified"})
    return recs


def _kin_negatives(rows):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import validate_r9_evalset as V
    return {r["id"] for r in rows if r.get("is_negative") and any(k in r["text"] for k in V.KIN_MULTI)}


def high_risk_groups(rep, st, compliance=None):
    """고위험 군(문장 id 집합). 원문은 쓰지 않는다."""
    g = defaultdict(set)
    g["kin_only_negatives_rep"] = _kin_negatives(rep)
    g["kin_only_negatives_stress"] = _kin_negatives(st)
    for r in rep + st:
        if any(s["label"] == "PERSON_ALIAS" for s in r["spans"]):
            g["alias_sentences"].add(r["id"])
        if len([s for s in r["spans"] if s["label"] in ("PERSON", "PERSON_ALIAS")]) >= 2:
            g["multi_person"].add(r["id"])
        if r.get("context_entities"):
            g["context_entities_loc_org"].add(r["id"])
        t = r.get("primary_stress_type")
        if t in ("A3", "A3N", "D1", "D1P", "D2", "C2", "D3"):
            g[f"stress_{t}"].add(r["id"])
    if compliance:
        ids = compliance["ids"]
        for k, name in (("REP_UNLABELED_NAME_SUSPECT_OTHER", "unlabeled_name_suspect_rep"), ("STRESS_UNLABELED_NAME_SUSPECT_OTHER", "unlabeled_name_suspect_stress"),
                        ("REP_POSSIBLE_ONE_CHAR_NAME_PATTERN", "one_char_name_suspect_rep"), ("STRESS_POSSIBLE_ONE_CHAR_NAME_PATTERN", "one_char_name_suspect_stress"),
                        ("REP_PERSON_FOLLOWED_BY_KIN_TERM", "kin_adjacent_rep")):
            if k in ids:
                g[name] = set(ids[k])
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rep", required=True, type=Path)
    ap.add_argument("--stress", required=True, type=Path)
    ap.add_argument("--ledger", type=Path)
    ap.add_argument("--compliance", type=Path, help="check_r9_compliance.py 결과 JSON(고위험 id 목록)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--min-total", type=int, default=240)
    ap.add_argument("--target-total", type=int, default=300)
    a = ap.parse_args(argv)
    rep, st = load_rows(a.rep), load_rows(a.stress)
    meta = {r["id"]: ("representative" if r in rep else "stress") for r in rep + st}
    by_id = {r["id"]: r for r in rep + st}
    total_rows = len(by_id)
    out = {"dataset_rows": total_rows, "ledger_provided": bool(a.ledger), "required_min_total": a.min_total, "target_total": a.target_total}
    if not a.ledger:
        out.update({"reviewed_unique_valid": 0, "note": "검수 기록이 없어 완료로 집계한 문장은 0건이다. 미검수 문장을 정상으로 간주하지 않는다."})
        text = json.dumps(out, ensure_ascii=False)
        if a.out:
            a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(text)
        return 0
    recs = read_ledger(a.ledger)
    last = {}
    dups = Counter(r["id"] for r in recs)
    invalid = []
    for r in recs:
        if r["id"] not in by_id:
            invalid.append(r["id"])
        else:
            last[r["id"]] = r
    bad_verdict = [r["id"] for r in last.values() if r["verdict"] not in VERDICTS]
    verdicts = Counter(r["verdict"] for r in last.values())
    sets = Counter(meta[i] for i in last)
    stress_types = Counter(by_id[i].get("primary_stress_type") for i in last if meta[i] == "stress")
    cats = Counter(by_id[i]["category"] for i in last if meta[i] == "representative")
    comp = json.loads(a.compliance.read_text(encoding="utf-8")) if a.compliance else None
    groups = high_risk_groups(rep, st, comp)
    cov = {k: {"size": len(v), "reviewed": len(v & set(last)), "reviewed_pct": round(100 * len(v & set(last)) / len(v), 1) if v else None} for k, v in sorted(groups.items())}
    out.update({
        "ledger_records": len(recs), "reviewed_unique_valid": len(last), "reviewed_share_pct": round(100 * len(last) / total_rows, 1),
        "invalid_ids_not_in_dataset": sorted(set(invalid)), "duplicate_ids": sorted(i for i, c in dups.items() if c > 1), "unknown_verdict_ids": sorted(bad_verdict),
        "verdicts": dict(verdicts), "by_set": dict(sets), "stress_by_type": dict(stress_types), "representative_by_category": dict(cats),
        "meets_min_total": len(last) >= a.min_total, "meets_target_total": len(last) >= a.target_total,
        "unreviewed_rows": total_rows - len(last), "high_risk_coverage": cov,
        "note": "집계는 검수 기록에 있는 id 만이다. 기록이 없는 문장은 정상으로 간주하지 않는다. verdict 가 unspecified 면 '검수함'으로만 센다.",
    })
    if a.out:
        a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("reviewed_unique_valid", "reviewed_share_pct", "meets_min_total", "meets_target_total", "verdicts", "by_set", "invalid_ids_not_in_dataset", "duplicate_ids")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
