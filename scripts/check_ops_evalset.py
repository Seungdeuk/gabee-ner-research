"""R11 독립 평가셋 제작 후 자동 품질 검증(운영 교체 평가용). 문장 원문은 출력하지 않는다(건수·id 만).

**의미 정답은 검증하지 못한다**(자동 검사는 구조·수량·겹침·일관성만 본다). 사람 검수를 대체하지 않으며 결과에 항상 그 상태를 적는다.
모델 예측은 쓰지 않는다(평가셋을 모델 결과에 맞춰 고치는 것을 막기 위해).

검사: (1) 기존 검증기(`validate_r11_evalset.py --strict`: 스키마, span offset·BIO, 쌍 무결성, 중복·틀·4-gram, 기존 데이터 겹침) 실행 결과 (2) 총수·유형별 수량·허용 유형
(3) 쌍 300·pair_word 수 (4) PERSON 구간·음성 분포 (5) 평가 분모 충족(P6·P7·P8, evaluate_ops_final.precheck) (6) S1600 부분집합 생성 가능 여부
(7) 이름·대조쌍 단어 역할 누수(양성 PERSON 값이 다른 음성 문장에 그대로 등장하는 건수; 같은 쌍의 명사 문장은 정상이라 제외) (8) 봉인 가능 판정(오류 0 이고 위 항목 충족).
사용(API 가상환경, cwd = apps/api): PYTHONPATH=. python scripts/check_ops_evalset.py --data converted_trimmed.jsonl --out-dir <폴더> [--reference-dir <폴더> ...]
"""

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_ops_final as ops  # noqa: E402
import trim_r11_evalset as trim  # noqa: E402

NEG_TYPES = ("D1", "D2", "A3N", "GENERIC", "OTHER")
EXPECT_TOTAL = 2350
DEFAULT_REFS = [Path(r"C:\Users\<user>\venvs\ner-runs") / d for d in ("data_r11", "data_r10", "data_r9", "data_r8", "data")]


def person_values(r):
    out = []
    for sp in r.get("spans", []):
        if (sp.get("kind") or sp.get("label") or sp.get("mapped_kind")) in ("PERSON", "PERSON_ALIAS"):
            out.append(r["text"][sp["start"]:sp["end"]])
    return out


def run_validator(data, out_dir, refs):
    report = out_dir / "validator_report.json"
    cmd = [sys.executable, "-X", "utf8", str(HERE / "validate_r11_evalset.py"), "--data", str(data), "--set-kind", "stress", "--full-ids", "--strict", "--report-out", str(report)]
    for d in refs:
        if Path(d).exists():
            cmd += ["--reference-dir", str(d)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    rep = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {"errors": {"validator_crash": {"count": 1}}, "warnings": {}, "info": {}}
    return r.returncode, rep


def analyze(rows):
    """검증기와 별개의 구조 분석(건수만). 반환 사전과 문제 목록."""
    problems = []
    cnt = Counter(r.get("primary_stress_type") for r in rows)
    quota = dict(trim.FINAL, D1P=trim.PAIRS_FINAL, D1=trim.PAIRS_FINAL)
    if len(rows) != EXPECT_TOTAL:
        problems.append(f"총 문장 수 {len(rows)} != {EXPECT_TOTAL}")
    for t, n in quota.items():
        if cnt.get(t, 0) != n:
            problems.append(f"{t}: {cnt.get(t, 0)} != {n}")
    problems += [f"허용 목록 밖 유형 {t}" for t in cnt if t not in quota]
    pairs = defaultdict(dict)
    for r in rows:
        if r.get("pair_id"):
            pairs[r["pair_id"]][r.get("role")] = r
    full_pairs = [k for k, v in pairs.items() if set(v) == {"person", "noun"}]
    words = Counter(pairs[k]["person"].get("pair_word") for k in full_pairs)
    if len(full_pairs) < 300:
        problems.append(f"쌍 {len(full_pairs)} < 300")
    if len(words) < 75:
        problems.append(f"서로 다른 pair_word {len(words)} < 75")
    if words and max(words.values()) > 4:
        problems.append("같은 pair_word 가 4쌍 초과")
    n_person_spans = Counter()
    neg_rows = Counter()
    person_vals = defaultdict(set)
    for r in rows:
        vals = person_values(r)
        t = r.get("primary_stress_type")
        n_person_spans[t] += len(vals)
        if not vals:
            neg_rows[t] += 1
        for v in vals:
            if len(v) >= 2:
                person_vals[v].add(r["id"])
    # 이름 역할 누수: 양성 PERSON 값이 다른 음성 문장 본문에 그대로 등장(같은 쌍의 명사 문장은 정상)
    leaks = []
    pair_of = {r["id"]: r.get("pair_id") for r in rows}
    for r in rows:
        if person_values(r):
            continue
        for v, ids in person_vals.items():
            if v in r["text"] and not any(pair_of.get(i) and pair_of.get(i) == r.get("pair_id") for i in ids):
                leaks.append(r["id"])
                break
    info = {"rows": len(rows), "by_type": dict(sorted(cnt.items())), "pairs": len(full_pairs), "distinct_pair_words": len(words), "max_pairs_per_word": max(words.values()) if words else 0,
            "person_spans_by_type": dict(sorted(n_person_spans.items())), "person_spans_total": sum(n_person_spans.values()),
            "rows_without_person_by_type": dict(sorted(neg_rows.items())), "negative_rows_total": sum(neg_rows.values()),
            "name_role_leak_candidates": {"count": len(leaks), "ids": leaks[:200], "note": "휴리스틱(경고). 사람이 아닌 문맥의 동음 단어일 수 있어 사람 검토 대상"}}
    return info, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--reference-dir", action="append", type=Path, default=[])
    ap.add_argument("--skip-validator", action="store_true", help="시험용")
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in a.data.read_text(encoding="utf-8").splitlines() if x.strip()]
    info, problems = analyze(rows)
    code, rep = (0, {"errors": {}, "warnings": {}, "info": {}}) if a.skip_validator else run_validator(a.data, a.out_dir, a.reference_dir or DEFAULT_REFS)
    norm, dropped = ops.normalize_rows(rows)
    try:
        den = ops.precheck(norm, rehearsal=False)
        den_ok, den_err = True, None
    except SystemExit as e:  # INVALID_DATASET
        den, den_ok, den_err = None, False, str(e)
    spec = json.loads((ops.OPS / "ops_decision_criteria.json").read_text(encoding="utf-8"))["supplementary_analysis"]["S1600_regression_weighted"]
    ids, why = ops.select_subset(norm, spec)
    s1600 = {"available": ids is not None, "rows": len(ids) if ids else 0, "short": why}
    validator_clean = (code == 0 and not rep.get("errors"))
    sealable = validator_clean and not problems and den_ok and s1600["available"]
    report = {"label": "자동 검증 기반 · 사람 검수 미완료 · 의미 정답은 검증되지 않음", "structure": info, "structure_problems": problems,
              "validator": {"exit_code": code, "errors": rep.get("errors"), "warnings": {k: {"count": v.get("count"), "unique_ids": v.get("unique_ids")} for k, v in (rep.get("warnings") or {}).items()}, "reference_check": (rep.get("info") or {}).get("reference"), "within_set": (rep.get("info") or {}).get("within_set")},
              "ops_denominators": den, "ops_denominators_ok": den_ok, "ops_denominators_error": den_err, "rows_excluded_flagged": dropped, "S1600": s1600,
              "sealable_by_automatic_checks": sealable,
              "not_verified": ["라벨 의미의 정확성(사람 이름 여부, 숨은 PERSON 누락)", "사람 문장·명사 문장의 문맥 판별 가능성", "말투·장면의 실제 가족 메신저 대표성", "라벨 오류율(표본 검수 필요)"],
              "human_review": "미실시"}
    (a.out_dir / "ops_evalset_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"sealable_by_automatic_checks": sealable, "structure_problems": problems, "validator_errors": list((rep.get("errors") or {}).keys()), "ops_denominators_ok": den_ok, "S1600": s1600}, ensure_ascii=False, indent=1))
    return 0 if sealable else 1


if __name__ == "__main__":
    sys.exit(main())
