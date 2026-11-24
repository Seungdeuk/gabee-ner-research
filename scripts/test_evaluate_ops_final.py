"""운영 교체 평가 실행기 시험(모델·평가셋 없이 합성 집계로). 판정 논리, 쌍체 부트스트랩, 재실행 방지, 기준·모델 해시 고정."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_ops_final as e  # noqa: E402

CRIT = json.loads((e.OPS / "ops_decision_criteria.json").read_text(encoding="utf-8"))
TYPES = ["A1", "A3", "B2", "C1", "D3", "B1", "C2", "D1", "D1P", "LONGNAME", "NEWNAME", "ORGLOC_MIX", "NEG_X", "NEG_Y"]
ROW = {"A1": (100, 70, 0, 10, 20, 70, 10, 0, 0, 0)}  # n_gold, ex, bo, pa, fu, tp, fpg, fpn, neg_n, neg_flag


def table(exposure, recall_ex, neg_flag=2, d1_flag=20, extra_fp=10):
    """유형별 합계 배열. exposure: 노출 수(100 중), recall_ex: exact 수."""
    arr = np.zeros((len(TYPES), len(e.COLS)), dtype=np.int64)
    for i, t in enumerate(TYPES):
        if t.startswith("NEG"):
            arr[i] = [0, 0, 0, 0, 0, 0, 0, neg_flag, 100, neg_flag]
        elif t == "D1":
            arr[i] = [0, 0, 0, 0, 0, 0, 0, d1_flag, 100, d1_flag]
        else:
            arr[i] = [100, recall_ex, 100 - exposure - recall_ex, exposure // 2, exposure - exposure // 2, recall_ex, extra_fp, 0, 0, 0]
    return arr


def point_ci(mA, mB, half=0.5):
    ci = {k: [mB[k] - mA[k] - half / 100, mB[k] - mA[k] + half / 100] for k in mA if not k.endswith("n_gold") and k != "n_gold"}
    return ci


def run_decide(a_arr, b_arr, pa=(100, 12), pb=(100, 40), half=0.5):
    mA, mB = e.metrics_from(a_arr, np.array(pa), TYPES), e.metrics_from(b_arr, np.array(pb), TYPES)
    ci = point_ci(mA, mB, half)
    return e.decide(CRIT, mA, mB, ci, TYPES)


def test_clear_improvement_passes_all_primary_checks():
    verdict, checks, _ = run_decide(table(exposure=30, recall_ex=50, extra_fp=40), table(exposure=8, recall_ex=84, extra_fp=10))
    assert verdict == "PRIMARY_ALL_PASS" and all(checks.values()), checks


def test_target_type_worsening_beyond_limit_blocks_replacement():
    a, b = table(30, 50, extra_fp=40), table(8, 84, extra_fp=10)
    i = TYPES.index("B1")
    b[i][3], b[i][4] = 20, 20  # B1 노출 40 으로 악화(S0 30)
    verdict, checks, _ = run_decide(a, b)
    assert verdict == "KEEP_S0" and not checks["P6_target_types_exposure"]


def test_negative_fpr_increase_blocks_even_with_big_recall_gain():
    verdict, checks, _ = run_decide(table(30, 50, neg_flag=2, extra_fp=40), table(8, 84, neg_flag=10, extra_fp=10))
    assert verdict == "KEEP_S0" and not checks["P5_negative_fpr"]


def test_ci_crossing_threshold_means_keep_s0():
    a, b = table(30, 50, extra_fp=40), table(18, 62, extra_fp=10)  # 점추정 +12pp 이지만 구간 하한이 5 미만이 되도록 넓은 구간
    verdict, checks, _ = run_decide(a, b, half=8.0)
    assert verdict == "KEEP_S0" and not checks["P1_recall"]


def test_paired_bootstrap_is_deterministic_and_pairs_models():
    rng = np.random.default_rng(1)
    n = 60
    A = rng.integers(0, 3, (n, 2, len(e.COLS)))
    B = A.copy()
    B[:, :, 1] += 1  # 모든 군집에서 B 가 exact 하나 더
    pa = np.zeros((n, 2), dtype=np.int64)
    ci1 = e.bootstrap({"A": (A, pa), "B": (B, pa)}, ["T1", "T2"], 200, 7)
    ci2 = e.bootstrap({"A": (A, pa), "B": (B, pa)}, ["T1", "T2"], 200, 7)
    assert str(ci1) == str(ci2)  # NaN 포함 비교
    assert ci1["exact_recall"][0] > 0  # 같은 재표본을 쓰므로 일관된 개선은 0 을 포함하지 않는다


def test_criteria_hash_must_match_lock(tmp_path):
    crit = tmp_path / "c.json"
    crit.write_text(json.dumps(CRIT), encoding="utf-8")
    crit.write_text(crit.read_text(encoding="utf-8") + " ", encoding="utf-8")  # 기준 파일이 바뀐 상황
    with pytest.raises(SystemExit):
        e.main(["--rehearsal-data", str(tmp_path / "x.jsonl"), "--criteria", str(crit), "--out", str(tmp_path / "o.json")])


def test_sealed_mode_refuses_second_run(tmp_path, monkeypatch):
    root = tmp_path / "ops"
    sealed = root / "sealed"
    sealed.mkdir(parents=True)
    f = sealed / "r11_stress_v1.jsonl"
    f.write_text("", encoding="utf-8")
    (sealed / "MANIFEST.sha256").write_text(f"{e.sha(f)} *{f.name}\n", encoding="utf-8")
    (root / "EVALUATED.lock").write_text("{}", encoding="utf-8")  # 이미 한 번 평가한 상황
    monkeypatch.setattr(e, "verify_models", lambda lock: None)
    monkeypatch.setattr(e, "precheck", lambda rows, rehearsal: {})
    with pytest.raises(SystemExit) as ex:
        e.main(["--sealed-dir", str(sealed), "--out", str(tmp_path / "o.json")])
    assert "EVALUATED.lock" in str(ex.value)


def test_seal_tampering_is_detected(tmp_path):
    d = tmp_path / "sealed"
    d.mkdir()
    (d / "a.jsonl").write_text("x", encoding="utf-8")
    (d / "MANIFEST.sha256").write_text(f"{e.sha(d / 'a.jsonl')} *a.jsonl\n", encoding="utf-8")
    e.verify_seal(d)
    (d / "a.jsonl").write_text("y", encoding="utf-8")
    with pytest.raises(SystemExit):
        e.verify_seal(d)


def test_normalize_rows_maps_spec_fields_and_drops_whole_pair_when_one_side_excluded():
    rows = [{"id": "1", "primary_stress_type": "D1P", "pair_id": "p1", "role": "person"},
            {"id": "2", "primary_stress_type": "D1", "pair_id": "p1", "role": "noun", "exclude": True},
            {"id": "3", "primary_stress_type": "B1"}]
    out, dropped = e.normalize_rows(rows)
    assert [r["id"] for r in out] == ["3"] and dropped == 2
    out2, _ = e.normalize_rows([rows[0]])
    assert out2[0]["type"] == "D1P" and out2[0]["pair_role"] == "person"


def test_criteria_hash_is_line_ending_independent(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_bytes(b'{"x": 1}\n{"y": 2}\n')
    b.write_bytes(b'{"x": 1}\r\n{"y": 2}\r\n')
    assert e.sha_lf(a) == e.sha_lf(b) and e.sha(a) != e.sha(b)
    lock = json.loads((e.OPS / "ops_model_lock.json").read_text(encoding="utf-8"))
    assert lock["criteria_sha256"] == e.sha_lf(e.OPS / "ops_decision_criteria.json")


def test_p5_uses_fixed_negative_types_and_ignores_others():
    types = ["A3N", "D2", "GENERIC", "OTHER", "D1", "NEG_Z"]
    arr = np.zeros((6, len(e.COLS)), dtype=np.int64)
    for i, flag in enumerate([10, 20, 30, 40, 99, 99]):
        arr[i] = [0, 0, 0, 0, 0, 0, 0, flag, 100, flag]
    m = e.metrics_from(arr, np.array([0, 0]), types, e.P5_NEG_TYPES)
    assert abs(m["neg_fpr_macro"] - 0.25) < 1e-9 and abs(m["D1_fpr"] - 0.99) < 1e-9


def test_invalid_dataset_stops_before_lock_is_created():
    assert e.dataset_valid({"D1P": 300, "B1": 99, "C2": 300, "NEWNAME": 300, "LONGNAME": 120, "ORGLOC_MIX": 100}, None, 300) == ["B1"]
    assert e.dataset_valid({t: 100 for t in e.TARGET_TYPES}, None, 249) == ["D1_sentences"]
    assert e.dataset_valid({t: 100 for t in e.TARGET_TYPES}, None, 250) == []
    rows = [{"id": str(i), "type": "B1", "spans": [{"kind": "PERSON", "start": 0, "end": 2}]} for i in range(5)]
    with pytest.raises(SystemExit) as ex:
        e.precheck(rows, rehearsal=False)
    assert "INVALID_DATASET" in str(ex.value)


def _types_arr(n_gold_p7):
    arr = table(exposure=30, recall_ex=50, extra_fp=40)
    return arr


def test_p7_small_type_uses_point_estimate_and_missing_type_fails():
    a, b = table(30, 50, extra_fp=40), table(8, 84, extra_fp=10)
    i = TYPES.index("A3")
    # A3 의 정답 구간을 30개(<50)로 줄이고 노출이 +4pp 악화(점추정 ≤ +5pp → 통과, 구간은 쓰지 않는다)
    a[i] = [30, 15, 0, 3, 12, 15, 0, 0, 0, 0]
    b[i] = [30, 14, 0, 4, 12, 14, 0, 0, 0, 0]
    verdict, checks, detail = run_decide(a, b, half=20.0)  # 구간이 매우 넓어도 소형 유형은 점추정만
    assert detail["A3"] is True
    b[i] = [30, 10, 0, 5, 15, 10, 0, 0, 0, 0]  # +16.7pp 악화 → 점추정 한도 초과
    verdict, checks, detail = run_decide(a, b, half=0.5)
    assert detail["A3"] is False and not checks["P7_regression_types_exposure"]
    # 유형이 비어 있으면(정답 구간 0) 침묵 통과하지 않는다
    b2, a2 = table(8, 84, extra_fp=10), table(30, 50, extra_fp=40)
    j = TYPES.index("D3")
    a2[j] = 0
    b2[j] = 0
    _, checks2, detail2 = run_decide(a2, b2)
    assert detail2["D3"] is False and not checks2["P7_regression_types_exposure"]


def _rows(counts):
    out = []
    for ty, n in counts.items():
        for i in range(n):
            out.append({"id": f"{ty}-{i}", "type": ty, "spans": [{"kind": "PERSON", "start": 0, "end": 2}] if ty not in ("D1", "D2", "A3N", "GENERIC", "OTHER") else []})
    return out


FULL = {"A1": 150, "A3": 100, "B1": 240, "B2": 80, "C1": 100, "C2": 200, "D1": 300, "D1P": 300, "D2": 100, "D3": 100, "A3N": 60, "GENERIC": 40, "OTHER": 50, "NEWNAME": 310, "LONGNAME": 120, "ORGLOC_MIX": 100}


def test_precheck_invalid_conditions_do_not_conflict_with_small_p7_rule():
    e.precheck(_rows(FULL), rehearsal=False)  # 정상 구성은 통과
    small = dict(FULL, A3=30)  # P7 유형이 1~49 → 평가 가능(점추정 규칙), INVALID 아님
    e.precheck(_rows(small), rehearsal=False)
    for bad in (dict(FULL, A3=0), dict(FULL, D1=200), dict(FULL, B1=90), dict(FULL, XYZ=10)):
        with pytest.raises(SystemExit) as ex:
            e.precheck(_rows({k: v for k, v in bad.items() if v}), rehearsal=False)
        assert "INVALID_DATASET" in str(ex.value)


def _full_rows_with_pairs():
    rows = _rows({k: v for k, v in FULL.items() if k not in ("D1", "D1P")})
    for i in range(300):
        rows.append({"id": f"P{i}-p", "type": "D1P", "pair_id": f"pr{i}", "pair_role": "person", "spans": []})
        rows.append({"id": f"P{i}-n", "type": "D1", "pair_id": f"pr{i}", "pair_role": "noun", "spans": []})
    return rows


def test_subset_selection_is_deterministic_has_1600_rows_and_keeps_pairs_together():
    spec = CRIT["supplementary_analysis"]["S1600_regression_weighted"]
    rows = _full_rows_with_pairs()
    ids, why = e.select_subset(rows, spec)
    assert why == [] and len(ids) == 1600
    ids2, _ = e.select_subset(list(reversed(rows)), spec)  # 입력 순서와 무관
    assert ids == ids2
    for i in range(300):
        assert (f"P{i}-p" in ids) == (f"P{i}-n" in ids)  # 쌍은 함께 선택
    assert sum(1 for r in rows if r["id"] in ids and r["type"] == "D1P") == 180


def test_subset_unavailable_when_quota_not_met():
    spec = CRIT["supplementary_analysis"]["S1600_regression_weighted"]
    rows = [r for r in _full_rows_with_pairs() if not (r["type"] == "OTHER" and int(r["id"].split("-")[1]) >= 49)]
    ids, why = e.select_subset(rows, spec)
    assert ids is None and any(w.startswith("OTHER") for w in why)
