"""R11-G12 판정 로직·집계·일치율 계산 시험(합성 수치). 측정 자체의 통합 시험은 S0 대 S0 로 별도 실행한다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ml" / "ner" / "train"))

import g12_parity as gp  # noqa: E402
import g12_r11 as g  # noqa: E402

S0 = {"p50_ms": 10.0, "p95_ms": 20.0, "mean_ms": 11.0, "model_size_mb": 54.0, "rss_mb_after": 300.0}


def cand(**kw):
    return {**S0, **kw}


def test_same_as_s0_passes():
    r = g.judge(100.0, cand(), S0)
    assert r["G12"] is True and abs(r["values"]["latency_p50_ratio"] - 1.0) < 1e-12


def test_boundary_plus_30_percent_passes_and_just_above_fails():
    assert g.judge(100.0, cand(p50_ms=13.0), S0)["G12"] is True  # +30.0% 경계는 통과
    assert g.judge(100.0, cand(p50_ms=13.01), S0)["pass_by_item"]["latency_p50"] is False


def test_each_item_failure_fails_g12():
    assert g.judge(99.89, cand(), S0)["pass_by_item"]["parity"] is False  # 일치율 99.9% 미만
    assert g.judge(99.9, cand(), S0)["pass_by_item"]["parity"] is True
    assert g.judge(100.0, cand(model_size_mb=70.5), S0)["G12"] is False
    assert g.judge(100.0, cand(rss_mb_after=391.0), S0)["G12"] is False


def test_p95_and_mean_are_reported_but_not_gating():
    r = g.judge(100.0, cand(p95_ms=60.0, mean_ms=30.0), S0)
    assert r["G12"] is True and r["values"]["latency_p95_ratio(report)"] == 3.0


def test_aggregate_uses_median_over_runs():
    runs = [{**S0, "p50_ms": 9.0}, {**S0, "p50_ms": 30.0}, {**S0, "p50_ms": 10.0}]
    assert g.aggregate_runs(runs)["p50_ms"] == 10.0  # 이상치 한 번에 흔들리지 않는다


def test_measurement_order_is_interleaved_three_each():
    assert g.ORDER.count("s0") == 3 and g.ORDER.count("cand") == 3 and g.ORDER[0] != g.ORDER[1]


def test_token_agreement_counts_and_length_mismatch():
    ag, tot, same, lm = gp.agreement([[1, 2, 3], [4, 5]], [[1, 2, 9], [4, 5, 6]])
    assert (ag, tot, same, lm) == (4, 5, 0, 1)  # 첫 문장 1개 불일치, 둘째 문장은 길이 불일치라 완전 일치 아님
