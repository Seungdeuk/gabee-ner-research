"""판정 기준 JSON 이 사용자가 승인한 수치와 같은지, 실행 코드의 상수와 일치하는지 고정하는 시험(조용한 수정 방지)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_ops_final as e  # noqa: E402

CRIT = json.loads((e.OPS / "ops_decision_criteria.json").read_text(encoding="utf-8"))


def test_criteria_values_match_user_approved_thresholds():
    pc, sc = CRIT["primary_checks"], CRIT["secondary_checks"]
    assert (pc["P1_recall"]["delta_pp_min"], pc["P1_recall"]["ci_lower_pp_gt"]) == (10.0, 5.0)
    assert (pc["P2_total_exposure"]["delta_pp_max"], pc["P2_total_exposure"]["ci_upper_pp_lt"]) == (-8.0, -4.0)
    assert (pc["P3_f1_official"]["delta_pp_min"], pc["P3_f1_official"]["ci_lower_pp_gt"]) == (8.0, 0.0)
    assert (pc["P4_precision_all_noninferior"]["delta_pp_min"], pc["P4_precision_all_noninferior"]["ci_lower_pp_ge"]) == (-3.0, -6.0)
    p5 = pc["P5_negative_fpr"]
    assert (p5["absolute_max"], p5["delta_pp_max"], p5["ci_upper_pp_le"]) == (0.06, 2.0, 4.0)
    assert tuple(p5["neg_types"]) == e.P5_NEG_TYPES
    p6, p7 = pc["P6_target_types_exposure"], pc["P7_regression_types_exposure"]
    assert (p6["delta_pp_max"], p6["ci_upper_pp_le"]) == (1.0, 3.0) and tuple(p6["types"]) == e.TARGET_TYPES
    assert (p7["delta_pp_max"], p7["ci_upper_pp_le"], p7["min_gold_spans_for_ci"], p7["small_type_point_delta_pp_max"]) == (2.0, 5.0, 50, 5.0)
    p8, p9 = pc["P8_D1_noun_fpr"], pc["P9_pair_context_accuracy"]
    assert (p8["absolute_max"], p8["delta_pp_max"], p8["ci_upper_pp_le"]) == (0.5015, 10.0, 15.0)
    assert (p9["delta_pp_min"], p9["absolute_min"]) == (0.0, 0.235)
    assert (sc["S1_klue_val_f1"]["delta_pp_min"], sc["S1_klue_val_f1"]["absolute_min"]) == (-1.0, 0.6854)
    s2 = sc["S2_g12_onnx_cpu"]
    assert (s2["parity_min_pct"], s2["p50_ratio_max"], s2["size_ratio_max"], s2["rss_ratio_max"]) == (99.9, 1.30, 1.30, 1.30)
    vm = CRIT["dataset_validity_minimums"]
    assert CRIT["statistics"]["resamples"] == 5000 and (vm["gold_spans_per_P6_type_min"], vm["D1_noun_sentences_min"]) == (e.MIN_GOLD_P6, e.MIN_D1_SENT)


def test_p7_types_documented_equal_code_exclusions():
    """코드가 P7 에서 빼는 유형(P6 + D1·D2·A3N·GENERIC·OTHER)이 문서의 정의와 같다."""
    assert set(CRIT["primary_checks"]["P6_target_types_exposure"]["types"]) == set(e.TARGET_TYPES)
    text = CRIT["primary_checks"]["P7_regression_types_exposure"]["types"]
    for t in ("A1", "A3", "B2", "C1", "D3"):
        assert t in text


def test_p7_types_subset_quota_and_allowed_types_are_consistent():
    p7 = CRIT["primary_checks"]["P7_regression_types_exposure"]
    assert tuple(p7["types"]) == e.P7_TYPES
    assert tuple(CRIT["dataset_validity_minimums"]["allowed_types"]) == e.ALLOWED_TYPES
    sub = CRIT["supplementary_analysis"]["S1600_regression_weighted"]
    assert sum(sub["quota_rows"].values()) == sub["total_rows"] == 1600
    assert CRIT["secondary_checks"]["S2_g12_onnx_cpu"]["S2b"]["status"].startswith("UNVERIFIED")
    assert "S2b" in CRIT["verdict"]["OPS_NOT_READY"]
