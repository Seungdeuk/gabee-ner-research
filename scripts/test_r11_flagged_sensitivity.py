"""의심 행 제외 민감도 계산의 산술 시험(합성 개수): 전체 개수에서 부분 개수를 빼 파생 지표를 다시 계산하는 규칙이 맞는지 확인한다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import r11_flagged_sensitivity as s  # noqa: E402


def counts(n, exact, bo, part, full, tp, fp_gold, fp_neg, neg_n, neg_flag, nsent):
    return {"n_gold_spans": n, "exact": exact, "boundary_only": bo, "partial": part, "full": full, "tp": tp, "fn": n - exact, "fp_in_gold_rows": fp_gold, "fp_in_neg_rows": fp_neg,
            "n_neg_sentences": neg_n, "neg_flagged": neg_flag, "n_sentences": nsent}


def test_derive_matches_definitions():
    d = s.derive(counts(100, 70, 10, 5, 15, 70, 10, 5, 20, 4, 120))
    assert d["exact_recall"] == 0.7 and d["mask_rate"] == 0.8 and d["total_exposure_rate"] == 0.2
    assert abs(d["precision_official"] - 70 / 80) < 1e-12 and abs(d["precision_all"] - 70 / 85) < 1e-12 and d["nonperson_fpr"] == 0.2
    # 항등식: 가림률 - Recall == 경계만 틀림
    assert abs(d["mask_rate"] - d["exact_recall"] - d["boundary_only_rate"]) < 1e-12


def test_sub_counts_subtracts_part_and_handles_missing_type():
    full = counts(100, 70, 10, 5, 15, 70, 10, 5, 20, 4, 120)
    part = counts(4, 1, 1, 1, 1, 1, 0, 0, 0, 0, 4)
    r = s.sub_counts(full, part)
    assert r["n_gold_spans"] == 96 and r["exact"] == 69 and r["full"] == 14 and r["n_sentences"] == 116
    assert s.sub_counts(full, None) == {k: full[k] for k in s.CNT}


def test_exclusion_rate_change_is_exactly_the_subtracted_counts():
    full = counts(1497, 1100, 120, 50, 227, 1100, 150, 40, 500, 15, 1920)
    part = counts(10, 4, 2, 2, 2, 4, 1, 0, 0, 0, 12)
    base, after = s.derive(full), s.derive(s.sub_counts(full, part))
    assert abs(after["exact_recall"] - (1100 - 4) / (1497 - 10)) < 1e-12
    assert abs(after["total_exposure_rate"] - ((50 - 2) + (227 - 2)) / (1497 - 10)) < 1e-12
    assert abs(after["exact_recall"] - base["exact_recall"]) < 0.01  # 의심 10구간 제외의 영향은 1%p 미만
