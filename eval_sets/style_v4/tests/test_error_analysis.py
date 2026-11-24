from app.shared.privacy.eval.style_v4.error_analysis import classify_miss
from app.shared.privacy.ner import NerSpan


def test_classify_miss_returns_complete_miss_when_nothing_overlaps():
    raw_spans = [NerSpan("PERSON", 20, 23, 0.9)]  # 전혀 다른 위치
    assert classify_miss(0, 3, raw_spans) == "완전미탐"


def test_classify_miss_returns_complete_miss_on_empty_spans():
    assert classify_miss(0, 3, []) == "완전미탐"


def test_classify_miss_returns_wrong_label_when_only_other_kind_overlaps():
    raw_spans = [NerSpan("ORGANIZATION", 0, 3, 0.9)]  # 이름 위치를 기관명으로 잡음
    assert classify_miss(0, 3, raw_spans) == "라벨매핑"


def test_classify_miss_returns_boundary_when_person_label_only_partially_covers():
    raw_spans = [NerSpan("PERSON", 0, 2, 0.9)]  # 정답은 [0,3) 인데 [0,2) 만
    assert classify_miss(0, 3, raw_spans) == "경계오류"


def test_classify_miss_prefers_boundary_over_wrong_label_when_both_overlap():
    raw_spans = [
        NerSpan("ORGANIZATION", 0, 1, 0.9),
        NerSpan("PERSON", 1, 2, 0.9),
    ]
    assert classify_miss(0, 3, raw_spans) == "경계오류"
