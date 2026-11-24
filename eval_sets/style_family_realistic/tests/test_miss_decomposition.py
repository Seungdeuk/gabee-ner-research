from app.shared.privacy.eval.style_family_realistic.miss_decomposition import (
    classify_complete_miss,
)
from app.shared.privacy.ner import NerSpan


def test_no_overlapping_person_span_is_no_candidate():
    text = "민준이 학교 갔어"
    raw = [NerSpan("ORGANIZATION", 4, 6, 0.9)]  # 다른 라벨만 겹침, PERSON류는 없음
    assert classify_complete_miss(text, 0, 2, raw) == "후보자체미생성"


def test_overlapping_person_below_min_score_is_score_dropped():
    text = "민준이 학교 갔어"
    raw = [NerSpan("PERSON", 0, 3, 0.4)]  # NER_MIN_SCORE(0.5) 미만
    assert classify_complete_miss(text, 0, 3, raw) == "점수탈락"


def test_short_fragment_after_trim_is_postproc_altered():
    text = "윤이인이랑 진기호 둘이 또 투닥거리네"
    # "윤이인"이 한 글자씩 쪼개져 각각 2글자 미만 PERSON 으로 버려지는 경우
    raw = [
        NerSpan("PERSON", 0, 1, 0.8),
        NerSpan("PERSON_ALIAS", 1, 2, 0.6),
        NerSpan("PERSON", 2, 3, 0.9),
    ]
    assert classify_complete_miss(text, 0, 3, raw) == "후처리변형"


def test_valid_candidate_overlapping_after_trim_is_merge_stage():
    text = "오늘 민준 생일이야"
    # 점수 0.5 이상·2글자 이상·가족호칭 아님 -> 다듬은 뒤에도 정답과 겹침(실제로는 merge_ner 의 규칙-겹침 처리 때문에 최종엔 안 남는 경우)
    raw = [NerSpan("PERSON", 3, 5, 0.8)]
    assert classify_complete_miss(text, 3, 5, raw) == "후처리변형(병합단계)"
