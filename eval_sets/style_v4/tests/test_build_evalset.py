import pytest

from app.shared.privacy.eval.style_v4.build_evalset import (
    assert_no_overlap,
    skeleton_from_template,
)


def test_assert_no_overlap_fails_on_shared_name():
    rows = [
        {
            "category": "가족대화",
            "text": "김철수가 왔어",
            "spans": [{"kind": "PERSON", "start": 0, "end": 3}],
        }
    ]
    with pytest.raises(SystemExit):
        assert_no_overlap(rows, excluded_names={"김철수"}, template_texts=set())


def test_assert_no_overlap_passes_on_fresh_name():
    rows = [
        {
            "category": "가족대화",
            "text": "최유일이 왔어",
            "spans": [{"kind": "PERSON", "start": 0, "end": 3}],
        }
    ]
    assert_no_overlap(rows, excluded_names={"김철수"}, template_texts=set())  # 예외 없이 통과


def test_assert_no_overlap_catches_given_name_reused_via_suffix():
    """given("하연")이 제외 집합의 어떤 전체 이름("송하연")의 끝부분이면, 홑 given 사용도 겹침이다."""
    rows = [
        {
            "category": "이름만",
            "text": "하연이 왔어",
            "spans": [{"kind": "PERSON", "start": 0, "end": 2}],
        }
    ]
    with pytest.raises(SystemExit):
        assert_no_overlap(rows, excluded_names={"송하연"}, template_texts=set())


def test_skeleton_from_template_expands_synth_josa_marker():
    """synth.py 는 조사를 {슬롯} 밖에 ``@이/가`` 로 적는다 — 두 조사 변형 모두 뼈대 집합에 들어가야 한다."""
    variants = skeleton_from_template("{G}@이/가 숙제 다 했대")
    assert "이 숙제 다 했대" in variants
    assert "가 숙제 다 했대" in variants


def test_assert_no_overlap_catches_synth_josa_style_collision():
    row = {
        "category": "가족대화",
        "text": "백도율이 숙제 다 했대",
        "spans": [{"kind": "PERSON", "start": 0, "end": 3}],
    }
    template_texts = skeleton_from_template("{G}@이/가 숙제 다 했대")
    with pytest.raises(SystemExit):
        assert_no_overlap([row], excluded_names=set(), template_texts=template_texts)
