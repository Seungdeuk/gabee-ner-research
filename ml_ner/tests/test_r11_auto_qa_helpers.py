"""자동 품질 검증 보조 함수 시험(합성 문자열): 이모티콘이 만드는 괄호 불균형은 오류가 아니고, 진짜 불균형은 잡는다. 받침 판정."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

import pytest  # noqa: E402

pytest.importorskip("tokenizers")
import r11_auto_qa as qa  # noqa: E402
import review_r11_data as rv  # noqa: E402


def test_emoticon_parenthesis_is_not_unbalanced():
    assert qa.bad_brackets("잘 했어 :)") is False
    assert qa.bad_brackets("고마워 ㅠㅠ)") is False
    assert qa.bad_brackets("(오늘 모임은 취소") is True  # 진짜 불균형
    assert qa.bad_brackets('그가 "안녕 하고 말했어') is True


def test_batchim_detection():
    assert rv.batchim("민") is True and rv.batchim("수") is False and rv.batchim("a") is None
