"""문체별 평가셋에서 놓친 이름(none/partial)이 왜 놓쳤는지 분류한다(로드맵 2단계: 오류 분석).

``app.shared.privacy.ner.spans_for`` 가 돌려주는 원본 예측(라벨 필터링·규칙 병합 이전, LOCATION·ORGANIZATION
포함)을 정답 구간과 대조해 세 가지로 나눈다:

- 완전미탐: 그 구간에 어떤 라벨의 예측도 전혀 겹치지 않음.
- 라벨매핑: PERSON/PERSON_ALIAS 가 아닌 다른 라벨(LOCATION·ORGANIZATION 등)만 겹침 — 모델이 이름을 다른 것으로 봤다.
- 경계오류: PERSON/PERSON_ALIAS 라벨이 겹치긴 하는데 정답 구간을 전부 덮지 못함(잘림).
"""

from collections.abc import Sequence

from app.shared.privacy.ner import NerSpan

PERSON_LABELS = ("PERSON", "PERSON_ALIAS")


def classify_miss(gold_start: int, gold_end: int, raw_spans: Sequence[NerSpan]) -> str:
    """정답 구간 [gold_start, gold_end) 를 완전히 덮지 못한 경우의 원인 한 가지를 돌려준다.

    같은 자리에 PERSON류 라벨이 하나라도 겹치면(일부만 덮어도) "경계오류" — 이름으로는 봤지만 구간이 잘렸다는
    뜻이라 라벨매핑보다 더 구체적인 원인이다. 그다음으로 다른 라벨이라도 겹치면 "라벨매핑", 아무 것도 안 겹치면
    "완전미탐".
    """

    def overlaps(span: NerSpan) -> bool:
        return max(span.start, gold_start) < min(span.end, gold_end)

    overlapping = [s for s in raw_spans if overlaps(s)]
    if any(s.label in PERSON_LABELS for s in overlapping):
        return "경계오류"
    if overlapping:
        return "라벨매핑"
    return "완전미탐"
