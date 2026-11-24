"""메신저체 완전미탐(none)을 세 원인으로 가른다(Codex 2순위 제안 실행).

``error_analysis.py``(로드맵 2단계)는 라벨 종류·경계만 본다. 여기서는 완전미탐(최종 출력에 그 정답 구간을
덮는 PERSON류가 전혀 없는 경우)만 대상으로, ``app.shared.privacy.ner.merge_ner`` 내부에서 **어느 단계에서
버려졌는지**를 가른다:

- 후보자체미생성: ``spans_for``(원본 예측, 라벨 필터·점수 하한 전)에 PERSON/PERSON_ALIAS 라벨조차 그 구간과
  겹치는 게 하나도 없다 — 모델이 애초에 아무 것도 못 봤다.
- 점수탈락: PERSON류 겹치는 예측은 있지만 전부 ``NER_MIN_SCORE``(0.5) 미만이라 ``merge_ner`` 가 후보로도
  안 본다.
- 후처리변형: 점수 0.5 이상인 PERSON류 겹치는 예측이 있지만, 다듬기(공백·구두점 제거)·2글자 미만 PERSON
  버림·가족호칭(KINSHIP_TERMS) 버림 중 하나에 걸려 최종 구간이 정답과 안 겹치게 됐다.
"""

from collections.abc import Sequence

from app.shared.privacy.detectors import KINSHIP_TERMS
from app.shared.privacy.ner import NER_MIN_SCORE, NerSpan, _trim
from app.shared.privacy.types import Kind

PERSON_LABELS = ("PERSON", "PERSON_ALIAS")


def classify_complete_miss(
    text: str, gold_start: int, gold_end: int, raw_spans: Sequence[NerSpan]
) -> str:
    """완전미탐 정답 구간 하나의 원인. 호출 전 "최종 출력에 겹치는 PERSON 류가 없다"(완전미탐)를 이미 확인했다고 가정한다."""

    def overlaps(span: NerSpan) -> bool:
        return max(span.start, gold_start) < min(span.end, gold_end)

    person_over = [s for s in raw_spans if s.label in PERSON_LABELS and overlaps(s)]
    if not person_over:
        return "후보자체미생성"
    if all(s.score < NER_MIN_SCORE for s in person_over):
        return "점수탈락"
    # 점수 0.5 이상인 후보가 있다 -> merge_ner 과 같은 다듬기를 재현해 왜 안 남았는지 확인
    for s in person_over:
        if s.score < NER_MIN_SCORE:
            continue
        a, b = _trim(text, s.start, s.end, Kind.PERSON)
        value = text[a:b]
        if b <= a or value in KINSHIP_TERMS or len(value) < 2:
            continue  # 이 후보는 후처리에서 버려짐 -> 다른 후보가 혹시 살아남는지 계속 확인
        if max(a, gold_start) < min(b, gold_end):
            # 다듬은 뒤에도 정답과 겹치는데 최종 출력엔 없다 -> merge_ner 의 규칙-겹침 처리(병합 로직) 때문
            return "후처리변형(병합단계)"
    return "후처리변형"
