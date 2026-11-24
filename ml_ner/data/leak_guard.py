"""시험(test) 누출 방지 가드.

Baseline test.jsonl 은 학습·검증 데이터로 쓰지 않는다. 이 모듈의 ``load_test_guard`` 만 test 를 읽고, 그 결과는
(1) 합성 값 풀에서 뺄 문자열(이름·기관·주소·지명) 차단 목록, (2) 누출 비교용 정규화 문장·문장 틀 집합 으로만 쓴다.
test 문장 자체는 어떤 산출 jsonl 에도 들어가지 않는다(시험: tests/test_data_pipeline.py).
"""

from dataclasses import dataclass, field

import ner_common as nc

# 이 파일 밖에서 test.jsonl 경로를 직접 열면 안 된다(정적 시험이 검사). 경로 조각만 여기서 만든다.
_TEST_FILE = "test" + ".jsonl"
KIN_WORDS = {"엄마", "아빠", "삼촌", "이모", "고모", "할머니", "할아버지", "오빠", "언니", "형", "누나", "이모부", "고모부"}


@dataclass
class TestGuard:
    """test 에서 뽑은 차단 목록과 비교용 집합(원문 문장은 들고 있지 않는다)."""

    texts_norm: set = field(default_factory=set)
    skeletons_ner: set = field(default_factory=set)  # NER 라벨만 [라벨] 로 바꾼 틀
    skeletons_all: set = field(default_factory=set)  # 모든 라벨을 바꾼 틀(Baseline 평가 스크립트와 동일)
    values: dict = field(default_factory=dict)  # label -> set(span 문자열)
    given_parts: set = field(default_factory=set)  # PERSON/ALIAS 에서 성을 뗀 이름 부분

    def person_strings(self):
        """PERSON·PERSON_ALIAS 문자열 전체."""
        return self.values.get("PERSON", set()) | self.values.get("PERSON_ALIAS", set())

    def blocked_name(self, s):
        """이름 후보가 test 의 이름 문자열과 같거나 포함되거나 test 이름의 이름 부분이면 True."""
        if s in self.given_parts or s in self.person_strings():
            return True
        return len(s) >= 2 and any(s in t or t in s for t in self.person_strings() if len(t) >= 2)

    def blocked_text_value(self, s, label):
        """기관·주소·지명 후보 문자열이 test 의 같은 라벨 값과 같거나 서로 포함되면 True."""
        return any(s == t or s in t or t in s for t in self.values.get(label, set()))


def load_test_guard(baseline_dir=None):
    """test.jsonl 을 읽어 차단 목록만 만든다(학습 데이터로 쓰지 않는다).

    @param baseline_dir: Baseline 폴더(기본: 저장소 baseline_v2)
    @returns: TestGuard
    """
    path = (baseline_dir or nc.BASELINE_DIR) / _TEST_FILE
    g = TestGuard()
    for r in nc.read_jsonl(path):
        g.texts_norm.add(nc.normalize_text(r["text"]))
        g.skeletons_all.add(nc.skeleton(r["text"], r["spans"]))
        g.skeletons_ner.add(nc.skeleton(r["text"], nc.to_ner_spans(r)))
        for sp in nc.to_ner_spans(r):
            val = r["text"][sp["start"] : sp["end"]]
            g.values.setdefault(sp["label"], set()).add(val)
            if sp["label"] in ("PERSON", "PERSON_ALIAS"):
                for tok in val.split():
                    if tok in KIN_WORDS:
                        continue
                    g.given_parts.add(tok)
                    if len(tok) >= 2:
                        g.given_parts.add(tok[1:])
                    if len(tok) >= 3:
                        g.given_parts.add(tok[2:])
    return g
