"""우선순위 2 · 1단계: 템플릿에 취약한 7단어(가을·미소·사랑·새벽·수정·여름·이슬)에 대한
합성 데이터 초안(DRAFT)을 만든다. **이 스크립트의 출력은 `ml/ner/data/synth.py`에 추가되지 않고
어떤 학습에도 쓰이지 않는다** — 1단계는 설계 초안까지만이고, 2단계(파인튜닝)는 별도 승인 후.

대상 선정 근거: `analyze_candidate_gap_step1.py` 결과 두 모델(서비스·R1) 공통으로 PERSON 템플릿
3개 중 적어도 하나에서 원시 후보 자체가 안 생기는 단어 7개. PERSON 문장은 기존 평가셋(T1~T3)이
덜 쓴 문형(여격 "한테"·목적격 "을/를"·호격·"제일" 서술)으로 새로 썼고, 비PERSON 문장도 기존
`build_ambiguity_probe.py`의 문장과 겹치지 않게 새로 썼다(README 설계 원칙 4).
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def has_batchim(ch: str) -> bool:
    return (ord(ch) - 0xAC00) % 28 != 0


def subj(w: str) -> str:
    return w + ("이" if has_batchim(w[-1]) else "가")


def obj(w: str) -> str:
    return w + ("을" if has_batchim(w[-1]) else "를")


def topic(w: str) -> str:
    return w + ("은" if has_batchim(w[-1]) else "는")


def voc(w: str) -> str:
    return w + ("아" if has_batchim(w[-1]) else "야")


WORDS = ["가을", "미소", "사랑", "새벽", "수정", "여름", "이슬"]

PERSON_TEMPLATES = [
    lambda w: f"{w}한테 생일 선물로 뭘 줄지 아직 못 정했어",
    lambda w: f"{subj(w)} 이번 학기 반장 선거에서 뽑혔대",
    lambda w: f"선생님이 {obj(w)} 칭찬하셨어",
    lambda w: f"{subj(w)} 요즘 유치원에서 제일 인기 많아",
    lambda w: f"{voc(w)}, 저녁에 떡볶이 먹으러 갈래?",
]

NONPERSON_SENTENCES = {
    "가을": ["다음 주부터 가을 학기가 시작된다", "가을걷이가 한창이다", "올해는 가을이 유난히 짧았다"],
    "미소": ["사진 속 그녀는 은은한 미소를 띠고 있었다", "면접관이 미소로 긴장을 풀어줬다", "미소 하나로 분위기가 밝아졌다"],
    "사랑": ["두 사람의 사랑은 변함이 없었다", "사랑이란 말로 다 표현할 수 없었다", "책에서 사랑에 대한 구절을 읽었다"],
    "새벽": ["새벽 기차를 타고 출발했다", "그는 새벽마다 운동을 한다", "새벽부터 줄을 서서 기다렸다"],
    "수정": ["문서의 오타를 수정했다", "일정을 조금 수정해야겠다", "수정이 끝난 뒤 다시 보냈다"],
    "여름": ["매년 여름이면 바다에 간다", "올여름은 유난히 더웠다", "여름휴가 계획을 세웠다"],
    "이슬": ["새벽 산길에 이슬이 가득했다", "풀잎 위 이슬방울이 반짝였다", "이슬에 바지가 다 젖었다"],
}


def find_span(text: str, token: str) -> tuple[int, int]:
    idx = text.find(token)
    if idx < 0:
        raise ValueError(f"토큰을 못 찾음: {token!r} in {text!r}")
    return idx, idx + len(token)


def build() -> list[dict]:
    rows = []
    for w in WORDS:
        for i, tmpl in enumerate(PERSON_TEMPLATES):
            text = tmpl(w)
            s, e = find_span(text, w)
            rows.append(
                {
                    "id": f"draft-{w}-person-{i+1}", "word": w, "context_type": "PERSON",
                    "gold_label": "PERSON", "text": text, "spans": [{"kind": "PERSON", "start": s, "end": e}],
                }
            )
        for i, sent in enumerate(NONPERSON_SENTENCES[w]):
            s, e = find_span(sent, w)
            rows.append(
                {
                    "id": f"draft-{w}-nonperson-{i+1}", "word": w, "context_type": "NONPERSON",
                    "gold_label": "O", "text": sent, "spans": [],
                }
            )
    return rows


def main() -> None:
    rows = build()
    texts = [r["text"] for r in rows]
    assert len(texts) == len(set(texts)), "문장 중복 있음"
    for r in rows:
        for g in r["spans"]:
            assert r["text"][g["start"] : g["end"]] == r["word"]
    out = HERE / "synth_draft.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"초안 {len(rows)}문장(단어 {len(WORDS)}개 × PERSON 5 + 비PERSON 3) 저장: {out}")


if __name__ == "__main__":
    main()
