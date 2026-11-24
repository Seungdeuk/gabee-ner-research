"""의미적 중의성 진단셋(180문장, 전부 합성): 이름이기도 한 단어 30개 × PERSON 문맥 3 + 비PERSON 문맥 3.

**이 폴더는 "진단셋"이다, v4~v8 같은 "독립 평가셋"이 아니다.** 단어·문장틀이 기존 v7/v8 등과
겹치는 걸 일부러 막지 않았다(겹침 여부는 `check_overlap.py`로 기록만 한다). 목적은 특정 취약점(기관명
충돌형 vs 일반단어 충돌형)을 들여다보는 것이지, 새 성능 수치를 공식 벤치마크로 쓰려는 게 아니다.
**모델·규칙을 고치면 이 셋이 아니라 새 독립 평가셋(v9)으로 재검증해야 한다.**

30개 단어, 3개 그룹:
- 기관·지역(4): 연준·상원·진주·동해 — 비PERSON 문맥의 정답은 ORGANIZATION 또는 LOCATION
- 자연·계절(13): 하늘·바다·이슬·노을·새벽·별·여름·가을·은하·태양·우주·장미·초원 — 비PERSON 정답은 O
- 일상·추상(13): 지수·정원·수정·미래·사랑·보라·하나·지혜·지성·미소·선율·연수·다정 — 비PERSON 정답은 O

PERSON 문맥 3문장은 공통 템플릿(조사 유무·조사 종류를 다르게) 3개를 쓰고, 비PERSON 문맥 3문장은 단어마다
자연스러운 문장을 직접 썼다(생성 템플릿을 억지로 맞추면 부자연스러워짐). `known_names`는 전부 비워
모델 자체의 문맥 구분 능력만 본다(조건 7).

사용: ``python build_ambiguity_probe.py <출력.jsonl>``
"""

import argparse
import json
from pathlib import Path

GROUPS = {
    "연준": ("기관지역", "ORGANIZATION"),
    "상원": ("기관지역", "ORGANIZATION"),
    "진주": ("기관지역", "LOCATION"),
    "동해": ("기관지역", "LOCATION"),
    "하늘": ("자연계절", "O"), "바다": ("자연계절", "O"), "이슬": ("자연계절", "O"),
    "노을": ("자연계절", "O"), "새벽": ("자연계절", "O"), "별": ("자연계절", "O"),
    "여름": ("자연계절", "O"), "가을": ("자연계절", "O"), "은하": ("자연계절", "O"),
    "태양": ("자연계절", "O"), "우주": ("자연계절", "O"), "장미": ("자연계절", "O"),
    "초원": ("자연계절", "O"),
    "지수": ("일상추상", "O"), "정원": ("일상추상", "O"), "수정": ("일상추상", "O"),
    "미래": ("일상추상", "O"), "사랑": ("일상추상", "O"), "보라": ("일상추상", "O"),
    "하나": ("일상추상", "O"), "지혜": ("일상추상", "O"), "지성": ("일상추상", "O"),
    "미소": ("일상추상", "O"), "선율": ("일상추상", "O"), "연수": ("일상추상", "O"),
    "다정": ("일상추상", "O"),
}

# 비PERSON 문맥(단어마다 직접 쓴 자연스러운 문장 3개). {w} 는 build() 에서 치환한다.
NONPERSON_SENTENCES: dict[str, list[str]] = {
    "연준": ["오늘 {w}이 기준금리를 동결했다", "{w} 의장이 금리 인상을 시사했다", "{w}의 통화정책 회의 결과가 발표됐다"],
    "상원": ["오늘 {w}이 법안을 통과시켰다", "{w} 의원들이 표결을 진행했다", "{w}에서 예산안이 부결됐다"],
    "진주": ["{w}는 경상남도에 있는 도시다", "{w} 시내에 새 다리가 생겼다", "{w}행 기차표를 예매했다"],
    "동해": ["{w}에서 해돋이를 봤다", "{w} 바닷가로 여행을 갔다", "{w}안전행정부 발표에 따르면 수온이 올랐다"],
    "하늘": ["{w}이 오늘따라 유난히 맑다", "{w} 위로 비행기가 지나갔다", "{w}에 별이 가득했다"],
    "바다": ["{w}가 오늘따라 잔잔하다", "{w} 위로 갈매기가 날았다", "{w}에서 수영을 했다"],
    "이슬": ["{w}이 풀잎에 맺혀 있었다", "아침 {w}이 차갑게 느껴졌다", "{w}을 밟으며 산책했다"],
    "노을": ["{w}이 붉게 물들었다", "{w}을 보며 사진을 찍었다", "{w}이 지고 어두워졌다"],
    "새벽": ["{w}에 비가 내렸다", "{w} 공기가 차가웠다", "{w}까지 공부를 했다"],
    "별": ["{w}이 밤하늘에 빛났다", "{w}을 보며 소원을 빌었다", "{w}이 유난히 반짝였다"],
    "여름": ["{w}이 벌써 끝나간다", "{w}에는 에어컨이 필수다", "{w} 방학이 시작됐다"],
    "가을": ["{w}이 되니 날씨가 선선하다", "{w}에는 단풍이 아름답다", "{w}바람이 선선하게 불었다"],
    "은하": ["{w}는 수많은 별들로 이루어져 있다", "{w}의 중심에는 블랙홀이 있다", "{w}를 관측하는 망원경을 샀다"],
    "태양": ["{w}이 뜨겁게 내리쬐었다", "{w}이 서쪽으로 지고 있었다", "{w} 아래서 땀을 흘렸다"],
    "우주": ["{w}는 끝없이 넓다", "{w}를 탐사하는 로켓이 발사됐다", "{w}에 대한 다큐멘터리를 봤다"],
    "장미": ["{w}가 정원에 활짝 피었다", "{w} 꽃다발을 선물했다", "{w}에는 가시가 있다"],
    "초원": ["{w} 위로 바람이 불었다", "{w}에서 양들이 풀을 뜯었다", "{w}이 끝없이 펼쳐졌다"],
    "지수": ["주가 {w}가 올랐다", "물가 {w}를 발표했다", "체질량 {w}를 측정했다"],
    "정원": ["{w}에 꽃을 심었다", "{w}을 가꾸는 게 취미다", "{w}에서 차를 마셨다"],
    "수정": ["보고서를 {w}했다", "계획을 {w}하기로 했다", "{w}된 내용을 확인했다"],
    "미래": ["{w}를 준비해야 한다", "{w}에는 어떤 일이 생길까", "{w}가 기대된다"],
    "사랑": ["{w}은 소중한 감정이다", "{w}에 빠졌다", "{w}을 표현했다"],
    "보라": ["{w}색 옷을 입었다", "{w}빛 노을이 아름다웠다", "{w}색을 가장 좋아한다"],
    "하나": ["{w}부터 열까지 다 챙겼다", "{w}만 더 주세요", "{w}씩 정리했다"],
    "지혜": ["{w}를 발휘해야 할 때다", "{w}로운 선택을 했다", "{w}가 필요한 순간이다"],
    "지성": ["{w}을 갖춘 사람이다", "{w}과 감성을 겸비했다", "{w}미가 느껴졌다"],
    "미소": ["{w}를 지었다", "{w}가 아름다웠다", "따뜻한 {w}를 보였다"],
    "선율": ["아름다운 {w}이 흘러나왔다", "{w}에 맞춰 춤을 췄다", "{w}이 귀에 맴돌았다"],
    "연수": ["해외 {w}를 다녀왔다", "신입사원 {w}를 받았다", "{w} 프로그램에 참여했다"],
    "다정": ["{w}한 말투로 이야기했다", "{w}하게 안아주었다", "{w}한 성격이다"],
}

PERSON_TEMPLATES = [
    "{subj} 오늘 피아노 학원 다녀왔대",
    "어제 {word}한테 전화했는데 안 받더라",
    "우리 반에서 {subj} 제일 웃겨",
]


def has_batchim(ch: str) -> bool:
    return (ord(ch) - 0xAC00) % 28 != 0


def josa(word: str) -> str:
    return word + ("이" if has_batchim(word[-1]) else "가")


def find_span(text: str, token: str) -> tuple[int, int]:
    idx = text.find(token)
    if idx < 0:
        raise ValueError(f"토큰을 못 찾음: {token!r} in {text!r}")
    return idx, idx + len(token)


def build_rows() -> list[dict]:
    rows = []
    for word, (group, nonperson_label) in GROUPS.items():
        for i, tmpl in enumerate(PERSON_TEMPLATES):
            subj = josa(word) if "{subj}" in tmpl else word
            text = tmpl.format(subj=subj, word=word)
            s, e = find_span(text, word)
            rows.append(
                {
                    "id": f"amb-{word}-person-{i+1}",
                    "word": word,
                    "group": group,
                    "context_type": "PERSON",
                    "gold_label": "PERSON",
                    "text": text,
                    "spans": [{"kind": "PERSON", "start": s, "end": e}],
                    "known_names": [],
                    "ambiguous_flag": False,
                }
            )
        for i, tmpl in enumerate(NONPERSON_SENTENCES[word]):
            text = tmpl.format(w=word)
            s, e = find_span(text, word)
            rows.append(
                {
                    "id": f"amb-{word}-nonperson-{i+1}",
                    "word": word,
                    "group": group,
                    "context_type": "NONPERSON",
                    "gold_label": nonperson_label,
                    "text": text,
                    "spans": [{"kind": nonperson_label, "start": s, "end": e}] if nonperson_label != "O" else [],
                    "known_names": [],
                    "ambiguous_flag": False,
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    rows = build_rows()
    assert len(rows) == 180, f"180문장이어야 하는데 {len(rows)}개"
    texts = [r["text"] for r in rows]
    dupes = len(texts) - len(set(texts))
    assert dupes == 0, f"문장 중복 {dupes}건"
    for r in rows:
        for g in r["spans"]:
            assert r["text"][g["start"] : g["end"]] == r["word"], f"구간 검증 실패: {r['id']}"

    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], {"PERSON": 0, "NONPERSON": 0})[r["context_type"]] += 1
    print("그룹별 문장 수:", by_group)

    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"총 {len(rows)}문장 저장: {args.out}")


if __name__ == "__main__":
    main()
