"""v11 — R5 최종 잠금 독립 평가셋(2026-11-16).

목적: R5가 **학습·모델 선정(baseline_val)에 전혀 쓰지 않은 신규 20단어**(복합명사형 10 + 단일형 10)
에서 전체 이름 노출률 <5% 등 55절 목표를 달성했는지 **딱 한 번** 확인한다. v9·v10·R3·R4·R5(학습+
holdout+의미범주)·48절·draft56·GIVEN/NICK 풀·baseline/v4~v8 어디에도 없는 단어로만 구성했다
(`leak_check`가 자동 확인). 결과를 보고 R5를 다시 튜닝하지 않는다.

구성(사용자 지시 2026-11-16):
- 5개 문형(목적격·비교서술·기본·여격·호격) 포함
- 두 문체를 구분해 집계: **중의성문맥**(통제된 최소 문형, v9/v10류와 동일한 엄격함) vs
  **가족채팅체**(더 자연스러운 가족 대화체 — 같은 문형이지만 표현이 느슨함)
- 복합명사형·단일형 균형(10/10)
- PERSON + 비PERSON 문장 포함
- 중의성 오탐은 48절(ambiguity_probe)을 그대로 재사용해 교차 확인한다(별도 재구축 안 함,
  55절 표준 정의와 일관성 유지).
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

COMPOUND_WORDS = ["산그늘", "길섶", "나무그늘", "돌계단", "벽돌담", "장독대", "마룻바닥", "다락방", "빨랫줄", "지붕끝"]
SINGLE_WORDS = ["모래알", "빗장", "창틀", "문턱", "굴뚝", "금빛", "놋그릇", "질그릇", "숯덩이", "메주"]
WORDS = COMPOUND_WORDS + SINGLE_WORDS
WORD_TYPE = {w: "복합명사형" for w in COMPOUND_WORDS} | {w: "단일형" for w in SINGLE_WORDS}


def has_batchim(ch: str) -> bool:
    return (ord(ch) - 0xAC00) % 28 != 0


def obj(w: str) -> str:
    return w + ("을" if has_batchim(w[-1]) else "를")


def subj(w: str) -> str:
    return w + ("이" if has_batchim(w[-1]) else "가")


def voc(w: str) -> str:
    return w + ("아" if has_batchim(w[-1]) else "야")


# 중의성문맥: 통제된 최소 문형(v9/v10과 같은 엄격함, 새 문장)
STRICT_TEMPLATES = [
    ("목적격", "삼촌이 {Nobj} 유독 귀여워했어"),
    ("비교서술", "{Nsubj} 동네에서 제일 눈에 띄어"),
    ("기본", "{N} 소식을 오랜만에 들었어"),
    ("여격", "{N}한테 숙제 다 했는지 확인해봤어"),
    ("호격", "{Nvoc}, 얼른 씻고 나와"),
]
# 가족채팅체: 같은 문형, 더 자연스러운 대화체(새 문장)
CASUAL_TEMPLATES = [
    ("목적격", "엄마가 {Nobj} 데리러 간다고 했어"),
    ("비교서술", "{Nsubj} 우리 가족 중에서 제일 부지런해"),
    ("기본", "{Nsubj} 오늘 기분 좋아 보이더라"),
    ("여격", "{N}한테 사진 좀 보내달라고 했어"),
    ("호격", "{Nvoc} 밥 먹었어?"),
]

NONPERSON_SENTENCES: dict[str, list[str]] = {
    "산그늘": ["산그늘이 마을까지 내려왔다", "산그늘 아래가 서늘했다"],
    "길섶": ["길섶에 들꽃이 피어 있었다", "길섶을 따라 걸었다"],
    "나무그늘": ["나무그늘에서 더위를 피했다", "나무그늘이 시원했다"],
    "돌계단": ["절까지 돌계단이 이어졌다", "돌계단을 천천히 올랐다"],
    "벽돌담": ["집 둘레에 벽돌담이 있었다", "벽돌담에 담쟁이가 있었다"],
    "장독대": ["마당 한쪽에 장독대가 있었다", "장독대에서 된장을 꺼냈다"],
    "마룻바닥": ["마룻바닥이 반질반질했다", "마룻바닥을 걸레로 닦았다"],
    "다락방": ["다락방에 옛 물건이 쌓여 있었다", "다락방 계단이 좁았다"],
    "빨랫줄": ["마당에 빨랫줄을 매달았다", "빨랫줄에 이불을 널었다"],
    "지붕끝": ["지붕끝에 고드름이 달렸다", "지붕끝이 비바람에 낡았다"],
    "모래알": ["모래알이 발 사이로 빠져나갔다", "모래알이 반짝였다"],
    "빗장": ["문에 빗장을 걸었다", "빗장이 녹슬어 있었다"],
    "창틀": ["창틀에 먼지가 쌓였다", "창틀을 깨끗이 닦았다"],
    "문턱": ["문턱에 걸려 넘어질 뻔했다", "문턱이 꽤 높았다"],
    "굴뚝": ["굴뚝에서 연기가 올라왔다", "굴뚝 청소를 했다"],
    "금빛": ["저녁 하늘이 금빛으로 물들었다", "금빛 들판이 펼쳐졌다"],
    "놋그릇": ["놋그릇을 반짝이게 닦았다", "할머니 댁엔 놋그릇이 많았다"],
    "질그릇": ["질그릇에 된장을 담았다", "질그릇이 투박하지만 정겨웠다"],
    "숯덩이": ["화로에 숯덩이를 넣었다", "숯덩이가 빨갛게 타올랐다"],
    "메주": ["장독에 메주를 매달았다", "메주 냄새가 집 안에 퍼졌다"],
}
NEG_BACKGROUND = [
    "오늘 세차를 했어", "주말에 손빨래를 했어", "보일러가 말썽이야", "창문 방충망을 교체했어",
    "에어컨 필터를 갈았어", "오늘 비가 많이 왔어", "택배 상자를 정리했어", "주방 수납장을 정리했어",
    "오늘 지하철이 붐볐어", "현관 센서등이 고장났어", "냄비를 새로 샀어", "욕실 수건을 바꿨어",
    "오늘 유독 졸렸어", "책상 조명을 바꿨어", "커튼을 세탁했어", "자동차 기름을 넣었어",
    "오늘 계단에서 미끄러질 뻔했어", "장바구니를 깜빡했어", "노트북이 또 느려졌어", "거실 러그를 치웠어",
]


def fill(tpl: str, n: str) -> str:
    return tpl.format(N=n, Nobj=obj(n), Nsubj=subj(n), Nvoc=voc(n))


def find_span(text: str, token: str) -> tuple[int, int]:
    idx = text.find(token)
    if idx < 0:
        raise ValueError(f"토큰을 못 찾음: {token!r} in {text!r}")
    return idx, idx + len(token)


def build_rows() -> list[dict]:
    rows = []
    rid = 0
    for style, templates in (("중의성문맥", STRICT_TEMPLATES), ("가족채팅체", CASUAL_TEMPLATES)):
        for cat, tpl in templates:
            for w in WORDS:
                text = fill(tpl, w)
                s, e = find_span(text, w)
                rid += 1
                rows.append({
                    "id": f"v11-{rid:04d}", "group": "PERSON", "style": style, "category": cat,
                    "word_type": WORD_TYPE[w], "word": w, "text": text,
                    "spans": [{"kind": "PERSON", "start": s, "end": e}], "known_names": [],
                })
    for w, sents in NONPERSON_SENTENCES.items():
        for text in sents:
            rid += 1
            rows.append({"id": f"v11-{rid:04d}", "group": "NONPERSON_WORD", "style": None, "category": "nonperson",
                         "word_type": WORD_TYPE[w], "word": w, "text": text, "spans": [], "known_names": []})
    for text in NEG_BACKGROUND:
        rid += 1
        rows.append({"id": f"v11-{rid:04d}", "group": "BACKGROUND_NEG", "style": None, "category": "background",
                     "word_type": None, "word": None, "text": text, "spans": [], "known_names": []})
    return rows


def leak_check(rows: list[dict]) -> None:
    import sys

    root = HERE.parents[6]
    blocked_words = {
        "진달래", "민들레", "억새", "솔잎", "눈송이", "빗방울", "들꽃", "메아리", "솔바람", "꽃잎",
        "윤슬", "구름", "바람", "물결", "달빛", "노래", "그림", "서리", "안개", "단풍", "여울", "반디",
        "눈꽃", "물방울", "풀잎", "꽃망울", "나뭇잎", "샘물", "봄꽃", "물거품", "조각달", "손길", "나뭇가지", "풀꽃",
        "돌탑", "조약돌", "나뭇결", "물줄기", "나룻배", "촛불", "반딧불", "바닷가", "들국화", "담쟁이",
        "초승달", "과꽃", "채송화", "나팔꽃", "분꽃", "산수유",
        "돌다리", "바위틈", "산울림", "징검다리", "물레", "처마끝", "낙엽", "구슬", "우물", "고드름",
        "장작불", "등잔불", "디딤돌", "모닥불", "풍경소리", "종소리",
        "다홍", "연두", "청록", "진홍", "열정", "평화", "성실", "용기",
        "연준", "상원", "진주", "동해", "하늘", "바다", "이슬", "노을", "새벽", "별", "여름", "가을",
        "은하", "태양", "우주", "장미", "초원", "지수", "정원", "수정", "미래", "사랑", "보라", "하나",
        "지혜", "지성", "미소", "선율", "연수", "다정",
    }
    assert not (set(WORDS) & blocked_words), f"v11 단어가 기존 단어와 겹침: {set(WORDS) & blocked_words}"

    sys.path.insert(0, str(root / "ml" / "ner" / "data"))
    import synth  # noqa: E402

    given_pools = set(synth.GIVEN1) | set(synth.GIVEN2) | set(synth.NICKS)
    assert not (set(WORDS) & given_pools), f"v11 단어가 GIVEN/NICK 풀과 겹침: {set(WORDS) & given_pools}"

    frozen_texts = set()
    rels = [
        "apps/api/app/shared/privacy/eval/baseline_v2/train.jsonl",
        "apps/api/app/shared/privacy/eval/baseline_v2/val.jsonl",
        "apps/api/app/shared/privacy/eval/baseline_v2/test.jsonl",
        "apps/api/app/shared/privacy/eval/style_v4/evalset_v4.jsonl",
        "apps/api/app/shared/privacy/eval/style_v5/locked_test_v5.jsonl",
        "apps/api/app/shared/privacy/eval/style_v6_blind/blind_v6.jsonl",
        "apps/api/app/shared/privacy/eval/style_v7_namefreq/v7_namefreq.jsonl",
        "apps/api/app/shared/privacy/eval/style_v8_independent/v8_independent.jsonl",
        "apps/api/app/shared/privacy/eval/style_ambiguity_probe/ambiguity_probe.jsonl",
        "apps/api/app/shared/privacy/eval/step2_synth_draft/synth_draft.jsonl",
        "apps/api/app/shared/privacy/eval/style_v9_locked/v9_locked.jsonl",
        "apps/api/app/shared/privacy/eval/style_v10_locked/v10_locked.jsonl",
    ]
    for rel in rels:
        p = root / rel
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                frozen_texts.add(json.loads(line)["text"].replace(" ", ""))
    for path_str in (
        r"C:\Users\<user>\venvs\ner-runs\data_r3_phrase\r3_train_extra.jsonl",
        r"C:\Users\<user>\venvs\ner-runs\data_r3_phrase\r3_val_extra.jsonl",
        r"C:\Users\<user>\venvs\ner-runs\data_r3_phrase\r3_heldout_eval.jsonl",
        r"C:\Users\<user>\venvs\ner-runs\data_r4_phrase\r4_train_extra.jsonl",
        r"C:\Users\<user>\venvs\ner-runs\data_r4_phrase\r4_val_extra.jsonl",
        r"C:\Users\<user>\venvs\ner-runs\data_r4_phrase\r4_heldout_eval.jsonl",
        r"C:\Users\<user>\venvs\ner-runs\data_r5_phrase\r5_train_extra.jsonl",
        r"C:\Users\<user>\venvs\ner-runs\data_r5_phrase\r5_val_extra.jsonl",
        r"C:\Users\<user>\venvs\ner-runs\data_r5_phrase\r5_heldout_eval.jsonl",
    ):
        from pathlib import Path as _P
        p2 = _P(path_str)
        if p2.exists():
            for line in p2.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    frozen_texts.add(json.loads(line)["text"].replace(" ", ""))

    new_texts = {r["text"].replace(" ", "") for r in rows}
    overlap = new_texts & frozen_texts
    print(f"기존 평가셋/R3·R4·R5 학습데이터와 문장 겹침: {len(overlap)}개", overlap or "(없음, OK)")
    assert not overlap, "v11 문장이 기존 데이터와 겹친다"
    dup = len(rows) - len(new_texts)
    assert dup == 0, f"v11 내부 문장 중복 {dup}건"
    print("v11 내부 중복: 0, OK")


def main() -> None:
    rows = build_rows()
    leak_check(rows)
    for r in rows:
        for g in r["spans"]:
            assert r["text"][g["start"]:g["end"]] == r["word"]
    out = HERE / "v11_locked.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    by_group = {}
    for r in rows:
        key = r["group"] if r["group"] != "PERSON" else f"PERSON_{r['style']}"
        by_group[key] = by_group.get(key, 0) + 1
    print(f"v11 총 {len(rows)}문장: {by_group}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
