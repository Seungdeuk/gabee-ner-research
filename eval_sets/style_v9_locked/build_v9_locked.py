"""v9 — 최종 잠금 독립 평가셋(우선순위 2 · 2단계, 사용자 지시 2026-11-16, 조건 7).

목적: R3 문형 보강(목적격·여격·호격·비교서술)이 **학습에 전혀 쓰지 않은 새 충돌 단어**에도
일반화되는지 딱 한 번 확인한다. 이 결과를 보고 모델을 다시 튜닝하지 않는다(사용자 지시).

단어 10개(진달래·민들레·억새·솔잎·눈송이·빗방울·들꽃·메아리·솔바람·꽃잎)는 `build_r3_phrase_
collision_data.py` 작성 시 같은 방식으로 뽑은 22개 "깨끗한" 후보 중 R3 학습(8)·holdout(4)에 안 쓴
나머지 10개다 — 즉 R3 어떤 데이터에도, 48절 진단셋·draft56·GIVEN/NICK 풀·v4~v8 평가셋 어디에도 없다.
문장 틀은 R3 학습/holdout 틀과도 전부 다르게 새로 썼다(틀 자체 중복도 없음).

구성:
- A: 새 충돌단어 10개 x PERSON 신규 문형 4개(R3 학습 문형과 다르게 새로 씀) = 40문장
- A-neg: 같은 10개 단어의 비PERSON 문장(단어마다 2개, 48절/R3와 다른 새 문장) = 20문장
- B: 같은 10개 단어 x 기존에 이미 잘 통하던 "기본 문형"(주어+서술, T1 계열) 2개 = 20문장(회귀 확인용)
- D: 충돌 단어와 무관한 일반 배경 비인명 문장 20개(배경 오탐률 측정용)

안전 조건: `ner.py`·모델 파일·DB·설정 미수정, 외부 LLM 미호출. 이 평가셋은 **한 번만** 쓴다(결과를
보고 추가 튜닝하지 않는다).
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

WORDS = ["진달래", "민들레", "억새", "솔잎", "눈송이", "빗방울", "들꽃", "메아리", "솔바람", "꽃잎"]


def has_batchim(ch: str) -> bool:
    return (ord(ch) - 0xAC00) % 28 != 0


def obj(w: str) -> str:
    return w + ("을" if has_batchim(w[-1]) else "를")


def subj(w: str) -> str:
    return w + ("이" if has_batchim(w[-1]) else "가")


def voc(w: str) -> str:
    return w + ("아" if has_batchim(w[-1]) else "야")


PERSON_TEMPLATES_NEW = [
    ("목적격", "할머니가 {Nobj} 많이 예뻐하셨어"),
    ("여격", "{N}한테 급식 맛있었는지 물어봤어"),
    ("호격", "{Nvoc}, 이리 와서 앉아봐"),
    ("비교서술", "{Nsubj} 우리 학원에서 제일 성실해"),
]
PERSON_TEMPLATES_BASIC = [
    ("기본_주어서술", "{Nsubj} 오늘 친구 생일파티 다녀왔대"),
    ("기본_단순언급", "{N} 사진을 가족방에 올렸어"),
]
NONPERSON_SENTENCES: dict[str, list[str]] = {
    "진달래": ["산기슭에 진달래가 활짝 피었다", "진달래꽃으로 화전을 만들었다"],
    "민들레": ["길가에 민들레가 피어 있었다", "민들레 홀씨가 바람에 날렸다"],
    "억새": ["가을 들판에 억새가 흔들렸다", "억새밭 사이로 길이 나 있었다"],
    "솔잎": ["솔잎 냄새가 은은하게 퍼졌다", "바닥에 솔잎이 수북이 쌓였다"],
    "눈송이": ["창밖으로 눈송이가 흩날렸다", "눈송이가 소복이 쌓였다"],
    "빗방울": ["창문에 빗방울이 맺혔다", "빗방울 떨어지는 소리가 들렸다"],
    "들꽃": ["들판에 들꽃이 가득 피었다", "들꽃을 꺾어 꽃다발을 만들었다"],
    "메아리": ["산에서 메아리가 울려 퍼졌다", "메아리 소리가 점점 작아졌다"],
    "솔바람": ["소나무 숲에 솔바람이 불었다", "솔바람 소리가 시원했다"],
    "꽃잎": ["바람에 꽃잎이 흩날렸다", "꽃잎이 바닥에 떨어졌다"],
}
NEG_BACKGROUND = [
    "오늘 저녁 메뉴는 카레로 정했어", "주말에 캠핑 가기로 했어", "세탁기 소리가 너무 시끄러워",
    "냉장고에 반찬 꺼내놨어", "내일 병원 예약 시간 확인해줘", "택배가 문 앞에 왔다고 문자왔어",
    "오늘따라 길이 많이 막혔어", "에어컨 필터 청소할 때 됐어", "마트에서 장 보고 들어갈게",
    "거실 조명이 하나 나갔어", "분리수거는 내가 할게", "오늘 운동 다녀왔어",
    "휴대폰 배터리가 금방 닳네", "창고 정리 좀 해야겠어", "내일 일찍 일어나야 해",
    "커피머신 고장난 것 같아", "주차장 자리가 하나도 없었어", "보일러 온도 좀 낮춰줘",
    "책상 서랍 정리했어", "빨래 널어놨으니까 걷어줘",
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
    for cat, tpl in PERSON_TEMPLATES_NEW:
        for w in WORDS:
            text = fill(tpl, w)
            s, e = find_span(text, w)
            rid += 1
            rows.append({"id": f"v9-{rid:04d}", "group": "A_new_phrase", "category": cat, "word": w,
                         "text": text, "spans": [{"kind": "PERSON", "start": s, "end": e}], "known_names": []})
    for w, sents in NONPERSON_SENTENCES.items():
        for text in sents:
            rid += 1
            rows.append({"id": f"v9-{rid:04d}", "group": "A_neg", "category": "nonperson", "word": w,
                         "text": text, "spans": [], "known_names": []})
    for cat, tpl in PERSON_TEMPLATES_BASIC:
        for w in WORDS:
            text = fill(tpl, w)
            s, e = find_span(text, w)
            rid += 1
            rows.append({"id": f"v9-{rid:04d}", "group": "B_basic_phrase", "category": cat, "word": w,
                         "text": text, "spans": [{"kind": "PERSON", "start": s, "end": e}], "known_names": []})
    for text in NEG_BACKGROUND:
        rid += 1
        rows.append({"id": f"v9-{rid:04d}", "group": "D_background_neg", "category": "background", "word": None,
                     "text": text, "spans": [], "known_names": []})
    return rows


def leak_check(rows: list[dict]) -> None:
    import sys

    root = HERE.parents[6]
    r3_words = {"윤슬", "구름", "바람", "물결", "달빛", "노래", "그림", "서리", "안개", "단풍", "여울", "반디"}
    probe_words = {
        "연준", "상원", "진주", "동해", "하늘", "바다", "이슬", "노을", "새벽", "별", "여름", "가을",
        "은하", "태양", "우주", "장미", "초원", "지수", "정원", "수정", "미래", "사랑", "보라", "하나",
        "지혜", "지성", "미소", "선율", "연수", "다정",
    }
    assert not (set(WORDS) & r3_words), f"v9 단어가 R3 학습/holdout 단어와 겹침: {set(WORDS) & r3_words}"
    assert not (set(WORDS) & probe_words), f"v9 단어가 48절 진단셋과 겹침: {set(WORDS) & probe_words}"

    sys.path.insert(0, str(root / "ml" / "ner" / "data"))
    import synth  # noqa: E402

    given_pools = set(synth.GIVEN1) | set(synth.GIVEN2) | set(synth.NICKS)
    assert not (set(WORDS) & given_pools), f"v9 단어가 GIVEN/NICK 풀과 겹침: {set(WORDS) & given_pools}"

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
    ]
    for rel in rels:
        p = root / rel
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                frozen_texts.add(json.loads(line)["text"].replace(" ", ""))
    data_dir = Path(r"C:\Users\<user>\venvs\ner-runs\data_r3_phrase")
    for name in ("r3_train_extra.jsonl", "r3_val_extra.jsonl", "r3_heldout_eval.jsonl"):
        p = data_dir / name
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    frozen_texts.add(json.loads(line)["text"].replace(" ", ""))

    new_texts = {r["text"].replace(" ", "") for r in rows}
    overlap = new_texts & frozen_texts
    print(f"기존 평가셋/R3 학습데이터와 문장 겹침: {len(overlap)}개", overlap or "(없음, OK)")
    assert not overlap, "v9 문장이 기존 데이터와 겹친다"
    dup = len(rows) - len(new_texts)
    assert dup == 0, f"v9 내부 문장 중복 {dup}건"
    print("v9 내부 중복: 0, OK")


def main() -> None:
    rows = build_rows()
    leak_check(rows)
    for r in rows:
        for g in r["spans"]:
            assert r["text"][g["start"]:g["end"]] == r["word"]
    out = HERE / "v9_locked.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    by_group = {}
    for r in rows:
        by_group[r["group"]] = by_group.get(r["group"], 0) + 1
    print(f"v9 총 {len(rows)}문장: {by_group}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
