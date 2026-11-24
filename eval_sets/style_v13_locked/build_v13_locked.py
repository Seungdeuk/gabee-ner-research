"""v13 — R6+B(최종 후보) 최종 잠금 독립 평가셋(2026-11-17).

목적: "R6 모델 가중치(그대로) + 실험 B 병합 완화 규칙"을 **학습·모델 선정에 전혀 쓰지 않은 신규
20단어**(복합명사형 10 + 단일형 10)에서 55절 8개 표준 지표로 **딱 한 번** 검증한다. v9~v12·
R3~R7a(학습+holdout)·48절·draft56·GIVEN/NICK 풀·baseline/v4~v8 어디에도 없는 단어로만 구성했다
(`leak_check` 자동 확인). 결과를 보고 재튜닝하지 않는다. R7(실험 A 재학습)은 보류됐으므로 이번
v13은 R6+B만 검증한다(실험 A는 대상 아님).

문형·문체 구성은 v11·v12와 동일한 틀(5문형 x 2문체)이되, 문장 표현은 전부 새로 썼다.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

COMPOUND_WORDS = ["빨래집게", "다락문", "부엌문", "장작간", "처마밑", "기와지붕", "외양채", "토담", "장독뚜껑", "바지랑대"]
SINGLE_WORDS = ["삼태기", "쇠스랑", "삼베", "누룽지", "엿가락", "다식", "약과", "떡살", "갈퀴", "맷돌"]
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


STRICT_TEMPLATES = [
    ("목적격", "큰아버지가 {Nobj} 많이 예뻐하셨대"),
    ("비교서술", "{Nsubj} 동네에서 유독 성실해"),
    ("기본", "{N}은 요즘 뜸하네"),
    ("여격", "{N}한테 오랜만에 연락해봤어"),
    ("호격", "{Nvoc}, 나랑 같이 가자"),
]
CASUAL_TEMPLATES = [
    ("목적격", "할아버지가 {Nobj} 유난히 챙기셔"),
    ("비교서술", "{Nsubj} 요즘 보면 제일 성실해"),
    ("기본", "{Nsubj} 요즘 뭐 하고 지내나 몰라"),
    ("여격", "{N}한테 밥 먹었냐고 물어봤어"),
    ("호격", "{Nvoc} 자니?"),
]

NONPERSON_SENTENCES: dict[str, list[str]] = {
    "빨래집게": ["빨래집게로 옷을 널었다", "빨래집게가 부러졌다"],
    "다락문": ["다락문을 열었다", "다락문이 삐걱거렸다"],
    "부엌문": ["부엌문을 닫았다", "부엌문 앞에 신발이 있었다"],
    "장작간": ["장작간에 나무를 쌓아두었다", "장작간이 비어 있었다"],
    "처마밑": ["처마밑에서 비를 피했다", "처마밑에 제비집이 있었다"],
    "기와지붕": ["기와지붕이 비에 젖었다", "기와지붕 위로 햇살이 비쳤다"],
    "외양채": ["외양채에 소를 들였다", "외양채가 낡았다"],
    "토담": ["토담을 따라 걸었다", "토담이 허물어졌다"],
    "장독뚜껑": ["장독뚜껑을 열었다", "장독뚜껑에 먼지가 쌓였다"],
    "바지랑대": ["바지랑대로 빨랫줄을 받쳤다", "바지랑대가 기울었다"],
    "삼태기": ["삼태기에 흙을 담았다", "삼태기가 낡았다"],
    "쇠스랑": ["쇠스랑으로 땅을 골랐다", "쇠스랑이 녹슬었다"],
    "삼베": ["삼베로 옷을 지었다", "삼베 이불이 시원했다"],
    "누룽지": ["누룽지를 끓여 먹었다", "누룽지 냄새가 고소했다"],
    "엿가락": ["엿가락을 늘렸다", "엿가락이 달라붙었다"],
    "다식": ["다식을 접시에 담았다", "다식이 알록달록했다"],
    "약과": ["약과를 맛있게 먹었다", "약과가 바삭했다"],
    "떡살": ["떡살로 무늬를 찍었다", "떡살이 오래됐다"],
    "갈퀴": ["갈퀴로 낙엽을 모았다", "갈퀴가 휘어졌다"],
    "맷돌": ["맷돌로 콩을 갈았다", "맷돌이 무거웠다"],
}
NEG_BACKGROUND = [
    "오늘 안경을 새로 맞췄어", "주말에 집들이를 했어", "세탁기 소리가 이상해", "현관 신발장을 정리했어",
    "오늘 미세먼지가 괜찮았어", "자전거 타이어가 펑크났어", "전자레인지가 고장났어", "주말에 등산화를 샀어",
    "오늘 버스를 놓쳤어", "거실 커튼을 바꿨어", "책상 서랍을 정리했어", "오늘 늦잠을 잤어",
    "주차장이 꽉 찼었어", "이불을 세탁 맡겼어", "휴대폰 케이스를 바꿨어", "오늘따라 날씨가 좋았어",
    "냉장고 청소를 했어", "창문을 새로 닦았어", "식탁보를 새로 샀어", "오늘 장을 많이 봤어",
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
                    "id": f"v13-{rid:04d}", "group": "PERSON", "style": style, "category": cat,
                    "word_type": WORD_TYPE[w], "word": w, "text": text,
                    "spans": [{"kind": "PERSON", "start": s, "end": e}], "known_names": [],
                })
    for w, sents in NONPERSON_SENTENCES.items():
        for text in sents:
            rid += 1
            rows.append({"id": f"v13-{rid:04d}", "group": "NONPERSON_WORD", "style": None, "category": "nonperson",
                         "word_type": WORD_TYPE[w], "word": w, "text": text, "spans": [], "known_names": []})
    for text in NEG_BACKGROUND:
        rid += 1
        rows.append({"id": f"v13-{rid:04d}", "group": "BACKGROUND_NEG", "style": None, "category": "background",
                     "word_type": None, "word": None, "text": text, "spans": [], "known_names": []})
    return rows


ALL_PRIOR_WORDS = {
    "진달래", "민들레", "억새", "솔잎", "눈송이", "빗방울", "들꽃", "메아리", "솔바람", "꽃잎",
    "윤슬", "구름", "바람", "물결", "달빛", "노래", "그림", "서리", "안개", "단풍", "여울", "반디",
    "눈꽃", "물방울", "풀잎", "꽃망울", "나뭇잎", "샘물", "봄꽃", "물거품", "조각달", "손길", "나뭇가지", "풀꽃",
    "돌탑", "조약돌", "나뭇결", "물줄기", "나룻배", "촛불", "반딧불", "바닷가", "들국화", "담쟁이",
    "초승달", "과꽃", "채송화", "나팔꽃", "분꽃", "산수유",
    "돌다리", "바위틈", "산울림", "징검다리", "물레", "처마끝", "낙엽", "구슬", "우물", "고드름",
    "장작불", "등잔불", "디딤돌", "모닥불", "풍경소리", "종소리",
    "다홍", "연두", "청록", "진홍", "열정", "평화", "성실", "용기",
    "산그늘", "길섶", "나무그늘", "돌계단", "벽돌담", "장독대", "마룻바닥", "다락방", "빨랫줄", "지붕끝",
    "모래알", "빗장", "창틀", "문턱", "굴뚝", "금빛", "놋그릇", "질그릇", "숯덩이", "메주",
    "서랍", "벼루", "바늘", "연못", "찻잔", "먹물", "빗물", "눈발", "마당", "지게", "뒤뜰", "앞뜰", "화덕", "절구",
    "부뚜막", "쪽마루", "외양간", "나막신", "책꽂이", "대들보", "들창", "토방", "뒷간", "곁방",
    "베개", "이불", "장롱", "화로", "섬돌", "삽", "호미", "쟁반", "소쿠리", "키",
    "연탄불", "아궁이", "짚더미", "가마솥", "다듬이", "문고리", "빗자루", "두레박", "디딜방아", "베틀",
    "자갈밭", "풀숲", "등불", "촛대",
    "연준", "상원", "진주", "동해", "하늘", "바다", "이슬", "노을", "새벽", "별", "여름", "가을",
    "은하", "태양", "우주", "장미", "초원", "지수", "정원", "수정", "미래", "사랑", "보라", "하나",
    "지혜", "지성", "미소", "선율", "연수", "다정",
}


def leak_check(rows: list[dict]) -> None:
    import sys

    root = HERE.parents[6]
    assert not (set(WORDS) & ALL_PRIOR_WORDS), f"v13 단어가 기존 단어와 겹침: {set(WORDS) & ALL_PRIOR_WORDS}"

    sys.path.insert(0, str(root / "ml" / "ner" / "data"))
    import synth  # noqa: E402

    given_pools = set(synth.GIVEN1) | set(synth.GIVEN2) | set(synth.NICKS)
    assert not (set(WORDS) & given_pools), f"v13 단어가 GIVEN/NICK 풀과 겹침: {set(WORDS) & given_pools}"

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
        "apps/api/app/shared/privacy/eval/style_v11_locked/v11_locked.jsonl",
        "apps/api/app/shared/privacy/eval/style_v12_locked/v12_locked.jsonl",
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
        r"C:\Users\<user>\venvs\ner-runs\data_r6_phrase\r6_train_extra.jsonl",
        r"C:\Users\<user>\venvs\ner-runs\data_r6_phrase\r6_val_extra.jsonl",
        r"C:\Users\<user>\venvs\ner-runs\data_r6_phrase\r6_heldout_eval.jsonl",
        r"C:\Users\<user>\venvs\ner-runs\data_r7a_phrase\r7a_train_extra.jsonl",
        r"C:\Users\<user>\venvs\ner-runs\data_r7a_phrase\r7a_val_extra.jsonl",
        r"C:\Users\<user>\venvs\ner-runs\data_r7a_phrase\r7a_heldout_eval.jsonl",
    ):
        from pathlib import Path as _P
        p2 = _P(path_str)
        if p2.exists():
            for line in p2.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    frozen_texts.add(json.loads(line)["text"].replace(" ", ""))

    new_texts = {r["text"].replace(" ", "") for r in rows}
    overlap = new_texts & frozen_texts
    print(f"기존 평가셋/R3~R7a 학습데이터와 문장 겹침: {len(overlap)}개", overlap or "(없음, OK)")
    assert not overlap, "v13 문장이 기존 데이터와 겹친다"
    dup = len(rows) - len(new_texts)
    assert dup == 0, f"v13 내부 문장 중복 {dup}건"
    print("v13 내부 중복: 0, OK")


def main() -> None:
    rows = build_rows()
    leak_check(rows)
    for r in rows:
        for g in r["spans"]:
            assert r["text"][g["start"]:g["end"]] == r["word"]
    out = HERE / "v13_locked.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    by_group = {}
    for r in rows:
        key = r["group"] if r["group"] != "PERSON" else f"PERSON_{r['style']}"
        by_group[key] = by_group.get(key, 0) + 1
    print(f"v13 총 {len(rows)}문장: {by_group}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
