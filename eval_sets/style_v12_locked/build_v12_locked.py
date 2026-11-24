"""v12 — R6 최종 잠금 독립 평가셋(2026-11-17).

목적: R6(ORG/LOC 대비 + 기본 문형 증량 + 2글자 단일형 경계 보강)이 **학습·모델 선정에 전혀 쓰지
않은 신규 20단어**(복합명사형 10 + 단일형 10)에서 55절 8개 지표 목표를 달성했는지 확인한다.
v9~v11·R3~R6(학습+holdout)·48절·draft56·GIVEN/NICK 풀·baseline/v4~v8 어디에도 없는 단어로만
구성했다(`leak_check` 자동 확인). 결과를 보고 R6을 다시 튜닝하지 않는다.

"기본" 문형의 중의성문맥 템플릿은 R6 학습이 쓴 "~소식이 궁금해서..."/"~얘기가 화제였대"와도,
v11의 "~소식을 오랜만에 들었어"와도 다른 **세 번째 새 표현**("~안부가 궁금해졌어")을 써서,
ORG/LOC 구별 보강이 특정 문장을 외운 게 아니라 일반화됐는지를 본다.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

COMPOUND_WORDS = ["부뚜막", "쪽마루", "외양간", "나막신", "책꽂이", "대들보", "들창", "토방", "뒷간", "곁방"]
SINGLE_WORDS = ["베개", "이불", "장롱", "화로", "섬돌", "삽", "호미", "쟁반", "소쿠리", "키"]
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
    ("목적격", "할머니가 {Nobj} 아주 예뻐하셨대"),
    ("비교서술", "{Nsubj} 모임에서 유난히 눈에 띄어"),
    ("기본", "{N} 안부가 궁금해졌어"),
    ("여격", "{N}한테 잘 지내는지 물어봤어"),
    ("호격", "{Nvoc}, 조심히 가"),
]
CASUAL_TEMPLATES = [
    ("목적격", "아빠가 {Nobj} 챙겨서 나갔어"),
    ("비교서술", "{Nsubj} 요즘 제일 바빠 보여"),
    ("기본", "{Nsubj} 요즘 통 소식이 없네"),
    ("여격", "{N}한테 도착했는지 물어봤어"),
    ("호격", "{Nvoc} 왔어?"),
]

NONPERSON_SENTENCES: dict[str, list[str]] = {
    "부뚜막": ["부뚜막에 솥을 걸었다", "부뚜막이 따뜻했다"],
    "쪽마루": ["쪽마루에 앉아 쉬었다", "쪽마루가 햇볕에 데워졌다"],
    "외양간": ["외양간에 소를 키웠다", "외양간 냄새가 났다"],
    "나막신": ["나막신을 신고 걸었다", "나막신이 딸깍거렸다"],
    "책꽂이": ["책꽂이에 책을 정리했다", "책꽂이가 가득 찼다"],
    "대들보": ["대들보가 집을 받치고 있었다", "대들보에 먼지가 쌓였다"],
    "들창": ["들창을 열어 바람을 쐤다", "들창으로 햇살이 들어왔다"],
    "토방": ["토방에 신발을 벗어두었다", "토방이 서늘했다"],
    "뒷간": ["뒷간이 마당 구석에 있었다", "뒷간에 다녀왔다"],
    "곁방": ["곁방에 짐을 쌓아두었다", "곁방이 비어 있었다"],
    "베개": ["베개를 베고 누웠다", "베개가 폭신했다"],
    "이불": ["이불을 덮고 잤다", "이불을 걷어찼다"],
    "장롱": ["장롱에 옷을 넣었다", "장롱 문이 뻑뻑했다"],
    "화로": ["화로에 불을 담았다", "화로가 따뜻했다"],
    "섬돌": ["섬돌 위에 신발을 벗었다", "섬돌이 미끄러웠다"],
    "삽": ["삽으로 땅을 팠다", "삽이 녹슬었다"],
    "호미": ["호미로 밭을 갈았다", "호미가 무뎌졌다"],
    "쟁반": ["쟁반에 과일을 담았다", "쟁반을 떨어뜨렸다"],
    "소쿠리": ["소쿠리에 나물을 담았다", "소쿠리가 가벼웠다"],
    "키": ["키로 곡식을 까불렀다", "키가 낡아 보였다"],
}
NEG_BACKGROUND = [
    "오늘 머리를 잘랐어", "주말에 영화를 봤어", "에어컨 리모컨을 잃어버렸어", "현관 매트를 새로 깔았어",
    "오늘 운동 안 갔어", "냉동실 정리를 했어", "휴대폰 액정이 깨졌어", "주차 딱지를 뗐어",
    "오늘따라 입맛이 없었어", "선풍기를 꺼냈어", "책상을 치웠어", "자전거 자물쇠를 바꿨어",
    "오늘 택배가 두 번 왔어", "거울을 새로 달았어", "커피를 쏟았어", "스피커가 고장났어",
    "오늘 알람을 못 들었어", "화분을 옮겼어", "이어폰을 잃어버렸어", "식기세척기를 돌렸어",
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
                    "id": f"v12-{rid:04d}", "group": "PERSON", "style": style, "category": cat,
                    "word_type": WORD_TYPE[w], "word": w, "text": text,
                    "spans": [{"kind": "PERSON", "start": s, "end": e}], "known_names": [],
                })
    for w, sents in NONPERSON_SENTENCES.items():
        for text in sents:
            rid += 1
            rows.append({"id": f"v12-{rid:04d}", "group": "NONPERSON_WORD", "style": None, "category": "nonperson",
                         "word_type": WORD_TYPE[w], "word": w, "text": text, "spans": [], "known_names": []})
    for text in NEG_BACKGROUND:
        rid += 1
        rows.append({"id": f"v12-{rid:04d}", "group": "BACKGROUND_NEG", "style": None, "category": "background",
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
        "산그늘", "길섶", "나무그늘", "돌계단", "벽돌담", "장독대", "마룻바닥", "다락방", "빨랫줄", "지붕끝",
        "모래알", "빗장", "창틀", "문턱", "굴뚝", "금빛", "놋그릇", "질그릇", "숯덩이", "메주",
        "서랍", "벼루", "바늘", "연못", "찻잔", "먹물", "빗물", "눈발", "마당", "지게", "뒤뜰", "앞뜰", "화덕", "절구",
        "연준", "상원", "진주", "동해", "하늘", "바다", "이슬", "노을", "새벽", "별", "여름", "가을",
        "은하", "태양", "우주", "장미", "초원", "지수", "정원", "수정", "미래", "사랑", "보라", "하나",
        "지혜", "지성", "미소", "선율", "연수", "다정",
    }
    assert not (set(WORDS) & blocked_words), f"v12 단어가 기존 단어와 겹침: {set(WORDS) & blocked_words}"

    sys.path.insert(0, str(root / "ml" / "ner" / "data"))
    import synth  # noqa: E402

    given_pools = set(synth.GIVEN1) | set(synth.GIVEN2) | set(synth.NICKS)
    assert not (set(WORDS) & given_pools), f"v12 단어가 GIVEN/NICK 풀과 겹침: {set(WORDS) & given_pools}"

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
    ):
        from pathlib import Path as _P
        p2 = _P(path_str)
        if p2.exists():
            for line in p2.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    frozen_texts.add(json.loads(line)["text"].replace(" ", ""))

    new_texts = {r["text"].replace(" ", "") for r in rows}
    overlap = new_texts & frozen_texts
    print(f"기존 평가셋/R3~R6 학습데이터와 문장 겹침: {len(overlap)}개", overlap or "(없음, OK)")
    assert not overlap, "v12 문장이 기존 데이터와 겹친다"
    dup = len(rows) - len(new_texts)
    assert dup == 0, f"v12 내부 문장 중복 {dup}건"
    print("v12 내부 중복: 0, OK")


def main() -> None:
    rows = build_rows()
    leak_check(rows)
    for r in rows:
        for g in r["spans"]:
            assert r["text"][g["start"]:g["end"]] == r["word"]
    out = HERE / "v12_locked.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    by_group = {}
    for r in rows:
        key = r["group"] if r["group"] != "PERSON" else f"PERSON_{r['style']}"
        by_group[key] = by_group.get(key, 0) + 1
    print(f"v12 총 {len(rows)}문장: {by_group}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
