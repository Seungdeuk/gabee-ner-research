"""v10 — 최종 잠금 독립 평가셋 2(R4 검증용, 2026-11-16).

목적: R4(복합명사형 x 목적격 보강)가 **학습·모델 선정에 전혀 쓰지 않은 처음 보는 이름·문장**에서
이름 전체 노출률을 5% 미만으로 낮췄는지 딱 한 번 확인한다. v9에서 놓친 이름을 그대로 재사용하지
않는다(사용자 지시) — v9·R3·R4 학습/holdout·48절·draft56·GIVEN/NICK 풀·v4~v8 어디에도 없는
신규 단어 16개로만 구성했다(`leak_check`가 자동 확인).

52절 제안대로 목적격·비교서술 비중을 올리고(각 약 29%) 여격·호격·기본은 회귀 확인용으로 유지했다
(각 14~14%). 52절이 밝힌 "복합명사형 vs 단일/opaque형" 구분도 절반씩 섞어서, R4가 복합명사형에
집중 보강했을 때 단일형 쪽을 희생시키지 않는지도 같이 본다.

**이 평가셋은 한 번만 쓴다.** 결과를 보고 R4 모델/데이터를 추가로 조정하지 않는다.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 복합명사형(8) + 단일/opaque형(8)
COMPOUND_WORDS = ["돌탑", "조약돌", "나뭇결", "물줄기", "나룻배", "촛불", "반딧불", "바닷가"]
SINGLE_WORDS = ["들국화", "담쟁이", "초승달", "과꽃", "채송화", "나팔꽃", "분꽃", "산수유"]
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


PERSON_TEMPLATES = [
    ("목적격", "이웃들이 {Nobj} 많이 아껴줬대"),
    ("목적격", "담임선생님이 {Nobj} 특별히 챙겨주셨어"),
    ("비교서술", "{Nsubj} 동아리에서 가장 열심이야"),
    ("비교서술", "{Nsubj} 이번 대회에서 제일 잘했대"),
    ("여격", "{N}한테 주말 계획 물어봤어"),
    ("호격", "{Nvoc}, 간식 먹고 갈래?"),
    ("기본", "{Nsubj} 오늘 늦게까지 놀다 왔대"),
]

NONPERSON_SENTENCES: dict[str, list[str]] = {
    "돌탑": ["산길에 돌탑이 쌓여 있었다", "돌탑을 하나씩 쌓아 올렸다"],
    "조약돌": ["냇가에서 조약돌을 주웠다", "조약돌이 반질반질했다"],
    "나뭇결": ["책상에 나뭇결이 선명했다", "나뭇결을 따라 무늬가 났다"],
    "물줄기": ["폭포에서 물줄기가 쏟아졌다", "가느다란 물줄기가 흘렀다"],
    "나룻배": ["강가에 나룻배가 매여 있었다", "나룻배를 타고 강을 건넜다"],
    "촛불": ["케이크에 촛불을 켰다", "촛불이 바람에 흔들렸다"],
    "반딧불": ["여름밤 반딧불이 날아다녔다", "반딧불 빛이 은은했다"],
    "바닷가": ["바닷가를 따라 산책했다", "바닷가에 파도가 밀려왔다"],
    "들국화": ["가을 들판에 들국화가 피었다", "들국화 향이 은은했다"],
    "담쟁이": ["담벼락에 담쟁이가 뒤덮였다", "담쟁이 잎이 붉게 물들었다"],
    "초승달": ["밤하늘에 초승달이 떠 있었다", "초승달이 가늘게 빛났다"],
    "과꽃": ["화단에 과꽃이 활짝 피었다", "과꽃 씨앗을 심었다"],
    "채송화": ["마당에 채송화가 피어 있었다", "채송화 색이 알록달록했다"],
    "나팔꽃": ["담장에 나팔꽃이 피었다", "나팔꽃이 아침에만 핀다"],
    "분꽃": ["화단에 분꽃이 피었다", "분꽃 씨앗이 까맣다"],
    "산수유": ["마을에 산수유가 노랗게 피었다", "산수유 열매를 땄다"],
}
NEG_BACKGROUND = [
    "오늘 요가 수업 다녀왔어", "주말에 집 대청소 했어", "냉장고 문이 잘 안 닫혀", "강아지 산책 다녀왔어",
    "커튼을 새로 달았어", "이번 주말에 등산 가기로 했어", "텔레비전 리모컨이 안 돼", "식탁 의자를 하나 더 샀어",
    "오늘 버스가 많이 늦었어", "화분에 물을 줬어", "안경을 새로 맞췄어", "신발장 정리를 했어",
    "오늘 미세먼지가 심했어", "자전거 바퀴에 바람을 넣었어", "거울을 하나 샀어", "책장을 새로 짰어",
    "오늘 장보기를 늦게 했어", "전등을 LED로 바꿨어", "주방 세제가 떨어졌어", "현관문 비밀번호를 바꿨어",
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
    for cat, tpl in PERSON_TEMPLATES:
        for w in WORDS:
            text = fill(tpl, w)
            s, e = find_span(text, w)
            rid += 1
            rows.append({"id": f"v10-{rid:04d}", "group": "PERSON", "category": cat, "word_type": WORD_TYPE[w],
                         "word": w, "text": text, "spans": [{"kind": "PERSON", "start": s, "end": e}], "known_names": []})
    for w, sents in NONPERSON_SENTENCES.items():
        for text in sents:
            rid += 1
            rows.append({"id": f"v10-{rid:04d}", "group": "NONPERSON_WORD", "category": "nonperson", "word_type": WORD_TYPE[w],
                         "word": w, "text": text, "spans": [], "known_names": []})
    for text in NEG_BACKGROUND:
        rid += 1
        rows.append({"id": f"v10-{rid:04d}", "group": "BACKGROUND_NEG", "category": "background", "word_type": None,
                     "word": None, "text": text, "spans": [], "known_names": []})
    return rows


def leak_check(rows: list[dict]) -> None:
    import sys

    root = HERE.parents[6]
    blocked_words = {
        "진달래", "민들레", "억새", "솔잎", "눈송이", "빗방울", "들꽃", "메아리", "솔바람", "꽃잎",
        "윤슬", "구름", "바람", "물결", "달빛", "노래", "그림", "서리", "안개", "단풍", "여울", "반디",
        "눈꽃", "물방울", "풀잎", "꽃망울", "나뭇잎", "샘물", "봄꽃", "물거품", "조각달", "손길", "나뭇가지", "풀꽃",
        "연준", "상원", "진주", "동해", "하늘", "바다", "이슬", "노을", "새벽", "별", "여름", "가을",
        "은하", "태양", "우주", "장미", "초원", "지수", "정원", "수정", "미래", "사랑", "보라", "하나",
        "지혜", "지성", "미소", "선율", "연수", "다정",
    }
    assert not (set(WORDS) & blocked_words), f"v10 단어가 기존 단어와 겹침: {set(WORDS) & blocked_words}"

    sys.path.insert(0, str(root / "ml" / "ner" / "data"))
    import synth  # noqa: E402

    given_pools = set(synth.GIVEN1) | set(synth.GIVEN2) | set(synth.NICKS)
    assert not (set(WORDS) & given_pools), f"v10 단어가 GIVEN/NICK 풀과 겹침: {set(WORDS) & given_pools}"

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
    ):
        from pathlib import Path as _P
        p2 = _P(path_str)
        if p2.exists():
            for line in p2.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    frozen_texts.add(json.loads(line)["text"].replace(" ", ""))

    new_texts = {r["text"].replace(" ", "") for r in rows}
    overlap = new_texts & frozen_texts
    print(f"기존 평가셋/R3·R4 학습데이터와 문장 겹침: {len(overlap)}개", overlap or "(없음, OK)")
    assert not overlap, "v10 문장이 기존 데이터와 겹친다"
    dup = len(rows) - len(new_texts)
    assert dup == 0, f"v10 내부 문장 중복 {dup}건"
    print("v10 내부 중복: 0, OK")


def main() -> None:
    rows = build_rows()
    leak_check(rows)
    for r in rows:
        for g in r["spans"]:
            assert r["text"][g["start"]:g["end"]] == r["word"]
    out = HERE / "v10_locked.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    by_group = {}
    for r in rows:
        by_group[r["group"]] = by_group.get(r["group"], 0) + 1
    print(f"v10 총 {len(rows)}문장: {by_group}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
