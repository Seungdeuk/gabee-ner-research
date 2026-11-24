"""실험 A — PERSON/PERSON_ALIAS 라벨 일관성 보강(60절에서 확인한 "복합어 음절 간 라벨 흔들림"을
학습 데이터로 완화). R6(base+R3+R4+R5+R6) 위에 이어붙인다. `ner.py`/`merge_ner`는 전혀 안 건드린다
— 이건 모델 쪽 개선(실험 A)이고, 병합 쪽 개선(실험 B)은 별도 평가 스크립트에서 ner.py 를 고치지
않고 비교한다.

60절에서 "책꽂이·토방·부뚜막"처럼 3음절 내외 복합어가 음절마다 PERSON/PERSON_ALIAS를 오가며
쪼개지는 걸 확인했다(둘 다 합성 데이터가 한 span에 하나의 라벨만 주므로, 학습 자체가 모순된
신호를 준 적은 없다 — 못 본 단어로 일반화가 덜 된 것으로 보인다, 추정). 이 스크립트는 **새
3음절 위주 복합어 10개**를 5개 문형에 고르게 노출시켜 "여러 음절이 전부 하나의 PERSON"이라는
신호를 넓힌다. holdout 4개는 학습에 전혀 안 씀.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

TRAIN_WORDS = ["연탄불", "아궁이", "짚더미", "가마솥", "다듬이", "문고리", "빗자루", "두레박", "디딜방아", "베틀"]
HOLDOUT_WORDS = ["자갈밭", "풀숲", "등불", "촛대"]


def has_batchim(ch: str) -> bool:
    return (ord(ch) - 0xAC00) % 28 != 0


def obj(w: str) -> str:
    return w + ("을" if has_batchim(w[-1]) else "를")


def subj(w: str) -> str:
    return w + ("이" if has_batchim(w[-1]) else "가")


def voc(w: str) -> str:
    return w + ("아" if has_batchim(w[-1]) else "야")


PERSON_TEMPLATES = [
    ("목적격", "train", "식구들이 {Nobj} 오래 아꼈대"),
    ("목적격", "val", "다 같이 {Nobj} 소중히 여겼어"),
    ("비교서술", "train", "{Nsubj} 집안에서 제일 야물어"),
    ("비교서술", "val", "{Nsubj} 늘 믿음이 간다고 했어"),
    ("기본", "train", "{N} 얘기는 언제나 반갑더라"),
    ("기본", "val", "{N} 생각이 문득 났어"),
    ("여격", "train", "{N}한테 별일 없는지 물어봤어"),
    ("여격", "val", "{N}한테 요즘 바쁜지 여쭤봤어"),
    ("호격", "train", "{Nvoc}, 여기로 와봐"),
    ("호격", "val", "{Nvoc}, 잠깐 이리 와"),
]

NONPERSON_TRAIN_WORDS: dict[str, list[tuple[str, str]]] = {
    "연탄불": [("연탄불을 피웠다", "train"), ("연탄불이 발갛게 달아올랐다", "train"), ("연탄불 냄새가 났다", "val")],
    "아궁이": [("아궁이에 불을 지폈다", "train"), ("아궁이 앞에서 밥을 지었다", "train"), ("아궁이가 따뜻했다", "val")],
    "짚더미": [("마당에 짚더미가 쌓여 있었다", "train"), ("짚더미 위에 앉았다", "train"), ("짚더미가 축축했다", "val")],
    "가마솥": [("가마솥에 밥을 지었다", "train"), ("가마솥이 펄펄 끓었다", "train"), ("가마솥 뚜껑을 열었다", "val")],
    "다듬이": [("다듬이로 옷을 다졌다", "train"), ("다듬이 소리가 들렸다", "train"), ("다듬이가 낡았다", "val")],
    "문고리": [("문고리를 잡고 열었다", "train"), ("문고리가 차가웠다", "train"), ("문고리가 헐거워졌다", "val")],
    "빗자루": [("빗자루로 마당을 쓸었다", "train"), ("빗자루가 닳았다", "train"), ("빗자루를 세워두었다", "val")],
    "두레박": [("두레박으로 물을 퍼올렸다", "train"), ("두레박이 우물에 빠졌다", "train"), ("두레박줄이 끊어졌다", "val")],
    "디딜방아": [("디딜방아로 곡식을 찧었다", "train"), ("디딜방아가 삐걱거렸다", "train"), ("디딜방아 소리가 났다", "val")],
    "베틀": [("베틀로 천을 짰다", "train"), ("베틀이 덜컹거렸다", "train"), ("베틀 앞에 앉았다", "val")],
}

HOLDOUT_TEMPLATES = [
    "{N} 얘기 들으니 반가웠어",
    "{Nvoc}, 조심해서 와",
]
NONPERSON_HOLDOUT_WORDS: dict[str, list[str]] = {
    "자갈밭": ["자갈밭을 밟으며 걸었다", "자갈밭이 울퉁불퉁했다"],
    "풀숲": ["풀숲에서 벌레 소리가 났다", "풀숲을 헤치고 지나갔다"],
    "등불": ["등불을 켜고 길을 나섰다", "등불이 희미하게 흔들렸다"],
    "촛대": ["촛대에 초를 꽂았다", "촛대가 은은하게 빛났다"],
}


def fill(tpl: str, n: str) -> str:
    return tpl.format(N=n, Nobj=obj(n), Nsubj=subj(n), Nvoc=voc(n))


def find_span(text: str, token: str) -> tuple[int, int]:
    idx = text.find(token)
    if idx < 0:
        raise ValueError(f"토큰을 못 찾음: {token!r} in {text!r}")
    return idx, idx + len(token)


def make_span_row(rid: str, text: str, name: str, category: str, split: str) -> dict:
    s, e = find_span(text, name)
    return {
        "id": rid, "text": text, "spans": [{"label": "PERSON", "start": s, "end": e}],
        "source": "synth", "template_id": f"r7a-{category}", "category": category, "split": split, "augment": True,
    }


def make_neg_row(rid: str, text: str, category: str, split: str) -> dict:
    return {
        "id": rid, "text": text, "spans": [],
        "source": "synth", "template_id": f"r7a-{category}-neg", "category": category, "split": split, "augment": True,
    }


def make_eval_row(rid: str, text: str, name: str | None, category: str) -> dict:
    spans = []
    if name:
        s, e = find_span(text, name)
        spans.append({"kind": "PERSON", "start": s, "end": e})
    return {"id": rid, "category": category, "text": text, "spans": spans, "known_names": [], "template_id": category}


FROZEN_JSONLS = [
    "apps/api/app/shared/privacy/eval/baseline_v2/train.jsonl",
    "apps/api/app/shared/privacy/eval/baseline_v2/val.jsonl",
    "apps/api/app/shared/privacy/eval/baseline_v2/test.jsonl",
    "apps/api/app/shared/privacy/eval/style_family_realistic/realistic_family.jsonl",
    "apps/api/app/shared/privacy/eval/style_family_realistic/diag_structures.jsonl",
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
FROZEN_DATA_DIR_FILES = [
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
]
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
    "연준", "상원", "진주", "동해", "하늘", "바다", "이슬", "노을", "새벽", "별", "여름", "가을",
    "은하", "태양", "우주", "장미", "초원", "지수", "정원", "수정", "미래", "사랑", "보라", "하나",
    "지혜", "지성", "미소", "선율", "연수", "다정",
}


def leak_check(all_texts: list[str], collision_words: list[str]) -> None:
    frozen_texts_norm = set()
    for rel in FROZEN_JSONLS:
        p = ROOT / rel
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                frozen_texts_norm.add(json.loads(line)["text"].replace(" ", ""))
    for path_str in FROZEN_DATA_DIR_FILES:
        p = Path(path_str)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    frozen_texts_norm.add(json.loads(line)["text"].replace(" ", ""))

    new_texts_norm = {t.replace(" ", "") for t in all_texts}
    overlap_text = new_texts_norm & frozen_texts_norm
    print(f"동결 평가셋/R3~R6 데이터와 문장 겹침: {len(overlap_text)}개", overlap_text or "(없음, OK)")
    assert not overlap_text, "동결 데이터와 문장이 겹친다"

    given_pools = set()
    try:
        import synth  # noqa

        given_pools = set(synth.GIVEN1) | set(synth.GIVEN2) | set(synth.NICKS)
    except Exception:
        pass
    word_overlap = set(collision_words) & (given_pools | ALL_PRIOR_WORDS)
    print(f"실험A 단어(학습+holdout)가 기존 풀과 겹침: {len(word_overlap)}개", word_overlap or "(없음, OK)")
    assert not word_overlap, "실험A 단어가 기존 이름 풀/평가셋과 겹친다"

    overlaps = set(TRAIN_WORDS) & set(HOLDOUT_WORDS)
    assert not overlaps
    print("학습/holdout 단어 겹침: 없음, OK")


def write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_texts: list[str] = []
    train_rows, val_rows = [], []
    rid = 0

    for cat, split, tpl in PERSON_TEMPLATES:
        for w in TRAIN_WORDS:
            text = fill(tpl, w)
            all_texts.append(text)
            rid += 1
            row = make_span_row(f"r7a-pos-{rid:04d}", text, w, cat, split)
            (train_rows if split == "train" else val_rows).append(row)

    for w, sents in NONPERSON_TRAIN_WORDS.items():
        for text, split in sents:
            all_texts.append(text)
            rid += 1
            row = make_neg_row(f"r7a-neg-{rid:04d}", text, "main_neg", split)
            (train_rows if split == "train" else val_rows).append(row)

    leak_check(all_texts, TRAIN_WORDS + HOLDOUT_WORDS)

    write(out / "r7a_train_extra.jsonl", train_rows)
    write(out / "r7a_val_extra.jsonl", val_rows)
    print(f"학습 추가 train: {len(train_rows)}행, val: {len(val_rows)}행 -> {out}")

    holdout_rows = []
    rid = 0
    for tpl in HOLDOUT_TEMPLATES:
        for w in HOLDOUT_WORDS:
            text = fill(tpl, w)
            rid += 1
            holdout_rows.append(make_eval_row(f"r7a-ho-{rid:04d}", text, w, "label_consistency_holdout"))
    for w, sents in NONPERSON_HOLDOUT_WORDS.items():
        for text in sents:
            rid += 1
            holdout_rows.append(make_eval_row(f"r7a-ho-{rid:04d}", text, None, "label_consistency_holdout_neg"))
    write(out / "r7a_heldout_eval.jsonl", holdout_rows)
    print(f"held-out 평가 전용(학습 미사용): {len(holdout_rows)}행 -> {out}")


if __name__ == "__main__":
    main()
