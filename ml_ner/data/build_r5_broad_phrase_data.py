"""R5(비교서술·기본 중심 복합명사 확장 + 중의성 오탐 억제 + 회귀 리허설) 학습 데이터 생성.

배경: `apps/api/docs/security/NER_VALIDATION.md` 54절 R5 전략 제안을 그대로 구현한다.
- A) PERSON 탐지: R4가 목적격만 보강했던 것과 달리 **비교서술·기본을 우선** 보강하고, 목적격·여격·
  호격도 가볍게 포함해 기존 성능을 지킨다(신규 12단어, 복합명사형 6 + 단일형 6).
- B) 비PERSON 대비: 같은 12단어의 일반명사 문장 + **색채·추상 의미범주의 신규 8단어**(PERSON+비
  PERSON 쌍)를 추가해, R4에서 확인된 "의미범주 전체로 번지는 오탐"(54절 4번)을 억제한다 — 48절
  진단셋의 실제 단어는 절대 재사용하지 않고 **같은 범주의 새 단어**만 쓴다.
- C) 회귀 리허설: 54절 5번에서 확인된, R4가 깨뜨린 패턴("~이/가 아침부터...", "방금 통화한 사람이
  X였어/이었어", "~한테...")을 **안전한 기존 이름(GIVEN2)**으로 다시 학습 섞음에 포함해 파국적
  망각을 줄인다.
- holdout 4단어는 학습에 전혀 안 쓴다(일반화 확인용).

**R4 데이터(base+R3+R4 보강) 위에 이어붙인다** — "R4와 최대한 동일한 조건"을 유지한 채 R5 증분만의
효과를 분리하기 위함(사용자 지시).

사용: ``python build_r5_broad_phrase_data.py --out-dir <data_r5_phrase 폴더>``
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

TRAIN_WORDS = ["돌다리", "바위틈", "산울림", "징검다리", "물레", "처마끝", "낙엽", "구슬", "우물", "고드름", "장작불", "등잔불"]
HOLDOUT_WORDS = ["디딤돌", "모닥불", "풍경소리", "종소리"]
SEMANTIC_NEG_WORDS = ["다홍", "연두", "청록", "진홍", "열정", "평화", "성실", "용기"]
REHEARSAL_NAME_SLICE = slice(5, 5 + 16 * 9, 9)  # synth.GIVEN2 결정적 표본(R3의 다른 슬라이스와 겹치지 않게)


def has_batchim(ch: str) -> bool:
    return (ord(ch) - 0xAC00) % 28 != 0


def obj(w: str) -> str:
    return w + ("을" if has_batchim(w[-1]) else "를")


def subj(w: str) -> str:
    return w + ("이" if has_batchim(w[-1]) else "가")


def voc(w: str) -> str:
    return w + ("아" if has_batchim(w[-1]) else "야")


def copula_past(w: str) -> str:
    return w + ("이었어" if has_batchim(w[-1]) else "였어")


# 주력 12단어용 문형: 비교서술(3) + 기본(2) + 목적격(1) + 여격(1) + 호격(1) = 8틀
PERSON_TEMPLATES = [
    ("비교서술", "train", "{Nsubj} 반에서 늘 믿음직하대"),
    ("비교서술", "train", "{Nsubj} 이번에도 제일 돋보였어"),
    ("비교서술", "val", "선생님도 {Nsubj} 가장 야무지다고 하셨어"),
    ("기본", "train", "{Nsubj} 요즘 부쩍 말이 많아졌어"),
    ("기본", "val", "{N} 얘기를 자주 들었어"),
    ("목적격", "train", "다들 {Nobj} 참 많이 챙겨줘"),
    ("여격", "train", "{N}한테 저녁 뭐 먹을지 물어봤어"),
    ("호격", "train", "{Nvoc}, 이제 그만 자자"),
]

NONPERSON_TRAIN_WORDS: dict[str, list[tuple[str, str]]] = {
    "돌다리": [("냇가에 돌다리가 놓여 있었다", "train"), ("돌다리를 건너 학교에 갔다", "train"), ("돌다리가 미끄러웠다", "val")],
    "바위틈": [("바위틈에 풀이 자라 있었다", "train"), ("바위틈으로 물이 흘렀다", "train"), ("바위틈에 게가 숨어 있었다", "val")],
    "산울림": [("산에서 산울림이 들려왔다", "train"), ("산울림 소리가 신기했다", "train"), ("산울림이 메아리처럼 퍼졌다", "val")],
    "징검다리": [("개울에 징검다리가 놓여 있었다", "train"), ("징검다리를 조심조심 건넜다", "train"), ("징검다리가 물에 잠겼다", "val")],
    "물레": [("할머니가 물레를 돌리셨다", "train"), ("물레로 실을 뽑았다", "train"), ("물레 돌아가는 소리가 들렸다", "val")],
    "처마끝": [("처마끝에 고드름이 달렸다", "train"), ("처마끝에서 비가 떨어졌다", "train"), ("처마끝이 낮았다", "val")],
    "낙엽": [("길에 낙엽이 쌓였다", "train"), ("낙엽을 밟으며 걸었다", "train"), ("낙엽이 바람에 날렸다", "val")],
    "구슬": [("아이가 구슬을 가지고 놀았다", "train"), ("구슬이 바닥에 굴러다녔다", "train"), ("구슬치기를 했다", "val")],
    "우물": [("마을에 오래된 우물이 있었다", "train"), ("우물에서 물을 길었다", "train"), ("우물이 깊었다", "val")],
    "고드름": [("처마에 고드름이 달렸다", "train"), ("고드름을 똑 떼어냈다", "train"), ("고드름이 녹아 떨어졌다", "val")],
    "장작불": [("마당에서 장작불을 피웠다", "train"), ("장작불이 활활 타올랐다", "train"), ("장작불 앞에서 몸을 녹였다", "val")],
    "등잔불": [("방에 등잔불을 켰다", "train"), ("등잔불이 흔들렸다", "train"), ("등잔불 밑에서 책을 읽었다", "val")],
}

# 색채·추상 의미범주 8단어: 같은 8틀 중 대표 2개(비교서술·기본)만 적용 + 비PERSON 2문장
SEMANTIC_PERSON_TEMPLATES = [
    ("비교서술", "train", "{Nsubj} 반에서 늘 믿음직하대"),
    ("기본", "val", "{N} 얘기를 자주 들었어"),
]
NONPERSON_SEMANTIC_WORDS: dict[str, list[tuple[str, str]]] = {
    "다홍": [("다홍색 치마를 입었다", "train"), ("다홍빛 저녁놀이 번졌다", "val")],
    "연두": [("연두색 새싹이 돋았다", "train"), ("연두빛 봄이 왔다", "val")],
    "청록": [("청록색 바다가 펼쳐졌다", "train"), ("청록빛 숲이 짙었다", "val")],
    "진홍": [("진홍색 장미가 피었다", "train"), ("진홍빛 노을이 깔렸다", "val")],
    "열정": [("열정을 다해 연습했다", "train"), ("열정이 느껴지는 무대였다", "val")],
    "평화": [("평화를 지키려 노력했다", "train"), ("평화로운 오후였다", "val")],
    "성실": [("성실을 다해 일했다", "train"), ("성실한 태도가 돋보였다", "val")],
    "용기": [("용기를 내어 손을 들었다", "train"), ("용기 있는 행동이었다", "val")],
}

HOLDOUT_TEMPLATES = [
    "{Nsubj} 요즘 들어 유난히 씩씩해졌어",
    "{N}한테 생일 축하한다고 전해줘",
]
NONPERSON_HOLDOUT_WORDS: dict[str, list[str]] = {
    "디딤돌": ["마당에 디딤돌을 놓았다", "디딤돌을 밟고 지나갔다"],
    "모닥불": ["캠핑장에서 모닥불을 피웠다", "모닥불 주위에 둘러앉았다"],
    "풍경소리": ["절에서 풍경소리가 들렸다", "바람에 풍경소리가 울렸다"],
    "종소리": ["멀리서 종소리가 울렸다", "종소리에 잠에서 깼다"],
}

# 회귀 리허설(54절 5번): R4가 깨뜨린 패턴을 안전한 기존 이름(GIVEN2)으로 재학습. train/val 분리.
REHEARSAL_TEMPLATES = [
    ("기본_주어서술", "train", "{Nsubj} 아침부터 기분이 좋아 보이더라"),
    ("기본_서술종결", "train", "방금 통화한 사람이 {copula}"),
    ("여격_안부", "val", "{N}한테 문자 보냈는데 아직 답이 없어"),
]


def fill(tpl: str, n: str) -> str:
    return tpl.format(N=n, Nobj=obj(n), Nsubj=subj(n), Nvoc=voc(n), copula=copula_past(n))


def find_span(text: str, token: str) -> tuple[int, int]:
    idx = text.find(token)
    if idx < 0:
        raise ValueError(f"토큰을 못 찾음: {token!r} in {text!r}")
    return idx, idx + len(token)


def make_span_row(rid: str, text: str, name: str, category: str, split: str) -> dict:
    s, e = find_span(text, name)
    return {
        "id": rid, "text": text, "spans": [{"label": "PERSON", "start": s, "end": e}],
        "source": "synth", "template_id": f"r5-{category}", "category": category, "split": split, "augment": True,
    }


def make_neg_row(rid: str, text: str, category: str, split: str) -> dict:
    return {
        "id": rid, "text": text, "spans": [],
        "source": "synth", "template_id": f"r5-{category}-neg", "category": category, "split": split, "augment": True,
    }


def make_eval_row(rid: str, text: str, name: str | None, category: str) -> dict:
    spans = []
    if name:
        s, e = find_span(text, name)
        spans.append({"kind": "PERSON", "start": s, "end": e})
    return {"id": rid, "category": category, "text": text, "spans": spans, "known_names": [], "template_id": category}


def rehearsal_names() -> list[str]:
    import synth  # noqa: E402

    pool = sorted(set(synth.GIVEN2))
    return pool[REHEARSAL_NAME_SLICE]


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
]
FROZEN_DATA_DIR_FILES = [
    r"C:\Users\<user>\venvs\ner-runs\data_r3_phrase\r3_train_extra.jsonl",
    r"C:\Users\<user>\venvs\ner-runs\data_r3_phrase\r3_val_extra.jsonl",
    r"C:\Users\<user>\venvs\ner-runs\data_r3_phrase\r3_heldout_eval.jsonl",
    r"C:\Users\<user>\venvs\ner-runs\data_r4_phrase\r4_train_extra.jsonl",
    r"C:\Users\<user>\venvs\ner-runs\data_r4_phrase\r4_val_extra.jsonl",
    r"C:\Users\<user>\venvs\ner-runs\data_r4_phrase\r4_heldout_eval.jsonl",
]


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
    print(f"동결 평가셋/R3·R4 데이터와 문장 겹침: {len(overlap_text)}개", overlap_text or "(없음, OK)")
    assert not overlap_text, "동결 데이터와 문장이 겹친다"

    import importlib.util

    def load_mod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    synth_mod = load_mod("_synth_check", ROOT / "ml/ner/data/synth.py")
    r3_mod = load_mod("_r3_check", ROOT / "ml/ner/data/build_r3_phrase_collision_data.py")
    r4_mod = load_mod("_r4_check", ROOT / "ml/ner/data/build_r4_compound_objective_data.py")

    existing_tpl_skel = {
        t.replace("{P}", "").replace("{G}", "").replace("{NK}", "").replace("{AL}", "")
        for _c, t in synth_mod.AUG_TEMPLATES + synth_mod.AUG_TEMPLATES_2 + synth_mod.AUG_TEMPLATES_3
    } | {t for _c, t in synth_mod.TEMPLATES}
    for _c, _s, t in r3_mod.PERSON_TEMPLATES:
        existing_tpl_skel.add(t.replace("{N}", "").replace("{Nobj}", "").replace("{Nsubj}", "").replace("{Nvoc}", ""))
    for t in r3_mod.HOLDOUT_TEMPLATES:
        existing_tpl_skel.add(t.replace("{N}", "").replace("{Nobj}", "").replace("{Nsubj}", "").replace("{Nvoc}", ""))
    for _c, _s, t in r4_mod.PERSON_TEMPLATES:
        existing_tpl_skel.add(t.replace("{N}", "").replace("{Nobj}", ""))
    for t in r4_mod.HOLDOUT_TEMPLATES:
        existing_tpl_skel.add(t.replace("{N}", "").replace("{Nobj}", ""))

    def strip_placeholders(t):
        return t.replace("{N}", "").replace("{Nobj}", "").replace("{Nsubj}", "").replace("{Nvoc}", "").replace("{copula}", "")

    new_tpl_skel = {strip_placeholders(t) for _c, _s, t in PERSON_TEMPLATES}
    new_tpl_skel |= {strip_placeholders(t) for t in HOLDOUT_TEMPLATES}
    new_tpl_skel |= {strip_placeholders(t) for _c, _s, t in REHEARSAL_TEMPLATES}
    overlap_tpl = new_tpl_skel & existing_tpl_skel
    print(f"기존/R3/R4 템플릿과 뼈대 겹침: {len(overlap_tpl)}개", overlap_tpl or "(없음, OK)")
    assert not overlap_tpl, "기존 템플릿과 뼈대가 겹친다"

    probe_words = {
        "연준", "상원", "진주", "동해", "하늘", "바다", "이슬", "노을", "새벽", "별", "여름", "가을",
        "은하", "태양", "우주", "장미", "초원", "지수", "정원", "수정", "미래", "사랑", "보라", "하나",
        "지혜", "지성", "미소", "선율", "연수", "다정",
    }
    v9_words = {"진달래", "민들레", "억새", "솔잎", "눈송이", "빗방울", "들꽃", "메아리", "솔바람", "꽃잎"}
    v10_words = {
        "돌탑", "조약돌", "나뭇결", "물줄기", "나룻배", "촛불", "반딧불", "바닷가",
        "들국화", "담쟁이", "초승달", "과꽃", "채송화", "나팔꽃", "분꽃", "산수유",
    }
    r3_words = set(r3_mod.TRAIN_WORDS) | set(r3_mod.HOLDOUT_WORDS)
    r4_words = set(r4_mod.TRAIN_WORDS) | set(r4_mod.HOLDOUT_WORDS)
    given_pools = set(synth_mod.GIVEN1) | set(synth_mod.GIVEN2) | set(synth_mod.NICKS)
    all_blocked = probe_words | given_pools | v9_words | v10_words | r3_words | r4_words
    word_overlap = set(collision_words) & all_blocked
    print(f"R5 충돌단어(학습+holdout+의미범주)가 기존 풀과 겹침: {len(word_overlap)}개", word_overlap or "(없음, OK)")
    assert not word_overlap, "R5 충돌단어가 기존 이름 풀/평가셋과 겹친다"

    overlaps = set(TRAIN_WORDS) & set(HOLDOUT_WORDS) | set(TRAIN_WORDS) & set(SEMANTIC_NEG_WORDS) | set(HOLDOUT_WORDS) & set(SEMANTIC_NEG_WORDS)
    assert not overlaps, f"R5 내부 단어 집합 겹침: {overlaps}"
    print("R5 내부(학습/holdout/의미범주) 단어 겹침: 없음, OK")


def write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    names = rehearsal_names()
    assert len(names) == 16, f"리허설 이름 표본이 16개가 아님: {len(names)}"
    print("회귀 리허설용 GIVEN2 표본(16):", names)

    all_texts: list[str] = []
    train_rows, val_rows = [], []
    rid = 0

    # 1) 주력 12단어 x PERSON 문형 8개
    for cat, split, tpl in PERSON_TEMPLATES:
        for w in TRAIN_WORDS:
            text = fill(tpl, w)
            all_texts.append(text)
            rid += 1
            row = make_span_row(f"r5-pos-{rid:04d}", text, w, cat, split)
            (train_rows if split == "train" else val_rows).append(row)

    # 2) 주력 12단어 x 비PERSON(단어마다 3개)
    for w, sents in NONPERSON_TRAIN_WORDS.items():
        for text, split in sents:
            all_texts.append(text)
            rid += 1
            row = make_neg_row(f"r5-neg-{rid:04d}", text, "main_neg", split)
            (train_rows if split == "train" else val_rows).append(row)

    # 3) 색채·추상 의미범주 8단어 x PERSON 2틀 + 비PERSON 2문장
    for cat, split, tpl in SEMANTIC_PERSON_TEMPLATES:
        for w in SEMANTIC_NEG_WORDS:
            text = fill(tpl, w)
            all_texts.append(text)
            rid += 1
            row = make_span_row(f"r5-sem-pos-{rid:04d}", text, w, f"semantic_{cat}", split)
            (train_rows if split == "train" else val_rows).append(row)
    for w, sents in NONPERSON_SEMANTIC_WORDS.items():
        for text, split in sents:
            all_texts.append(text)
            rid += 1
            row = make_neg_row(f"r5-sem-neg-{rid:04d}", text, "semantic_neg", split)
            (train_rows if split == "train" else val_rows).append(row)

    # 4) 회귀 리허설(R4가 깨뜨린 패턴, 안전한 GIVEN2 이름)
    for cat, split, tpl in REHEARSAL_TEMPLATES:
        for n in names:
            text = fill(tpl, n)
            all_texts.append(text)
            rid += 1
            row = make_span_row(f"r5-rehearsal-{rid:04d}", text, n, cat, split)
            (train_rows if split == "train" else val_rows).append(row)

    leak_check(all_texts, TRAIN_WORDS + HOLDOUT_WORDS + SEMANTIC_NEG_WORDS)

    write(out / "r5_train_extra.jsonl", train_rows)
    write(out / "r5_val_extra.jsonl", val_rows)
    print(f"학습 추가 train: {len(train_rows)}행, val: {len(val_rows)}행 -> {out}")

    # 5) holdout 4개 - 학습에 전혀 안 씀
    holdout_rows = []
    rid = 0
    for tpl in HOLDOUT_TEMPLATES:
        for w in HOLDOUT_WORDS:
            text = fill(tpl, w)
            rid += 1
            holdout_rows.append(make_eval_row(f"r5-ho-{rid:04d}", text, w, "broad_holdout"))
    for w, sents in NONPERSON_HOLDOUT_WORDS.items():
        for text in sents:
            rid += 1
            holdout_rows.append(make_eval_row(f"r5-ho-{rid:04d}", text, None, "broad_holdout_neg"))
    write(out / "r5_heldout_eval.jsonl", holdout_rows)
    print(f"held-out 평가 전용(학습 미사용): {len(holdout_rows)}행 -> {out}")


if __name__ == "__main__":
    main()
