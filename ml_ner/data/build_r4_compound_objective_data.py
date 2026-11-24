"""R4(복합명사형 이름 x 목적격 문형 보강) 학습 데이터 생성 + 동결 평가셋과의 중복 검사.

배경: `apps/api/docs/security/NER_VALIDATION.md` 52절 — R3의 v9 잔존 완전노출(12건) 중 10건이
"복합명사형(N+N) 자연물 단어"와 "목적격(~을/를)" 문형의 교집합에 몰려 있었다(후처리 제거는 0건 —
병목은 원시 모델의 후보 생성 자체). 이 스크립트는 그 교집합만 정밀 타격한다(사용자 지시
2026-11-16): 목적격 문장에 복합명사형 이름을 등장시키고, **같은 단어를 일반 명사로 쓰는 문장을
똑같은 수만큼** 짝지어 과마스킹을 막는다.

설계:
- **학습용 복합명사형 단어 8개**(눈꽃·물방울·풀잎·꽃망울·나뭇잎·샘물·봄꽃·물거품): 전부 N+N 구조
  (예: 물+방울, 꽃+망울). v9 10단어·R3 학습/holdout 12단어·48절 30단어·draft56 7단어·GIVEN1/GIVEN2/
  NICKS·기관/주소 풀·baseline·v4~v8·v9 전부와 중복 없음(leak_check 자동 확인).
- **일반화 검증용 4개(holdout, 학습 전혀 안 씀)**: 조각달·손길·나뭇가지·풀꽃 — 학습 틀과 다른
  문형으로 `r4_heldout_eval.jsonl`에만 넣는다.
- 목적격 문형만 쓴다(사용자 지시대로 범위를 좁게 유지) — 변형 3개(train) + 1개(val).
- 단어마다 일반 명사 뜻 비PERSON 문장 3개(새로 씀, 기존 어떤 평가셋과도 중복 없음) — PERSON 문장과
  거의 같은 수로 짝지었다(조건: 오탐 억제).
- **R3 데이터 위에 이어붙인다**(R4 = R3 데이터 + 이번 증분) — "R3와 동일한 데이터 조건"을 유지한 채
  이번 증분만의 효과를 분리해서 보기 위함(R1/R2처럼 베이스 단독이 아니라 R3 대비 누적 비교가
  사용자의 목적이므로).

사용: ``python build_r4_compound_objective_data.py --out-dir <data_r4_phrase 폴더>``
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

TRAIN_WORDS = ["눈꽃", "물방울", "풀잎", "꽃망울", "나뭇잎", "샘물", "봄꽃", "물거품"]
HOLDOUT_WORDS = ["조각달", "손길", "나뭇가지", "풀꽃"]


def has_batchim(ch: str) -> bool:
    return (ord(ch) - 0xAC00) % 28 != 0


def obj(w: str) -> str:
    return w + ("을" if has_batchim(w[-1]) else "를")


PERSON_TEMPLATES = [
    ("목적격", "train", "선생님이 {Nobj} 많이 칭찬해주셨어"),
    ("목적격", "train", "할머니가 {Nobj} 유난히 예뻐하셨어"),
    ("목적격", "train", "친구들이 {Nobj} 다 좋아했대"),
    ("목적격", "val", "엄마가 {Nobj} 유치원에 데려다줬어"),
]

NONPERSON_TRAIN_WORDS: dict[str, list[tuple[str, str]]] = {
    "눈꽃": [("창문에 눈꽃이 피었다", "train"), ("겨울 산에 눈꽃이 가득했다", "train"), ("눈꽃 사진을 찍었다", "val")],
    "물방울": [("나뭇잎에 물방울이 맺혔다", "train"), ("물방울이 또르르 굴러떨어졌다", "train"), ("유리창에 물방울이 흘러내렸다", "val")],
    "풀잎": [("바람에 풀잎이 흔들렸다", "train"), ("풀잎을 엮어 반지를 만들었다", "train"), ("풀잎 냄새가 싱그러웠다", "val")],
    "꽃망울": [("나무에 꽃망울이 맺혔다", "train"), ("꽃망울이 터질 듯 부풀었다", "train"), ("꽃망울 사진을 찍었다", "val")],
    "나뭇잎": [("바람에 나뭇잎이 떨어졌다", "train"), ("나뭇잎이 붉게 물들었다", "train"), ("나뭇잎을 책갈피로 썼다", "val")],
    "샘물": [("바위틈에서 샘물이 솟았다", "train"), ("샘물을 떠서 마셨다", "train"), ("샘물이 맑고 시원했다", "val")],
    "봄꽃": [("공원에 봄꽃이 활짝 피었다", "train"), ("봄꽃 축제에 다녀왔다", "train"), ("봄꽃이 하나둘 지기 시작했다", "val")],
    "물거품": [("파도가 물거품을 남기고 사라졌다", "train"), ("물거품이 모래사장에 번졌다", "train"), ("비눗방울이 물거품처럼 터졌다", "val")],
}

HOLDOUT_TEMPLATES = [
    "선생님이 {Nobj} 유독 아껴주셨대",
    "친척들이 {Nobj} 참 예뻐했어",
]
NONPERSON_HOLDOUT_WORDS: dict[str, list[str]] = {
    "조각달": ["하늘에 조각달이 떠 있었다", "조각달이 구름 사이로 보였다"],
    "손길": ["할머니의 손길이 따뜻했다", "세심한 손길이 느껴졌다"],
    "나뭇가지": ["바람에 나뭇가지가 흔들렸다", "나뭇가지를 주워 모았다"],
    "풀꽃": ["길가에 풀꽃이 피어 있었다", "풀꽃으로 꽃다발을 만들었다"],
}


def fill(tpl: str, n: str) -> str:
    return tpl.format(N=n, Nobj=obj(n))


def find_span(text: str, token: str) -> tuple[int, int]:
    idx = text.find(token)
    if idx < 0:
        raise ValueError(f"토큰을 못 찾음: {token!r} in {text!r}")
    return idx, idx + len(token)


def make_span_row(rid: str, text: str, name: str, category: str, split: str) -> dict:
    s, e = find_span(text, name)
    return {
        "id": rid, "text": text, "spans": [{"label": "PERSON", "start": s, "end": e}],
        "source": "synth", "template_id": f"r4-{category}", "category": category, "split": split, "augment": True,
    }


def make_neg_row(rid: str, text: str, category: str, split: str) -> dict:
    return {
        "id": rid, "text": text, "spans": [],
        "source": "synth", "template_id": f"r4-{category}-neg", "category": category, "split": split, "augment": True,
    }


def make_eval_row(rid: str, text: str, name: str | None, category: str) -> dict:
    spans = []
    if name:
        s, e = find_span(text, name)
        spans.append({"kind": "PERSON", "start": s, "end": e})
    return {"id": rid, "category": category, "text": text, "spans": spans, "known_names": [], "template_id": category}


def skeleton_from_spans(text: str, spans: list[dict]) -> str:
    out, pos = [], 0
    for sp in sorted(spans, key=lambda s: s["start"]):
        out.append(text[pos:sp["start"]])
        pos = sp["end"]
    out.append(text[pos:])
    return "".join(out)


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
]
FROZEN_DATA_DIR_FILES = [
    r"C:\Users\<user>\venvs\ner-runs\data_r3_phrase\r3_train_extra.jsonl",
    r"C:\Users\<user>\venvs\ner-runs\data_r3_phrase\r3_val_extra.jsonl",
    r"C:\Users\<user>\venvs\ner-runs\data_r3_phrase\r3_heldout_eval.jsonl",
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
    print(f"동결 평가셋/R3 데이터와 문장 겹침: {len(overlap_text)}개", overlap_text or "(없음, OK)")
    assert not overlap_text, "동결 데이터와 문장이 겹친다"

    import importlib.util

    spec = importlib.util.spec_from_file_location("_synth_check", ROOT / "ml/ner/data/synth.py")
    synth_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(synth_mod)
    existing_tpl_skel = {
        t.replace("{P}", "").replace("{G}", "").replace("{NK}", "").replace("{AL}", "")
        for _c, t in synth_mod.AUG_TEMPLATES + synth_mod.AUG_TEMPLATES_2 + synth_mod.AUG_TEMPLATES_3
    } | {t for _c, t in synth_mod.TEMPLATES}
    r3_tpl_skel = set()
    r3_py = ROOT / "ml/ner/data/build_r3_phrase_collision_data.py"
    r3_spec = importlib.util.spec_from_file_location("_r3_check", r3_py)
    r3_mod = importlib.util.module_from_spec(r3_spec)
    r3_spec.loader.exec_module(r3_mod)
    for _c, _s, t in r3_mod.PERSON_TEMPLATES:
        r3_tpl_skel.add(t.replace("{N}", "").replace("{Nobj}", "").replace("{Nsubj}", "").replace("{Nvoc}", ""))
    for t in r3_mod.HOLDOUT_TEMPLATES:
        r3_tpl_skel.add(t.replace("{N}", "").replace("{Nobj}", "").replace("{Nsubj}", "").replace("{Nvoc}", ""))

    new_tpl_skel = {t.replace("{N}", "").replace("{Nobj}", "") for _c, _s, t in PERSON_TEMPLATES}
    new_tpl_skel |= {t.replace("{N}", "").replace("{Nobj}", "") for t in HOLDOUT_TEMPLATES}
    overlap_tpl = new_tpl_skel & (existing_tpl_skel | r3_tpl_skel)
    print(f"기존/R3 템플릿과 뼈대 겹침: {len(overlap_tpl)}개", overlap_tpl or "(없음, OK)")
    assert not overlap_tpl, "기존 템플릿과 뼈대가 겹친다"

    probe_words = {
        "연준", "상원", "진주", "동해", "하늘", "바다", "이슬", "노을", "새벽", "별", "여름", "가을",
        "은하", "태양", "우주", "장미", "초원", "지수", "정원", "수정", "미래", "사랑", "보라", "하나",
        "지혜", "지성", "미소", "선율", "연수", "다정",
    }
    v9_words = {"진달래", "민들레", "억새", "솔잎", "눈송이", "빗방울", "들꽃", "메아리", "솔바람", "꽃잎"}
    r3_words = set(r3_mod.TRAIN_WORDS) | set(r3_mod.HOLDOUT_WORDS)
    given_pools = set(synth_mod.GIVEN1) | set(synth_mod.GIVEN2) | set(synth_mod.NICKS)
    all_blocked = probe_words | given_pools | v9_words | r3_words
    word_overlap = set(collision_words) & all_blocked
    print(f"R4 충돌단어(학습+holdout)가 기존 풀과 겹침: {len(word_overlap)}개", word_overlap or "(없음, OK)")
    assert not word_overlap, "R4 충돌단어가 기존 이름 풀/평가셋과 겹친다"

    train_holdout_overlap = set(TRAIN_WORDS) & set(HOLDOUT_WORDS)
    assert not train_holdout_overlap
    print("학습 단어와 holdout 단어 겹침: 없음, OK")


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
            row = make_span_row(f"r4-pos-{rid:04d}", text, w, cat, split)
            (train_rows if split == "train" else val_rows).append(row)

    for w, sents in NONPERSON_TRAIN_WORDS.items():
        for text, split in sents:
            all_texts.append(text)
            rid += 1
            row = make_neg_row(f"r4-neg-{rid:04d}", text, "compound_obj_neg", split)
            (train_rows if split == "train" else val_rows).append(row)

    leak_check(all_texts, TRAIN_WORDS + HOLDOUT_WORDS)

    write(out / "r4_train_extra.jsonl", train_rows)
    write(out / "r4_val_extra.jsonl", val_rows)
    print(f"학습 추가 train: {len(train_rows)}행, val: {len(val_rows)}행 -> {out}")

    holdout_rows = []
    rid = 0
    for tpl in HOLDOUT_TEMPLATES:
        for w in HOLDOUT_WORDS:
            text = fill(tpl, w)
            rid += 1
            holdout_rows.append(make_eval_row(f"r4-ho-{rid:04d}", text, w, "compound_obj_holdout"))
    for w, sents in NONPERSON_HOLDOUT_WORDS.items():
        for text in sents:
            rid += 1
            holdout_rows.append(make_eval_row(f"r4-ho-{rid:04d}", text, None, "compound_obj_holdout_neg"))
    write(out / "r4_heldout_eval.jsonl", holdout_rows)
    print(f"held-out 평가 전용(학습 미사용): {len(holdout_rows)}행 -> {out}")


if __name__ == "__main__":
    main()
