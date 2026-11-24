"""R6(ORG/LOC 대비 데이터 + 기본 문형 증량 + 2글자 단일형 경계 보강) 학습 데이터 생성.

배경: `NER_VALIDATION.md` 58절 R6 설계를 사용자 조건(2026-11-17)대로 구현한다.
- **조건 1(ORG/LOC 구별)**: 네거티브만 늘리지 않고, "~소식..." 계열의 새 문형(v11의 "N 소식을
  오랜만에 들었어"와는 다른 문장) 안에서 **PERSON/ORGANIZATION/LOCATION 세 라벨을 같은 틀로 대비**
  시킨다. ORG/LOCATION 예시는 `synth.py`의 기존 안전한 풀(HOSP/SCHOOL/COMPANY 접두사+접미사,
  CITIES/LANDMARKS/STATIONS)에서만 뽑아 새 단어를 지어내지 않았다.
- **조건 2(2글자 경계 보강)**: `merge_ner`의 PERSON 2자 미만 필터는 건드리지 않고(운영 코드 미수정),
  **학습 데이터로만** 해결한다 — 학습 단어 10개를 전부 2글자 단일형으로 골라 "전체 2글자를 PERSON
  으로 예측"하는 신호를 강하게 반복 노출한다.
- **조건 3(R5 데이터 유지 + 증분 분리)**: R5 데이터(base+R3+R4+R5) 위에 그대로 이어붙인다. 기반
  사전학습 가중치에서 매번 새로 학습하는 R3~R5와 동일한 방식을 유지한다(아래 train 스크립트
  실행 기록에 근거 명시).
- holdout 4개는 학습에 전혀 안 씀(일반화 1차 확인).

사용: ``python build_r6_org_loc_contrast_data.py --out-dir <data_r6_phrase 폴더>``
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

TRAIN_WORDS = ["서랍", "벼루", "바늘", "연못", "찻잔", "먹물", "빗물", "눈발", "마당", "지게"]
HOLDOUT_WORDS = ["뒤뜰", "앞뜰", "화덕", "절구"]

# synth.py 기존 안전 풀에서만 뽑은 ORGANIZATION/LOCATION 대비 예시(새 단어를 지어내지 않음)
ORG_EXAMPLES = ["한빛초등학교", "대양전자", "참사랑병원", "새솔유치원", "동림물산", "으뜸치과"]
LOC_EXAMPLES = ["해운대역", "코엑스", "강남역", "경복궁", "남산타워", "서울숲"]


def has_batchim(ch: str) -> bool:
    return (ord(ch) - 0xAC00) % 28 != 0


def obj(w: str) -> str:
    return w + ("을" if has_batchim(w[-1]) else "를")


def subj(w: str) -> str:
    return w + ("이" if has_batchim(w[-1]) else "가")


def voc(w: str) -> str:
    return w + ("아" if has_batchim(w[-1]) else "야")


# 기본 문형 4개(조건 1·2 핵심, v9~v11 어떤 틀과도 다른 새 문장) + 목적격/비교서술/여격/호격 1개씩(가벼운 유지)
PERSON_TEMPLATES = [
    ("기본_ORGLOC대비1", "train", "{N} 소식이 궁금해서 연락해봤어"),
    ("기본_ORGLOC대비2", "train", "{N} 얘기가 오늘 화제였대"),
    ("기본_증량1", "train", "요즘 {N} 어떻게 지내는지 모르겠어"),
    ("기본_증량2", "val", "{N} 근황을 듣게 됐어"),
    ("목적격", "train", "다들 {Nobj} 많이 보고 싶어 했어"),
    ("비교서술", "train", "{Nsubj} 모임에서 제일 활발해"),
    ("여격", "train", "{N}한테 날짜 맞는지 물어봤어"),
    ("호격", "train", "{Nvoc}, 이제 집에 가자"),
]

# 조건 1: 같은 "기본_ORGLOC대비" 두 틀에 ORG/LOCATION 실제 예시를 넣어 라벨을 대비시킨다.
ORGLOC_TEMPLATES = [
    ("기본_ORGLOC대비1", "train", "{N} 소식이 궁금해서 연락해봤어"),
    ("기본_ORGLOC대비2", "val", "{N} 얘기가 오늘 화제였대"),
]

NONPERSON_TRAIN_WORDS: dict[str, list[tuple[str, str]]] = {
    "서랍": [("서랍에 옷을 정리했다", "train"), ("서랍을 열어 봤다", "train"), ("서랍 손잡이가 헐거웠다", "val")],
    "벼루": [("벼루에 먹을 갈았다", "train"), ("벼루와 붓을 꺼냈다", "train"), ("벼루가 묵직했다", "val")],
    "바늘": [("바늘에 실을 꿰었다", "train"), ("바늘이 어디 있는지 찾았다", "train"), ("바늘로 옷을 기웠다", "val")],
    "연못": [("연못에 물고기가 있었다", "train"), ("연못 주위를 산책했다", "train"), ("연못이 꽁꽁 얼었다", "val")],
    "찻잔": [("찻잔에 차를 따랐다", "train"), ("찻잔을 깨끗이 씻었다", "train"), ("찻잔이 예뻤다", "val")],
    "먹물": [("먹물이 손에 묻었다", "train"), ("먹물로 글씨를 썼다", "train"), ("먹물 냄새가 났다", "val")],
    "빗물": [("빗물이 고였다", "train"), ("빗물을 받아 화분에 줬다", "train"), ("빗물 떨어지는 소리가 들렸다", "val")],
    "눈발": [("눈발이 날리기 시작했다", "train"), ("눈발이 거세졌다", "train"), ("눈발 사이로 길을 걸었다", "val")],
    "마당": [("마당에 빨래를 널었다", "train"), ("마당을 쓸었다", "train"), ("마당이 넓었다", "val")],
    "지게": [("지게에 짐을 실었다", "train"), ("지게를 지고 산을 올랐다", "train"), ("지게가 무거웠다", "val")],
}

HOLDOUT_TEMPLATES = [
    "{N} 소식 듣고 반가웠어",
    "{Nvoc}, 밥 먹었어?",
]
NONPERSON_HOLDOUT_WORDS: dict[str, list[str]] = {
    "뒤뜰": ["뒤뜰에 텃밭을 가꿨다", "뒤뜰에서 고양이를 봤다"],
    "앞뜰": ["앞뜰에 화분을 놓았다", "앞뜰이 아늑했다"],
    "화덕": ["화덕에 불을 지폈다", "화덕에서 빵을 구웠다"],
    "절구": ["절구에 쌀을 찧었다", "절구가 무거웠다"],
}


def fill(tpl: str, n: str) -> str:
    return tpl.format(N=n, Nobj=obj(n), Nsubj=subj(n), Nvoc=voc(n))


def find_span(text: str, token: str) -> tuple[int, int]:
    idx = text.find(token)
    if idx < 0:
        raise ValueError(f"토큰을 못 찾음: {token!r} in {text!r}")
    return idx, idx + len(token)


def make_span_row(rid: str, text: str, name: str, label: str, category: str, split: str) -> dict:
    s, e = find_span(text, name)
    return {
        "id": rid, "text": text, "spans": [{"label": label, "start": s, "end": e}],
        "source": "synth", "template_id": f"r6-{category}", "category": category, "split": split, "augment": True,
    }


def make_neg_row(rid: str, text: str, category: str, split: str) -> dict:
    return {
        "id": rid, "text": text, "spans": [],
        "source": "synth", "template_id": f"r6-{category}-neg", "category": category, "split": split, "augment": True,
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
    print(f"동결 평가셋/R3~R5 데이터와 문장 겹침: {len(overlap_text)}개", overlap_text or "(없음, OK)")
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
    r5_mod = load_mod("_r5_check", ROOT / "ml/ner/data/build_r5_broad_phrase_data.py")

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
    for _c, _s, t in r5_mod.PERSON_TEMPLATES:
        existing_tpl_skel.add(t.replace("{N}", "").replace("{Nobj}", "").replace("{Nsubj}", "").replace("{Nvoc}", "").replace("{copula}", ""))
    for t in r5_mod.HOLDOUT_TEMPLATES:
        existing_tpl_skel.add(t.replace("{N}", "").replace("{Nobj}", "").replace("{Nsubj}", "").replace("{Nvoc}", ""))
    for _c, _s, t in r5_mod.REHEARSAL_TEMPLATES:
        existing_tpl_skel.add(t.replace("{N}", "").replace("{Nobj}", "").replace("{Nsubj}", "").replace("{Nvoc}", "").replace("{copula}", ""))
    for _c, _s, t in r5_mod.SEMANTIC_PERSON_TEMPLATES:
        existing_tpl_skel.add(t.replace("{N}", "").replace("{Nobj}", "").replace("{Nsubj}", "").replace("{Nvoc}", ""))

    def strip_placeholders(t):
        return t.replace("{N}", "").replace("{Nobj}", "").replace("{Nsubj}", "").replace("{Nvoc}", "")

    new_tpl_skel = {strip_placeholders(t) for _c, _s, t in PERSON_TEMPLATES}
    new_tpl_skel |= {strip_placeholders(t) for t in HOLDOUT_TEMPLATES}
    overlap_tpl = new_tpl_skel & existing_tpl_skel
    print(f"기존/R3~R5 템플릿과 뼈대 겹침: {len(overlap_tpl)}개", overlap_tpl or "(없음, OK)")
    assert not overlap_tpl, "기존 템플릿과 뼈대가 겹친다"

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
        "연준", "상원", "진주", "동해", "하늘", "바다", "이슬", "노을", "새벽", "별", "여름", "가을",
        "은하", "태양", "우주", "장미", "초원", "지수", "정원", "수정", "미래", "사랑", "보라", "하나",
        "지혜", "지성", "미소", "선율", "연수", "다정",
    }
    given_pools = set(synth_mod.GIVEN1) | set(synth_mod.GIVEN2) | set(synth_mod.NICKS)
    all_blocked = given_pools | ALL_PRIOR_WORDS
    word_overlap = set(collision_words) & all_blocked
    print(f"R6 충돌단어(학습+holdout)가 기존 풀과 겹침: {len(word_overlap)}개", word_overlap or "(없음, OK)")
    assert not word_overlap, "R6 충돌단어가 기존 이름 풀/평가셋과 겹친다"

    overlaps = set(TRAIN_WORDS) & set(HOLDOUT_WORDS)
    assert not overlaps
    print("R6 학습/holdout 단어 겹침: 없음, OK")

    # ORG/LOCATION 예시가 synth.py 기존 풀에서 왔는지 확인
    org_pool_strings = set()
    for p in synth_mod.HOSP_PREFIX:
        for s in synth_mod.HOSP_SUFFIX:
            org_pool_strings.add(p + s)
    for p in synth_mod.SCHOOL_PREFIX:
        for s in synth_mod.SCHOOL_SUFFIX:
            org_pool_strings.add(p + s)
    for p in synth_mod.COMPANY_PREFIX:
        for s in synth_mod.COMPANY_SUFFIX:
            org_pool_strings.add(p + s)
    loc_pool_strings = set(synth_mod.CITIES) | set(synth_mod.LANDMARKS) | set(synth_mod.STATIONS)
    missing_org = set(ORG_EXAMPLES) - org_pool_strings
    missing_loc = set(LOC_EXAMPLES) - loc_pool_strings
    print(f"ORG 예시가 기존 풀 밖인 것: {missing_org or '없음, OK'}")
    print(f"LOCATION 예시가 기존 풀 밖인 것: {missing_loc or '없음, OK'}")
    assert not missing_org and not missing_loc, "ORG/LOCATION 예시가 synth.py 기존 풀에 없다"


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

    # 1) 주력 10단어 x PERSON 문형 8개
    for cat, split, tpl in PERSON_TEMPLATES:
        for w in TRAIN_WORDS:
            text = fill(tpl, w)
            all_texts.append(text)
            rid += 1
            row = make_span_row(f"r6-pos-{rid:04d}", text, w, "PERSON", cat, split)
            (train_rows if split == "train" else val_rows).append(row)

    # 2) 조건 1: 같은 "기본_ORGLOC대비" 틀에 ORGANIZATION/LOCATION 실제 예시(기존 풀) 라벨 대비
    for cat, split, tpl in ORGLOC_TEMPLATES:
        for name in ORG_EXAMPLES:
            text = fill(tpl, name)
            all_texts.append(text)
            rid += 1
            row = make_span_row(f"r6-org-{rid:04d}", text, name, "ORGANIZATION", f"{cat}_ORG대비", split)
            (train_rows if split == "train" else val_rows).append(row)
        for name in LOC_EXAMPLES:
            text = fill(tpl, name)
            all_texts.append(text)
            rid += 1
            row = make_span_row(f"r6-loc-{rid:04d}", text, name, "LOCATION", f"{cat}_LOC대비", split)
            (train_rows if split == "train" else val_rows).append(row)

    # 3) 주력 10단어 x 비PERSON(단어마다 3개)
    for w, sents in NONPERSON_TRAIN_WORDS.items():
        for text, split in sents:
            all_texts.append(text)
            rid += 1
            row = make_neg_row(f"r6-neg-{rid:04d}", text, "main_neg", split)
            (train_rows if split == "train" else val_rows).append(row)

    leak_check(all_texts, TRAIN_WORDS + HOLDOUT_WORDS)

    write(out / "r6_train_extra.jsonl", train_rows)
    write(out / "r6_val_extra.jsonl", val_rows)
    print(f"학습 추가 train: {len(train_rows)}행, val: {len(val_rows)}행 -> {out}")

    # 4) holdout 4개 - 학습에 전혀 안 씀
    holdout_rows = []
    rid = 0
    for tpl in HOLDOUT_TEMPLATES:
        for w in HOLDOUT_WORDS:
            text = fill(tpl, w)
            rid += 1
            holdout_rows.append(make_eval_row(f"r6-ho-{rid:04d}", text, w, "broad_holdout"))
    for w, sents in NONPERSON_HOLDOUT_WORDS.items():
        for text in sents:
            rid += 1
            holdout_rows.append(make_eval_row(f"r6-ho-{rid:04d}", text, None, "broad_holdout_neg"))
    write(out / "r6_heldout_eval.jsonl", holdout_rows)
    print(f"held-out 평가 전용(학습 미사용): {len(holdout_rows)}행 -> {out}")


if __name__ == "__main__":
    main()
