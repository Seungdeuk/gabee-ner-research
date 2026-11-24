"""R3(문형 대비 보강: 목적격·여격·호격·비교서술) 학습 데이터 생성 + 동결 평가셋과의 중복 검사.

배경: `apps/api/docs/security/NER_VALIDATION.md` 50절 — KoELECTRA PERSON 탐지가 목적격("~을/를")·
여격("~한테")·호격("~아/야")·비교서술("~가 제일/가장")에서 유독 약하다는 걸 진단셋으로 확인했다.
이 스크립트는 그 4개 문형을 학습 데이터에 보강한다.

설계(우선순위 2 · 2단계, 사용자 지시 2026-11-16):
- **학습용 충돌 단어 8개**(윤슬·구름·바람·물결·달빛·노래·그림·서리): 48절 중의성 진단셋 30단어·
  `step2_synth_draft`의 7단어·`synth.py`의 GIVEN1/GIVEN2/NICKS·기관/주소 풀·baseline train/val/test·
  v4~v8 평가셋 어디에도 없는 새 단어다(leak_check 로 자동 검사). **이 8단어는 어떤 평가에도 다시 쓰지
  않는다** — 48절 진단셋·draft56 은 여전히 "회귀 진단용"으로만 쓰고, 이번 학습의 "독립 평가 근거"로
  삼지 않는다(사용자 지시 조건 1).
- **일반화 검증용 충돌 단어 4개**(안개·단풍·여울·반디): 학습에 전혀 안 쓰고 `r3_heldout_eval.jsonl`
  에만 들어간다(다른 문형으로 썼다 — R1 패턴과 동일하게 틀 자체의 일반화도 같이 본다).
- **폭넓은 이름 적용**(사용자 지시 조건 2): 같은 4개 문형을 충돌 단어에만 쓰지 않고, `synth.py`의
  GIVEN2(안전한 일반 이름) 16개에도 똑같이 적용해 "이 문형=PERSON" 자체를 넓게 가르친다.
- **대비 데이터**(조건 3): 충돌 단어 8개 + holdout 4개 각각, 같은 표면형을 일반명사로 쓴 비PERSON
  문장을 PERSON 문장 수만큼 짝지었다(기존 48절/draft56 문장과 겹치지 않는 새 문장).

사용: ``python build_r3_phrase_collision_data.py --out-dir <data_r3_phrase 폴더>``
  <폴더>/r3_train_extra.jsonl, r3_val_extra.jsonl 을 build_ner_data.py 가 만든 synth_train.jsonl/
  synth_val.jsonl 뒤에 이어붙인다(R1 과 동일한 통합 방식). <폴더>/r3_heldout_eval.jsonl 은 학습에
  쓰지 않는 일반화 전용 평가셋이다.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

# ---- 학습용 충돌 단어(8개) ----
TRAIN_WORDS = ["윤슬", "구름", "바람", "물결", "달빛", "노래", "그림", "서리"]
# ---- 일반화 검증용(학습 미사용, 4개) ----
HOLDOUT_WORDS = ["안개", "단풍", "여울", "반디"]
# ---- 폭넓은 적용용 안전 이름(GIVEN2 표본, 16개, 결정적 슬라이스) ----
BROAD_NAME_SLICE = slice(3, 3 + 16 * 11, 11)  # synth.GIVEN2 에서 11개 간격으로 16개(결정적)


def has_batchim(ch: str) -> bool:
    return (ord(ch) - 0xAC00) % 28 != 0


def obj(w: str) -> str:
    return w + ("을" if has_batchim(w[-1]) else "를")


def subj(w: str) -> str:
    return w + ("이" if has_batchim(w[-1]) else "가")


def voc(w: str) -> str:
    return w + ("아" if has_batchim(w[-1]) else "야")


# 문형 4종 x 2(train/val). {N}=단어, {obj}/{subj}/{voc}=조사 붙은 꼴.
PERSON_TEMPLATES = [
    ("목적격", "train", "선생님이 {Nobj} 칭찬하셨어"),
    ("여격", "train", "{N}한테 생일 선물로 뭘 줄지 아직 못 정했어"),
    ("호격", "train", "{Nvoc}, 저녁에 떡볶이 먹으러 갈래?"),
    ("비교서술", "train", "{Nsubj} 요즘 유치원에서 제일 인기 많아"),
    ("목적격", "val", "엄마가 {Nobj} 불렀는데 못 들었나봐"),
    ("여격", "val", "{N}한테 숙제 다 했냐고 물어봤어"),
    ("호격", "val", "{Nvoc}, 가방 챙겼어?"),
    ("비교서술", "val", "우리 반에서 {Nsubj} 가장 씩씩해"),
]

# 충돌 단어 8개의 비PERSON(일반명사) 문장 — 새로 씀(48절/draft56과 중복 없음), 단어마다 3개
NONPERSON_TRAIN_WORDS: dict[str, list[tuple[str, str]]] = {
    "윤슬": [("강물에 윤슬이 반짝였다", "train"), ("호수 위로 윤슬이 번졌다", "train"), ("사진에 윤슬이 담겼다", "val")],
    "구름": [("하늘에 구름이 잔뜩 꼈다", "train"), ("구름 사이로 해가 비쳤다", "train"), ("비행기가 구름 위를 날았다", "val")],
    "바람": [("창문 틈으로 바람이 들어왔다", "train"), ("바람이 세게 불어 간판이 흔들렸다", "train"), ("바람을 쐬러 밖에 나갔다", "val")],
    "물결": [("호수에 물결이 일었다", "train"), ("바람에 물결이 거세졌다", "train"), ("물결 소리가 잔잔했다", "val")],
    "달빛": [("달빛이 창문으로 들어왔다", "train"), ("달빛 아래서 산책을 했다", "train"), ("달빛이 유난히 밝았다", "val")],
    "노래": [("아이가 노래를 흥얼거렸다", "train"), ("라디오에서 노래가 흘러나왔다", "train"), ("노래 가사를 다 외웠다", "val")],
    "그림": [("벽에 그림을 걸었다", "train"), ("아이가 그림을 그렸다", "train"), ("전시회에서 그림을 봤다", "val")],
    "서리": [("아침에 서리가 내렸다", "train"), ("창문에 서리가 끼었다", "train"), ("서리 맞은 배추를 뽑았다", "val")],
}

# holdout 4개 — 학습 템플릿과 다른 문형(R1 과 동일 원칙), 전부 r3_heldout_eval 전용
HOLDOUT_TEMPLATES = [
    "{Nobj} 유치원 선생님이 칭찬했대",
    "{N}한테 저녁 메뉴 뭐 먹을지 물어봤어",
    "{Nvoc}, 손 씻고 밥 먹자",
    "{Nsubj} 반에서 제일 그림을 잘 그려",
]
NONPERSON_HOLDOUT_WORDS: dict[str, list[str]] = {
    "안개": ["아침에 안개가 자욱했다", "안개 때문에 앞이 잘 안 보였다"],
    "단풍": ["산에 단풍이 곱게 들었다", "단풍 구경하러 산에 갔다"],
    "여울": ["여울 소리가 들려왔다", "여울을 건너 길을 갔다"],
    "반디": ["여름밤 반디가 날아다녔다", "반디 불빛이 은은했다"],
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
        "source": "synth", "template_id": f"r3-{category}", "category": category, "split": split, "augment": True,
    }


def make_neg_row(rid: str, text: str, category: str, split: str) -> dict:
    return {
        "id": rid, "text": text, "spans": [],
        "source": "synth", "template_id": f"r3-{category}-neg", "category": category, "split": split, "augment": True,
    }


def make_eval_row(rid: str, text: str, name: str | None, category: str) -> dict:
    spans = []
    if name:
        s, e = find_span(text, name)
        spans.append({"kind": "PERSON", "start": s, "end": e})
    return {"id": rid, "category": category, "text": text, "spans": spans, "known_names": [], "template_id": category}


def broad_names() -> list[str]:
    import synth  # noqa: E402

    pool = sorted(set(synth.GIVEN2))
    return pool[BROAD_NAME_SLICE]


def skeleton_from_spans(text: str, spans: list[dict]) -> str:
    out, pos = [], 0
    key = "start"
    for sp in sorted(spans, key=lambda s: s[key]):
        out.append(text[pos : sp["start"]])
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
]


def leak_check(all_texts: list[str], collision_words: list[str]) -> None:
    import synth  # noqa: E402

    frozen_texts_norm = set()
    frozen_skel = set()
    for rel in FROZEN_JSONLS:
        p = ROOT / rel
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            spans = [{"start": sp["start"], "end": sp["end"]} for sp in row.get("spans", [])]
            frozen_skel.add(skeleton_from_spans(row["text"], spans))
            frozen_texts_norm.add(row["text"].replace(" ", ""))

    new_texts_norm = {t.replace(" ", "") for t in all_texts}
    overlap_text = new_texts_norm & frozen_texts_norm
    print(f"동결 평가셋과 문장(공백 제거) 겹침: {len(overlap_text)}개", overlap_text or "(없음, OK)")
    assert not overlap_text, "동결 평가셋과 문장이 겹친다"

    import importlib.util

    spec = importlib.util.spec_from_file_location("_synth_check", ROOT / "ml/ner/data/synth.py")
    synth_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(synth_mod)
    existing_tpl_skel = {
        t.replace("{P}", "").replace("{G}", "").replace("{NK}", "").replace("{AL}", "")
        for _c, t in synth_mod.AUG_TEMPLATES + synth_mod.AUG_TEMPLATES_2 + synth_mod.AUG_TEMPLATES_3
    } | {t for _c, t in synth_mod.TEMPLATES}
    new_tpl_skel = {
        t.replace("{N}", "").replace("{Nobj}", "").replace("{Nsubj}", "").replace("{Nvoc}", "")
        for _c, _s, t in PERSON_TEMPLATES
    } | {t.replace("{N}", "").replace("{Nobj}", "").replace("{Nsubj}", "").replace("{Nvoc}", "") for t in HOLDOUT_TEMPLATES}
    overlap_tpl = new_tpl_skel & existing_tpl_skel
    print(f"기존 TEMPLATES/AUG_TEMPLATES 와 뼈대 겹침: {len(overlap_tpl)}개", overlap_tpl or "(없음, OK)")
    assert not overlap_tpl, "기존 템플릿과 뼈대가 겹친다"

    probe_words = {
        "연준", "상원", "진주", "동해", "하늘", "바다", "이슬", "노을", "새벽", "별", "여름", "가을",
        "은하", "태양", "우주", "장미", "초원", "지수", "정원", "수정", "미래", "사랑", "보라", "하나",
        "지혜", "지성", "미소", "선율", "연수", "다정",
    }
    given_pools = set(synth_mod.GIVEN1) | set(synth_mod.GIVEN2) | set(synth_mod.NICKS)
    all_blocked = probe_words | given_pools
    word_overlap = set(collision_words) & all_blocked
    print(f"R3 충돌단어(학습+holdout)가 48절 진단셋·GIVEN/NICK 풀과 겹침: {len(word_overlap)}개", word_overlap or "(없음, OK)")
    assert not word_overlap, "R3 충돌단어가 기존 이름 풀/진단셋과 겹친다"

    train_holdout_overlap = set(TRAIN_WORDS) & set(HOLDOUT_WORDS)
    assert not train_holdout_overlap, f"학습 단어와 holdout 단어가 겹친다: {train_holdout_overlap}"
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

    names = broad_names()
    assert len(names) == 16, f"폭넓은 이름 표본이 16개가 아님: {len(names)}"
    print("폭넓은 적용용 GIVEN2 표본(16):", names)

    all_texts: list[str] = []
    train_rows, val_rows = [], []
    rid = 0

    # 1) 충돌 단어 8개 x PERSON 템플릿 8개(목적격/여격/호격/비교서술, 각 train1+val1)
    for cat, split, tpl in PERSON_TEMPLATES:
        for w in TRAIN_WORDS:
            text = fill(tpl, w)
            all_texts.append(text)
            rid += 1
            row = make_span_row(f"r3-pos-{rid:04d}", text, w, f"collision_{cat}", split)
            (train_rows if split == "train" else val_rows).append(row)

    # 2) 충돌 단어 8개 x 비PERSON 대비 문장(단어마다 3개, 2 train + 1 val)
    for w, sents in NONPERSON_TRAIN_WORDS.items():
        for text, split in sents:
            all_texts.append(text)
            rid += 1
            row = make_neg_row(f"r3-neg-{rid:04d}", text, "collision_neg", split)
            (train_rows if split == "train" else val_rows).append(row)

    # 3) 폭넓은 이름 16개 x 같은 PERSON 템플릿 8개(안전한 이름으로 문형 자체를 넓게 학습)
    for cat, split, tpl in PERSON_TEMPLATES:
        for n in names:
            text = fill(tpl, n)
            all_texts.append(text)
            rid += 1
            row = make_span_row(f"r3-broad-{rid:04d}", text, n, f"broad_{cat}", split)
            (train_rows if split == "train" else val_rows).append(row)

    leak_check(all_texts, TRAIN_WORDS + HOLDOUT_WORDS)

    write(out / "r3_train_extra.jsonl", train_rows)
    write(out / "r3_val_extra.jsonl", val_rows)
    print(f"학습 추가 train: {len(train_rows)}행, val: {len(val_rows)}행 -> {out}")

    # 4) holdout 4개 - 학습에 전혀 안 씀, 별도 문형으로 일반화 전용 평가
    holdout_rows = []
    rid = 0
    for tpl in HOLDOUT_TEMPLATES:
        for w in HOLDOUT_WORDS:
            text = fill(tpl, w)
            rid += 1
            holdout_rows.append(make_eval_row(f"r3-ho-{rid:04d}", text, w, "collision_holdout"))
    for w, sents in NONPERSON_HOLDOUT_WORDS.items():
        for text in sents:
            rid += 1
            holdout_rows.append(make_eval_row(f"r3-ho-{rid:04d}", text, None, "collision_holdout_neg"))
    write(out / "r3_heldout_eval.jsonl", holdout_rows)
    print(f"held-out 평가 전용(학습 미사용): {len(holdout_rows)}행 -> {out}")


if __name__ == "__main__":
    main()
