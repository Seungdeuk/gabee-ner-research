"""v14 — R7(+B) 최종 독립 평가셋. 학습·개발셋(R7 train/holdout, style_r7_dev, ambiguity_probe,
v9~v13)과 이름·단어·문장이 전혀 겹치지 않는 완전히 새 이름·단어·문형으로 만든다. 실사용·중의성·
스트레스 세 영역 + 1글자 이름을 전부 포함해 평가 전에 고정(커밋)한다. 평가·모델 선정에는 쓰지
않고, 최종 후보(R7+B 등)를 한 번만 검증하는 데만 쓴다.
"""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[6]
sys.path.insert(0, str(ROOT / "ml" / "ner" / "data"))

import synth  # noqa: E402
import build_r7_realistic_data as r7data  # noqa: E402


def has_batchim(ch: str) -> bool:
    return (ord(ch) - 0xAC00) % 28 != 0


def obj(w): return w + ("을" if has_batchim(w[-1]) else "를")
def subj(w): return w + ("이" if has_batchim(w[-1]) else "가")
def voc(w): return w + ("아" if has_batchim(w[-1]) else "야")
def dat(w): return w + "한테"


def fill(tpl, n):
    return tpl.format(N=n, Nobj=obj(n), Nsubj=subj(n), Nvoc=voc(n), Ndat=dat(n))


def find_span(text, token):
    idx = text.find(token)
    if idx < 0:
        raise ValueError(f"토큰을 못 찾음: {token!r} in {text!r}")
    return idx, idx + len(token)


def row(rid, text, spans, category, style):
    return {"id": rid, "category": category, "style": style, "text": text,
            "spans": [{"kind": k, "start": s, "end": e} for s, e, k in spans],
            "known_names": [], "template_id": category}


# --------------------------------------------------------------------------- 실사용(새 이름)
# R7 학습(TRAIN_NAMES 20)·R7 홀드아웃(HOLDOUT_NAMES 10)과 전혀 다른 GIVEN2 이름 14개 + NICKS 3개.
USE_NAMES = ["도하", "선호", "태오", "현서", "은찬", "시후", "민재", "준혁", "동혁", "성훈", "상철", "경훈", "재민", "승우"]
USE_NICKS = ["쭈쭈", "순돌이", "까미"]
assert not (set(USE_NAMES) & (set(r7data.TRAIN_NAMES) | set(r7data.HOLDOUT_NAMES)))
assert not (set(USE_NICKS) & (set(r7data.TRAIN_NICKS) | set(r7data.HOLDOUT_NICKS)))
for n in USE_NAMES:
    assert n in synth.GIVEN2, n
for n in USE_NICKS:
    assert n in synth.NICKS, n

SINGLE_TEMPLATES = [
    ("기본", "{Nsubj} 요즘 많이 피곤해 보여"),
    ("목적격", "삼촌이 {Nobj} 데리러 온대"),
    ("비교서술", "{Nsubj} 가족 중에 제일 다정해"),
    ("여격", "{Ndat} 숙제 끝났는지 물어봤어"),
    ("호격", "{Nvoc}! 밥 먹자"),
]
PAIR_TEMPLATES = ["{AB}가 같이 등교했대", "{AB}한테 모임 시간 알려줬어"]


# --------------------------------------------------------------------------- 중의성(신규 12단어)
AMBIG_TRAIN = ["행복", "자유", "믿음", "인연", "온정", "나눔", "다짐", "기억"]
AMBIG_HOLDOUT = ["정성", "온기", "진심", "마음"]
AMBIG_PERSON_TEMPLATES = [
    ("기본", "{Nsubj} 요즘 연락이 뜸하네"),
    ("목적격", "이모가 {Nobj} 많이 예뻐하셔"),
    ("호격", "{Nvoc}, 같이 놀자"),
]
AMBIG_NONPERSON: dict[str, list[str]] = {
    "행복": ["행복이 가득한 하루였다", "소소한 행복을 느꼈다"],
    "자유": ["자유를 만끽했다", "자유로운 분위기였다"],
    "믿음": ["믿음이 깊어졌다", "서로의 믿음을 확인했다"],
    "인연": ["인연이 닿았다", "귀한 인연이라고 했다"],
    "온정": ["온정을 베풀었다", "따뜻한 온정이 느껴졌다"],
    "나눔": ["나눔을 실천했다", "작은 나눔이 큰 힘이 됐다"],
    "다짐": ["다짐을 새로 했다", "굳은 다짐을 적어봤다"],
    "기억": ["기억이 희미해졌다", "좋은 기억으로 남았다"],
    "정성": ["정성을 다했다", "정성 가득한 음식이었다"],
    "온기": ["온기가 남아 있었다", "따뜻한 온기를 느꼈다"],
    "진심": ["진심이 통했다", "진심 어린 사과였다"],
    "마음": ["마음이 편안해졌다", "고마운 마음을 전했다"],
}
AMBIG_HOLDOUT_TEMPLATES = ["{N} 얘기 들으니 반가웠어", "{Nvoc}, 조심해서 와"]


# --------------------------------------------------------------------------- 스트레스(신규 12단어)
STRESS_TRAIN = ["뒤주", "다듬잇돌", "물레방아", "귀틀집", "흙벽", "초가지붕", "곳간문", "장독"]
STRESS_HOLDOUT = ["멍석", "절구공이", "사랑채", "빨랫방망이"]
STRESS_PERSON_TEMPLATES = [
    ("기본", "{Nsubj} 요즘 뜸하네"),
    ("목적격", "큰아버지가 {Nobj} 많이 예뻐하셨대"),
    ("호격", "{Nvoc}, 나랑 같이 가자"),
]
STRESS_NONPERSON: dict[str, list[str]] = {
    "뒤주": ["뒤주에 쌀을 채웠다"], "다듬잇돌": ["다듬잇돌 위에 옷을 올렸다"],
    "물레방아": ["물레방아가 돌아갔다"], "귀틀집": ["귀틀집이 산속에 있었다"],
    "흙벽": ["흙벽이 갈라졌다"], "초가지붕": ["초가지붕을 새로 엮었다"],
    "곳간문": ["곳간문을 열었다"], "장독": ["장독 뚜껑을 덮었다"],
    "멍석": ["멍석을 마당에 깔았다"], "절구공이": ["절구공이로 곡식을 찧었다"],
    "사랑채": ["사랑채에 손님을 모셨다"], "빨랫방망이": ["빨랫방망이로 빨래를 두드렸다"],
}


# --------------------------------------------------------------------------- 1글자(GIVEN1 재사용, 새 문형)
ONE_CHAR_TEMPLATES = ["{N} 요즘 잘 지내?", "{Nvoc}, 오늘 늦는대"]


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
    "apps/api/app/shared/privacy/eval/style_v13_locked/v13_locked.jsonl",
    "apps/api/app/shared/privacy/eval/style_r7_dev/r7_dev_realistic.jsonl",
    "apps/api/app/shared/privacy/eval/style_r7_dev/r7_dev_1char.jsonl",
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
    r"C:\Users\<user>\venvs\ner-runs\data_r7a_phrase\r7a_train_extra.jsonl",
    r"C:\Users\<user>\venvs\ner-runs\data_r7a_phrase\r7a_val_extra.jsonl",
    r"C:\Users\<user>\venvs\ner-runs\data_r7a_phrase\r7a_heldout_eval.jsonl",
    r"C:\Users\<user>\venvs\ner-runs\data_r7_phrase\r7_train_extra.jsonl",
    r"C:\Users\<user>\venvs\ner-runs\data_r7_phrase\r7_val_extra.jsonl",
    r"C:\Users\<user>\venvs\ner-runs\data_r7_phrase\r7_heldout_eval.jsonl",
]
ALL_PRIOR_WORDS = r7data.ALL_PRIOR_WORDS


def _ambiguity_probe_words():
    p = ROOT / "apps/api/app/shared/privacy/eval/style_ambiguity_probe/build_ambiguity_probe.py"
    spec = importlib.util.spec_from_file_location("build_ambiguity_probe", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return set(mod.GROUPS.keys())


def leak_check(all_texts, new_words):
    frozen = set()
    for rel in FROZEN_JSONLS:
        pth = ROOT / rel
        for line in pth.read_text(encoding="utf-8").splitlines():
            if line.strip():
                frozen.add(json.loads(line)["text"].replace(" ", ""))
    for path_str in FROZEN_DATA_DIR_FILES:
        pth = Path(path_str)
        if pth.exists():
            for line in pth.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    frozen.add(json.loads(line)["text"].replace(" ", ""))
    new = {t.replace(" ", "") for t in all_texts}
    overlap = new & frozen
    print(f"동결 데이터와 문장 겹침: {len(overlap)}개", overlap or "(없음, OK)")
    assert not overlap

    cae_path = ROOT / "apps/api/scripts/check_adjacent_entity_merge.py"
    cae_text = cae_path.read_text(encoding="utf-8").replace(" ", "")
    cae_hits = [t for t in all_texts if t.replace(" ", "") in cae_text]
    print(f"check_adjacent_entity_merge.py 문구 겹침: {len(cae_hits)}개", cae_hits or "(없음, OK)")
    assert not cae_hits

    given_pools = set(synth.GIVEN1) | set(synth.GIVEN2) | set(synth.NICKS) | set(synth.WORDLIKE_GIVEN)
    amb30 = _ambiguity_probe_words()
    r7b_words = set(r7data.TRAIN_WORDS_B) | set(r7data.HOLDOUT_WORDS_B)
    word_overlap = set(new_words) & (given_pools | amb30 | r7b_words | ALL_PRIOR_WORDS)
    print(f"v14 신규 단어가 기존 전체 풀과 겹침: {len(word_overlap)}개", word_overlap or "(없음, OK)")
    assert not word_overlap

    dup = len(all_texts) - len(set(all_texts))
    print(f"내부 중복 문장: {dup}개")
    assert dup == 0


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def build_realistic():
    rows = []
    rid = 0
    for cat, tpl in SINGLE_TEMPLATES:
        for w in USE_NAMES + USE_NICKS:
            label = "PERSON" if w in USE_NAMES else "PERSON_ALIAS"
            text = fill(tpl, w)
            rid += 1
            s, e = find_span(text, w)
            rows.append(row(f"v14-r-{rid:04d}", text, [(s, e, label)], cat, "실사용"))
    pairs = [(USE_NAMES[i], USE_NAMES[i + 1]) for i in range(0, 8, 2)]
    pairs += [(USE_NAMES[8], USE_NICKS[0]), (USE_NAMES[9], USE_NICKS[1])]
    for i, (a, b) in enumerate(pairs):
        tpl = PAIR_TEMPLATES[i % len(PAIR_TEMPLATES)]
        ab = a + b
        text = tpl.format(AB=ab)
        label_b = "PERSON" if b in USE_NAMES else "PERSON_ALIAS"
        rid += 1
        rows.append(row(f"v14-r-{rid:04d}", text, [(0, len(a), "PERSON"), (len(a), len(a) + len(b), label_b)], "무공백인접", "실사용"))
    neg_words = synth.GENERIC_NOUNS[10:20] + synth.GENERIC_PLACES[10:20]
    neg_tpls = ["{w} 때문에 정신없었어", "{w}을 확인했어", "{w}이 늦게 왔어"]
    for i, w in enumerate(neg_words):
        text = neg_tpls[i % len(neg_tpls)].format(w=w)
        rid += 1
        rows.append(row(f"v14-r-{rid:04d}", text, [], "background", "실사용"))
    return rows


def build_ambiguity():
    rows = []
    rid = 0
    for cat, tpl in AMBIG_PERSON_TEMPLATES:
        for w in AMBIG_TRAIN + AMBIG_HOLDOUT:
            text = fill(tpl, w)
            rid += 1
            s, e = find_span(text, w)
            rows.append(row(f"v14-a-{rid:04d}", text, [(s, e, "PERSON")], cat, "중의성"))
    for w, sents in AMBIG_NONPERSON.items():
        for text in sents:
            rid += 1
            rows.append(row(f"v14-a-{rid:04d}", text, [], "nonperson", "중의성"))
    return rows


def build_stress():
    rows = []
    rid = 0
    for cat, tpl in STRESS_PERSON_TEMPLATES:
        for w in STRESS_TRAIN + STRESS_HOLDOUT:
            text = fill(tpl, w)
            rid += 1
            s, e = find_span(text, w)
            rows.append(row(f"v14-s-{rid:04d}", text, [(s, e, "PERSON")], cat, "스트레스"))
    for w, sents in STRESS_NONPERSON.items():
        for text in sents:
            rid += 1
            rows.append(row(f"v14-s-{rid:04d}", text, [], "nonperson", "스트레스"))
    return rows


def build_one_char():
    rows = []
    rid = 0
    for i, n in enumerate(synth.GIVEN1):
        for tpl in ONE_CHAR_TEMPLATES:
            text = fill(tpl, n)
            rid += 1
            s, e = find_span(text, n)
            rows.append(row(f"v14-1c-{rid:04d}", text, [(s, e, "PERSON")], "1글자이름", "1글자"))
    return rows


def main():
    realistic = build_realistic()
    ambiguity = build_ambiguity()
    stress = build_stress()
    one_char = build_one_char()

    new_words = AMBIG_TRAIN + AMBIG_HOLDOUT + STRESS_TRAIN + STRESS_HOLDOUT
    all_texts = [r["text"] for r in realistic + ambiguity + stress + one_char]
    leak_check(all_texts, new_words)

    write(HERE / "v14_realistic.jsonl", realistic)
    write(HERE / "v14_ambiguity.jsonl", ambiguity)
    write(HERE / "v14_stress.jsonl", stress)
    write(HERE / "v14_one_char.jsonl", one_char)
    print(f"v14 실사용 {len(realistic)}행 / 중의성 {len(ambiguity)}행 / 스트레스 {len(stress)}행 / 1글자 {len(one_char)}행")


if __name__ == "__main__":
    main()
