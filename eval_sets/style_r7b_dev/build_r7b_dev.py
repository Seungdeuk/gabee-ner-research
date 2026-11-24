"""R7b 개발셋 — v14·R7 학습/홀드아웃·style_r7_dev·48절 ambiguity_probe·v9~v13과 이름·단어·문장이
전혀 겹치지 않는 완전 신규 데이터. P+Amb 다중 시드 검증과 R7b 후보 선정에만 쓴다(모델 선정용
개발셋 — v15과는 별개, v15는 후보가 선정된 뒤에만 새로 만든다).

4개 평가 영역(사용자 지시 그대로):
1. 일반 가족 대화·한국어 이름/별명(실사용)
2. 이름과 일반 명사가 겹치는 중의성 문맥
3. 무공백 인접 이름·별명 + 조사 결합
4. 신규 복합명사형 이름 스트레스 문맥
+ 1글자 이름(GIVEN1 재사용, 정책 지표 전용, 통합 점수에 안 넣음)
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

sys.path.insert(0, str(ROOT / "apps" / "api" / "app" / "shared" / "privacy" / "eval" / "style_v14_locked"))
import build_v14_locked as v14  # noqa: E402


def has_batchim(ch): return (ord(ch) - 0xAC00) % 28 != 0
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


# --------------------------------------------------------------------------- 1. 실사용(신규 이름)
USE_NAMES = ["시율", "라온", "로운", "가온", "노아", "이든", "하진", "도겸", "규민", "세민", "영호", "병철", "기훈", "진우"]
USE_NICKS = ["똘이", "몽글이", "호야"]
_used = set(r7data.TRAIN_NAMES) | set(r7data.HOLDOUT_NAMES) | set(v14.USE_NAMES)
_used_n = set(r7data.TRAIN_NICKS) | set(r7data.HOLDOUT_NICKS) | set(v14.USE_NICKS)
assert not (set(USE_NAMES) & _used), set(USE_NAMES) & _used
assert not (set(USE_NICKS) & _used_n), set(USE_NICKS) & _used_n
for n in USE_NAMES:
    assert n in synth.GIVEN2, n
for n in USE_NICKS:
    assert n in synth.NICKS, n

SINGLE_TEMPLATES = [
    ("기본", "{Nsubj} 요즘 연락이 잘 안 돼"),
    ("목적격", "이모부가 {Nobj} 많이 챙겨주셔"),
    ("비교서술", "{Nsubj} 형제 중에 제일 차분해"),
    ("여격", "{Ndat} 시험 잘 봤는지 물어봤어"),
    ("호격", "{Nvoc}! 저녁 먹자"),
]

# --------------------------------------------------------------------------- 3. 무공백 인접 + 조사 결합
PAIR_TEMPLATES_PARTICLE = [
    ("와", "{A}와 {B}가 같이 숙제했대"),       # 공백 있는 통제문(비교용)
    ("무공백기본", "{AB}가 같이 숙제했대"),
    ("무공백랑", "{AB}랑 나도 같이 놀고 싶었어"),
    ("무공백한테", "{AB}한테 숙제 다 했냐고 물어봤어"),
    ("무공백목적격", "선생님이 {AB}를 집에 데려다줬어"),
]


# --------------------------------------------------------------------------- 2. 중의성(신규 12단어)
AMBIG_TRAIN = ["여유", "고요", "웃음", "다복", "뜨락", "인정", "배려", "도움"]
AMBIG_HOLDOUT = ["성원", "격려", "끈기", "여운"]
AMBIG_PERSON_TEMPLATES = [
    ("기본", "{Nsubj} 요즘 통 소식이 없네"),
    ("목적격", "삼촌이 {Nobj} 유독 챙기셔"),
    ("호격", "{Nvoc}, 이리 와서 앉아"),
]
AMBIG_NONPERSON = {
    "여유": ["여유가 생겼다", "마음의 여유를 가졌다"],
    "고요": ["고요가 흘렀다", "방 안이 고요했다"],
    "웃음": ["웃음이 터졌다", "웃음소리가 들렸다"],
    "다복": ["다복한 가정이었다", "다복을 빌었다"],
    "뜨락": ["뜨락에 꽃이 피었다", "뜨락을 거닐었다"],
    "인정": ["인정이 많았다", "따뜻한 인정을 느꼈다"],
    "배려": ["배려가 느껴졌다", "작은 배려에 감동했다"],
    "도움": ["도움이 됐다", "큰 도움을 받았다"],
    "성원": ["성원에 감사했다", "뜨거운 성원을 보냈다"],
    "격려": ["격려가 힘이 됐다", "따뜻한 격려를 건넸다"],
    "끈기": ["끈기가 필요했다", "끈기 있게 버텼다"],
    "여운": ["여운이 남았다", "깊은 여운을 느꼈다"],
}
AMBIG_HOLDOUT_TEMPLATES = ["{N} 얘기 들으니 반가웠어", "{Nvoc}, 조심해서 와"]


# --------------------------------------------------------------------------- 4. 스트레스(신규 12단어)
STRESS_TRAIN = ["짚신", "멍에", "쟁기", "도리깨", "가마니", "똬리", "우물가", "장작더미"]
STRESS_HOLDOUT = ["지붕밑", "대나무밭", "물지게", "싸리문"]
STRESS_PERSON_TEMPLATES = [
    ("기본", "{Nsubj} 요즘 뜸하네"),
    ("목적격", "큰아버지가 {Nobj} 많이 예뻐하셨대"),
    ("호격", "{Nvoc}, 나랑 같이 가자"),
]
STRESS_NONPERSON = {
    "짚신": ["짚신을 삼았다"], "멍에": ["멍에를 짊어졌다"], "쟁기": ["쟁기로 밭을 갈았다"],
    "도리깨": ["도리깨로 콩을 털었다"], "가마니": ["가마니에 곡식을 담았다"], "똬리": ["똬리를 틀고 앉았다"],
    "우물가": ["우물가에서 물을 길었다"], "장작더미": ["장작더미가 쌓여 있었다"],
    "지붕밑": ["지붕밑에 제비가 둥지를 틀었다"], "대나무밭": ["대나무밭이 바람에 흔들렸다"],
    "물지게": ["물지게를 지고 날랐다"], "싸리문": ["싸리문을 열고 들어왔다"],
}

ONE_CHAR_TEMPLATES = ["{N} 요즘 바쁘대", "{Nvoc}, 밥은 먹었어?"]

FROZEN_JSONLS = v14.FROZEN_JSONLS + [
    "apps/api/app/shared/privacy/eval/style_v14_locked/v14_realistic.jsonl",
    "apps/api/app/shared/privacy/eval/style_v14_locked/v14_ambiguity.jsonl",
    "apps/api/app/shared/privacy/eval/style_v14_locked/v14_stress.jsonl",
    "apps/api/app/shared/privacy/eval/style_v14_locked/v14_one_char.jsonl",
]
FROZEN_DATA_DIR_FILES = v14.FROZEN_DATA_DIR_FILES + [
    r"C:\Users\<user>\venvs\ner-runs\data_r7abl_p_phrase\r7_train_extra.jsonl",
    r"C:\Users\<user>\venvs\ner-runs\data_r7abl_amb_phrase\r7_train_extra.jsonl",
    r"C:\Users\<user>\venvs\ner-runs\data_r7abl_pamb_phrase\r7_train_extra.jsonl",
    r"C:\Users\<user>\venvs\ner-runs\data_r7abl_pamb_phrase\r7_val_extra.jsonl",
]


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
    v14_words = set(v14.AMBIG_TRAIN) | set(v14.AMBIG_HOLDOUT) | set(v14.STRESS_TRAIN) | set(v14.STRESS_HOLDOUT)
    word_overlap = set(new_words) & (given_pools | amb30 | r7b_words | v14_words | r7data.ALL_PRIOR_WORDS)
    print(f"신규 단어가 기존 전체 풀과 겹침: {len(word_overlap)}개", word_overlap or "(없음, OK)")
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
            rows.append(row(f"r7bdev-r-{rid:04d}", text, [(s, e, label)], cat, "실사용"))
    neg_words = synth.GENERIC_NOUNS[20:] + synth.GENERIC_PLACES[20:]
    neg_tpls = ["{w} 소식을 들었다", "{w}에 다녀왔다"]
    for i, w in enumerate(neg_words):
        text = neg_tpls[i % len(neg_tpls)].format(w=w)
        rid += 1
        rows.append(row(f"r7bdev-r-{rid:04d}", text, [], "background", "실사용"))
    return rows


def build_adjacent():
    rows = []
    rid = 0
    names = USE_NAMES[:10]
    pairs = [(names[i], names[i + 1]) for i in range(0, 10, 2)]
    pairs += [(USE_NAMES[10], USE_NICKS[0]), (USE_NAMES[11], USE_NICKS[1]), (USE_NAMES[12], USE_NICKS[2])]
    for i, (a, b) in enumerate(pairs):
        label, tpl = PAIR_TEMPLATES_PARTICLE[i % len(PAIR_TEMPLATES_PARTICLE)]
        label_b = "PERSON" if b in USE_NAMES else "PERSON_ALIAS"
        if label == "와":
            text = tpl.format(A=a, B=b)
            s1, e1 = find_span(text, a)
            s2, e2 = find_span(text[e1:], b)
            spans = [(s1, e1, "PERSON"), (e1 + s2, e1 + e2, label_b)]
        else:
            ab = a + b
            text = tpl.format(AB=ab)
            spans = [(0, len(a), "PERSON"), (len(a), len(a) + len(b), label_b)]
        rid += 1
        rows.append(row(f"r7bdev-adj-{rid:04d}", text, spans, label, "무공백인접"))
    return rows, rid


def main():
    realistic = build_realistic()
    adjacent, rid = build_adjacent()
    # 별명+별명(서로 다른 두 사람), 새 문구
    a1, a2 = "가온형", "세민누나"
    text = f"{a1}{a2}가 같이 왔대"
    rid += 1
    adjacent.append(row(f"r7bdev-adj-{rid:04d}", text, [(0, len(a1), "PERSON_ALIAS"), (len(a1), len(a1) + len(a2), "PERSON_ALIAS")], "별명무공백", "무공백인접"))

    ambiguity = []
    rid2 = 0
    for cat, tpl in AMBIG_PERSON_TEMPLATES:
        for w in AMBIG_TRAIN + AMBIG_HOLDOUT:
            text = fill(tpl, w)
            rid2 += 1
            s, e = find_span(text, w)
            ambiguity.append(row(f"r7bdev-a-{rid2:04d}", text, [(s, e, "PERSON")], cat, "중의성"))
    for w, sents in AMBIG_NONPERSON.items():
        for text in sents:
            rid2 += 1
            ambiguity.append(row(f"r7bdev-a-{rid2:04d}", text, [], "nonperson", "중의성"))

    stress = []
    rid3 = 0
    for cat, tpl in STRESS_PERSON_TEMPLATES:
        for w in STRESS_TRAIN + STRESS_HOLDOUT:
            text = fill(tpl, w)
            rid3 += 1
            s, e = find_span(text, w)
            stress.append(row(f"r7bdev-s-{rid3:04d}", text, [(s, e, "PERSON")], cat, "스트레스"))
    for w, sents in STRESS_NONPERSON.items():
        for text in sents:
            rid3 += 1
            stress.append(row(f"r7bdev-s-{rid3:04d}", text, [], "nonperson", "스트레스"))

    one_char = []
    for i, n in enumerate(synth.GIVEN1):
        for tpl in ONE_CHAR_TEMPLATES:
            text = fill(tpl, n)
            i += 1
            s, e = find_span(text, n)
            one_char.append(row(f"r7bdev-1c-{i:04d}", text, [(s, e, "PERSON")], "1글자이름", "1글자"))

    new_words = AMBIG_TRAIN + AMBIG_HOLDOUT + STRESS_TRAIN + STRESS_HOLDOUT
    all_texts = [r["text"] for r in realistic + adjacent + ambiguity + stress + one_char]
    leak_check(all_texts, new_words)

    write(HERE / "r7b_realistic.jsonl", realistic)
    write(HERE / "r7b_adjacent.jsonl", adjacent)
    write(HERE / "r7b_ambiguity.jsonl", ambiguity)
    write(HERE / "r7b_stress.jsonl", stress)
    write(HERE / "r7b_one_char.jsonl", one_char)
    print(f"실사용 {len(realistic)} / 무공백인접 {len(adjacent)} / 중의성 {len(ambiguity)} / "
          f"스트레스 {len(stress)} / 1글자 {len(one_char)}")


if __name__ == "__main__":
    main()
