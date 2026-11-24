"""R7 — 실사용 중심 보강(가족 대화에서 실제 쓰이는 이름·별명 중심). R6(base+R3+R4+R5+R6) 위에
이어붙인다. `ner.py`/`merge_ner`는 전혀 안 건드린다.

사용자가 지정한 R7 우선순위를 그대로 따른다(R7 지시사항 2절):
- A. 실사용 데이터: 기존 이름·별명 풀(`synth.GIVEN2`/`synth.NICKS`/`synth.KINS`)만 쓴다 — 새
  이름을 지어내지 않는다. 62~64절에서 확인한 "무공백 인접 두 사람" 절단·과잉병합 문제를 직접
  겨냥해, 공백 없이 붙은 두 사람(PERSON+PERSON, PERSON+PERSON_ALIAS)을 각자 정확한 경계로 라벨링한
  학습 신호를 추가한다. 일정·학교·가족연락 문맥과 짧은 메신저체도 추가한다.
- B. 중의성 대비 데이터: 48절 `ambiguity_probe`(읽기 전용 진단셋, 학습 금지 — 이 스크립트가
  `GROUPS` 단어와 겹치지 않는지 직접 검사한다)와 `synth.WORDLIKE_GIVEN`(기존 학습용 중의성 이름
  풀)과도 겹치지 않는 **새 중의성 단어 8개**를 골라 PERSON/비PERSON 대비 문장을 만든다.
- C. 스트레스(복합명사형) 데이터는 **추가하지 않는다** — R6까지의 기존 보강분은 그대로 두되, 이
  라운드에서 새로 늘리지 않는다(지시사항: "그 유형만을 맞히기 위해 전체 모델을 최적화하지 말 것").

이름/별명 자체는 기존 풀을 그대로 재사용하므로 학습 단어 중복 검사 대상이 아니다(여러 라운드가
항상 이 풀을 공유해 왔다). 새로 추가하는 건 (1) 문장 템플릿 텍스트(기존 평가셋·이전 라운드 데이터·
`check_adjacent_entity_merge.py` 문장과 안 겹치는지 확인), (2) 8개 신규 중의성 단어(기존 모든
이름·중의성 풀과 안 겹치는지 확인)뿐이다.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import synth  # noqa: E402

# --------------------------------------------------------------------------- 공통 헬퍼


def has_batchim(ch: str) -> bool:
    return (ord(ch) - 0xAC00) % 28 != 0


def obj(w: str) -> str:
    return w + ("을" if has_batchim(w[-1]) else "를")


def subj(w: str) -> str:
    return w + ("이" if has_batchim(w[-1]) else "가")


def voc(w: str) -> str:
    return w + ("아" if has_batchim(w[-1]) else "야")


def dat(w: str) -> str:  # 여격 "~한테"는 받침 상관없이 그대로 붙는다
    return w + "한테"


def fill(tpl: str, n: str) -> str:
    return tpl.format(N=n, Nobj=obj(n), Nsubj=subj(n), Nvoc=voc(n), Ndat=dat(n))


def find_span(text: str, token: str) -> tuple[int, int]:
    idx = text.find(token)
    if idx < 0:
        raise ValueError(f"토큰을 못 찾음: {token!r} in {text!r}")
    return idx, idx + len(token)


def make_span_row(rid: str, text: str, spans: list[tuple[int, int, str]], category: str, split: str) -> dict:
    return {
        "id": rid, "text": text,
        "spans": [{"label": label, "start": s, "end": e} for s, e, label in spans],
        "source": "synth", "template_id": f"r7-{category}", "category": category, "split": split, "augment": True,
    }


def make_neg_row(rid: str, text: str, category: str, split: str) -> dict:
    return {
        "id": rid, "text": text, "spans": [],
        "source": "synth", "template_id": f"r7-{category}-neg", "category": category, "split": split, "augment": True,
    }


def make_eval_row(rid: str, text: str, spans: list[tuple[int, int, str]], category: str) -> dict:
    return {
        "id": rid, "category": category, "text": text,
        "spans": [{"kind": label, "start": s, "end": e} for s, e, label in spans],
        "known_names": [], "template_id": category,
    }


# --------------------------------------------------------------------------- A. 실사용 — 이름 풀
# 기존 synth.GIVEN2/NICKS 에서만 뽑는다(새 이름을 지어내지 않음). 홀드아웃은 "이 라운드의 새 패턴
# (무공백 인접, 새 문형)"을 전혀 학습하지 않은 이름으로, 일반화 확인용이다(기존 베이스라인 학습에는
# GIVEN2 전체가 이미 쓰이고 있으므로, 이름 자체를 처음 보는 건 아니다 — "이 패턴"을 처음 보는 것).
TRAIN_NAMES = ["민준", "서윤", "지호", "하은", "도윤", "수아", "예준", "하린", "시우", "가은",
               "준서", "나윤", "우진", "소율", "태민", "은서", "현우", "지안", "윤서", "재원"]
HOLDOUT_NAMES = ["지우", "하윤", "채원", "연우", "시현", "건우", "승민", "정우", "도현", "해원"]
TRAIN_NICKS = ["콩이", "밤톨이", "몽이", "쭈니", "찡이", "초롱이"]
HOLDOUT_NICKS = ["뽀미", "달봉이"]
ALIAS_KIN = ["형", "누나", "오빠", "언니"]  # 이름/별명 + 가족호칭 = PERSON_ALIAS (LABELING_GUIDE)

assert not (set(TRAIN_NAMES) & set(HOLDOUT_NAMES))
assert not (set(TRAIN_NICKS) & set(HOLDOUT_NICKS))
for _n in TRAIN_NAMES + HOLDOUT_NAMES:
    assert _n in synth.GIVEN2, _n
for _n in TRAIN_NICKS + HOLDOUT_NICKS:
    assert _n in synth.NICKS, _n


def alias(name: str, kin: str) -> str:
    return name + kin  # "민준" + "형" -> "민준형" (LABELING_GUIDE: 이름+가족호칭 = PERSON_ALIAS)


# A-1. 무공백 인접 두 사람(62~64절 절단·과잉병합 대응). check_adjacent_entity_merge.py 와는 다른
# 문장(아래 템플릿은 그 스크립트의 CASES 문구와 겹치지 않게 새로 썼다).
PAIR_TEMPLATES_PP = [  # (A,B) 둘 다 PERSON, 공백 없이 붙임
    "{AB}가 숙제 같이 끝냈대",
    "선생님이 {AB}를 집에 데려다줬어",
    "{AB}한테 내일 준비물 알려줬어",
]
PAIR_TEMPLATES_PA = [  # A=PERSON, B=PERSON_ALIAS(별명), 공백 없이 붙임
    "{AB}가 놀이터에서 만났대",
    "{AB}한테 간식 나눠줬어",
]
TRIPLE_TEMPLATE = "{ABC}가 다 같이 영화 보러 갔어"  # 3인 무공백(지시사항: 인접 엔티티 회귀 커버)


def pair_rows(rid_start: int, train_a: list[str], train_b: list[str], templates: list[str],
              label_a: str, label_b: str, category: str) -> tuple[list[dict], int]:
    rows = []
    rid = rid_start
    n = min(len(train_a), len(train_b), len(templates) * 4)
    pairs = list(zip(train_a, train_b[::-1]))[: max(6, n // 2)]
    for i, (a, b) in enumerate(pairs):
        tpl = templates[i % len(templates)]
        ab = a + b
        text = tpl.format(AB=ab)
        s1, e1 = 0, len(a)
        s2, e2 = len(a), len(a) + len(b)
        split = "val" if i % 5 == 4 else "train"
        rid += 1
        rows.append(make_span_row(f"r7-pos-{rid:04d}", text, [(s1, e1, label_a), (s2, e2, label_b)], category, split))
    return rows, rid


# A-2. 일정·학교·가족연락 문맥(자연스러운 대화, 기존 R6 템플릿과 다른 문형)
SCHEDULE_TEMPLATES = [
    ("기본", "train", "{N} 학원 몇 시에 끝나는지 물어봤어"),
    ("기본", "val", "{N} 체육대회 날짜 확인해줘"),
    ("목적격", "train", "{Nobj} 내일 학교 준비물 챙기라고 했어"),
    ("목적격", "val", "선생님이 {Nobj} 상담 때문에 찾으셨어"),
    ("여격", "train", "{Ndat} 학교 끝나면 바로 오라고 했어"),
    ("여격", "val", "{Ndat} 병원 예약 잡아놨다고 전해줬어"),
    ("호격", "train", "{Nvoc}, 숙제 다 했는지 확인해볼게"),
    ("호격", "val", "{Nvoc}, 성적표 나왔는지 선생님한테 물어봤어"),
]

# A-3. 짧은 메신저체(존댓말·조사 생략, 극도로 짧은 실사용 문장)
SHORT_MSG_TEMPLATES = [
    ("messenger", "train", "{N} 뭐해"),
    ("messenger", "train", "{N} 밥 먹었어?"),
    ("messenger", "val", "{N} 어디야"),
    ("messenger", "train", "{N} 오늘 늦는대"),
    ("messenger", "val", "{N} 자나봐"),
]


# --------------------------------------------------------------------------- B. 중의성 대비 — 신규 단어
# ambiguity_probe(48절, 읽기 전용)와 synth.WORDLIKE_GIVEN(기존 학습용 중의성 풀) 둘 다와 겹치지
# 않는 새 단어만 쓴다(아래 leak_check 가 동적으로 두 곳 다 불러와 검사한다).
TRAIN_WORDS_B = ["유리", "소라", "보석", "소망", "기쁨", "보배", "다래", "겨울"]
HOLDOUT_WORDS_B = ["매화", "난초", "산호", "자두"]

PERSON_TEMPLATES_B = [
    ("목적격", "train", "오늘 {Nobj} 학원 끝나고 만났어"),
    ("여격", "train", "{Ndat} 숙제 다 했냐고 물어봤어"),
    ("호격", "val", "{Nvoc}, 같이 놀이터 가자"),
]
NONPERSON_TRAIN_WORDS_B: dict[str, list[tuple[str, str]]] = {
    "유리": [("유리가 깨졌다", "train"), ("유리창을 깨끗이 닦았다", "train"), ("유리 조각을 조심히 치웠다", "val")],
    "소라": [("소라를 바닷가에서 주웠다", "train"), ("소라 껍데기가 예뻤다", "train"), ("소라 속에서 소리가 들렸다", "val")],
    "보석": [("보석이 반짝반짝 빛났다", "train"), ("보석함을 열어봤다", "train"), ("보석가게에 들렀다", "val")],
    "소망": [("소망이 이루어졌다", "train"), ("간절한 소망을 빌었다", "train"), ("새해 소망을 적어봤다", "val")],
    "기쁨": [("기쁨이 가득했다", "train"), ("기쁨의 눈물을 흘렸다", "train"), ("합격 소식에 기쁨을 감추지 못했다", "val")],
    "보배": [("그건 집안의 보배였다", "train"), ("보배 같은 존재라고 했다", "train"), ("보배롭게 여겼다", "val")],
    "다래": [("다래를 산에서 땄다", "train"), ("다래 넝쿨이 담장을 덮었다", "train"), ("다래 맛이 새콤달콤했다", "val")],
    "겨울": [("겨울이 성큼 다가왔다", "train"), ("겨울 날씨가 많이 추웠다", "train"), ("겨울옷을 꺼내 입었다", "val")],
}
HOLDOUT_TEMPLATES_B = [
    "{N} 얘기 들으니 반가웠어",
    "{Nvoc}, 조심해서 와",
]
NONPERSON_HOLDOUT_WORDS_B: dict[str, list[str]] = {
    "매화": ["매화가 활짝 피었다", "매화 향이 은은하게 퍼졌다"],
    "난초": ["난초에 물을 줬다", "난초 잎이 길게 자랐다"],
    "산호": ["산호가 하얗게 변했다", "바다 속 산호초를 봤다"],
    "자두": ["자두를 한 입 베어 물었다", "자두가 새콤달콤했다"],
}

for _w in TRAIN_WORDS_B + HOLDOUT_WORDS_B:
    assert _w not in synth.GIVEN1 and _w not in synth.GIVEN2 and _w not in synth.NICKS and _w not in synth.WORDLIKE_GIVEN, _w


# --------------------------------------------------------------------------- 중복 검사(leak_check)

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
]
# R6/R7a 의 ALL_PRIOR_WORDS 에 R7a 자신의 단어(당시엔 미포함) + v13 20개 단어를 더한 버전.
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
    # R7a 자신의 단어(10 train + 4 holdout)
    "연탄불", "아궁이", "짚더미", "가마솥", "다듬이", "문고리", "빗자루", "두레박", "디딜방아", "베틀",
    "자갈밭", "풀숲", "등불", "촛대",
    # v13 20개 단어
    "빨래집게", "다락문", "부엌문", "장작간", "처마밑", "기와지붕", "외양채", "토담", "장독뚜껑", "바지랑대",
    "삼태기", "쇠스랑", "삼베", "누룽지", "엿가락", "다식", "약과", "떡살", "갈퀴", "맷돌",
}


def _load_ambiguity_probe_words() -> set[str]:
    p = ROOT / "apps/api/app/shared/privacy/eval/style_ambiguity_probe/build_ambiguity_probe.py"
    spec = importlib.util.spec_from_file_location("build_ambiguity_probe", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return set(mod.GROUPS.keys())


def leak_check(all_texts: list[str]) -> None:
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
    print(f"동결 평가셋/R3~R7a 데이터와 문장 겹침: {len(overlap_text)}개", overlap_text or "(없음, OK)")
    assert not overlap_text, "동결 데이터와 문장이 겹친다"

    # check_adjacent_entity_merge.py(FROZEN_JSONLS 밖, 스크립트 내부 상수)와도 수동 비교
    cae_path = ROOT / "apps/api/scripts/check_adjacent_entity_merge.py"
    cae_text = cae_path.read_text(encoding="utf-8")
    cae_hits = [t for t in all_texts if t.replace(" ", "") and t.replace(" ", "") in cae_text.replace(" ", "")]
    print(f"check_adjacent_entity_merge.py 문구와 겹침: {len(cae_hits)}개", cae_hits or "(없음, OK)")
    assert not cae_hits, "인접엔티티 스트레스 테스트 문장을 그대로 학습에 복사했다"

    ambiguity_words = _load_ambiguity_probe_words()
    wordlike_given = set(synth.WORDLIKE_GIVEN)
    b_words = set(TRAIN_WORDS_B) | set(HOLDOUT_WORDS_B)
    word_overlap = b_words & (ambiguity_words | wordlike_given | ALL_PRIOR_WORDS | set(synth.GIVEN1) | set(synth.GIVEN2) | set(synth.NICKS))
    print(f"R7 중의성 신규 단어가 기존 풀(ambiguity_probe/WORDLIKE_GIVEN/이전 라운드)과 겹침: {len(word_overlap)}개", word_overlap or "(없음, OK)")
    assert not word_overlap, "R7 중의성 단어가 기존 풀과 겹친다"

    assert not (set(TRAIN_WORDS_B) & set(HOLDOUT_WORDS_B))
    print("중의성 학습/holdout 단어 겹침: 없음, OK")


def write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--blocks", default="p,r,amb",
                     help="쉼표로 구분한 포함 블록: p(A-1 무공백 인접 쌍), r(A-2+A-3 실사용 단일개체), "
                          "amb(B 중의성 대비). 기본값은 전체(R7과 동일)")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blocks = set(a.blocks.split(","))

    all_texts: list[str] = []
    train_rows, val_rows = [], []
    rid = 0

    if "p" in blocks:
        # A-1. 무공백 인접 쌍 (PERSON+PERSON, PERSON+PERSON_ALIAS)
        rows, rid = pair_rows(rid, TRAIN_NAMES, TRAIN_NAMES, PAIR_TEMPLATES_PP, "PERSON", "PERSON", "pair_pp")
        for r in rows:
            all_texts.append(r["text"])
            (train_rows if r["split"] == "train" else val_rows).append(r)

        alias_names = [alias(n, k) for n, k in zip(TRAIN_NAMES[:8], (ALIAS_KIN * 2)[:8])]
        rows, rid = pair_rows(rid, TRAIN_NAMES[8:16], alias_names, PAIR_TEMPLATES_PA, "PERSON", "PERSON_ALIAS", "pair_pa")
        for r in rows:
            all_texts.append(r["text"])
            (train_rows if r["split"] == "train" else val_rows).append(r)

        # 별명+별명(서로 다른 두 사람, 둘 다 PERSON_ALIAS) — 61~62절 "민준이형지호형이"류 과잉병합
        # 방지용, 문구는 다르게 구성
        alias_pairs = [(alias(TRAIN_NAMES[16], "형"), alias(TRAIN_NAMES[17], "누나")),
                       (alias(TRAIN_NAMES[18], "오빠"), alias(TRAIN_NAMES[19], "언니"))]
        for i, (a1, a2) in enumerate(alias_pairs):
            text = f"{a1}{a2}가 같이 숙제했대"
            s1, e1 = 0, len(a1)
            s2, e2 = len(a1), len(a1) + len(a2)
            split = "val" if i == 1 else "train"
            rid += 1
            row = make_span_row(f"r7-pos-{rid:04d}", text, [(s1, e1, "PERSON_ALIAS"), (s2, e2, "PERSON_ALIAS")], "pair_aa", split)
            all_texts.append(text)
            (train_rows if split == "train" else val_rows).append(row)

        # 3인 무공백
        triple_names = TRAIN_NAMES[:3]
        abc = "".join(triple_names)
        text = TRIPLE_TEMPLATE.format(ABC=abc)
        spans = []
        offset = 0
        for n in triple_names:
            spans.append((offset, offset + len(n), "PERSON"))
            offset += len(n)
        rid += 1
        row = make_span_row(f"r7-pos-{rid:04d}", text, spans, "triple_nospace", "train")
        all_texts.append(text)
        train_rows.append(row)

    if "r" in blocks:
        # A-2. 일정·학교·가족연락 문맥
        for cat, split, tpl in SCHEDULE_TEMPLATES:
            for w in TRAIN_NAMES[:12] + TRAIN_NICKS:
                label = "PERSON" if w in TRAIN_NAMES else "PERSON_ALIAS"
                text = fill(tpl, w)
                all_texts.append(text)
                rid += 1
                row = make_span_row(f"r7-pos-{rid:04d}", text, [(*find_span(text, w), label)], cat, split)
                (train_rows if split == "train" else val_rows).append(row)

        # A-3. 짧은 메신저체
        for cat, split, tpl in SHORT_MSG_TEMPLATES:
            for w in TRAIN_NAMES[12:20] + TRAIN_NICKS:
                label = "PERSON" if w in TRAIN_NAMES else "PERSON_ALIAS"
                text = fill(tpl, w)
                all_texts.append(text)
                rid += 1
                row = make_span_row(f"r7-pos-{rid:04d}", text, [(*find_span(text, w), label)], cat, split)
                (train_rows if split == "train" else val_rows).append(row)

    if "amb" in blocks:
        # B. 중의성 대비(신규 8단어)
        for cat, split, tpl in PERSON_TEMPLATES_B:
            for w in TRAIN_WORDS_B:
                text = fill(tpl, w)
                all_texts.append(text)
                rid += 1
                row = make_span_row(f"r7-pos-{rid:04d}", text, [(*find_span(text, w), "PERSON")], cat, split)
                (train_rows if split == "train" else val_rows).append(row)

        for w, sents in NONPERSON_TRAIN_WORDS_B.items():
            for text, split in sents:
                all_texts.append(text)
                rid += 1
                row = make_neg_row(f"r7-neg-{rid:04d}", text, "ambiguity_neg", split)
                (train_rows if split == "train" else val_rows).append(row)

    print(f"포함 블록: {sorted(blocks)} | train {len(train_rows)}행 / val {len(val_rows)}행")
    leak_check(all_texts)

    write(out / "r7_train_extra.jsonl", train_rows)
    write(out / "r7_val_extra.jsonl", val_rows)
    print(f"학습 추가 train: {len(train_rows)}행, val: {len(val_rows)}행 -> {out}")

    # ---------- held-out(학습 미사용, 일반화 확인 전용) ----------
    holdout_rows = []
    rid = 0

    # 이름 홀드아웃: 이 라운드의 새 패턴(무공백 인접·새 문형)을 전혀 학습하지 않은 이름으로 확인
    pairs = [(HOLDOUT_NAMES[0], HOLDOUT_NAMES[1]), (HOLDOUT_NAMES[2], HOLDOUT_NAMES[3])]
    for a, b in pairs:
        text = f"{a}{b}가 숙제 같이 끝냈대"
        rid += 1
        holdout_rows.append(make_eval_row(f"r7-ho-{rid:04d}", text, [(0, len(a), "PERSON"), (len(a), len(a) + len(b), "PERSON")], "realistic_holdout"))
    a, b = HOLDOUT_NAMES[4], alias(HOLDOUT_NAMES[5], "형")
    text = f"{a}{b}한테 간식 나눠줬어"
    rid += 1
    holdout_rows.append(make_eval_row(f"r7-ho-{rid:04d}", text, [(0, len(a), "PERSON"), (len(a), len(a) + len(b), "PERSON_ALIAS")], "realistic_holdout"))
    for w in HOLDOUT_NAMES[6:] + HOLDOUT_NICKS:
        label = "PERSON" if w in HOLDOUT_NAMES else "PERSON_ALIAS"
        for tpl in ["{N} 학원 끝나고 전화해봐", "{Nvoc}, 밥 먹었어?"]:
            text = fill(tpl, w)
            rid += 1
            holdout_rows.append(make_eval_row(f"r7-ho-{rid:04d}", text, [(*find_span(text, w), label)], "realistic_holdout"))

    # 중의성 홀드아웃(4단어)
    for tpl in HOLDOUT_TEMPLATES_B:
        for w in HOLDOUT_WORDS_B:
            text = fill(tpl, w)
            rid += 1
            holdout_rows.append(make_eval_row(f"r7-ho-{rid:04d}", text, [(*find_span(text, w), "PERSON")], "ambiguity_holdout"))
    for w, sents in NONPERSON_HOLDOUT_WORDS_B.items():
        for text in sents:
            rid += 1
            holdout_rows.append(make_eval_row(f"r7-ho-{rid:04d}", text, [], "ambiguity_holdout_neg"))

    write(out / "r7_heldout_eval.jsonl", holdout_rows)
    print(f"held-out 평가 전용(학습 미사용): {len(holdout_rows)}행 -> {out}")


if __name__ == "__main__":
    main()
