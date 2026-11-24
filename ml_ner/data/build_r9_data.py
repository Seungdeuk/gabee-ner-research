"""R9 신규 학습 데이터 생성기(정책 r9-1.1). 사전 등록 NER_R9_PREREGISTRATION.md §3, §10.

문형 틀(`r9_frames/*.json`, 슬롯 `{N1:k}` 등, 명세 `FRAME_SPEC.md`)에 이름·별명·호칭·단어를 채워 학습 행을 만든다. 신규 데이터는 전부 합성이다.

정책 r9-1.1: 이름+가족호칭은 이름만 PERSON(호칭·조사·접미 `이`는 라벨 밖), PERSON_ALIAS 는 별명 자체만, 호칭만 있는 문장은 음성, 별명/일반 명사 동형은
문맥 대조쌍으로 둔다. 1글자 이름 없음. 모든 행은 변환기 `policy_issues` 검사를 통과해야 한다.

격리: **봉인 평가셋(R9 v1.2 두 파일)과 new180 은 프로그램 안에서만 읽어** 이름·문장·문형·구간 값이 겹치는 후보를 걸러내는 데만 쓴다. 내용은 출력하지 않고
건수만 보고하며 생성에 참고하지 않는다. 개발 평가셋·기존 학습 데이터와 겹치는 문장, 규격·지시서 예시 문자열도 걸러낸다.

산출(`--out-dir`, 기본 data_r9): 버킷별 학습 풀 r9_msg/errA/errB/errD/mixed_train, 실험군용 r9_E1~E4_train, r9_Ffull_train, 공통 검증 r9_val_extra,
빌드 보고서 r9_build_report.json. E5 는 E1~E4 결과 후에 만든다(`make_e5` 는 별도).
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import itertools

import build_r8_data as r8
import convert_common_to_train as conv
import synth

FRAMES = HERE / "r9_frames"
FRAME_FILES = {"msg_person", "msg_neg", "err_A", "err_B", "err_D", "mixed"}
SEAL = Path(r"C:\Users\<user>\Documents\gabee-ner-sealed-r9\sealed")
NEW180 = Path(
    r"C:\Users\<user>\Documents\gabee-ner-sealed-new180\gabee_ner_new180_v1.jsonl"
)
EXISTING_DIR = Path(r"C:\Users\<user>\venvs\ner-runs\data_r9")

# 규격·지시서·시험에 든 예시 문자열(학습 데이터에서 제외)
SPEC_EXAMPLES = {
    "서아",
    "지우",
    "도윤",
    "하린",
    "세인",
    "콩이",
    "강남역",
    "다온",
    "민준",
    "서윤",
    "하은",
}
KIN = [
    "엄마",
    "아빠",
    "할머니",
    "할아버지",
    "이모",
    "고모",
    "삼촌",
    "언니",
    "오빠",
    "누나",
    "동생",
    "외할머니",
    "큰이모",
    "외삼촌",
    "큰누나",
    "막내",
]
NICK_STEMS = list("콩밤똘찡뽀쭈몽솜달꾸쫑토별보두하호꼬뭉복초또순깜") + [
    "다람",
    "동글",
    "쌀",
    "도토",
    "단무",
    "감자",
    "고구마",
    "별똥",
]
NICK_SUFFIX = [
    "이",
    "돌이",
    "순이",
    "둥이",
    "미",
    "니",
    "리",
    "비",
    "쭈",
    "쿵",
    "동이",
    "똥이",
    "봉이",
    "롱이",
    "랑이",
    "보",
]
NAMEWORDS = [
    "가람",
    "보람",
    "나래",
    "다솜",
    "아름",
    "여울",
    "두리",
    "아라",
    "새봄",
    "다온",
    "해든",
    "미르",
    "한솔",
    "온새미",
    "라임",
    "소나",
    "별하",
    "나리",
    "사랑",
    "다운",
    "하나",
    "누리",
    "소담",
    "아롱",
    "푸름",
    "초롱",
    "다정",
    "고운",
    "보리",
    "소리",
    "마루",
    "보름",
    "열매",
    "새싹",
    "나루",
    "아침",
    "가을",
    "겨울",
    "노랑",
    "하양",
    "파랑",
    "오름",
    "한울",
    "이음",
    "다짐",
    "소망",
    "온유",
    "은혜",
    "보석",
    "희망",
    "미소",
    "선물",
    "하루",
    "믿음",
    "열정",
    "평화",
    "용기",
    "성실",
    "지혜",
    "슬기",
    "샘물",
    "시내",
    "바위",
    "구슬",
    "방울",
    "이랑",
    "마당",
    "나무",
    "들꽃",
    "노을",
    "구름",
    "바람",
    "이슬",
    "햇살",
    "달빛",
    "별빛",
    "초원",
    "하늘",
    "바다",
    "새벽",
    "여름",
    "봄비",
    "가랑",
    "해님",
    "달님",
    "별님",
    "산들",
    "꽃님",
    "푸른",
    "한결",
    "물결",
    "소나무",
    "단비",
    "보배",
    "다래",
    "나비",
    "여름",
    "모래",
    "은빛",
    "금빛",
    "윤슬",
    "이삭",
    "솔잎",
    "잎새",
    "울림",
    "반달",
    "나눔",
    "바름",
    "도담",
    "가온",
    "하늬",
    "보금",
    "다올",
    "샛별",
    "한가위",
    "정월",
    "나들이",
    "메아리",
    "동무",
    "까치",
    "제비",
    "버들",
    "느티",
    "그늘",
    "햇볕",
]
NATURE = [
    "바람결",
    "구름결",
    "햇빛",
    "달무리",
    "별무리",
    "은하수",
    "샛바람",
    "이슬비",
    "안개비",
    "가랑비",
    "소낙비",
    "함박눈",
    "진눈깨비",
    "눈보라",
    "서릿발",
    "시냇가",
    "강가",
    "호숫가",
    "바닷바람",
    "갯벌",
    "모래톱",
    "해변",
    "갈대",
    "숲길",
    "들길",
    "꽃길",
    "강물",
    "냇물",
    "샘터",
    "솔숲",
    "대숲",
    "갯바람",
    "봄비",
    "가을비",
    "겨울비",
    "아침놀",
    "저녁놀",
    "새털구름",
    "먹구름",
    "뭉게구름",
    "산새",
    "물새",
    "들새",
    "봄볕",
    "가을볕",
    "별똥별",
    "첫눈",
    "샛별",
    "해돋이",
    "해넘이",
    "별자리",
    "물안개",
    "서리꽃",
    "들녘",
    "벌판",
    "언덕",
    "골짜기",
    "개울",
    "계곡",
    "호수",
    "해무",
    "여우비",
    "소나기",
    "장마",
    "태풍",
    "무더위",
    "한파",
    "폭설",
    "황사",
    "미세먼지",
    "노을빛",
    "달맞이꽃",
    "잔디",
    "모닥불",
    "장작",
    "낙엽",
    "이끼",
    "갯바위",
    "모래사장",
    "파도소리",
    "달그림자",
]
OBJECTS = [
    "우산",
    "장화",
    "도시락",
    "풍경화",
    "사진첩",
    "커튼",
    "이불",
    "화분",
    "선풍기",
    "가습기",
    "냉장고",
    "세탁기",
    "청소기",
    "전기장판",
    "보온병",
    "물병",
    "책가방",
    "필통",
    "실내화",
    "방석",
    "베개",
    "수건",
    "칫솔",
    "샴푸",
    "비누",
    "휴지",
    "쌀",
    "라면",
    "김치",
    "계란",
    "우유",
    "빵",
    "과일",
    "사과",
    "감자",
    "양파",
    "두부",
    "고기",
    "생선",
    "반찬",
    "간식",
    "과자",
    "아이스크림",
    "케이크",
    "커피",
    "주스",
    "물티슈",
    "건전지",
    "충전기",
    "이어폰",
    "공책",
    "연필",
    "색연필",
    "가위",
    "테이프",
    "상자",
    "봉투",
    "장바구니",
    "우비",
    "모자",
    "장갑",
    "양말",
    "신발",
    "지갑",
    "열쇠",
    "손목시계",
    "안경",
    "마스크",
    "체온계",
    "밴드",
    "보조배터리",
    "리모컨",
    "도서",
    "달력",
    "영수증",
    "쿠폰",
]
# 실제 명사이면서 사람 이름으로도 쓰이는 단어(대조쌍은 이 단어들로만 만든다: 사람 문맥이면 PERSON, 명사 문맥이면 라벨 없음)
NAMEWORDS_NOUN = [
    "가람",
    "보람",
    "아름",
    "여울",
    "나리",
    "마루",
    "보리",
    "소리",
    "열매",
    "나루",
    "윤슬",
    "이삭",
    "솔잎",
    "잎새",
    "반달",
    "하늬",
    "샘물",
    "구슬",
    "방울",
    "라임",
    "바다",
    "하늘",
    "노을",
    "구름",
    "바람",
    "이슬",
    "햇살",
    "달빛",
    "별빛",
    "새벽",
    "단비",
    "나비",
    "초원",
    "물결",
    "한결",
    "보배",
    "다래",
    "달래",
    "소라",
    "보라",
    "하나",
    "가을",
    "여름",
    "겨울",
    "새봄",
    "은하",
    "샛별",
    "한솔",
    "솔빛",
    "새암",
    "그린",
    "라일락",
    "국화",
    "매화",
    "벚꽃",
    "목련",
    "튤립",
    "데이지",
    "클로버",
    "코스모스",
    "도토리",
    "호두",
    "유자",
    "시냇물",
    "노랑",
    "초록",
    "연두",
    "하양",
    "파랑",
    "주홍",
    "진달래",
    "민들레",
    "개나리",
]
SPOKEN_NAMEWORD_BLOCK = {"하재", "장보고"}
NAMEWORD_OBJ = {
    "라임",
    "유자",
    "튤립",
    "데이지",
    "클로버",
    "라일락",
    "코스모스",
    "매화",
    "목련",
}  # 사 오거나 보관하는 물건 문맥에도 자연스러운 이름 단어
WTYPE = (
    json.loads((FRAMES / "w_slot_types.json").read_text(encoding="utf-8"))
    if (FRAMES / "w_slot_types.json").exists()
    else {}
)


def hbucket(key, mod=100):
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % mod


# ----------------------------------------------------------------------------- 틀 읽기
SLOT_RX = re.compile(
    r"\{(N|AL|NK|K|W|NW|WN)(\d)(?::(\w+))?\}|\{(L|O|AD|PH|EM|AC)(\d)\}"
)
KEYS = {"sub", "top", "obj", "com", "dat", "voc", "pos", "bare", "lit"}
HANGUL = re.compile(r"[가-힣]")


def load_frames():
    out, problems = [], Counter()
    seen = set()
    for f in sorted(FRAMES.glob("*.json")):
        if f.stem not in FRAME_FILES:
            continue
        items = json.loads(f.read_text(encoding="utf-8"))
        for it in items:
            t = it.get("text", "")
            ms = list(SLOT_RX.finditer(t))
            names = [
                (m.group(1) or m.group(4)) + (m.group(2) or m.group(5)) for m in ms
            ]
            if len(set(names)) != len(names):
                problems["dup_slot_name"] += 1
                continue
            bad = False
            fixed, pos = [], 0
            for m in ms:
                fixed.append(t[pos : m.start()])
                kind, key = m.group(1), m.group(3)
                if kind and key and key not in KEYS:
                    bad = True
                    break
                nxt = t[m.end() : m.end() + 1]
                if kind and key == "bare" and nxt and HANGUL.match(nxt):
                    key = "lit"  # 바로 뒤에 한글이 이어지면 lit
                    problems["bare_to_lit"] += 1
                fixed.append(
                    m.group(0)
                    if not (kind and key != m.group(3))
                    else "{" + kind + m.group(2) + ":" + key + "}"
                )
                pos = m.end()
            if bad:
                problems["bad_key"] += 1
                continue
            fixed.append(t[pos:])
            t2 = "".join(fixed)
            if "{" in SLOT_RX.sub("", t2) or "}" in SLOT_RX.sub("", t2):
                problems["unparsed_brace"] += 1
                continue
            if not (8 <= len(SLOT_RX.sub("가나다", t2)) <= 130):
                problems["length"] += 1
                continue
            if t2 in seen:
                problems["dup_text"] += 1
                continue
            seen.add(t2)
            out.append(
                {
                    "id": it["id"],
                    "file": f.stem,
                    "scenario": it.get("scenario"),
                    "text": t2,
                    "tags": it.get("tags", []),
                    "slots": names,
                }
            )
    return out, dict(problems)


# ----------------------------------------------------------------------------- 봉인·개발·기존 데이터 필터(내용 출력 금지)
class Guards:
    def __init__(self):
        (
            self.sealed_sents,
            self.sealed_skels,
            self.sealed_grams,
            self.sealed_vals,
            self.sealed_names,
            self.sealed_given,
        ) = set(), set(), [], set(), set(), set()
        for p in list(SEAL.glob("r9_*_v1.2.jsonl")) + [NEW180]:
            for r in r8.read_jsonl(p):
                t = r["text"]
                self.sealed_sents.add(r8.norm(t))
                sp = [{"start": s["start"], "end": s["end"]} for s in r["spans"]]
                sk = r8.skeleton_of(t, sp)
                self.sealed_skels.add(sk)
                self.sealed_grams.append(r8.ngrams(sk))
                for s in r["spans"]:
                    v = t[s["start"] : s["end"]]
                    self.sealed_vals.add(v)
                    if s.get("label") in ("PERSON", "PERSON_ALIAS"):
                        self.sealed_names.add(v)
                        if len(v) >= 3:
                            self.sealed_given.add(v[1:])
                        if len(v) >= 4:
                            self.sealed_given.add(v[2:])
        self.gram_index = defaultdict(list)
        for i, g in enumerate(self.sealed_grams):
            for x in g:
                self.gram_index[x].append(i)
        self.dev_exact, self.dev_given, self.dev_words, self.dev_sents = (
            r8.collect_dev_blocks()
        )
        self.dev_skels = set()
        for f in list(r8.EVAL_DIR.rglob("*.jsonl")) + list(r8.NEMO_DIR.glob("*.jsonl")):
            for r in r8.read_jsonl(f):
                t = r.get("text")
                if isinstance(t, str):
                    self.dev_skels.add(
                        r8.skeleton_of(
                            t,
                            [
                                {"start": s["start"], "end": s["end"]}
                                for s in r.get("spans", [])
                                if "start" in s
                            ],
                        )
                    )
        self.common = self._common_words()
        self.existing_sents = set()
        for f in ("baseline_train", "baseline_val", "synth_train", "synth_val"):
            for r in r8.read_jsonl(EXISTING_DIR / f"{f}.jsonl"):
                self.existing_sents.add(r8.norm(r["text"]))
        self.rejected = Counter()

    @staticmethod
    def _common_words():
        """KLUE 학습 문장에서 라벨이 없는 어절(조사 떼기 포함)로 3번 이상 나온 문자열 집합(일반 명사일 가능성이 큰 것)."""
        cnt = Counter()
        josa = [
            "은",
            "는",
            "이",
            "가",
            "을",
            "를",
            "의",
            "에",
            "로",
            "과",
            "와",
            "도",
            "에서",
            "으로",
        ]
        for r in r8.read_jsonl(EXISTING_DIR / "klue_train.jsonl"):
            t = r["text"]
            labeled = [False] * len(t)
            for sp in r["spans"]:
                for i in range(sp["start"], sp["end"]):
                    labeled[i] = True
            pos = 0
            for tok in t.split(" "):
                a, b = pos, pos + len(tok)
                pos = b + 1
                if not tok or any(labeled[a:b]):
                    continue
                stems = {tok}
                for j in josa:
                    if tok.endswith(j) and len(tok) > len(j):
                        stems.add(tok[: -len(j)])
                for st in stems:
                    if 2 <= len(st) <= 4:
                        cnt[st] += 1
        return {w for w, c in cnt.items() if c >= 3}

    def name_ok(self, full, given=None, allow_common=False):
        given = given or full
        if not allow_common and (full in self.common or given in self.common):
            return False
        if full in SPEC_EXAMPLES or given in SPEC_EXAMPLES:
            return False
        if (
            full in self.dev_exact
            or given in self.dev_given
            or given in self.dev_exact
            or full in self.dev_words
        ):
            return False
        if (
            full in self.sealed_names
            or given in self.sealed_given
            or given in self.sealed_names
            or full in self.sealed_vals
        ):
            return False
        for v in self.sealed_names:  # 포함 관계(강한 격리)
            if (
                len(v) >= 2
                and len(given) >= 2
                and (
                    given == v
                    or (given in v and len(given) >= 3)
                    or (v in full and len(v) >= 3)
                )
            ):
                return False
        return True

    def text_ok(self, text, spans):
        n = r8.norm(text)
        if n in self.sealed_sents:
            self.rejected["sealed_sentence"] += 1
            return False
        sk = r8.skeleton_of(text, spans)
        if sk in self.sealed_skels:
            self.rejected["sealed_skeleton"] += 1
            return False
        g = r8.ngrams(sk)
        if g:
            share = Counter()
            for x in g:
                for i in self.gram_index.get(x, ()):
                    share[i] += 1
            for i, c in share.most_common(20):
                if (
                    c >= 0.6 * len(g)
                    and len(g & self.sealed_grams[i]) / len(g | self.sealed_grams[i])
                    >= 0.6
                ):
                    self.rejected["sealed_skeleton_similar"] += 1
                    return False
        for s in spans:
            v = text[s["start"] : s["end"]]
            if v in self.sealed_vals and len(v) >= 2:
                self.rejected["sealed_value"] += 1
                return False
        if n in self.dev_sents:
            self.rejected["dev_sentence"] += 1
            return False
        if sk in self.dev_skels:
            self.rejected["dev_skeleton"] += 1
            return False
        if n in self.existing_sents:
            self.rejected["existing_training_sentence"] += 1
            return False
        for ex in SPEC_EXAMPLES:
            if ex in text:
                self.rejected["spec_example_string"] += 1
                return False
        return True


# ----------------------------------------------------------------------------- 이름 풀
A_SYL = "서하지도시예유은이채다소수윤아나가민현준건우재태승규상성영진원혜경광대동종창연한슬미세주보라로새선정인형희효해범석기명"
B_SYL = "윤우준현은영서진아연희민호원빈율하린솔나지경훈찬성수재안온정인유혁환리라미담결람별슬채후완규주건국태용"
PLACE_A = "경도영상병대동성진한수광혜창철부금태용형안인"
PLACE_B = "원주산천진동성대영화남북관교문장수구포령평양"
RISKY_END = "은이애인일한남니"
SURN = [x for x in synth.SURNAMES if len(x) == 1]
COMPOUND = ["남궁", "황보", "선우", "제갈", "사공", "독고"]
FOREIGN3 = [
    "레이나",
    "클로이",
    "아리아",
    "엘리사",
    "하이든",
    "로이나",
    "미카엘",
    "다니엘",
    "제이든",
    "에이든",
    "루시아",
    "엘라나",
    "로아나",
    "카일라",
    "이레나",
    "아델라",
    "소피아",
    "리나",
    "노엘라",
    "에밀리",
]
FOLK3 = [
    "빛나리",
    "한가람",
    "아라나",
    "으뜸이",
    "한울림",
    "푸른솔",
    "고운별",
    "새나래",
    "다솜이",
    "해오름",
    "한소리",
    "나누리",
    "다온빛",
    "초롱별",
    "하늬바람",
]


class Names:
    """이름 풀. 분할(train/val)은 **이름 부분(given)** 의 해시로 정해, 성만 다른 이름이 두 분할에 걸치지 않게 한다."""

    CAP = 4

    def __init__(self, g, rng):
        self.g, self.rng = g, rng
        self.used = Counter()
        self.pools = {}
        self.split = {}
        self.idx = Counter()

    def _build(self):
        rng, g = self.rng, self.g
        wordblock = (
            set(NATURE)
            | set(NAMEWORDS)
            | set(KIN)
            | set(synth.GENERIC_NOUNS)
            | set(synth.GENERIC_PLACES)
        )

        def combos(a, b):
            out = [x + y for x in a for y in b if x != y]
            rng.shuffle(out)
            return out

        def good(n):
            return g.name_ok(n) and n not in wordblock

        def by_split(lst):
            return {
                "train": [x for x in lst if hbucket(x) >= 16],
                "val": [x for x in lst if hbucket(x) < 16],
            }

        def with_surnames(givens, per, comp=False):
            out = []
            for gname in givens:
                for _ in range(per):
                    sname = rng.choice(COMPOUND) if comp else rng.choice(SURN)
                    full = sname + gname
                    if g.name_ok(full, gname):
                        out.append(full)
            return list(dict.fromkeys(out))

        G = by_split(
            [
                n
                for n in combos(
                    A_SYL + "솔별결율찬빛늘림단혜누담겸휘온",
                    B_SYL + "솔별결담겸휘빛늘림단혜누",
                )
                if good(n)
            ]
        )
        R = by_split([n for n in combos(A_SYL, RISKY_END) if good(n)])
        P = by_split([n for n in combos(PLACE_A, PLACE_B) if good(n)])
        sp = {}
        for sp_ in ("train", "val"):
            gl = G[sp_]
            n1, n2 = int(len(gl) * 0.42), int(len(gl) * 0.85)
            given2, full_base, rare_base = gl[:n1], gl[n1:n2], gl[n2:]
            pl = P[sp_]
            p2, p3 = pl[: int(len(pl) * 0.6)], pl[int(len(pl) * 0.6) :]
            foreign = (
                [x for x in FOREIGN3 + FOLK3 if hbucket(x) < 16]
                if sp_ == "val"
                else [x for x in FOREIGN3 + FOLK3 if hbucket(x) >= 16]
            )
            sp[sp_] = {
                "given2": given2,
                "full3": with_surnames(full_base, 2),
                "rare3": with_surnames(rare_base, 2),
                "full3risky": with_surnames(R[sp_], 2),
                "place2": p2,
                "place3": with_surnames(p3, 3),
                "full4": with_surnames(
                    given2[: max(10, len(given2) // 5)], 1, comp=True
                )
                + [
                    s_ + x
                    for x in foreign
                    for s_ in rng.sample(SURN, 3)
                    if g.name_ok(s_ + x, x)
                ],
            }
        wordish = [
            w
            for w in [
                "소망",
                "은혜",
                "보석",
                "희망",
                "자유",
                "정의",
                "진실",
                "효도",
                "재주",
                "가치",
                "온유",
                "기쁨",
                "미리",
                "소리",
                "다짐",
                "정성",
                "행복",
                "믿음",
                "푸른",
                "달님",
                "별님",
            ]
            if g.name_ok(w, allow_common=True)
        ]
        nameword = [
            w
            for w in dict.fromkeys(NAMEWORDS_NOUN)
            if g.name_ok(w, allow_common=True) and w not in SPOKEN_NAMEWORD_BLOCK
        ]
        nicks = []
        for st in NICK_STEMS:
            for suf in NICK_SUFFIX:
                n = st + suf
                if (
                    2 <= len(n) <= 4
                    and g.name_ok(n)
                    and n not in wordblock
                    and n not in synth.GIVEN2
                ):
                    nicks.append(n)
        rng.shuffle(nicks)
        for k, v in (
            ("wordish", by_split(wordish)),
            ("nameword", by_split(nameword)),
            ("nickname", by_split(nicks[:700])),
        ):
            for sp_ in ("train", "val"):
                sp[sp_][k] = v[sp_]
        self.split = {
            k: {"train": sp["train"][k], "val": sp["val"][k]} for k in sp["train"]
        }
        self.pools = {k: v["train"] + v["val"] for k, v in self.split.items()}

    def pick(self, kind, split):
        if not self.pools:
            self._build()
        pool = self.split[kind][split]
        if not pool:
            raise RuntimeError(f"이름 풀 비어 있음: {kind}/{split}")
        for _ in range(len(pool)):
            n = pool[self.idx[(kind, split)] % len(pool)]
            self.idx[(kind, split)] += 1
            if self.used[n] < self.CAP:
                self.used[n] += 1
                return n
        n = pool[
            self.idx[(kind, split)] % len(pool)
        ]  # 풀이 작은 종류는 상한을 넘겨 재사용(보고서에 최대 재사용 횟수로 기록)
        self.idx[(kind, split)] += 1
        self.used[n] += 1
        return n


# ----------------------------------------------------------------------------- 조사
def josa_name(name, key, rng):
    if key in ("bare", "lit"):
        return ""
    return r8.attach(name, key, rng)[len(name) :]


NOUN_J = {
    "sub": ("이", "가"),
    "top": ("은", "는"),
    "obj": ("을", "를"),
    "com": ("과", "와"),
    "dat": ("에게", "에게"),
    "voc": ("아", "야"),
    "pos": ("의", "의"),
}


def josa_noun(w, key, rng):
    if key in ("bare", "lit"):
        return ""
    if key == "com":
        return (
            rng.choice(["이랑", "하고", "과"])
            if r8.batchim(w[-1])
            else rng.choice(["랑", "하고", "와"])
        )
    if key == "dat":
        return rng.choice(["한테", "에게"])
    j = NOUN_J[key]
    return j[0] if r8.batchim(w[-1]) else j[1]


def josa_kin(k, key, rng):
    if key in ("bare", "lit", "voc"):
        return ""
    return josa_noun(k, key, rng)


def josa_alias(w, key, rng):
    if key in ("bare", "lit"):
        return ""
    if key == "voc":
        return "야" if not r8.batchim(w[-1]) else "아"
    return josa_noun(w, key, rng)


# ----------------------------------------------------------------------------- 채우기
LEADS = [
    "아 ",
    "근데 ",
    "참 ",
    "혹시 ",
    "저기 ",
    "야 ",
    "그리고 ",
    "아 맞다 ",
    "음.. ",
    "ㅇㅇ ",
    "일단 ",
    "그럼 ",
]
TAILS = [
    " ㅋㅋ",
    " ㅠㅠ",
    "~",
    "!!",
    " ㅎㅎ",
    "..",
    " ㅇㅇ",
    " ㅜㅜ",
    "?!",
    " 🙏",
    " 😭",
    " 😊",
]
TYPO = [
    ("할게", "할께"),
    ("됐어", "됬어"),
    ("했어", "했엉"),
    ("있어", "있엉"),
    ("진짜", "진짜루"),
    ("그런데", "근데"),
    ("괜찮", "괜춘"),
    ("아니야", "아니여"),
]


def fill(frame, bucket_cfg, names, split, rng, g, roles):
    """frame 을 채워 (text, spans, rule_spans, meta). roles: 슬롯별 이름 종류 선택 함수."""
    t = frame["text"]
    segs, pos = [], 0
    meta = {"names": [], "kin": False}
    for m in SLOT_RX.finditer(t):
        segs.append((t[pos : m.start()], None))
        pos = m.end()
        kind, num, key = m.group(1), m.group(2), m.group(3)
        if kind in ("N", "NW"):
            name = roles(kind, num)
            segs.append((name, "PERSON"))
            sfx = josa_name(name, key, rng)
            if sfx:
                segs.append((sfx, None))
            meta["names"].append(name)
        elif kind == "AL":
            name = names.pick("nickname", split)
            segs.append((name, "PERSON_ALIAS"))
            sfx = josa_alias(name, key, rng)
            if sfx:
                segs.append((sfx, None))
            meta["names"].append(name)
        elif kind == "NK":
            name = roles("N", num)
            k = rng.choice(KIN[:8] if rng.random() < 0.85 else KIN)
            sep = "" if rng.random() < 0.15 else " "
            segs.append((name, "PERSON"))
            segs.append((sep + k, None))
            sfx = josa_kin(k, key, rng)
            if sfx:
                segs.append((sfx, None))
            meta["names"].append(name)
            meta["kin"] = True
        elif kind == "K":
            k = rng.choice(KIN)
            segs.append((k, None))
            sfx = josa_kin(k, key, rng)
            if sfx:
                segs.append((sfx, None))
            meta["kin"] = True
        elif kind in ("W", "WN"):
            w = roles(kind, num)
            segs.append((w, None))
            sfx = josa_noun(w, key, rng)
            if sfx:
                segs.append((sfx, None))
        elif m.group(4) == "L":
            segs.append((rng.choice(bucket_cfg["locs"][split]), "LOCATION"))
        elif m.group(4) == "O":
            segs.append((rng.choice(bucket_cfg["orgs"][split]), "ORGANIZATION"))
        elif m.group(4) == "AD":
            segs.append((r8.addr(rng), "ADDRESS"))
        elif m.group(4) == "PH":
            segs.append((r8.phone(rng), "R_PHONE"))
        elif m.group(4) == "EM":
            segs.append((r8.email(rng), "R_EMAIL"))
        elif m.group(4) == "AC":
            segs.append((r8.account(rng), "R_ACCOUNT"))
    segs.append((t[pos:], None))
    return segs, meta


def apply_noise(segs, rng, p_lead, p_tail, p_space, p_typo):
    segs = [list(x) for x in segs]
    if rng.random() < p_typo:
        for i, (s, lab) in enumerate(segs):
            if lab is None:
                for a, b in TYPO:
                    if a in s and rng.random() < 0.5:
                        segs[i][0] = s.replace(a, b, 1)
                        break
    if rng.random() < p_space:
        cands = [
            i
            for i, (s, lab) in enumerate(segs)
            if lab is None and s.strip(" ").count(" ") >= 1 and len(s) > 4
        ]
        if cands:
            i = rng.choice(cands)
            s = segs[i][0]
            inner = [j for j, ch in enumerate(s) if ch == " " and 0 < j < len(s) - 1]
            if inner:
                j = rng.choice(inner)
                segs[i][0] = s[:j] + s[j + 1 :]
    if rng.random() < p_lead:
        segs.insert(0, [rng.choice(LEADS), None])
    if rng.random() < p_tail:
        segs.append([rng.choice(TAILS), None])
    return [tuple(x) for x in segs]


def finalize(segs):
    text, spans, rule = "", [], []
    for s, lab in segs:
        a = len(text)
        text += s
        if lab is None or not s:
            continue
        if lab.startswith("R_"):
            rule.append({"kind": lab[2:], "start": a, "end": len(text)})
        else:
            spans.append({"label": lab, "start": a, "end": len(text)})
    return text, spans, rule


# ----------------------------------------------------------------------------- 버킷 구성
# 군별 (파일, 태그 또는 None, 학습 행 수)
PLAN = {
    "msg": [("msg_person", None, 1463), ("msg_neg", None, 413)],
    "errA": [("err_A", "A1", 350), ("err_A", "A3p", 220), ("err_A", "A3L", 212)],
    "errB": [
        ("err_B", "B1", 320),
        ("err_B", "C1", 200),
        ("err_B", "C2", 160),
        ("err_B", "B2", 102),
    ],
    "errD": [
        ("err_D", "D1n", 200),
        ("err_D", "D1p", 120),
        ("err_D", "D1w", 120),
        ("err_D", "D2", 140),
        ("err_D", "ALn", 100),
        ("err_D", "kin", 102),
    ],
    "mixed": [("mixed", None, 468)],
}
NOISE = {
    "msg": (0.30, 0.35, 0.12, 0.10),
    "errA": (0.25, 0.30, 0.10, 0.08),
    "errB": (0.25, 0.30, 0.10, 0.08),
    "errD": (0.25, 0.30, 0.10, 0.08),
    "mixed": (0.04, 0.02, 0.02, 0.02),
}
VAL_FRACTION = 0.18


def name_roles(tag, file, names, split, rng, frame_id=None):
    """슬롯 종류별로 이름을 고르는 함수를 돌려준다(틀 파일·태그에 따라 이름 유형 비율이 다르다)."""

    def pick_person(kind_w):
        kinds = [k for k, _ in kind_w]
        ws = [w for _, w in kind_w]
        return names.pick(rng.choices(kinds, ws)[0], split)

    def roles(kind, num):
        if kind == "N":
            if file == "err_A" and tag == "A1":
                return pick_person([("given2", 70), ("rare3", 20), ("wordish", 10)])
            if file == "err_A" and tag == "A3p":
                return pick_person([("place2", 55), ("place3", 45)])
            if file == "err_B" and tag in ("B1",):
                return pick_person([("full3risky", 45), ("full3", 20), ("full4", 35)])
            if file == "err_B":
                return pick_person([("given2", 45), ("full3", 40), ("rare3", 15)])
            if file == "mixed":
                return pick_person(
                    [("full3", 55), ("given2", 15), ("full4", 15), ("rare3", 15)]
                )
            return pick_person(
                [
                    ("given2", 33),
                    ("full3", 36),
                    ("full4", 8),
                    ("rare3", 6),
                    ("place2", 5),
                    ("wordish", 4),
                    ("full3risky", 8),
                ]
            )
        if kind == "NW":
            return names.pick("nameword", split)
        if kind in ("W", "WN"):
            typ = WTYPE.get(frame_id, "any")
            if kind == "WN":
                allw = names.split["nameword"][split]
                sub = (
                    [w for w in allw if (w in NAMEWORD_OBJ) == (typ == "obj")]
                    if typ in ("obj", "nat")
                    else allw
                )
                return rng.choice(sub or allw)
            if typ == "obj":
                return rng.choice(names.objects[split])
            if typ == "nat":
                return rng.choice(names.nature[split])
            return rng.choice(names.objects[split] + names.nature[split])
        raise ValueError(kind)

    return roles


def build_bucket(bucket, frames, names, g, guards_rejected, seed, bucket_cfg):
    rng = random.Random(seed + hbucket(bucket, 10000))
    seen = set()
    pools_by_split = {"train": [], "val": []}
    for fname, tag, quota in PLAN[bucket]:
        fs = [
            f
            for f in frames
            if f["file"] == fname and (tag is None or tag in f["tags"])
        ]
        if not fs:
            raise RuntimeError(f"틀 없음: {fname}/{tag}")
        for split, q in (("train", quota), ("val", round(quota * VAL_FRACTION))):
            use = [f for f in fs if (hbucket(f["id"]) < 15) == (split == "val")]
            if len(use) < 2:
                use = fs if split == "train" else fs[: max(3, len(fs) // 6)]
            made, tries = 0, 0
            ring = 0
            while made < q and tries < q * 40:
                tries += 1
                fr = use[ring % len(use)]
                ring += 1
                roles = name_roles(tag, fname, names, split, rng, fr["id"])
                segs, _meta = fill(fr, bucket_cfg, names, split, rng, g, roles)
                n_lead, n_tail, n_space, n_typo = NOISE[bucket]
                segs = apply_noise(segs, rng, n_lead, n_tail, n_space, n_typo)
                text, spans, rule = finalize(segs)
                key = r8.norm(text)
                if key in seen:
                    continue
                if not g.text_ok(text, spans):
                    continue
                row = {
                    "id": "",
                    "text": text,
                    "spans": spans,
                    "source": "synth",
                    "template_id": fr["id"],
                    "category": f"r9-{bucket}-{tag or fname}",
                    "split": split,
                    "augment": True,
                    "r9": True,
                    "train_bucket": {
                        "msg": "msg",
                        "errA": "err_A",
                        "errB": "err_B",
                        "errD": "err_D",
                        "mixed": "mixed",
                    }[bucket],
                    "frame_file": fname,
                    "frame_tag": tag or "",
                }
                if rule:
                    row["rule_spans"] = rule
                seen.add(key)
                pools_by_split[split].append(row)
                made += 1
            if made < q:
                raise RuntimeError(
                    f"행 부족: {bucket}/{fname}/{tag}/{split} {made}/{q}"
                )
    return pools_by_split


def validate(rows, g):
    problems = []
    for r in rows:
        t = r["text"]
        sp = sorted(r["spans"], key=lambda s: s["start"])
        for a, b in itertools.pairwise(sp):
            if b["start"] < a["end"]:
                problems.append(("overlap", r["id"]))
        for s in sp:
            if not (0 <= s["start"] < s["end"] <= len(t)):
                problems.append(("bounds", r["id"]))
            v = t[s["start"] : s["end"]]
            if s["label"] in ("PERSON", "PERSON_ALIAS") and len(v) < 2:
                problems.append(("one_char_name", r["id"]))
            if s["label"] in ("PERSON", "PERSON_ALIAS") and conv.kin_inside(v):
                problems.append(("kin_inside_label", r["id"]))
            if s["label"] not in (
                "PERSON",
                "PERSON_ALIAS",
                "ADDRESS",
                "LOCATION",
                "ORGANIZATION",
            ):
                problems.append(("label", r["id"]))
        for rs in r.get("rule_spans", []):
            if not (0 <= rs["start"] < rs["end"] <= len(t)):
                problems.append(("rule_bounds", r["id"]))
        if conv.policy_issues(
            {
                "spans": [
                    {"label": s["label"], "value": t[s["start"] : s["end"]]} for s in sp
                ]
            }
        ):
            problems.append(("policy", r["id"]))
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=EXISTING_DIR)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument(
        "--dry-run", action="store_true", help="파일을 쓰지 않고 보고서만 낸다"
    )
    a = ap.parse_args()
    frames, fprob = load_frames()
    g = Guards()
    rng = random.Random(a.seed)
    names = Names(g, rng)
    names._build()
    names.nature = {
        "train": [
            w
            for w in dict.fromkeys(NATURE)
            if hbucket(w) >= 16
            and w not in g.dev_words
            and w not in g.dev_exact
            and w not in g.sealed_vals
        ],
        "val": [
            w
            for w in dict.fromkeys(NATURE)
            if hbucket(w) < 16
            and w not in g.dev_words
            and w not in g.dev_exact
            and w not in g.sealed_vals
        ],
    }
    names.objects = {
        "train": [
            w
            for w in dict.fromkeys(OBJECTS)
            if hbucket(w) >= 16
            and w not in g.dev_words
            and w not in g.dev_exact
            and w not in g.sealed_vals
        ],
        "val": [
            w
            for w in dict.fromkeys(OBJECTS)
            if hbucket(w) < 16
            and w not in g.dev_words
            and w not in g.dev_exact
            and w not in g.sealed_vals
        ],
    }
    locs = [
        x
        for x in synth.CITIES + synth.STATIONS + synth.LANDMARKS
        if x not in g.dev_exact and x not in g.sealed_vals
    ]
    orgs = []
    for p in synth.HOSP_PREFIX:
        for s in synth.HOSP_SUFFIX:
            orgs.append(p + s)
    for p in synth.SCHOOL_PREFIX:
        for s in synth.SCHOOL_SUFFIX:
            orgs.append(p + s)
    for p in synth.COMPANY_PREFIX:
        for s in synth.COMPANY_SUFFIX:
            orgs.append(p + s)
    orgs = [x for x in orgs if x not in g.dev_exact and x not in g.sealed_vals]
    bucket_cfg = {
        "locs": {
            "train": [x for x in locs if hbucket(x) >= 16],
            "val": [x for x in locs if hbucket(x) < 16],
        },
        "orgs": {
            "train": [x for x in orgs if hbucket(x) >= 16],
            "val": [x for x in orgs if hbucket(x) < 16],
        },
    }
    pools = {}
    for bucket in PLAN:
        pools[bucket] = build_bucket(
            bucket, frames, names, g, g.rejected, a.seed, bucket_cfg
        )
    # id 부여, 정렬 고정
    allrows = []
    n = 0
    for bucket in PLAN:
        for split in ("train", "val"):
            for r in pools[bucket][split]:
                n += 1
                r["id"] = f"r9-{bucket}-{split}-{n:05d}"
                allrows.append(r)
    problems = validate(allrows, g)

    # 실험군 구성(중첩): E1 = msg 782 부분집합, E2~E4 = 오류 풀 전체, F-full = msg 1876 + A 274 + B 235 + D 273 + mixed 468
    def sub(rows, k, salt):
        r = random.Random(a.seed + salt)
        idx = r.sample(range(len(rows)), k)
        return [rows[i] for i in sorted(idx)]

    tr = {b: pools[b]["train"] for b in PLAN}
    files = {
        "r9_msg_train": tr["msg"],
        "r9_errA_train": tr["errA"],
        "r9_errB_train": tr["errB"],
        "r9_errD_train": tr["errD"],
        "r9_mixed_train": tr["mixed"],
    }
    e1 = sub(tr["msg"], 782, 1)
    (
        files["r9_E1_train"],
        files["r9_E2_train"],
        files["r9_E3_train"],
        files["r9_E4_train"],
    ) = e1, tr["errA"], tr["errB"], tr["errD"]
    ffull = (
        tr["msg"]
        + sub(tr["errA"], 274, 2)
        + sub(tr["errB"], 235, 3)
        + sub(tr["errD"], 273, 4)
        + tr["mixed"]
    )
    files["r9_Ffull_train"] = ffull
    files["r9_val_extra"] = [r for b in PLAN for r in pools[b]["val"]]

    # 보고서
    def stats(rows):
        c = Counter(s["label"] for r in rows for s in r["spans"])
        pers = [
            r["text"][s["start"] : s["end"]]
            for r in rows
            for s in r["spans"]
            if s["label"] in ("PERSON", "PERSON_ALIAS")
        ]
        skels = Counter(r8.skeleton_of(r["text"], r["spans"]) for r in rows)
        return {
            "rows": len(rows),
            "labels": dict(c),
            "negative_rows(no NER span)": sum(1 for r in rows if not r["spans"]),
            "rows_with_person_family": sum(
                1
                for r in rows
                if any(s["label"] in ("PERSON", "PERSON_ALIAS") for s in r["spans"])
            ),
            "multi_entity_rows": sum(1 for r in rows if len(r["spans"]) >= 2),
            "rule_span_rows": sum(1 for r in rows if r.get("rule_spans")),
            "person_len_dist": dict(sorted(Counter(len(x) for x in pers).items())),
            "distinct_person_values": len(set(pers)),
            "max_person_value_reuse": max(Counter(pers).values()) if pers else 0,
            "distinct_skeletons": len(skels),
            "top_skeleton_share_pct": round(
                100 * skels.most_common(1)[0][1] / len(rows), 2
            )
            if rows
            else None,
            "distinct_frames": len({r["template_id"] for r in rows}),
            "kin_adjacent_rows": sum(
                1
                for r in rows
                if any(
                    s["label"] in ("PERSON", "PERSON_ALIAS")
                    and r["text"][s["end"] : s["end"] + 8]
                    .lstrip("이 ")
                    .startswith(tuple(KIN))
                    for s in r["spans"]
                )
            ),
            "rows_with_kin_term_no_person": sum(
                1
                for r in rows
                if not any(s["label"] in ("PERSON", "PERSON_ALIAS") for s in r["spans"])
                and any(k in r["text"] for k in KIN)
            ),
        }

    train_names = {
        r["text"][s["start"] : s["end"]]
        for b in PLAN
        for r in pools[b]["train"]
        for s in r["spans"]
        if s["label"] in ("PERSON", "PERSON_ALIAS")
    }
    val_names = {
        r["text"][s["start"] : s["end"]]
        for b in PLAN
        for r in pools[b]["val"]
        for s in r["spans"]
        if s["label"] in ("PERSON", "PERSON_ALIAS")
    }
    texts = [r8.norm(r["text"]) for r in allrows]
    ratio = Counter(r["train_bucket"] for r in ffull)
    report = {
        "seed": a.seed,
        "policy": "r9-1.1",
        "frames": {
            "count": len(frames),
            "problems": fprob,
            "by_file": dict(Counter(f["file"] for f in frames)),
        },
        "files": {k: stats(v) for k, v in files.items()},
        "f_full_ratio": {
            "total": len(ffull),
            "by_bucket": dict(ratio),
            "internal_pct": {
                k: round(100 * v / len(ffull), 1) for k, v in ratio.items()
            },
            "msg_pct": round(100 * ratio["msg"] / len(ffull), 1),
            "err_pct": round(
                100 * (ratio["err_A"] + ratio["err_B"] + ratio["err_D"]) / len(ffull), 1
            ),
            "mixed_pct": round(100 * ratio["mixed"] / len(ffull), 1),
            "new_share_of_all_training_pct": round(
                100 * len(ffull) / (len(ffull) + 4689), 2
            ),
        },
        "duplicates_within_new": len(texts) - len(set(texts)),
        "train_val_name_overlap": len(train_names & val_names),
        "problems": len(problems),
        "problem_examples": problems[:10],
        "rejected_candidates_by_reason(counts only)": dict(g.rejected),
        "spec_examples_blocked": len(SPEC_EXAMPLES),
        "sealed_access": "프로그램 안에서 겹침 거르기 용도로만 읽음(내용 출력·참고 생성 없음)",
        "name_pools": {
            k: {"train": len(v["train"]), "val": len(v["val"])}
            for k, v in names.split.items()
        },
    }
    ok = (
        not problems
        and report["duplicates_within_new"] == 0
        and not (train_names & val_names)
    )
    if not a.dry_run:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        for k, v in files.items():
            (a.out_dir / f"{k}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in v),
                encoding="utf-8",
            )
        (a.out_dir / "r9_build_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "ok": ok,
                "frames": report["frames"],
                "f_full_ratio": report["f_full_ratio"],
                "problems": report["problems"],
                "dups": report["duplicates_within_new"],
                "train_val_name_overlap": report["train_val_name_overlap"],
                "rejected": report["rejected_candidates_by_reason(counts only)"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
