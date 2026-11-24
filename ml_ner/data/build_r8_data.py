"""R8 학습 데이터 생성: A1(후보 미생성)·A3(지명·기관 혼동)·B1(끝 글자 누락)·D(오탐) + 대조 문맥 + 다중 개인정보 문장.

사전 등록: docs/security/NER_R8_PREREGISTRATION.md §7. 전부 자체 작성 합성 문장이라 외부 데이터 라이선스 문제가 없다.

격리 규칙(위반하면 종료 코드 1):
- 봉인 평가셋 new180(문장·정규화 문장·문장 틀·글자 4-gram 유사 틀·구간 값)과 겹치면 안 된다. new180 원문은 이 스크립트가 메모리에서만 쓰고 어떤 출력에도
  내보내지 않는다(보고서에는 집계만). 처음 열람한 12행의 이름은 SEEN_NEW180_NAMES 로 추가 차단한다.
- 개발 평가셋(저장소 eval 폴더 전체 + Nemotron 개발 파일)과 기존 학습 풀의 이름·단어를 새 이름 풀에서 뺀다(개발 평가에서 이름을 외운 효과 방지).
- train/val 은 문형, 이름 풀, 단어 풀을 서로 분리한다.

라벨: NER 대상(PERSON, PERSON_ALIAS, ADDRESS, LOCATION, ORGANIZATION)은 문장 안 모두 표시하고, PHONE/EMAIL/ACCOUNT 는 규칙 몫이라 NER 라벨 O 로 두되 ``rule_spans`` 에 기록한다.

사용: ``python build_r8_data.py --out-dir <data_r8 폴더>``
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
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import synth  # noqa: E402

SEALED = Path(r"C:\Users\<user>\Documents\gabee-ner-sealed-new180\gabee_ner_new180_v1.jsonl")
NEMO_DIR = Path(r"C:\Users\<user>\Documents\gabee-ner-backup-20261118\scratch\nemotron")
EVAL_DIR = ROOT / "apps" / "api" / "app" / "shared" / "privacy" / "eval"
# 형식 확인 때 열람한 new180 앞 12행의 이름(이후 학습 데이터에 쓰지 않는다)
SEEN_NEW180_NAMES = {"서민결", "주라온", "문하겸", "유다빈", "차서율", "배이안", "고하린", "장유찬", "정가엘", "표도겸", "한지오", "오채린"}
NER_LABELS = {"PERSON", "PERSON_ALIAS", "ADDRESS", "LOCATION", "ORGANIZATION"}


# ----------------------------------------------------------------------------- 한글 도우미
def batchim(ch):
    return (ord(ch) - 0xAC00) % 28 != 0


JOSA = {
    # key: (받침 있을 때 후보, 받침 없을 때 후보)
    "sub": (["이", "이가", "이는"], ["가", "는"]),
    "top": (["은", "이는"], ["는"]),
    "obj": (["을", "이를"], ["를"]),
    "com": (["이랑", "이하고", "이와"], ["랑", "하고", "와"]),
    "dat": (["한테", "에게", "이한테"], ["한테", "에게"]),
    "voc": (["아", "이야"], ["야"]),
    "pos": (["의", "이의"], ["의"]),
    "bare": ([""], [""]),
}


NOUN_JOSA = {"sub": ("이", "가"), "top": ("은", "는"), "obj": ("을", "를"), "bare": ("", ""), "com": ("과", "와"), "dat": ("에게", "에게")}


def attach_noun(w, key):
    j = NOUN_JOSA[key]
    return w + (j[0] if batchim(w[-1]) else j[1])


def attach(name, key, rng):
    c = JOSA[key]
    pool = c[0] if batchim(name[-1]) else c[1]
    return name + rng.choice(pool)


# ----------------------------------------------------------------------------- 차단 목록
def norm(t):
    return re.sub(r"[\s\.,!?~ㅋㅎㅠ'\"\-]+", "", t)


def skeleton_of(text, spans):
    """구간을 라벨 토큰으로 바꾼 문장 틀(뒤 공백·구두점 제거)."""
    out, pos = [], 0
    for s in sorted(spans, key=lambda x: x["start"]):
        out.append(text[pos:s["start"]]); out.append("§"); pos = s["end"]
    out.append(text[pos:])
    return norm("".join(out))


def ngrams(s, n=4):
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def read_jsonl(p):
    out = []
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def collect_dev_blocks():
    """개발 평가셋 + 기존 풀에서 이름·단어 차단 목록과 문장 집합을 만든다."""
    exact, given, words, sents = set(), set(), set(), set()
    files = list(EVAL_DIR.rglob("*.jsonl")) + list(NEMO_DIR.glob("*.jsonl"))
    for f in files:
        for r in read_jsonl(f):
            t = r.get("text") or ""
            sents.add(norm(t))
            for sp in r.get("spans", []) or []:
                v = t[sp["start"]:sp["end"]] if "start" in sp else ""
                if v:
                    exact.add(v)
                    if len(v) >= 3:
                        given.add(v[1:])
                    if len(v) >= 4:
                        given.add(v[2:])
            for k in ("word", "name", "full_name", "gold_text"):
                v = r.get(k)
                if isinstance(v, str) and v:
                    words.add(v)
                    exact.add(v)
                    if len(v) >= 3:
                        given.add(v[1:])
    for g in synth.GIVEN2 + synth.GIVEN1 + synth.NICKS + list(synth.WORDLIKE_GIVEN):
        exact.add(g)
    return exact, given, words, sents


def load_sealed():
    rows = read_jsonl(SEALED)
    skels, sent_norm, values, grams = set(), set(), set(), []
    for r in rows:
        sent_norm.add(norm(r["text"]))
        sk = skeleton_of(r["text"], r["spans"])
        skels.add(sk)
        grams.append(ngrams(sk))
        for s in r["spans"]:
            values.add(s["value"])
    return {"rows": len(rows), "skels": skels, "sents": sent_norm, "values": values, "grams": grams}


# ----------------------------------------------------------------------------- 풀
SURN = [s for s in synth.SURNAMES if len(s) == 1]
COMPOUND = ["남궁", "황보", "선우", "제갈", "사공", "독고"]
A_SYL = "서하지도시예유은이채다소수윤아나가민현준건우재태승규상성영진원혜경광대동종창연한슬미세주보라로새선정인형희효해범석기명"
B_SYL = "윤우준현은영서진아연희민호원빈율하린솔나지경훈찬성수재안온정인유혁환리라미담결람별슬채후완규주건국태용"
PLACE_A = "경도영상병대동성진한수광혜창철부금태용형안인"
PLACE_B = "원주산천진동성대영화남북관교문장수구포령평양"
FOREIGN3 = ["레이나", "클로이", "아리아", "엘리사", "하이든", "로이나", "미카엘", "다니엘", "사라온", "제이든", "에이든", "루시아", "엘라나", "로아나", "카일라", "이레나"]
FOLK3 = ["빛나리", "한가람", "아라나", "으뜸이", "한울림", "푸른솔", "고운별", "새나래", "다솜이", "해오름", "한소리", "나누리"]
RISKY_END = list("은이애인일한남니")  # 분석에서 끝 글자 누락이 많았던 마지막 글자
WORDISH = ["소망", "은혜", "보석", "희망", "자유", "정의", "진실", "효도", "재주", "가치", "온유", "기쁨", "미리", "소리", "다짐", "정성", "행복", "믿음", "푸른", "고움", "꿈길", "달님", "별님"]
NAMEWORD = ["가람", "보람", "다솜", "아름", "여울", "두리", "아라", "새봄", "다온", "해든", "미르", "한솔", "온새미", "라임", "소나", "별하", "나리", "사랑", "다운", "하나", "누리", "소담", "아롱", "푸름"]
NAMEWORD_NOUN_OK = ["여울", "나리", "라임", "가람", "보람", "새봄", "누리"]  # 실제 보통명사로도 쓰이는 이름 단어(명사 대조 문맥에는 이것만 쓴다)
NATURE = ("바람결 구름결 햇빛 달무리 별무리 은하수 샛바람 이슬비 안개비 가랑비 소낙비 함박눈 진눈깨비 눈보라 서릿발 시냇가 강가 호숫가 "
          "바닷바람 갯벌 모래톱 해변 갈대 숲길 들길 꽃길 강물 냇물 샘터 솔숲 대숲 갯바람 봄비 가을비 겨울비 아침놀 저녁놀 새털구름 먹구름 뭉게구름 "
          "산새 물새 들새 봄볕 가을볕 달맞이 별똥별 첫눈 늦눈 봄눈 소나기구름 샛별 해돋이 해넘이 별자리 은빛 금빛 물안개 서리꽃 눈꽃길 "
          "들녘 벌판 언덕 골짜기 개울 계곡 호수 바다 하늘 노을 구름 파도 햇살 안개 단풍 무지개 이슬 별빛 눈송이").split()
NATURE_PRED = ["참 맑네", "정말 예쁘다", "오늘따라 선명하네", "한참 바라봤어", "사진으로 남겨 뒀어", "아침부터 계속 이래", "창밖으로 보이더라",
               "보고만 있어도 마음이 편해져", "오늘은 유독 깊어 보여", "어제와는 다르게 느껴졌어"]
SPOKEN_END = ["같이 먹재", "일찍 오래", "내일 만나재", "그만 놀래", "사진 보내 주재", "먼저 가래", "다시 해 보재", "조금만 기다리래", "숙제부터 하래", "집에 가재"]
LEADS = ["", "", "", "아 ", "근데 ", "그리고 ", "참 ", "혹시 ", "저기 "]
TAILS = ["", "", "", "", " ㅋㅋ", " ㅠㅠ", "~", "!!", " ㅎㅎ"]
DOMAINS = ["example.com", "test.kr", "sample.net", "mail.example.org"]
BANKS = ["국민", "신한", "우리", "하나", "농협", "기업"]
CITY_SI = ["수원시", "성남시", "용인시", "고양시", "창원시", "청주시", "전주시", "천안시", "서울시", "부산시", "대구시", "인천시", "대전시"]


# ----------------------------------------------------------------------------- 문형 뱅크
# (키, split, 문장 틀). {N1:sub} 처럼 사람 슬롯에 조사 종류를 붙인다. split 은 문형 단위 train/val 분리.
def _f(split, items):
    return [(split, t) for t in items]


PERSON_FRAMES = (
    _f("train", [
        "{N1:sub} 방금 도착했다고 카톡 왔어", "{N1:sub} 감기 걸려서 오늘 학교 못 갔대", "주말에 {N1:sub} 할머니 댁에 가기로 했어",
        "어제 {N1:sub} 학원 버스를 놓쳤다고 하네", "{N1:sub} 새로 산 신발이 마음에 든대", "{N1:sub} 내일 현장학습 도시락 싸 달래",
        "점심 먹고 {N1:sub} 도서관에 들렀대", "요즘 {N1:sub} 통 연락이 없네", "저녁 약속은 {N1:sub} 정하기로 했어",
        "{N1:top} 숙제를 벌써 다 끝냈더라", "오늘은 {N1:top} 일찍 자야 한다고 했어", "{N1:top} 매운 음식을 잘 못 먹어",
        "내 기억에 {N1:top} 작년에도 여기 왔었어", "{N1:top} 학교 앞 문구점에서 만나기로 했어",
        "내일 {N1:obj} 병원에 데려가야 해", "방금 {N1:obj} 학원 앞에서 봤어", "엄마가 {N1:obj} 많이 찾으셨어",
        "이따가 {N1:obj} 데리러 가기로 했어", "우리가 {N1:obj} 깜빡하고 안 불렀네",
        "{N1:com} 같이 영화 보러 갈 거야", "오늘 {N1:com} 놀이터에서 놀았어", "주말에 {N1:com} 캠핑 가기로 약속했어",
        "{N1:dat} 우산 좀 가져다 달라고 해 줘", "아까 {N1:dat} 문자 보냈는데 답이 없어", "내가 {N1:dat} 직접 물어볼게",
        "{N1:voc}, 지금 어디야?", "{N1:voc} 밥 먹으러 와", "{N1:voc}! 신발 신고 나가야지",
        "{N1:pos} 가방이 현관에 그대로 있네", "{N1:pos} 생일 선물은 뭐가 좋을까", "이건 {N1:pos} 우산인 것 같아",
        "{N1:bare} 오늘 학원 쉰대", "{N1:bare} 이번 주말에 이사한다더라", "아까 {N1:bare} 전화 왔었어",
        "{N1:bare} 방에 불이 켜져 있던데", "혹시 {N1:bare} 어디 갔는지 알아?", "{N1:bare} 병원 예약 시간 바꿨다던데",
        "{N1:bare} 학부모 면담은 금요일이야", "단톡방에서 {N1:bare} 얘기가 나왔어", "그럼 {N1:bare} 몫은 따로 남겨 둘게",
        "{N1:sub} 어린이집 가방을 두고 갔어", "배고프다고 {N1:sub} 칭얼거렸어", "{N1:top} 지난번에도 지각했잖아",
        "'{N1:bare}' 이름으로 예약해 놨어", "{N1:bare}라는 아이가 우리 반에 있대",
    ]),
    _f("val", [
        "{N1:sub} 요즘 피아노 연습을 열심히 하더라", "{N1:sub} 이번 방학에 수영 배운다던데", "{N1:top} 다음 주 시험 걱정 안 해도 될 것 같아",
        "선생님이 {N1:obj} 칭찬하셨다고 전해 줘", "{N1:com} 점심 먹고 들어올게", "{N1:dat} 이 사진 먼저 보여 줄까?",
        "{N1:voc} 숙제는 다 했니?", "{N1:pos} 담임 선생님한테서 전화 왔어", "{N1:bare} 지금 샤워 중이래", "{N1:bare} 숙제 검사 받는 날이야",
    ]),
)
PERSON_FRAMES = PERSON_FRAMES[0] + PERSON_FRAMES[1]

NOUN_FRAMES = (
    _f("train", ["{W:top} 보기만 해도 마음이 편해져", "{W:sub} 가득 퍼져 있었어", "오늘 {W:obj} 한참 바라봤어", "{W:bare} 사진을 찍어 뒀어",
                 "창밖으로 {W:obj} 보고 있었어", "{W:bare} 소리가 정말 좋더라", "여행 가서 {W:obj} 실컷 봤어", "{W:bare} {P}"]),
    _f("val", ["{W:sub} 이렇게 아름다운 줄 몰랐어", "{W:bare} 냄새가 아직도 생각나", "{W:top} 사계절 내내 볼 수 있대"]),
)
NOUN_FRAMES = NOUN_FRAMES[0] + NOUN_FRAMES[1]

LOC_FRAMES = _f("train", ["주말에 {L}에 다녀왔어", "{L}에서 내리면 바로 보여", "내일 {L} 근처에서 만나자", "{L} 가는 길에 비가 오더라", "{L}까지 가려면 한 시간쯤 걸려"]) \
    + _f("val", ["다음 달에 {L} 쪽으로 이사 갈지도 몰라", "{L}에서 전화 왔다고 전해 줘"])
ORG_FRAMES = _f("train", ["{O}에서 연락이 왔어", "{O} 예약은 내일 오전이야", "오늘 {O}에 서류 내러 갔다 왔어", "{O} 앞에서 기다릴게", "{O}는 이번 주 쉰대"]) \
    + _f("val", ["{O}에 전화해서 시간 물어봤어", "내일 {O} 가는 길에 들를게"])

# 다중 개인정보 문형(여러 개체 모두 라벨). 규칙 몫 값(PH/EM/AC)은 rule_spans 로 기록.
MULTI_FRAMES = {
    "PP": _f("train", ["{N1:sub} {N2:dat} 먼저 연락했대", "{N1:com} {N2:bare} 둘 다 감기 기운이 있대", "{N1:sub} {N2:obj} 데리러 가기로 했어",
                       "{N1:top} {N2:com} 같은 반이 됐어", "{N1:sub} {N2:pos} 숙제를 도와줬어", "오늘 {N1:com} {N2:com} 영화 보러 가"])
          + _f("val", ["{N1:sub} {N2:dat} 이 얘기를 전했대", "{N1:com} {N2:bare} 같이 발표한대"]),
    "PPP": _f("train", ["{N1:com} {N2:com} {N3:sub} 같이 갔다 왔대", "{N1:sub} {N2:dat} {N3:obj} 소개해 줬어"]) + _f("val", ["{N1:top} {N2:com} {N3:com} 한 팀이래"]),
    "PA": _f("train", ["{N1:dat} 택배 보낼 주소는 {A}, 맞지?", "{N1:sub} {A} 쪽으로 이사했다던데", "{A}에 있는 {N1:dat} 이 서류 전해 줘", "{N1:pos} 집 주소가 {A}라고 했어"])
          + _f("val", ["{N1:sub} 새로 이사한 곳이 {A} 쪽이래"]),
    "PO": _f("train", ["{N1:sub} {O}에 다닌대", "{O}에서 {N1:obj} 찾는 전화가 왔어", "{N1:pos} 진료는 {O}에서 받기로 했어"]) + _f("val", ["{N1:sub} {O} 선생님이랑 상담했대"]),
    "PL": _f("train", ["{N1:com} {L}에서 만나기로 했어", "{N1:sub} {L}에 놀러 갔다 왔대", "{L} 근처에서 {N1:obj} 봤어"]) + _f("val", ["{N1:sub} {L}까지 가는 길이래"]),
    "AO": _f("train", ["{O} 사무실은 {A}에 있어", "{O} 주소가 {A}라고 들었어"]) + _f("val", ["{A}에 있는 {O}에 다녀왔어"]),
    "PPH": _f("train", ["{N1:dat} 연락하려면 {PH}로 전화해", "{N1:pos} 번호는 {PH} 맞아"]) + _f("val", ["{N1:dat} {PH}로 문자 남겨 줘"]),
    "PEM": _f("train", ["{N1:pos} 이메일은 {EM} 맞아", "{N1:dat} {EM}로 자료 보내 줘"]) + _f("val", ["{N1:sub} {EM}로 메일 줬어"]),
    "PAC": _f("train", ["{N1:dat} 보낼 계좌는 {AC}, 확인해 줘", "{N1:pos} 입금 계좌가 {AC}라고 했어"]) + _f("val", ["{N1:sub} 알려 준 계좌는 {AC} 맞대"]),
    "PAPH": _f("train", ["{N1:dat} {A}로 보내고 {PH}로 연락해 줘", "{N1:sub} {A}에 산대, 번호는 {PH} 맞아"]) + _f("val", ["{N1:pos} 집은 {A}이고 번호는 {PH}라고 했어"]),
}


# ----------------------------------------------------------------------------- 생성기
def frame_text():
    parts = [t for _s, t in PERSON_FRAMES + NOUN_FRAMES + LOC_FRAMES + ORG_FRAMES]
    for v in MULTI_FRAMES.values():
        parts += [t for _s, t in v]
    return " ".join(parts) + " " + " ".join(SPOKEN_END) + " " + " ".join(NATURE_PRED)


class Ctx:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.exact, self.given, self.words, self.dev_sents = collect_dev_blocks()
        self.sealed = load_sealed()
        self.used_names = {}  # 이름 문자열 -> split
        self.all_entity_strings = set()
        self.word_block = set(NATURE) | set(NAMEWORD) | set(WORDISH) | set(synth.GENERIC_NOUNS) | set(synth.GENERIC_PLACES) | set(synth.KIN_TERMS) | set(synth.KINS)
        self.ft = frame_text() + " " + " ".join(NATURE)
        self.loc_pool = [x for x in synth.CITIES + synth.STATIONS + synth.LANDMARKS if x not in self.exact and x not in self.sealed["values"]]
        self.org_pool = self._org_pool()

    def _org_pool(self):
        out = []
        for p in synth.HOSP_PREFIX:
            for s in synth.HOSP_SUFFIX:
                out.append(p + s)
        for p in synth.SCHOOL_PREFIX:
            for s in synth.SCHOOL_SUFFIX:
                out.append(p + s)
        for p in synth.COMPANY_PREFIX:
            for s in synth.COMPANY_SUFFIX:
                out.append(p + s)
        return [x for x in out if x not in self.exact and x not in self.sealed["values"]]

    def name_ok(self, full, given):
        """개발·봉인 평가셋과 겹치지 않는 이름인지."""
        if full in self.exact or given in self.given or given in self.exact:
            return False
        if full in SEEN_NEW180_NAMES or given in SEEN_NEW180_NAMES:
            return False
        sv = self.sealed["values"]
        for v in sv:  # 봉인 세트 값과 같거나 서로 포함되면 제외(강한 격리)
            if len(v) >= 2 and (full == v or given == v or (len(given) >= 2 and (given in v or v in full))):
                return False
        if given in self.word_block or full in self.word_block:
            return False
        if given in self.ft or full in self.ft:
            return False
        return True

    def split_of(self, key):
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100
        return "val" if h < 16 else "train"


def sylnames(ctx, a_set, b_set, n, rng, ban_double=True):
    out, seen = [], set()
    cand = [a + b for a in a_set for b in b_set if not (ban_double and a == b)]
    rng.shuffle(cand)
    for g in cand:
        if g in seen or not ctx.name_ok(g, g):
            continue
        seen.add(g); out.append(g)
        if len(out) >= n:
            break
    return out


def full_names(ctx, givens, rng, comp=False):
    """given 목록에서 성+이름 후보(차단 통과)를 만든다. (full, given)"""
    out = []
    for g in givens:
        s = rng.choice(COMPOUND) if comp else rng.choice(SURN)
        full = s + g
        if ctx.name_ok(full, g):
            out.append((full, g))
    return out


def fill(frame, vals, rng):
    """frame 의 {K:조사} 슬롯을 채우고 (text, [(slot, start, end, value)]) 를 돌려준다."""
    parts, out, pos = [], [], 0
    text = ""
    for m in re.finditer(r"\{(\w+)(?::(\w+))?\}", frame):
        text += frame[pos:m.start()]
        slot, key = m.group(1), m.group(2)
        v = vals[slot]
        s = len(text)
        if slot.startswith("N") or slot == "W":
            surf = attach(v, key or "bare", rng) if slot != "W" else attach_noun(v, key or "bare")
            text += surf
            out.append((slot, s, s + len(v), v))
        else:
            text += v
            out.append((slot, s, s + len(v), v))
        pos = m.end()
    text += frame[pos:]
    return text, out


def decorate(text, rng, p=0.35):
    if rng.random() < p:
        text = rng.choice(LEADS) + text + rng.choice(TAILS)
    return text


def make_row(rid, text, spans, cat, split, rule_spans=None, source="r8"):
    row = {"id": rid, "text": text, "spans": spans, "source": "synth", "template_id": f"r8-{cat}", "category": cat, "split": split, "augment": True, "r8": True}
    if rule_spans:
        row["rule_spans"] = rule_spans
    return row


def addr(rng):
    k = rng.randrange(3)
    if k == 0:
        return f"{rng.choice(CITY_SI)} {rng.choice(synth.ROADS)} {rng.randint(3, 480)}"
    if k == 1:
        return f"{rng.choice(CITY_SI)} {rng.choice(synth.DONGS)} {rng.randint(1, 320)}-{rng.randint(1, 40)}"
    return f"{rng.choice(synth.ROADS)} {rng.randint(3, 300)} {rng.choice(synth.COMPLEXES)}아파트 {rng.randint(101, 140)}동 {rng.randint(101, 1804)}호"


def phone(rng):
    return f"010-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"


def email(rng):
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(5, 8))) + str(rng.randint(1, 99)) + "@" + rng.choice(DOMAINS)


def account(rng):
    return f"{rng.choice(BANKS)}은행 {rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(100000, 999999)}"


def build(seed, scale):
    ctx = Ctx(seed)
    rng = ctx.rng
    rows = []
    rid = [0]
    seen_text = set()
    rejected = Counter()

    def add(text, spans, cat, split, rule=None):
        for sp_ in spans:
            v_ = text[sp_["start"]:sp_["end"]]
            if any(len(x) >= 2 and (v_ == x or (sp_["label"] == "PERSON" and (v_ in x or x in v_))) for x in ctx.sealed["values"]):
                rejected["sealed-value-conflict"] += 1
                return
        key = norm(text)
        if key in seen_text:
            rejected["dup"] += 1
            return
        seen_text.add(key)
        rid[0] += 1
        rows.append(make_row(f"r8-{rid[0]:05d}", text, spans, cat, split, rule))

    frames_by_split = lambda frames, sp: [t for s, t in frames if s == sp]

    def person_rows(cat, pairs, frames_filter=None, n_per_name=1, decor=0.35):
        """pairs: [(full, given_or_None)] 사람 이름 목록."""
        for full, _g in pairs:
            sp = ctx.split_of(full)
            ctx.used_names[full] = sp
            fr = frames_by_split(PERSON_FRAMES, sp)
            for _ in range(n_per_name):
                f = rng.choice(fr if not frames_filter else [x for x in fr if frames_filter(x)] or fr)
                text, slots = fill(f, {"N1": full}, rng)
                text = decorate(text, rng, decor) if not text.startswith("'") else text
                if text != fill(f, {"N1": full}, rng)[0] and decor:
                    # 장식으로 앞 글자가 밀리므로 위치를 다시 찾는다
                    pass
                s = text.find(full)
                if s < 0:
                    rejected["lost-name"] += 1
                    continue
                add(text, [{"label": "PERSON", "start": s, "end": s + len(full)}], cat, sp)

    # --- A1: 성 없는 2글자 이름(일반 + 흔한 단어 모양) -------------------------------------------------
    given2 = sylnames(ctx, A_SYL, B_SYL, int(900 * scale), rng)
    bare_only = lambda f: ":bare" in f or ":sub" in f
    person_rows("A1_given2", [(g, g) for g in given2[: int(620 * scale)]], n_per_name=1)
    ft = frame_text()
    wish = [w for w in WORDISH if w not in ft and w not in " ".join(NATURE) and w not in ctx.exact and w not in ctx.sealed["values"]]  # 의도적으로 단어 모양 이름
    person_rows("A1_wordish", [(w, w) for w in wish for _ in range(max(1, int(4 * scale)))], n_per_name=1)
    # --- A1b: 흔한 3글자 / 낯선 3글자 이름 ----------------------------------------------------------------
    common3 = full_names(ctx, given2[int(620 * scale): int(620 * scale) + int(160 * scale)], rng)
    person_rows("A1_full3", common3)
    # --- B1: 끝 글자 위험(조사 없는 문형 위주) 3글자 + 4글자 -----------------------------------------------
    risky_given = sylnames(ctx, A_SYL, "".join(RISKY_END), int(300 * scale), rng)
    risky_full = full_names(ctx, risky_given, rng)
    no_josa = lambda f: ":bare" in f
    person_rows("B1_end3", risky_full, frames_filter=no_josa)
    comp = full_names(ctx, sylnames(ctx, A_SYL, B_SYL, int(90 * scale), rng), rng, comp=True)  # 복성 + 2글자 = 4글자
    f3 = [(rng.choice(SURN) + g, g) for g in FOREIGN3 + FOLK3]
    f3 = [(f, g) for f, g in f3 if ctx.name_ok(f, g)]
    person_rows("B1_end4", comp + f3 * max(1, int(3 * scale)), frames_filter=no_josa)
    # --- A3: 지명·기관 모양 이름 (사람 문맥) + 실제 지명·기관 (LOC/ORG 문맥) ---------------------------------
    place_given = sylnames(ctx, PLACE_A, PLACE_B, int(260 * scale), rng)
    person_rows("A3_placelike2", [(g, g) for g in place_given[: int(160 * scale)]])
    person_rows("A3_placelike3", full_names(ctx, place_given[int(160 * scale):], rng))
    for sp, locs, orgs in (("train", ctx.loc_pool, ctx.org_pool), ("val", ctx.loc_pool, ctx.org_pool)):
        locs_s = [x for x in locs if ctx.split_of(x) == sp]
        orgs_s = [x for x in orgs if ctx.split_of(x) == sp]
        lf, of = frames_by_split(LOC_FRAMES, sp), frames_by_split(ORG_FRAMES, sp)
        for x in rng.sample(locs_s, min(len(locs_s), int((80 if sp == "train" else 14) * scale))):
            t, sl = fill(rng.choice(lf), {"L": x}, rng); s = t.find(x)
            add(t, [{"label": "LOCATION", "start": s, "end": s + len(x)}], "A3_loc_context", sp); ctx.all_entity_strings.add(x)
        for x in rng.sample(orgs_s, min(len(orgs_s), int((80 if sp == "train" else 14) * scale))):
            t, sl = fill(rng.choice(of), {"O": x}, rng); s = t.find(x)
            add(t, [{"label": "ORGANIZATION", "start": s, "end": s + len(x)}], "A3_org_context", sp); ctx.all_entity_strings.add(x)
    # --- D: 자연어·이름 같은 단어 (O) / 단어 모양 이름 대조 -------------------------------------------------
    nat = [w for w in NATURE if w not in ft and w not in ctx.exact and w not in ctx.words and w not in ctx.sealed["values"]]
    ctx.rejected_nature = len(NATURE) - len(nat)
    for w in nat:
        sp = ctx.split_of(w)
        nf = frames_by_split(NOUN_FRAMES, sp)
        for _ in range(max(1, int((4 if len(w) == 2 else 2) * scale))):
            f = rng.choice(nf)
            p = rng.choice(NATURE_PRED)
            t, sl = fill(f.replace("{P}", p), {"W": w}, rng)
            add(decorate(t, rng, 0.25), [], "D1_nature_neg", sp)
    ft = ft + " " + " ".join(NATURE)
    nw = [w for w in NAMEWORD if w not in ft and w not in ctx.exact and w not in ctx.words and w not in ctx.sealed["values"]]
    for w in nw:
        sp = ctx.split_of(w)
        pf = frames_by_split(PERSON_FRAMES, sp)
        nf = frames_by_split(NOUN_FRAMES, sp)
        for _ in range(max(1, int(3 * scale))):
            f = rng.choice(pf)
            t, sl = fill(f, {"N1": w}, rng); s = t.find(w)
            add(t, [{"label": "PERSON", "start": s, "end": s + len(w)}], "D1_nameword_as_person", sp)
            ctx.used_names[w] = sp
        for _ in range(max(1, int(4 * scale)) if w in NAMEWORD_NOUN_OK else 0):
            f = rng.choice([x for x in nf if "{P}" not in x])
            t, sl = fill(f, {"W": w}, rng)
            add(t, [], "D1_nameword_as_noun", sp)
    for sp in ("train", "val"):
        for e in SPOKEN_END:
            if (sp == "val") != (ctx.split_of(e) == "val"):
                continue
            for pre in rng.sample(["엄마가 ", "선생님이 ", "삼촌이 ", "아빠가 ", "이모가 ", "친구가 ", "언니가 "], 3):
                add(decorate(pre + e, rng, 0.2), [], "D2_spoken_ending_neg", sp)
    kin_terms = ["엄마", "아빠", "할머니", "할아버지", "삼촌", "이모", "고모", "언니", "오빠", "누나", "형", "선생님", "친구", "동생", "이모부", "고모부"]
    for sp in ("train", "val"):
        for f in frames_by_split(PERSON_FRAMES, sp):
            for k in rng.sample(kin_terms, min(len(kin_terms), max(2, int(3 * scale)))):
                t, sl = fill(f, {"N1": k}, rng)
                add(decorate(t, rng, 0.2), [], "D3_kin_neg", sp)
    # --- M: 다중 개인정보 문장 ------------------------------------------------------------------------------
    pool_full = [f for f, _ in full_names(ctx, sylnames(ctx, A_SYL, B_SYL, int(420 * scale), rng), rng)]
    pool_given = sylnames(ctx, A_SYL, B_SYL, int(300 * scale), rng)
    people = pool_full + pool_given

    def pick_people(k):
        return rng.sample(people, k)

    counts = {"PP": 110, "PPP": 45, "PA": 70, "PO": 55, "PL": 45, "AO": 25, "PPH": 35, "PEM": 30, "PAC": 30, "PAPH": 35}
    for kind, n in counts.items():
        n = int(n * scale)
        for _ in range(n):
            names = pick_people(3 if kind == "PPP" else 2)
            sp = "val" if all(ctx.split_of(x) == "val" for x in names[:1]) else "train"
            frs = frames_by_split(MULTI_FRAMES[kind], sp) or frames_by_split(MULTI_FRAMES[kind], "train")
            f = rng.choice(frs)
            vals = {"N1": names[0], "N2": names[1] if len(names) > 1 else None, "N3": names[2] if len(names) > 2 else None}
            vals = {k: v for k, v in vals.items() if v}
            if "{A}" in f:
                vals["A"] = addr(rng)
            if "{O}" in f:
                vals["O"] = rng.choice([x for x in ctx.org_pool if ctx.split_of(x) == sp] or ctx.org_pool)
            if "{L}" in f:
                vals["L"] = rng.choice([x for x in ctx.loc_pool if ctx.split_of(x) == sp] or ctx.loc_pool)
            if "{PH}" in f:
                vals["PH"] = phone(rng)
            if "{EM}" in f:
                vals["EM"] = email(rng)
            if "{AC}" in f:
                vals["AC"] = account(rng)
            for slot in ("N1", "N2", "N3"):
                if slot in vals:
                    ctx.used_names[vals[slot]] = sp
            text, slots = fill(f, vals, rng)
            spans, rule = [], []
            for slot, s, e, v in slots:
                lab = {"N1": "PERSON", "N2": "PERSON", "N3": "PERSON", "A": "ADDRESS", "O": "ORGANIZATION", "L": "LOCATION"}.get(slot)
                if lab:
                    spans.append({"label": lab, "start": s, "end": e})
                else:
                    rule.append({"kind": {"PH": "PHONE", "EM": "EMAIL", "AC": "ACCOUNT"}[slot], "start": s, "end": e})
            add(text, spans, f"M_{kind}", sp, rule)
    return ctx, rows, rejected


# ----------------------------------------------------------------------------- 검증
def validate(ctx, rows):
    problems = []
    names_in_use = set(ctx.used_names)
    ent_strings = ctx.all_entity_strings
    seen = set()
    for r in rows:
        t = r["text"]
        if r["id"] in seen:
            problems.append(("dup-id", r["id"]))
        seen.add(r["id"])
        sp = sorted(r["spans"], key=lambda s: s["start"])
        for a, b in zip(sp, sp[1:]):
            if b["start"] < a["end"]:
                problems.append(("overlap", r["id"]))
        for s in sp:
            if not (0 <= s["start"] < s["end"] <= len(t)):
                problems.append(("bounds", r["id"]))
            if s["label"] not in NER_LABELS:
                problems.append(("label", r["id"], s["label"]))
        covered = [False] * len(t)
        for s in sp:
            for i in range(s["start"], s["end"]):
                covered[i] = True
        # 정답 누락: 이번에 쓴 이름/지명/기관 문자열이 문장에 있는데 라벨이 안 덮은 경우
        if r["category"] == "D1_nameword_as_noun":
            if r["spans"]:
                problems.append(("noun-contrast-has-span", r["id"]))
            continue  # 이름 단어를 일부러 명사로 쓴 대조 문장(라벨 없음이 정답)
        for nm in names_in_use | ent_strings:
            if len(nm) < 2:
                continue
            i = t.find(nm)
            while i >= 0:
                if not any(covered[i:i + len(nm)]):
                    # 라벨 구간과 한 글자도 안 겹치는 출현만 정답 누락으로 본다(다른 이름의 조사와 우연히 같은 경우는 제외)
                    problems.append(("unlabeled-entity", r["id"], nm, t))
                    break
                i = t.find(nm, i + 1)
        for rs in r.get("rule_spans", []):
            if not (0 <= rs["start"] < rs["end"] <= len(t)):
                problems.append(("rule-bounds", r["id"]))
        if r["category"].endswith("_neg") and r["spans"]:
            problems.append(("neg-has-span", r["id"]))
    return problems


def leak_check(ctx, rows):
    sealed = ctx.sealed
    out = {"sealed_rows_checked": sealed["rows"], "sentence_exact_or_norm": 0, "skeleton_equal": 0, "skeleton_4gram_sim_ge_0.6": 0,
           "span_value_overlap_with_sealed": 0, "dev_sentence_overlap": 0, "seen12_names": 0}
    for r in rows:
        n = norm(r["text"])
        if n in sealed["sents"]:
            out["sentence_exact_or_norm"] += 1
        sk = skeleton_of(r["text"], r["spans"])
        if sk in sealed["skels"]:
            out["skeleton_equal"] += 1
        g = ngrams(sk)
        if g and any(len(g & h) / max(1, len(g | h)) >= 0.6 for h in sealed["grams"]):
            out["skeleton_4gram_sim_ge_0.6"] += 1
        for s in r["spans"]:
            v = r["text"][s["start"]:s["end"]]
            if any(len(x) >= 2 and (v == x or (s["label"] == "PERSON" and (v in x or x in v))) for x in sealed["values"]):
                out["span_value_overlap_with_sealed"] += 1
            if v in SEEN_NEW180_NAMES:
                out["seen12_names"] += 1
        if n in ctx.dev_sents:
            out["dev_sentence_overlap"] += 1
    return out


def write(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ctx, rows, rejected = build(a.seed, a.scale)
    probs = validate(ctx, rows)
    leak = leak_check(ctx, rows)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    # 사용 이름이 train/val 서로소인지
    tn = {n for n, s in ctx.used_names.items() if s == "train"}
    vn = {n for n, s in ctx.used_names.items() if s == "val"}
    report = {
        "seed": a.seed, "scale": a.scale, "rows": len(rows), "train": len(train), "val": len(val),
        "by_category": dict(Counter((r["category"], r["split"]) for r in rows) and {f"{k[0]}|{k[1]}": v for k, v in Counter((r["category"], r["split"]) for r in rows).items()}),
        "label_counts": dict(Counter(s["label"] for r in rows for s in r["spans"])),
        "negatives": sum(1 for r in rows if not r["spans"]),
        "multi_entity_rows": sum(1 for r in rows if len(r["spans"]) >= 2),
        "rule_span_rows": sum(1 for r in rows if r.get("rule_spans")),
        "distinct_person_values": len(ctx.used_names), "train_val_name_overlap": len(tn & vn),
        "distinct_skeletons": len({skeleton_of(r["text"], r["spans"]) for r in rows}),
        "rejected": dict(rejected), "problems": len(probs), "problem_examples": probs[:40],
        "leak": leak, "nature_words_dropped_as_dev_overlap": getattr(ctx, "rejected_nature", None),
        "len_dist_person": dict(sorted(Counter(len(r["text"][s["start"]:s["end"]]) for r in rows for s in r["spans"] if s["label"] == "PERSON").items())),
    }
    write(out / "r8_train_extra.jsonl", train)
    write(out / "r8_val_extra.jsonl", val)
    (out / "r8_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rows", "train", "val", "negatives", "multi_entity_rows", "problems", "train_val_name_overlap", "leak", "rejected")}, ensure_ascii=False))
    bad = probs or any(v for k, v in leak.items() if k not in ("sealed_rows_checked",)) or tn & vn
    if bad:
        print("검증/격리 실패", file=sys.stderr)
        return 1
    print("검증·격리 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
