"""Baseline v2 평가 세트(train/val/test.jsonl) 생성기. 규칙의 한계를 드러내는 어려운 합성 데이터.

기존 pii_eval*.jsonl 은 규칙을 만든 쪽이 같이 쓴 템플릿이라 점수가 이미 거의 1.0 이다. 이 세트는 한글 숫자·OCR 오인식·
비정상 띄어쓰기·약식 주소·호칭 없는 이름·체크섬/Luhn 실패 번호·ORG 같은 **규칙이 못 잡는 유형**을 일부러 모았다.

- 구조: 범주(category)마다 문장 틀(frame) 5~10개 -> 틀 단위로 train/val/test 에 약 60/20/20 으로 나눈다(template_id = 틀).
  같은 틀의 변형(값만 다름)은 항상 같은 분할에 들어가므로 분할 사이에 템플릿이 겹치지 않는다.
  val/test 는 train 과 이름 풀·가족 호칭 풀·도로명 풀이 겹치지 않게 했다.
- 모든 값은 합성이다(난수). 전화번호는 010-0000 계열을 피한 임의 값, 이메일은 예약 도메인(example.*)만 쓴다.
- 라벨 기준은 LABELING_GUIDE.md. 시드 고정(SEED)이라 다시 만들면 같은 파일이 나온다.

프로필: ``safe``(공개 저장소용 안전 픽스처: 미래 출생 주민번호, 공개 시험 PAN, 010-0000-NNNN) 와 ``private``(유효한 전체
식별자, apps/api/data/private_eval/ 에만 쓴다, git 무시). 문장 틀·분할·라벨은 같고 값만 다르다(FIXTURE_SAFETY.md).
라벨은 세분 라벨(PHONE_PERSONAL/PUBLIC, LOCATION, PERSON_ALIAS, ORGANIZATION …) + 정책 필드(mask_default, context_sensitive).

사용(apps/api 에서): ``python app/shared/privacy/eval/baseline_v2/build_baseline_v2.py [--profile safe|private|both]``
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

SEED = 20260827
OUT_DIR = Path(__file__).parent
VARIANTS = 4  # 틀 하나당 변형 수
TOKEN = re.compile(r"<(?:([A-Z_]+):)?(\w+)>")


# ---------------------------------------------------------------------------
# 값 생성기 (합성). ctx: rng, split, known(가족 사전), tags(추가 태그)
# ---------------------------------------------------------------------------
class Ctx:
    def __init__(self, rng: random.Random, split: str, safe: bool = True) -> None:
        self.rng, self.split, self.safe = rng, split, safe
        self.known: list[str] = []
        self.tags: set[str] = set()


SURNAME = {
    "train": list("김이박최정강조윤"),
    "val": list("장임한오"),
    "test": list("서신권황송홍"),
}
GIVEN = {
    "train": ["민수", "영희", "철수", "지훈", "수빈", "현우", "서연", "도윤"],
    "val": ["재민", "서하", "은서", "태윤"],
    "test": ["승현", "유진", "다온", "시우", "주원", "예린"],
}
GIVEN_OPEN = {  # 받침 없는 이름(조사 구분이 단순)
    "train": ["지우", "서아", "하나", "다래", "보라", "유나", "소라", "미래"],
    "val": ["채아", "하라", "서린", "리아"],
    "test": ["도아", "예나", "시아", "나유"],
}
COMPOUND = {"train": ["남궁", "황보"], "val": ["제갈"], "test": ["선우", "독고"]}
KIN = {
    "train": ["엄마", "아빠", "할머니", "누나", "형", "동생"],
    "val": ["외할머니", "고모", "큰누나", "외삼촌"],
    "test": ["울 엄마", "큰아들", "막내딸", "시어머니", "장모님", "작은이모"],
}
NICK = {
    "train": ["콩이", "밤톨이", "똘이", "찡이", "뽀미", "쭈니"],
    "val": ["몽이", "솜이", "쭈쭈", "달봉이"],
    "test": ["새롬이", "다솜이", "꾸꾸", "몽실이", "쫑이"],
}
ROAD = {
    "train": ["테헤란로", "세종대로", "판교역로", "중앙로", "해운대해변로", "월드컵로"],
    "val": ["올림픽로", "학동로", "충장로"],
    "test": ["도산대로", "송파대로", "동백로", "대학로"],
}
DONG = {
    "train": ["합정동", "역삼동", "서초동", "연남동"],
    "val": ["잠실동", "망원동"],
    "test": ["성수동", "방배동", "상도동"],
}
REGION = [
    ("서울특별시", "강남구"),
    ("서울특별시", "마포구"),
    ("서울시", "송파구"),
    ("경기도 성남시", "분당구"),
    ("경기도 수원시", "영통구"),
    ("부산광역시", "해운대구"),
    ("대구광역시", "수성구"),
    ("인천광역시", "연수구"),
]
STATION = {
    "train": ["강남역", "홍대입구역", "잠실역", "수원역"],
    "val": ["신촌역", "판교역"],
    "test": ["건대입구역", "서울숲역", "광교중앙역"],
}
ORG_PRE = {
    "train": ["한빛", "새봄", "가람", "푸른마을", "미래"],
    "val": ["늘솜", "다솜"],
    "test": ["보람", "해오름", "누리"],
}
ORG_SUF = ["초등학교", "병원", "어린이집", "학원", "약국", "전자", "치과"]
BANKS = [
    "국민은행",
    "신한은행",
    "농협",
    "기업은행",
    "카카오뱅크",
    "토스뱅크",
    "하나은행",
    "우리은행",
]
ID_PREFIX = list("abcdefghjkmnpqrstuvwxyz")
WORDS = ["김치찌개", "박수", "정리", "최고", "이정표", "강아지", "조용히", "강한", "오늘", "한참"]
DIGIT_KO = "공일이삼사오육칠팔구"
OCR_MAP = {"0": "O", "1": "l", "5": "S", "8": "B"}


def _d(rng: random.Random, n: int, lo: int = 0) -> str:
    return "".join(str(rng.randint(lo, 9)) for _ in range(n))


def _num4(rng: random.Random) -> str:
    return str(rng.randint(1000, 9999))


def _mobile(c: "Ctx") -> tuple[str, str, str]:
    """휴대전화 (앞, 가운데, 끝). safe: 가운데 블록을 0000 으로 고정(010-0000-NNNN, 미확인 가정, README 참고).
    private: 임의의 실제 형식 번호."""
    rng = c.rng
    if c.safe:
        return "010", "0000", _num4(rng)
    p = rng.choice(["010", "010", "010", "011", "016"])
    mid = str(rng.randint(1000, 9999)) if p == "010" else str(rng.randint(200, 999))
    return p, mid, _num4(rng)


def _luhn(body: str) -> str:
    total, flip = 0, True
    for ch in reversed(body):
        d = int(ch)
        if flip:
            d *= 2
            d -= 9 if d > 9 else 0
        total += d
        flip = not flip
    return body + str((10 - total % 10) % 10)


def _luhn_ok(digits: str) -> bool:
    return _luhn(digits[:-1]) == digits


# 결제사 공개 문서(Stripe·PayPal·Braintree 등)에 시험용으로 실린 PAN. Luhn 을 통과하는 것만 쓴다.
PUBLIC_TEST_PANS = [
    p
    for p in (
        "4111111111111111",
        "4242424242424242",
        "4012888888881881",
        "4000056655665556",
        "4000000000000002",
        "5555555555554444",
        "5105105105105100",
        "5200828282828210",
        "2223003122003222",
        "6011111111111117",
        "6011000990139424",
        "3566002020360505",
        "3530111333300000",
    )
    if len(p) == 16 and _luhn_ok(p)
]


def _card16(c: "Ctx") -> str:
    """safe: 공개 시험 PAN 중 하나. private: 임의의 Luhn 유효 번호."""
    if c.safe:
        return c.rng.choice(PUBLIC_TEST_PANS)
    head = c.rng.choice(["4", "51", "52", "53"])
    return _luhn(head + _d(c.rng, 15 - len(head)))


def _rrn_parts(c: "Ctx") -> tuple[str, str, str]:
    """주민번호 (생년월일, 성별+일련 6자리, 검증번호). safe: 2030년 이후 출생(성별 3/4)이라 실존할 수 없다.
    private: 1955~2012년대 출생의 현실적인 값."""
    rng = c.rng
    if c.safe:
        yy, gender = rng.randint(30, 99), rng.choice([3, 4])
    else:
        yy = rng.randint(55, 99) if rng.random() < 0.7 else rng.randint(0, 12)
        gender = rng.choice([1, 2]) if yy >= 55 else rng.choice([3, 4])
    mm = rng.randint(1, 12)
    dd = rng.randint(1, 28)
    front = f"{yy:02d}{mm:02d}{dd:02d}"
    back = str(gender) + _d(rng, 5)
    return front, back, _rrn_check(front + back)


def _rrn_check(twelve: str) -> str:
    w = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)
    total = sum(int(d) * x for d, x in zip(twelve, w, strict=True))
    return str((11 - total % 11) % 10)


def _hangul(digits: str, rng: random.Random | None = None) -> str:
    return "".join(DIGIT_KO[int(c)] for c in digits)


def _ocr(s: str, rng: random.Random) -> str:
    cand = [i for i, c in enumerate(s) if c in OCR_MAP]
    if not cand:
        return s
    k = min(len(cand), rng.randint(2, 4))
    out = list(s)
    for i in rng.sample(cand, k):
        out[i] = OCR_MAP[out[i]]
    return "".join(out)


def g_phone_dash(c: Ctx) -> str:
    return "-".join(_mobile(c))


def g_phone_plain(c: Ctx) -> str:
    return "".join(_mobile(c))


def g_phone_space(c: Ctx) -> str:
    return " ".join(_mobile(c))


def g_phone_dot(c: Ctx) -> str:
    return ".".join(_mobile(c))


def g_phone_intl(c: Ctx) -> str:
    _, mid, last = _mobile(c)
    return f"+82-10-{mid}-{last}"


def g_phone_land(c: Ctx) -> str:
    area = c.rng.choice(["02", "031", "032", "051"])
    mid = "000" if c.safe else str(c.rng.randint(200, 999))
    return f"{area}-{mid}-{_num4(c.rng)}"


def g_phone_odd(c: Ctx) -> str:
    c.tags.add("odd_space")
    p, m, l_ = _mobile(c)
    return c.rng.choice(
        [
            f"{p} - {m} - {l_}",
            " ".join(p + m + l_),
            f"{p} {m[:2]} {m[2:]} {l_}",
            f"{p}-{m[:2]} {m[2:]}-{l_}",
            f"{p[:1]} {p[1:]} {m} {l_}",
            f"{p} {m} {l_[:2]} {l_[2:]}",
        ]
    )


def g_phone_hangul(c: Ctx) -> str:
    c.tags.add("hangul_num")
    p, m, l_ = _mobile(c)
    return " ".join([_hangul(p), _hangul(m), _hangul(l_)])


def g_phone_hangul_dash(c: Ctx) -> str:
    c.tags.add("hangul_num")
    p, m, l_ = _mobile(c)
    return "-".join([_hangul(p), _hangul(m), _hangul(l_)])


def g_phone_hangul_mix(c: Ctx) -> str:
    c.tags.add("hangul_num")
    p, m, l_ = _mobile(c)
    return c.rng.choice(
        [f"{p} {_hangul(m)} {_hangul(l_)}", _hangul(p + m + l_), f"{p}-{m}-{_hangul(l_)}"]
    )


def g_phone_ocr(c: Ctx) -> str:
    c.tags.add("ocr_noise")
    p, m, l_ = _mobile(c)
    sep = c.rng.choice(["-", " ", "·", "_", " - "])
    s = _ocr(sep.join([p, m, l_]), c.rng)
    return s if any(ch in s for ch in "OlSB") else s.replace("0", "O", 1)


def g_email(c: Ctx) -> str:
    rng = c.rng
    local = (
        rng.choice(ID_PREFIX)
        + rng.choice(["", "."])
        + "".join(rng.choices("abcdefghijkmnoprstuw", k=5))
    )
    return (
        f"{local}{rng.randint(1, 99)}@{rng.choice(['example.com', 'example.org', 'example.net'])}"
    )


def g_bank(c: Ctx) -> str:
    return c.rng.choice(BANKS)


def g_acct_dash(c: Ctx) -> str:
    r = c.rng
    return r.choice(
        [
            f"110-{r.randint(100, 999)}-{r.randint(100000, 999999)}",
            f"1002-{r.randint(100, 999)}-{r.randint(100000, 999999)}",
            f"{r.randint(100, 999)}-{r.randint(100000, 999999)}-{r.randint(10, 99)}-{r.randint(100, 999)}",
            f"{r.randint(1000, 9999)}-{r.randint(10, 99)}-{r.randint(100000, 999999)}",
        ]
    )


def g_acct_plain(c: Ctx) -> str:
    return _d(c.rng, c.rng.randint(11, 14), 1)


def g_acct_odd(c: Ctx) -> str:
    c.tags.add("odd_space")
    d = _d(c.rng, 12, 1)
    return c.rng.choice(
        [
            f"{d[:3]} {d[3:6]} {d[6:]}",
            f"{d[:3]} - {d[3:6]} - {d[6:]}",
            f"{d[:3]}-{d[3:7]} -{d[7:]}",
            " ".join(d[i : i + 2] for i in range(0, 12, 2)),
        ]
    )


def g_acct_hangul(c: Ctx) -> str:
    c.tags.add("hangul_num")
    d = _d(c.rng, 12, 1)
    return c.rng.choice(
        [
            f"{_hangul(d[:3])} {_hangul(d[3:6])} {_hangul(d[6:])}",
            _hangul(d),
            f"{d[:3]}-{_hangul(d[3:6])}-{_hangul(d[6:])}",
        ]
    )


def g_acct_ocr(c: Ctx) -> str:
    c.tags.add("ocr_noise")
    s = _ocr(g_acct_dash(c), c.rng)
    return s if any(ch in s for ch in "OlSB") else s.replace("1", "l", 1)


def g_card_dash(c: Ctx) -> str:
    d = _card16(c)
    return "-".join(d[i : i + 4] for i in range(0, 16, 4))


def g_card_space(c: Ctx) -> str:
    d = _card16(c)
    return " ".join(d[i : i + 4] for i in range(0, 16, 4))


def g_card_plain(c: Ctx) -> str:
    return _card16(c)


def g_card_odd(c: Ctx) -> str:
    c.tags.add("odd_space")
    d = _card16(c)
    g = [d[i : i + 4] for i in range(0, 16, 4)]
    return c.rng.choice(
        [" - ".join(g), f"{d[:8]} {d[8:]}", f"{g[0]} {g[1]}{g[2]} {g[3]}", " ".join(d)]
    )


def g_card_ocr(c: Ctx) -> str:
    c.tags.add("ocr_noise")
    s = _ocr(g_card_dash(c), c.rng)
    return s if any(ch in s for ch in "OlSB") else s.replace("0", "O", 1)


def _bad_luhn(c: Ctx) -> str:
    d = _card16(c)
    while True:
        out = d[:-1] + str((int(d[-1]) + c.rng.randint(1, 9)) % 10)
        if not _luhn_ok(out):
            return out


def g_card_badluhn(c: Ctx) -> str:
    c.tags.add("card_luhn_fail")
    d = _bad_luhn(c)
    return "-".join(d[i : i + 4] for i in range(0, 16, 4))


def g_card_badluhn_plain(c: Ctx) -> str:
    c.tags.add("card_luhn_fail")
    return _bad_luhn(c)


def g_card_typo(c: Ctx) -> str:
    """한 자리 빠뜨리거나(15자리) 이웃한 두 자리를 바꾼 오타. 결과가 Luhn 유효한 실존 가능 번호가 되지 않게 거른다."""
    c.tags.add("card_typo")
    while True:
        d = _card16(c)
        if c.rng.random() < 0.5:
            i = c.rng.randint(2, 14)
            out = d[:i] + d[i + 1 :]
            if not (_luhn_ok(out) and out[:2] in ("34", "37")):
                return "-".join([out[:4], out[4:8], out[8:12], out[12:]])
            continue
        i = c.rng.choice([i for i in range(0, 15) if d[i] != d[i + 1]])
        out = d[:i] + d[i + 1] + d[i] + d[i + 2 :]
        if not _luhn_ok(out):
            return "-".join(out[j : j + 4] for j in range(0, 16, 4))


def g_rrn_dash(c: Ctx) -> str:
    f, b, k = _rrn_parts(c)
    return f"{f}-{b}{k}"


def g_rrn_plain(c: Ctx) -> str:
    f, b, k = _rrn_parts(c)
    return f + b + k


def g_rrn_badchk(c: Ctx) -> str:
    c.tags.add("rrn_checksum_fail")
    f, b, k = _rrn_parts(c)
    k = str((int(k) + c.rng.randint(1, 9)) % 10)
    return f"{f}-{b}{k}" if c.rng.random() < 0.7 else f + b + k


def g_rrn_baddate(c: Ctx) -> str:
    c.tags.add("rrn_invalid_date")
    f, b, _ = _rrn_parts(c)
    bad = c.rng.choice(["13", "00"]) if c.rng.random() < 0.5 else None
    f = f[:2] + bad + f[4:] if bad else f[:4] + c.rng.choice(["32", "00", "35"])
    return f"{f}-{b}{_rrn_check(f + b)}"


def g_rrn_hangul(c: Ctx) -> str:
    c.tags.add("hangul_num")
    f, b, k = _rrn_parts(c)
    return f"{_hangul(f)}-{_hangul(b + k)}"


def g_rrn_ocr(c: Ctx) -> str:
    c.tags.add("ocr_noise")
    s = _ocr(g_rrn_dash(c), c.rng)
    return s if any(ch in s for ch in "OlSB") else s.replace("0", "O", 1)


def _name(c: Ctx, n: int) -> str:
    sp = SURNAME[c.split]
    if n == 2:
        return c.rng.choice(sp) + c.rng.choice(GIVEN[c.split])[0]
    if n == 3:
        return c.rng.choice(sp) + c.rng.choice(GIVEN[c.split])
    return c.rng.choice(COMPOUND[c.split]) + c.rng.choice(GIVEN[c.split])


def g_name2(c: Ctx) -> str:
    c.tags.add("name_len_2")
    return _name(c, 2)


def g_name3(c: Ctx) -> str:
    c.tags.add("name_len_3")
    return _name(c, 3)


def g_name4(c: Ctx) -> str:
    c.tags.add("name_len_4")
    return _name(c, 4)


def g_given_open(c: Ctx) -> str:
    c.tags.add("name_given_only")
    return c.rng.choice(GIVEN_OPEN[c.split])


def g_famname(c: Ctx) -> str:
    name = _name(c, 3)
    c.known.append(name)
    c.tags.add("name_known")
    return name


ALIAS_TITLE = {
    "train": ["엄마", "아빠", "할머니"],
    "val": ["이모", "고모"],
    "test": ["누나", "오빠", "삼촌"],
}


def g_alias_title(c: Ctx) -> str:
    """'지우 엄마'처럼 이름+가족 호칭으로 특정 사람을 가리키는 표현(PERSON_ALIAS)."""
    c.tags.add("alias_title")
    return c.rng.choice(GIVEN_OPEN[c.split]) + " " + c.rng.choice(ALIAS_TITLE[c.split])


def g_nick(c: Ctx) -> str:
    c.tags.add("kin_nickname_name")
    return c.rng.choice(NICK[c.split])


def g_kin(c: Ctx) -> str:
    return c.rng.choice(KIN[c.split])


def _road(c: Ctx) -> str:
    sido, gu = c.rng.choice(REGION)
    return f"{sido} {gu} {c.rng.choice(ROAD[c.split])} {c.rng.randint(1, 399)}"


def g_addr_road(c: Ctx) -> str:
    return _road(c)


def g_addr_road_unit(c: Ctx) -> str:
    return f"{_road(c)} {c.rng.randint(101, 125)}동 {c.rng.randint(101, 2004)}호"


def g_addr_jibun(c: Ctx) -> str:
    sido, gu = c.rng.choice(REGION)
    return (
        f"{sido} {gu} {c.rng.choice(DONG[c.split])} {c.rng.randint(1, 999)}-{c.rng.randint(1, 40)}"
    )


def g_addr_nospace(c: Ctx) -> str:
    c.tags.add("nospace")
    return _road(c).replace(" ", "")


def g_addr_station(c: Ctx) -> str:
    c.tags.add("addr_abbrev")
    st = c.rng.choice(STATION[c.split])
    return c.rng.choice([f"{st} 근처", f"{st} {c.rng.randint(1, 9)}번 출구 앞", f"{st} 뒤쪽 골목"])


def g_addr_apt(c: Ctx) -> str:
    c.tags.add("addr_abbrev")
    return f"아파트 {c.rng.randint(1, 30)}동 {c.rng.randint(101, 1504)}호"


def g_addr_road_short(c: Ctx) -> str:
    c.tags.add("addr_abbrev")
    return f"{c.rng.choice(ROAD[c.split])} {c.rng.randint(1, 399)}"


def g_addr_dongapt(c: Ctx) -> str:
    c.tags.add("addr_abbrev")
    return f"{c.rng.choice(DONG[c.split])} {c.rng.choice(['래미안', '푸르지오', '자이'])}아파트 {c.rng.randint(101, 130)}동"


def g_org(c: Ctx) -> str:
    return c.rng.choice(ORG_PRE[c.split]) + c.rng.choice(ORG_SUF)


def g_pw(c: Ctx) -> str:
    r = c.rng
    return r.choice("abcdefghjkmnp") + r.choice("XYZQ") + _d(r, 4, 1) + r.choice("!@#")


def g_otp(c: Ctx) -> str:
    return _d(c.rng, 6, 1)


def g_otp4(c: Ctx) -> str:
    return _d(c.rng, 4, 1)


# 음성 생성기
def g_order(c: Ctx) -> str:
    r = c.rng
    return r.choice(
        [
            f"2024{r.randint(1, 12):02d}{r.randint(1, 28):02d}-{r.randint(100000, 999999)}",
            f"ORD-{r.randint(2023, 2025)}-{r.randint(1000, 9999)}",
            f"{r.randint(10**13, 10**14 - 1)}",
        ]
    )


def g_waybill(c: Ctx) -> str:
    return _d(c.rng, 12, 1)


def g_date(c: Ctx) -> str:
    r = c.rng
    return r.choice(
        [
            f"{r.randint(2023, 2026)}년 {r.randint(1, 12)}월 {r.randint(1, 28)}일",
            f"{r.randint(2023, 2026)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}",
            f"{r.randint(22, 26)}.{r.randint(1, 12):02d}.{r.randint(1, 28):02d}",
        ]
    )


def g_time(c: Ctx) -> str:
    return f"{c.rng.randint(8, 21)}:{c.rng.choice(['00', '15', '30', '45'])}"


def g_ver(c: Ctx) -> str:
    r = c.rng
    return r.choice(
        [
            f"v{r.randint(1, 9)}.{r.randint(0, 30)}.{r.randint(0, 20)}",
            f"{r.randint(1, 9)}.{r.randint(0, 9)}.{r.randint(0, 9)}-rc{r.randint(1, 4)}",
            f"Android {r.randint(10, 15)}.{r.randint(0, 3)}.{r.randint(0, 9)}",
        ]
    )


def g_amount(c: Ctx) -> str:
    r = c.rng
    return r.choice(
        [
            f"{r.randint(1, 99)},{r.randint(100, 999)},{r.randint(100, 999)}원",
            f"{r.randint(1000000000, 9999999999)}원",
            f"{r.randint(1, 9)}억 {r.randint(1, 9)}천만원",
            f"{r.randint(100, 999)}만 {r.randint(1000, 9999)}원",
        ]
    )


def g_word(c: Ctx) -> str:
    return c.rng.choice(WORDS)


def g_service(c: Ctx) -> str:
    return c.rng.choice(["1588-1234", "1577-3321", "1644-9876", "1599-0012", "080-123-4567"])


def g_city(c: Ctx) -> str:
    return c.rng.choice(["서울", "부산", "제주", "대전", "강릉"])


def g_place(c: Ctx) -> str:
    return c.rng.choice(STATION[c.split] + ["코엑스", "한강공원", "시청 광장", "롯데월드"])


GENS = {k[2:]: v for k, v in globals().items() if k.startswith("g_")}


# 탐지 라벨(kind)과 마스킹 정책 필드를 분리한다. mask_default: 기본으로 가릴지, context_sensitive: 의료·금융 문맥에서 민감.
MASK_DEFAULT = {
    "PHONE_PERSONAL": True,
    "PHONE_PUBLIC": False,
    "EMAIL": True,
    "RRN": True,
    "CARD": True,
    "ACCOUNT": True,
    "SECRET": True,
    "ADDRESS": True,
    "LOCATION": False,
    "PERSON": True,
    "PERSON_ALIAS": True,
    "ORGANIZATION": False,
}
SENSITIVE_ORG = ("병원", "약국", "치과")


def policy(kind: str, value: str) -> dict:
    return {
        "mask_default": MASK_DEFAULT[kind],
        "context_sensitive": kind == "ORGANIZATION" and value.endswith(SENSITIVE_ORG),
    }


def render(template: str, ctx: Ctx) -> tuple[str, list[dict]]:
    text, spans, pos = "", [], 0
    for m in TOKEN.finditer(template):
        text += template[pos : m.start()]
        val = GENS[m.group(2)](ctx)
        if m.group(1):
            spans.append(
                {
                    "kind": m.group(1),
                    "start": len(text),
                    "end": len(text) + len(val),
                    **policy(m.group(1), val),
                }
            )
        text += val
        pos = m.end()
    return text + template[pos:], spans


# ---------------------------------------------------------------------------
# 범주와 문장 틀. (tag, 틀 목록). 틀은 문자열 또는 (문자열, 추가 태그).
# <KIND:gen> 은 정답 구간, <gen> 은 라벨 없는 값. 'known' 은 famname 생성기가 채운다.
# ---------------------------------------------------------------------------
CATS: list[tuple[str, list]] = [
    (
        "normal_phone",
        [
            "내 번호는 <PHONE_PERSONAL:phone_dash> 이야 저장해둬",
            "급하면 <PHONE_PERSONAL:phone_plain> 으로 전화해",
            "<kin> 핸드폰 번호 <PHONE_PERSONAL:phone_space> 맞지?",
            "해외에서는 <PHONE_PERSONAL:phone_intl> 로 걸어줘",
            "집 전화는 <PHONE_PERSONAL:phone_land> 이니까 저녁에 걸어",
        ],
    ),
    (
        "normal_email",
        [
            "메일은 <EMAIL:email> 로 보내줘",
            "<kin> 이메일 주소가 <EMAIL:email> 야",
            "확인 메일 받을 주소: <EMAIL:email>",
            "참조에 <EMAIL:email> 넣어줘",
            "학교 제출용 메일 <EMAIL:email> 로 첨부해서 보냈어",
        ],
    ),
    (
        "normal_account",
        [
            "<bank> <ACCOUNT:acct_dash> 로 보내줘",
            "송금할 계좌번호는 <bank> <ACCOUNT:acct_plain> 이야",
            "용돈은 <ACCOUNT:acct_dash> (<bank>) 로 넣어줘",
            "계좌 <ACCOUNT:acct_dash> <bank> 확인해줘",
            "<bank> <ACCOUNT:acct_dash> 로 이체 부탁",
        ],
    ),
    (
        "normal_card",
        [
            "카드번호 <CARD:card_dash> 유효기간 09/28",
            "결제는 <CARD:card_dash> 로 해줘",
            "<kin> 카드 <CARD:card_space> 등록했어",
            "신용카드 <CARD:card_plain> cvc 123",
            "체크카드 번호는 <CARD:card_dash> 야",
        ],
    ),
    (
        "normal_rrn",
        [
            "주민번호는 <RRN:rrn_dash> 야",
            "주민등록번호 <RRN:rrn_dash> 적어서 보내",
            "<kin> 주민번호 <RRN:rrn_plain> 맞아?",
            "본인확인용 <RRN:rrn_dash> 입력했어",
            "등본 떼려는데 주민번호 <RRN:rrn_dash> 뒷자리가 뭐였지",
        ],
    ),
    (
        "normal_address",
        [
            "우리 집 주소는 <ADDRESS:addr_road> 야",
            "택배는 <ADDRESS:addr_road_unit> 로 보내줘",
            "<kin> 댁 주소 <ADDRESS:addr_jibun> 맞지?",
            "배송지: <ADDRESS:addr_road>",
            "이사 간 곳은 <ADDRESS:addr_road_unit> 이래",
            "등기 받을 주소 <ADDRESS:addr_jibun> 로 해주세요",
            "병원 위치 <ADDRESS:addr_road> 니까 거기로 와",
            "주소 바뀌었어. <ADDRESS:addr_road_unit> 로 고쳐줘",
            "학교 주소가 <ADDRESS:addr_jibun> 라고 적혀 있어",
            "<ADDRESS:addr_road> 로 가면 돼",
        ],
    ),
    (
        "normal_name",
        [
            "<PERSON:famname> 씨 오늘 늦는대",
            "<PERSON:famname>님 예약 확인됐습니다",
            "<PERSON:famname> 선생님께 전화드렸어",
            "<PERSON:famname>씨 몫으로 하나 남겼어",
            "다음 주 <PERSON:famname> 씨 생일이야",
        ],
    ),
    (
        "nospace",
        [
            "내번호는<PHONE_PERSONAL:phone_plain>이야",
            "연락처<PHONE_PERSONAL:phone_plain>으로전화줘",
            "계좌번호<ACCOUNT:acct_plain>로입금해줘",
            "이메일<EMAIL:email>로보내줘",
            "주민번호<RRN:rrn_plain>입니다",
            "카드번호<CARD:card_plain>유효기간0928",
            "주소는<ADDRESS:addr_nospace>입니다",
            "제이름은<PERSON:name3>입니다",
            "<PERSON:name3>씨한테<PHONE_PERSONAL:phone_plain>로연락해줘",
            "택배는<ADDRESS:addr_nospace>로보내고연락처는<PHONE_PERSONAL:phone_plain>이야",
        ],
    ),
    (
        "odd_space",
        [
            "번호 <PHONE_PERSONAL:phone_odd> 야",
            "전화번호가 <PHONE_PERSONAL:phone_odd> 래 저장해",
            "계좌 <ACCOUNT:acct_odd> 로 보내줘",
            "<bank> <ACCOUNT:acct_odd> 입금 부탁",
            "카드번호 <CARD:card_odd> 결제해줘",
            "<kin> 번호는 <PHONE_PERSONAL:phone_odd> 로 저장돼 있어",
            "연락처: <PHONE_PERSONAL:phone_odd>",
            "송금 계좌 <ACCOUNT:acct_odd> (<bank>)",
            "결제 카드 <CARD:card_odd> 로 해줘",
            "문자 보낼 번호 <PHONE_PERSONAL:phone_odd> 확인해줘",
        ],
    ),
    (
        "name_no_title",
        [
            "<PERSON:name3> 어제 학교에서 상 받았대",
            "담임 선생님 이름이 <PERSON:name3> 이래",
            "<PERSON:name3>한테 숙제 물어봐",
            "내일 <PERSON:name3> 만나기로 했어",
            "병원 예약은 <PERSON:name3> 이름으로 해줘",
            "<PERSON:name3>도 같이 가기로 했어",
            "보호자 <PERSON:name3> 로 등록돼 있어",
            "<kin> 친구 <PERSON:name3> 아직 안 왔대",
            "수신인은 <PERSON:name3> 로 적어줘",
            "<PERSON:name3> 전화 왔었는데 받았어?",
        ],
    ),
    (
        "name_given_only",
        [
            "<PERSON:given_open>한테 간식 줬어",
            "오늘 <PERSON:given_open> 학교 일찍 끝나",
            "<PERSON:given_open>도 데려와",
            "어제 <PERSON:given_open>랑 통화했어",
            "<PERSON:given_open> 숙제는 다 했대",
        ],
    ),
    (
        "name_len_2",
        [
            "<PERSON:name2>씨한테 문자했어",
            "<PERSON:name2> 선생님이 전화하셨대",
            "<PERSON:name2> 님 예약 도와드렸어요",
            "오늘 <PERSON:name2>씨 휴가래",
            "<PERSON:name2>한테 물어보자",
        ],
    ),
    (
        "name_len_3",
        [
            "<PERSON:name3>씨 아직 안 왔네",
            "<PERSON:name3> 교수님 연구실로 가봐",
            "<PERSON:name3> 고객님 주문 접수됐어요",
            "어제 <PERSON:name3>씨랑 통화했어",
            "<PERSON:name3>에게 부탁해보자",
        ],
    ),
    (
        "name_len_4",
        [
            "<PERSON:name4>씨 계약서 보냈어요",
            "<PERSON:name4> 대표님 일정 확인해줘",
            "<PERSON:name4> 회원님 등록 완료",
            "내일 <PERSON:name4>씨 면담이야",
            "<PERSON:name4>에게 안부 전해줘",
        ],
    ),
    (
        "kin_nickname",
        [
            "<kin> 말로는 <PERSON:name3> 오늘 온대",
            ("<kin>한테 <PERSON:given_open> 얘기 했더니 놀라시더라", []),
            "<kin>도 <PERSON:name3> 연락처는 모르신대",
            "우리 <kin> 친구 이름이 <PERSON:name3>인데 잘 부탁드려요",
            "<kin> 옆에 <PERSON:given_open> 도 같이 있었어",
        ],
    ),
    (
        "alias_title",
        [
            "어제 <PERSON_ALIAS:alias_title> 만났어",
            "<PERSON_ALIAS:alias_title> 한테 전해줘",
            "<PERSON_ALIAS:alias_title> 전화번호 알아?",
            "내일 <PERSON_ALIAS:alias_title> 도 온대",
            "<PERSON_ALIAS:alias_title> 가 부탁했어",
        ],
    ),
    (
        "kin_nickname_name",
        [
            "<kin> 이 부르는 이름이 <PERSON_ALIAS:nick> 이래",
            "우리 <kin> 별명은 <PERSON_ALIAS:nick> 이야",
            "<PERSON_ALIAS:nick> 가 오늘 시험 본대",
            "<kin> 한테 <PERSON_ALIAS:nick> 숙제 봐달라고 해",
            "<PERSON_ALIAS:nick> 는 학원 갔어",
        ],
    ),
    (
        "hangul_num",
        [
            "내 번호는 <PHONE_PERSONAL:phone_hangul> 이야",
            "전화번호 <PHONE_PERSONAL:phone_hangul> 로 걸어줘",
            "<kin> 번호 <PHONE_PERSONAL:phone_hangul_mix>",
            "계좌번호 <ACCOUNT:acct_hangul> 로 보내줘",
            "<bank> <ACCOUNT:acct_hangul> 이체해줘",
            "연락처는 <PHONE_PERSONAL:phone_hangul_dash> 입니다",
            "주민번호 <RRN:rrn_hangul> 야",
            "카톡 말고 <PHONE_PERSONAL:phone_hangul> 로 문자 줘",
            "인증번호 말고 내 번호 <PHONE_PERSONAL:phone_hangul_mix> 로 연락해",
            "송금 계좌 <ACCOUNT:acct_hangul> (<bank>)",
        ],
    ),
    (
        "ocr_noise",
        [
            "스캔한 명함에 <PHONE_PERSONAL:phone_ocr> 라고 찍혔어",
            "사진 속 전화번호 <PHONE_PERSONAL:phone_ocr> 맞는지 봐줘",
            "영수증 계좌 <ACCOUNT:acct_ocr> 인식된 거야",
            "<bank> <ACCOUNT:acct_ocr> 이라고 나와",
            "카드 사진에서 읽은 번호 <CARD:card_ocr>",
            "OCR 결과 카드번호 <CARD:card_ocr> 이래",
            "신분증 스캔: 주민번호 <RRN:rrn_ocr>",
            "<kin> 번호 <PHONE_PERSONAL:phone_ocr> 로 읽혔어",
            "택배 라벨 연락처 <PHONE_PERSONAL:phone_ocr>",
            "캡처한 통장 <ACCOUNT:acct_ocr>",
        ],
    ),
    (
        "addr_abbrev",
        [
            "우리 집은 <LOCATION:addr_station> 야",
            "집이 <LOCATION:addr_station> 라서 가까워",
            "택배는 우리 <ADDRESS:addr_apt> 으로 부탁드려요",
            "나 지금 우리 <ADDRESS:addr_apt> 에 있어",
            "배송지 <ADDRESS:addr_road_short>",
            "사는 곳은 <ADDRESS:addr_road_short> 쪽이야",
            "이사한 집이 <ADDRESS:addr_dongapt> 래",
            "<kin> 집은 <LOCATION:addr_station> 에 있어",
            "문 앞에 놔주세요. 우리 <ADDRESS:addr_apt>",
            "우리 집 <ADDRESS:addr_road_short> 으로 와",
        ],
    ),
    (
        "rrn_checksum_fail",
        [
            "주민번호 <RRN:rrn_badchk> 라고 적었어",
            "<kin> 주민등록번호 <RRN:rrn_badchk> 맞나 확인해줘",
            "신청서에 <RRN:rrn_badchk> 라고 적었어",
            "본인 확인 번호는 <RRN:rrn_badchk> 입니다",
            "주민번호 뒷자리 포함해서 <RRN:rrn_badchk> 이야",
        ],
    ),
    (
        "rrn_invalid_date",
        [
            "주민번호 <RRN:rrn_baddate> 로 입력했어",
            "<kin> 주민번호 <RRN:rrn_baddate> 인가?",
            "서류에 <RRN:rrn_baddate> 라고 적혀 있어",
            "주민등록번호는 <RRN:rrn_baddate> 입니다",
            "본인확인 <RRN:rrn_baddate> 다시 해줘",
        ],
    ),
    (
        "card_luhn_fail",
        [
            "카드번호 <CARD:card_badluhn> 로 결제해줘",
            "결제 카드 <CARD:card_badluhn_plain> 야",
            "<kin> 신용카드 <CARD:card_badluhn> 등록했어",
            "체크카드 <CARD:card_badluhn_plain> cvc 456",
            "카드 <CARD:card_badluhn> 유효기간 11/27",
        ],
    ),
    (
        "card_typo",
        [
            "카드번호가 <CARD:card_typo> 로 찍혔어",
            "결제 카드 <CARD:card_typo> 야",
            "<kin> 카드 <CARD:card_typo> 등록",
            "신용카드 <CARD:card_typo> 유효기간 03/29",
            "카드 번호 잘못 쳤나 봐 <CARD:card_typo>",
        ],
    ),
    (
        "org_name",
        [
            "<kin>이 <ORGANIZATION:org>에서 일해",
            "<ORGANIZATION:org> 앞에서 만나",
            "내일 <ORGANIZATION:org> 상담 예약했어",
            "<ORGANIZATION:org> 선생님이 전화했어",
            "우리 애는 <ORGANIZATION:org> 다녀",
        ],
    ),
    (
        "secret",
        [
            "비밀번호는 <SECRET:pw> 야",
            "와이파이 비번 <SECRET:pw>",
            "인증번호 <SECRET:otp> 왔어",
            "공유기 비밀번호 <SECRET:pw> 로 바꿨어",
            "도어락 번호 <SECRET:otp4>",
        ],
    ),
    (
        "multi_pii",
        [
            "<PERSON:name3>씨 번호 <PHONE_PERSONAL:phone_dash> 이고 메일은 <EMAIL:email> 야",
            "받는 분 <PERSON:name3> / <PHONE_PERSONAL:phone_dash> / <ADDRESS:addr_road_unit>",
            "주민번호 <RRN:rrn_dash> 계좌 <bank> <ACCOUNT:acct_dash>",
            "카드 <CARD:card_dash> 비번 <SECRET:pw> 알려줄게",
            "<kin> 연락처 <PHONE_PERSONAL:phone_plain> , 집주소 <ADDRESS:addr_road>",
            "<PERSON:name3> 이름으로 <PHONE_PERSONAL:phone_space> 에 예약했고 <EMAIL:email> 로 확인메일 와",
            "<PERSON:name3>씨 <RRN:rrn_dash> / <ACCOUNT:acct_dash> (<bank>) / <PHONE_PERSONAL:phone_dash>",
            "<ORGANIZATION:org> 담당 <PERSON:name3> 연락처 <PHONE_PERSONAL:phone_dash>, <EMAIL:email>",
            "배송지 <ADDRESS:addr_road>, 수령인 <PERSON:name3>, 전화 <PHONE_PERSONAL:phone_dash>",
            "<PERSON:given_open> 번호 <PHONE_PERSONAL:phone_hangul> 계좌 <ACCOUNT:acct_odd>",
        ],
    ),
    # ---- 음성(라벨 없음)
    (
        "neg_order",
        [
            "주문번호 <order> 배송 조회해줘",
            "주문하신 <order> 상품 출고됐어요",
            "결제 승인번호 <order> 확인",
            "환불 요청 건 <order> 처리해주세요",
            "<kin>이 시킨 주문 <order> 언제 와?",
        ],
    ),
    (
        "neg_waybill",
        [
            "운송장 번호 <waybill> 조회해줘",
            "송장 <waybill> 도착 예정이래",
            "택배 <waybill> 어디쯤이야",
            "반품 송장번호는 <waybill> 이야",
            "<kin> 선물 운송장 <waybill>",
        ],
    ),
    (
        "neg_datetime",
        [
            "<date>에 만나자",
            "<date> <time> 에 병원 예약했어",
            "<kin> 생일은 <date> 야",
            "회의는 <date> <time> 로 옮겼어",
            "<date> 까지 서류 내야 해",
        ],
    ),
    (
        "neg_version",
        [
            "앱 버전 <ver> 로 업데이트해",
            "<ver> 에서 버그가 있대",
            "OS 는 <ver> 이야",
            "설치된 버전이 <ver> 라고 나와",
            "<kin> 폰은 <ver> 이라 안 돼",
        ],
    ),
    (
        "neg_amount",
        [
            "<amount> 입금했어요",
            "이번 달 <amount> 나왔어",
            "전세금이 <amount> 래",
            "<kin> 용돈으로 <amount> 보냈어",
            "견적이 <amount> 정도야",
        ],
    ),
    (
        "neg_word",
        [
            "오늘 저녁은 <word> 어때",
            "<word> 좀 해줘",
            "<word> 이라고 하더라",
            "<kin> 이 <word> 얘기 했어",
            "그건 <word> 아닌가",
        ],
    ),
    (
        "phone_public",
        [
            "고객센터 <PHONE_PUBLIC:service> 로 문의해",
            "대표번호 <PHONE_PUBLIC:service> 이래",
            "문의는 <PHONE_PUBLIC:service> 으로",
            "병원 대표전화 <PHONE_PUBLIC:service> 야",
            "<kin> 이 <PHONE_PUBLIC:service> 에 전화해보래",
        ],
    ),
    (
        "location_general",
        [
            "내일 <LOCATION:place> 에서 만나자",
            "<LOCATION:city> 날씨 어때",
            "이번 주말에 <LOCATION:city> 여행 갈래",
            "<LOCATION:place> 에 맛집 많대",
            "<kin> 이랑 <LOCATION:city> 갔다 왔어",
        ],
    ),
    (
        "neg_kin",
        [
            "<kin>한테 전화해봐",
            "<kin> 생신 선물 뭐 사지",
            "오늘 <kin> 이랑 저녁 먹었어",
            "<kin> 도 같이 가자",
            "<kin> 보고 싶다",
        ],
    ),
]

FILLER = [
    "오늘은 아침부터 비가 와서 우산을 챙겨 나갔다.",
    "점심에는 회사 근처 식당에서 된장찌개를 먹었는데 생각보다 맛있었다.",
    "퇴근길에 마트에 들러 우유와 달걀, 두부를 샀다.",
    "주말에는 가족이 모두 모여 집 정리를 하기로 했다.",
    "아이가 학교에서 돌아와 숙제를 먼저 끝내고 놀겠다고 했다.",
    "다음 달 가족 모임 장소를 어디로 정할지 아직 의견이 갈린다.",
    "세탁기가 요즘 소리가 커져서 수리 기사를 부를까 고민 중이다.",
    "냉장고에 있는 재료로 저녁 메뉴를 짜 보았다.",
    "이번 휴가에는 바다보다 산 쪽으로 가는 것이 좋겠다는 이야기가 나왔다.",
    "병원 진료는 오전이 한산하다고 해서 일찍 가기로 했다.",
    "베란다 화분에 물을 주고 시든 잎을 정리했다.",
    "공과금 납부일이 다가오니 이번 달 지출을 한 번 더 확인해 봐야겠다.",
    "학교 알림장에 적힌 준비물은 내일 아침에 챙기면 된다.",
    "산책 코스를 바꿔 보니 동네 분위기가 조금 달라 보였다.",
    "저녁 뉴스에서 내일 기온이 많이 떨어진다고 했다.",
    "책장에 쌓인 책들을 정리하다가 오래된 사진첩을 발견했다.",
    "가족 달력에 각자의 일정을 새로 적어 넣었다.",
    "엘리베이터 점검 때문에 오전에는 계단을 이용해야 한다고 한다.",
    "주말 장보기 목록에 과일과 간식을 추가했다.",
    "내일 회의 자료는 오늘 안으로 마무리하는 것이 좋겠다.",
]
LONG_FRAMES = [
    ["그리고 참고로 내 연락처는 <PHONE_PERSONAL:phone_dash> 이야."],
    ["아 맞다 받는 분 주소는 <ADDRESS:addr_road_unit> 로 적어줘."],
    ["송금은 <bank> <ACCOUNT:acct_dash> 로 부탁해.", "연락은 <EMAIL:email> 로 줘."],
    ["담당자 <PERSON:name3> 씨 번호는 <PHONE_PERSONAL:phone_space> 라고 했어."],
    ["본인 확인용 주민번호 <RRN:rrn_dash> 는 직접 입력해야 한대."],
]


def split_frames(tag: str, n: int) -> list[str]:
    """틀 번호 -> 분할. 시드 고정 셔플로 약 60/20/20."""
    n_test = max(1, round(n * 0.2))
    n_val = max(1, round(n * 0.2))
    order = list(range(n))
    random.Random(f"{SEED}:{tag}").shuffle(order)
    out = [""] * n
    for rank, idx in enumerate(order):
        out[idx] = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
    return out


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def build(safe: bool = True) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    seen: set[str] = set()

    def emit(tag: str, tid: str, split: str, v: int, frame: str, extra: list[str], fillers=False):
        for salt in range(50):
            ctx = Ctx(random.Random(f"{SEED}:{tid}:{v}:{salt}"), split, safe)
            if fillers:
                text, spans = render_long(frame, ctx)
            else:
                text, spans = render(frame, ctx)
            key = norm(text)
            if key not in seen:
                seen.add(key)
                break
        else:
            raise RuntimeError(f"중복을 피하지 못함: {tid}")
        tags = sorted({tag, *extra, *ctx.tags} | ({"long_text"} if fillers else set()))
        if not spans:
            tags = sorted(set(tags) | {"negative"})
        rows[split].append(
            {
                "text": text,
                "spans": spans,
                "known_names": ctx.known,
                "tags": tags,
                "template_id": tid,
                "split": split,
            }
        )

    def render_long(frame: str, ctx: Ctx):
        sents = frame.split("\n")
        pii = [render(s, ctx) for s in sents]
        pool = FILLER[:]
        ctx.rng.shuffle(pool)
        parts: list[tuple[str, list[dict]]] = []
        total, i = 0, 0
        while total < 520 or i < 6:
            parts.append((pool[i % len(pool)], []))
            total += len(pool[i % len(pool)]) + 1
            i += 1
        for t, s in pii:
            parts.insert(ctx.rng.randint(2, len(parts) - 1), (t, s))
        text, spans = "", []
        for t, s in parts:
            if text:
                text += " "
            for sp in s:
                spans.append({**sp, "start": sp["start"] + len(text), "end": sp["end"] + len(text)})
            text += t
        assert len(text) >= 500
        return text, spans

    for tag, frames in CATS:
        splits = split_frames(tag, len(frames))
        for i, fr in enumerate(frames):
            frame, extra = (fr, []) if isinstance(fr, str) else fr
            for v in range(VARIANTS):
                emit(tag, f"{tag}-{i + 1:02d}", splits[i], v, frame, extra)
    splits = split_frames("long_text", len(LONG_FRAMES))
    for i, fr in enumerate(LONG_FRAMES):
        for v in range(VARIANTS):
            emit(
                "long_text", f"long_text-{i + 1:02d}", splits[i], v, "\n".join(fr), [], fillers=True
            )
    for split, lst in rows.items():
        lst.sort(key=lambda r: (r["template_id"], r["text"]))
        for n, r in enumerate(lst, 1):
            r["id"] = f"v2-{split}-{n:04d}"
    return rows


PRIVATE_DIR = OUT_DIR.parents[4] / "data" / "private_eval" / "baseline_v2"


def write(rows: dict[str, list[dict]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    order = ["id", "text", "spans", "known_names", "tags", "template_id", "split"]
    for split, lst in rows.items():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in lst:
                f.write(json.dumps({k: r[k] for k in order}, ensure_ascii=False) + "\n")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--profile",
        choices=["safe", "private", "both"],
        default="both",
        help="safe=공개 저장소용 안전 픽스처, private=유효한 전체 식별자(로컬 전용, git 무시 폴더)",
    )
    args = ap.parse_args()
    for profile in ("safe", "private"):
        if args.profile not in (profile, "both"):
            continue
        rows = build(safe=profile == "safe")
        out = OUT_DIR if profile == "safe" else PRIVATE_DIR
        write(rows, out)
        total = sum(len(v) for v in rows.values())
        neg = sum(1 for v in rows.values() for r in v if not r["spans"])
        print(
            f"[{profile}] {out}: 총 {total}문장 { ({k: len(v) for k, v in rows.items()}) }, 음성 {neg}"
        )
        if profile == "safe":
            tags = Counter(t for v in rows.values() for r in v for t in r["tags"])
            print(dict(sorted(tags.items())))


if __name__ == "__main__":
    main()
