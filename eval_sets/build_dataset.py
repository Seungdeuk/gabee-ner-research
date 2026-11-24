# ruff: noqa: UP031
"""평가 데이터셋(pii_eval.jsonl) 생성기. 모든 값은 임의로 만든 합성 값이다(실존 번호와 우연히 일치할 수 있다). 이메일은 예약 도메인만 쓴다.

사용: ``PYTHONPATH=. python app/shared/privacy/eval/build_dataset.py`` (같은 시드면 같은 결과).
문장은 ``{{KIND|값}}`` 표시를 넣어 쓰고, 표시를 지운 글자와 구간(start, end)을 계산해 한 줄 JSON 으로 쓴다.
탐지 규칙에 맞춰 데이터를 고치지 않는다: 현실적인 변형과 일부 어려운 사례(띄어쓰기 오타, 한글 숫자, 사전에 없는 이름)를 일부러 넣었다.
"""

import json
import random
import re
from pathlib import Path

rng = random.Random(20261026)
MARK = re.compile(r"\{\{([A-Z]+)\|(.*?)\}\}")
OUT = Path(__file__).with_name("pii_eval.jsonl")
ROWS: list[dict] = []


def add(template: str, known: list[str] | None = None) -> None:
    text, spans, pos = "", [], 0
    for m in MARK.finditer(template):
        text += template[pos : m.start()]
        start = len(text)
        text += m.group(2)
        spans.append({"kind": m.group(1), "start": start, "end": len(text)})
        pos = m.end()
    text += template[pos:]
    ROWS.append(
        {"id": f"pii-{len(ROWS) + 1:04d}", "text": text, "spans": spans, "known_names": known or []}
    )


def digits(n: int) -> str:
    return "".join(str(rng.randrange(10)) for _ in range(n))


def pick(seq):
    return rng.choice(seq)


# ---------------------------------------------------------------- 값 생성
def mobile() -> str:
    a, b = digits(4), digits(4)
    if a[0] == "0":
        a = "5" + a[1:]
    return pick(
        [
            f"010-{a}-{b}", f"010-{a}-{b}", f"010-{a}-{b}", f"010.{a}.{b}", f"010 {a} {b}", f"010{a}{b}",
            f"011-{a[:3]}-{b}", f"016-{a[:3]}-{b}", f"+82 10-{a}-{b}", f"+82-10-{a}-{b}", f"+8210{a}{b}",
        ]
    )  # fmt: skip


def landline() -> str:
    area = pick(["02", "031", "032", "051", "053", "062", "042", "064", "070", "0507"])
    mid = digits(3 if area == "02" and rng.random() < 0.5 else 4)
    return pick(
        [f"{area}-{mid}-{digits(4)}", f"{area}-{mid}-{digits(4)}", f"{area} {mid} {digits(4)}"]
    )


def email() -> str:
    local = pick(
        [
            "kim.minsu",
            "yh_lee",
            "family2026",
            "mom1962",
            "papa.park",
            "sy.choi",
            "hello_home",
            "jihoo0412",
        ]
    )
    # 실존 도메인을 쓰지 않는다: 예약 도메인(RFC 2606)만
    dom = pick(
        ["example.com", "example.org", "example.net", "mail.test", "family.test", "home.example"]
    )
    return f"{local}{rng.randrange(10, 99)}@{dom}" if rng.random() < 0.5 else f"{local}@{dom}"


def rrn(valid: bool = True) -> str:
    g = pick([1, 2, 3, 4, 1, 2])
    year = rng.randrange(40, 99) if g in (1, 2) else rng.randrange(0, 25)
    mm, dd = rng.randrange(1, 13), rng.randrange(1, 29)
    front = f"{year:02d}{mm:02d}{dd:02d}"
    body = f"{front}{g}{digits(5)}"
    w = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)
    check = (11 - sum(int(d) * x for d, x in zip(body, w, strict=True)) % 11) % 10
    if not valid:
        check = (check + rng.randrange(1, 10)) % 10
    full = body + str(check)
    return pick(
        [f"{full[:6]}-{full[6:]}", f"{full[:6]}-{full[6:]}", full, f"{full[:6]} {full[6:]}"]
    )


def luhn_card(prefix: str, length: int) -> str:
    body = prefix + digits(length - len(prefix) - 1)
    total = 0
    for i, ch in enumerate(reversed(body)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return body + str((10 - total % 10) % 10)


def card() -> str:
    if rng.random() < 0.15:
        c = luhn_card(pick(["34", "37"]), 15)
        return f"{c[:4]}-{c[4:10]}-{c[10:]}"
    c = luhn_card(pick(["4539", "4916", "5412", "5105", "3530", "9410"]), 16)
    return pick(
        [
            f"{c[:4]}-{c[4:8]}-{c[8:12]}-{c[12:]}",
            f"{c[:4]} {c[4:8]} {c[8:12]} {c[12:]}",
            c,
            f"{c[:4]}-{c[4:8]}-{c[8:12]}-{c[12:]}",
        ]
    )


def account() -> str:
    return pick(
        [
            f"{digits(3)}-{digits(6)}-{digits(5)}", f"{digits(4)}-{digits(2)}-{digits(7)}",
            f"{digits(3)}-{digits(3)}-{digits(6)}", f"{digits(6)}-{digits(2)}-{digits(6)}", digits(12),
            f"{digits(3)}-{digits(2)}-{digits(6)}", digits(11),
        ]
    )  # fmt: skip


BANKS = [
    "국민은행",
    "신한은행",
    "우리은행",
    "하나은행",
    "농협",
    "기업은행",
    "카카오뱅크",
    "토스뱅크",
    "케이뱅크",
    "우체국",
    "새마을금고",
    "수협",
]
ROADS = [
    "테헤란로",
    "세종대로",
    "판교역로",
    "올림픽로",
    "가로수길",
    "중앙로",
    "동백로",
    "시청로",
    "충장로",
    "한강대로",
    "백범로",
    "양재대로",
]
DONGS = [
    "역삼동",
    "서초동",
    "논현동",
    "정자동",
    "망원동",
    "연남동",
    "우동",
    "범일동",
    "삼성동",
    "신사동",
]
SIDOS = [
    "서울시 강남구",
    "서울특별시 마포구",
    "경기도 성남시",
    "부산광역시 해운대구",
    "인천시 연수구",
    "대전 유성구",
    "서울 송파구",
    "경기도 수원시",
]
SURN = "김이박최정강조윤장임한오서신권황안송류전홍"
GIVEN = [
    "민수",
    "영희",
    "지후",
    "서연",
    "도윤",
    "하은",
    "준호",
    "수빈",
    "예린",
    "태민",
    "지안",
    "현우",
    "나은",
    "시우",
    "유진",
]


def road() -> str:
    pre = pick(["", "", SIDOS[rng.randrange(len(SIDOS))] + " "])
    r = pick(ROADS)
    num = f"{rng.randrange(1, 400)}" + (f"-{rng.randrange(1, 30)}" if rng.random() < 0.3 else "")
    return f"{pre}{r} {num}" if rng.random() < 0.85 else f"{pre}{r}{num}"


def jibun() -> str:
    d = pick(DONGS)
    return pick(
        [
            f"{d} {rng.randrange(1, 900)}-{rng.randrange(1, 40)}",
            f"{d} {rng.randrange(1, 900)}번지",
            f"{pick(SIDOS)} {d} {rng.randrange(1, 900)}-{rng.randrange(1, 40)}",
        ]
    )


def dongho() -> str:
    return pick(
        [
            f"{rng.randrange(101, 130)}동 {rng.randrange(101, 2504)}호",
            f"{rng.randrange(1, 20)}동{rng.randrange(101, 1805)}호",
        ]
    )


def fullname() -> str:
    return pick(SURN) + pick(GIVEN)


# ---------------------------------------------------------------- 양성: 전화
for t in [
    "내 번호는 {{PHONE|%s}} 이야 저장해둬",
    "엄마 폰 {{PHONE|%s}} 로 전화해봐",
    "택배 기사님 연락처 {{PHONE|%s}} 래",
    "연락 주세요 {{PHONE|%s}}",
    "학원 번호 {{PHONE|%s}} 인데 5시 이후엔 안 받아",
    "병원 예약하려면 {{PHONE|%s}} 으로 걸면 돼",
    "할머니 댁 전화 {{PHONE|%s}}",
    "혹시 급하면 {{PHONE|%s}} 로 문자 줘",
    "아빠 번호 바뀌었어. {{PHONE|%s}}",
    "{{PHONE|%s}} 이 번호 누구 거야?",
    "수리 기사 번호는 {{PHONE|%s}}, 오전에만 받는대",
    "연락처: {{PHONE|%s}}",
    "내일 {{PHONE|%s}} 로 전화드릴게요",
    "집 전화 {{PHONE|%s}}",
]:
    for _ in range(2):
        add(t % (landline() if "집 전화" in t or "학원" in t or "병원" in t else mobile()))
for _ in range(2):
    add("두 번호 다 알려줄게 {{PHONE|%s}} / {{PHONE|%s}}" % (mobile(), mobile()))
add("전화번호 {{PHONE|010-5521-8834}} 맞지? 끝에 8834")
add("{{PHONE|+82 10-4410-2297}} 이 번호로 해외에서 연락 와")
add("엄마 폰 {{PHONE|01055218834}} 이거 맞나")
add("지점 전화는 {{PHONE|02-318-4402}} 야")
add("{{PHONE|010.3321.7740}} 으로 보내줘")

# ---------------------------------------------------------------- 이메일
for t in [
    "메일 주소 {{EMAIL|%s}} 야",
    "{{EMAIL|%s}} 로 사진 보내줘",
    "이 이메일로 가입했어: {{EMAIL|%s}}",
    "신청서는 {{EMAIL|%s}} 으로 보내면 된대",
    "내 구글 계정 {{EMAIL|%s}}",
    "참조에 {{EMAIL|%s}} 도 넣어줘",
    "학교 메일 {{EMAIL|%s}} 확인해봐",
    "청구서 {{EMAIL|%s}}으로 왔어",
    "답장은 {{EMAIL|%s}}.",
    "아빠 이메일: {{EMAIL|%s}}",
]:
    for _ in range(2):
        add(t % email())

# ---------------------------------------------------------------- 주민번호
for t in [
    "주민번호 {{RRN|%s}} 야 적어줘",
    "신청서에 {{RRN|%s}} 쓰래",
    "내 주민등록번호 {{RRN|%s}}",
    "보험 가입하려면 {{RRN|%s}} 필요해",
    "아이 주민번호는 {{RRN|%s}} 입니다",
    "{{RRN|%s}} 이게 맞아?",
    "외국인등록번호 {{RRN|%s}} 확인 부탁",
    "주민번호 뒷자리까지 {{RRN|%s}} 보내드렸어요",
]:
    for valid in (True, True, False):
        add(t % rrn(valid))
add(
    "생년월일 900101 그리고 주민번호 {{RRN|900101-1234568}} 였나"
)  # 체크섬이 맞지 않는 번호(형식만 맞음)
add("{{RRN|2307153123456}} 이건 2020년 이후 번호 형식이야")

# ---------------------------------------------------------------- 카드
for t in [
    "카드번호 {{CARD|%s}} 야",
    "결제는 {{CARD|%s}} 로 해줘",
    "내 신용카드 {{CARD|%s}} 유효기간 08/29",
    "{{CARD|%s}} 이 카드로 긁었어",
    "카드 번호 불러줄게 {{CARD|%s}}",
    "엄마 카드 {{CARD|%s}} 한도 초과래",
    "체크카드 {{CARD|%s}} 분실했어",
    "법인카드 {{CARD|%s}} 사용 내역 확인해줘",
    "자동결제 카드가 {{CARD|%s}} 로 등록돼 있어",
    "이 번호로 결제 부탁: {{CARD|%s}}",
]:
    for _ in range(2):
        add(t % card())
add("카드 뒷자리 {{CARD|%s}}!" % card())

# ---------------------------------------------------------------- 계좌
for t in [
    "{{BANK}} {{ACC|%s}} 로 입금해줘",
    "계좌번호 {{ACC|%s}} ({{BANK}})",
    "{{BANK}} 계좌 {{ACC|%s}} 맞지?",
    "송금은 {{BANK}} {{ACC|%s}} 으로",
    "계좌 {{ACC|%s}} 로 용돈 보냈어",
    "내 계좌는 {{ACC|%s}} {{BANK}} 야",
    "축의금 계좌 {{BANK}} {{ACC|%s}} 예금주 본인",
]:
    for _ in range(3):
        bank = pick(BANKS)
        add((t % account()).replace("{{BANK}}", bank).replace("{{ACC|", "{{ACCOUNT|"))
add("{{ACCOUNT|3333-12-3456789}} 카카오뱅크로 보내")
add("입금 계좌 {{ACCOUNT|110-123-456789}} 신한은행")
add("계좌 번호가 {{ACCOUNT|1002345678901}} 이야")
add("농협 {{ACCOUNT|352-1234-5678-93}}")
add("하나은행 {{ACCOUNT|123-456789-01234}} 로 부탁해")

# ---------------------------------------------------------------- 비밀 정보
for t in [
    "와이파이 비밀번호는 {{SECRET|%s}} 야",
    "비번 {{SECRET|%s}} 로 바꿨어",
    "인증번호 {{SECRET|%s}} 왔는데 입력해줘",
    "OTP {{SECRET|%s}} 야 빨리",
    "공동현관 비밀번호: {{SECRET|%s}}",
    "패스워드는 {{SECRET|%s}} 입니다",
    "핀번호 {{SECRET|%s}} 기억해둬",
    "PIN {{SECRET|%s}} 입력하면 돼",
    "인증코드 {{SECRET|%s}} 5분 안에 넣어야 돼",
    "카드 비밀번호 {{SECRET|%s}} 이야",
    "보안코드 {{SECRET|%s}} 문자로 왔어",
    "password: {{SECRET|%s}}",
]:
    for _ in range(2):
        v = pick(
            [digits(4), digits(6), "gabee" + digits(4) + "!", "Qwer" + digits(3) + "#", digits(8)]
        )
        add(t % v)
add("인증 번호 {{SECRET|%s}} 이거 입력" % digits(6))
add("pw={{SECRET|Family2026!}} 로그인해봐")
add("비밀번호는 아직 못 정했어")  # 음성(값 없음) -> 아래 음성 목록과 별도로 라벨 없음
add("비밀번호 바꿔야 하는데 귀찮다")

# ---------------------------------------------------------------- 주소
for t in [
    "우리 집 주소는 {{ADDRESS|%s}} 야",
    "{{ADDRESS|%s}} 로 택배 보내줘",
    "약속 장소 {{ADDRESS|%s}} 앞",
    "이사 간 곳이 {{ADDRESS|%s}} 래",
    "학원 위치 {{ADDRESS|%s}}",
    "내비에 {{ADDRESS|%s}} 찍고 와",
    "할머니 댁 {{ADDRESS|%s}}",
    "배송지 변경: {{ADDRESS|%s}}",
]:
    for fn in (road, jibun, dongho):
        add(t % fn())
for _ in range(4):
    add("주소 {{ADDRESS|%s %s}} 입니다" % (road(), dongho()))
add("{{ADDRESS|서울시 강남구}} 쪽으로 이사 가려고")
add("{{ADDRESS|경기도 수원시}} 에 있는 병원이야")
add("{{ADDRESS|대전 유성구}} 에서 만나자")
add("주소는 {{ADDRESS|서울시강남구테헤란로 152}} 야")  # 어려운 사례: 붙여 쓴 주소(미탐 예상)

# ---------------------------------------------------------------- 이름(사전)
FAM = ["김민수", "이영희", "박지후", "최서연"]
for t in [
    "{{PERSON|%s}}이 오늘 늦는대",
    "{{PERSON|%s}}한테 전화해봐",
    "{{PERSON|%s}}가 시장 갔어",
    "{{PERSON|%s}}는 아직 학교야",
    "{{PERSON|%s}}에게 물어볼게",
    "{{PERSON|%s}}야 밥 먹자",
    "{{PERSON|%s}}랑 같이 갈게",
    "어제 {{PERSON|%s}}도 왔어",
]:
    full = pick(FAM)
    given = full[1:]
    use = pick([full, given])
    josa_fix = t % use
    add(josa_fix, FAM)
for _ in range(6):
    n = pick(FAM)
    add("오늘 장은 {{PERSON|%s}} 가 볼게" % n, FAM)
for _ in range(4):
    n = pick(FAM)
    add("{{PERSON|%s}}님 생일 선물 뭐가 좋을까" % n[1:], FAM)
add("{{PERSON|김민수}} 는 몇 시에 와?", FAM)
add("우리 엄마는 {{PERSON|이영희}}씨고 아빠는 {{PERSON|김민수}}씨야", FAM)

# ---------------------------------------------------------------- 이름(호칭 휴리스틱, 사전 없음)
for n in ["김철수", "박영숙", "최민정", "정하늘", "윤재석", "한지민", "오세훈", "강다영"]:
    add("담임 선생님은 {{PERSON|%s}}선생님이래" % n)
    add("{{PERSON|%s}}씨한테 먼저 연락해보자" % n)
for n in ["이수정", "조현우", "서민재", "권지아", "송태양"]:
    add("옆집 {{PERSON|%s}}님이 택배 맡아주셨대" % n)
add("친구 {{PERSON|배진우}}군이 놀러 왔어")
# 어려운 사례: 사전에 없고 호칭도 없는 이름(미탐 예상)
add("{{PERSON|철수}}한테 말해놨어", [])
add("{{PERSON|영숙}}이가 전화했어", [])
add("{{PERSON|민재}}랑 같이 갈게", [])
add("{{PERSON|도현}}이 숙제 봐줬어", [])
add("{{PERSON|서윤}}이 어린이집 갔어", [])
add("{{PERSON|하준}}이는 감기 걸렸대", [])

# ---------------------------------------------------------------- 복합
add("{{PHONE|010-7788-1234}} 로 연락하고 {{EMAIL|dad.park@example.org}} 로 서류 보내줘")
add(
    "{{ADDRESS|테헤란로 152}} 에서 {{PERSON|민수}}를 만나, 번호는 {{PHONE|010-2000-3000}}",
    ["김민수"],
)
add("비번 {{SECRET|abcd1234}} 이고 카드는 {{CARD|%s}}" % card())
add("계좌 {{ACCOUNT|110-123-456789}} 예금주 {{PERSON|이영희}}", ["이영희"])

# ---------------------------------------------------------------- 어려운 사례(한계를 드러낸다)
add("번호는 {{PHONE|공일공 일이삼사 오육칠팔}} 이야")  # 한글 숫자 전화번호(미탐 예상)
add("연락처 {{PHONE|010 - 3456 - 7890}} 으로")  # 하이픈 주변 띄어쓰기(부분 탐지 또는 미탐)
add("전화 {{PHONE|010-3456-78 90}} 해줘")  # 숫자 중간 공백(미탐 예상)
add("번호 {{PHONE|010-34567890}} 야")
add("이메일은 {{EMAIL|mom1962@example.net}} 입니당")
add("메일 주소 {{EMAIL|mom1962 골뱅이 example 닷컴}}")  # 말로 풀어 쓴 이메일(미탐 예상)
add("카드번호 {{CARD|4539 1488 0343 6467}} 구요")
add("비밀번호가 {{SECRET|ab12}} 인데 짧아")
add("pin번호는 {{SECRET|2580}}")
add("계좌 {{ACCOUNT|1234567890123}}")

# ---------------------------------------------------------------- 음성(개인정보 없음)
NEG = [
    "내일 오후 3시에 만나자",
    "2026-11-03 에 병원 예약했어",
    "11월 3일 오전 10:30 치과야",
    "회의는 14:00 부터래",
    "총 15,000원이 나왔어",
    "택배비 3,500원 더 내야 해",
    "월급이 3,200,000원 들어왔어",
    "보증금 5000만원이래",
    "사과 3개랑 배 2개 사와",
    "우유 1리터 두 팩 부탁",
    "계란 30구 한 판 사줘",
    "물 2L 6병",
    "주문번호 202611030012 로 조회해봐",
    "운송장 번호는 6012345678901 이래",
    "주문번호 ORD-20261103-0042",
    "접수번호 77412 로 문의했어",
    "3층에서 내려",
    "101호 앞에서 기다릴게",
    "2번 출구 앞이야",
    "지하 1층 주차장에 있어",
    "오늘 걸음 수 12345보 걸었어",
    "시험 점수 87점 나왔어",
    "키가 172cm 야",
    "몸무게 65kg",
    "강남역에서 만나자",
    "서울 가는 기차 탔어",
    "부산 여행 가자",
    "제주도 항공권 알아보자",
    "김치찌개 먹을까",
    "박수 쳐줄게",
    "이사 준비 하자",
    "정리 좀 해야겠다",
    "고객님 번호표 받으세요",
    "선생님이 숙제 내줬어",
    "사장님이 일찍 보내주셨어",
    "이사님 오셨대",
    "고추씨 좀 사와",
    "그대로 둬도 돼",
    "마음대로 해",
    "운동 30분 했어",
    "엄마 오늘 늦어?",
    "아빠 퇴근했어?",
    "할머니 보고 싶다",
    "동생 학교 갔어",
    "저녁 뭐 먹지",
    "오늘 날씨 좋다",
    "주말에 뭐 해?",
    "영화 보러 갈래?",
    "7호선 타고 가",
    "버스 402번 타",
    "엘리베이터 고장났대",
    "열쇠 어디 뒀지",
    "2026년 목표는 운동하기",
    "올해 1월 1일에 시작했어",
    "100m 달리기 했어",
    "3대 5로 졌어",
    "비밀번호 바꿔야 하는데 까먹었어",
    "인증번호가 안 와",
    "카드 한도가 다 찼어",
    "계좌 이체가 안 돼",
    "주소 알려줘",
    "이메일 확인했어?",
    "전화 받아",
    "번호 저장해둘게",
    "1588 번으로 전화하면 돼",
    "회사는 9시부터 6시까지",
    "점심은 12시 30분에 먹자",
    "다음 주 월요일 휴가야",
    "1.5배속으로 봤어",
    "할인율 30% 야",
    "평점 4.5점",
    "버전 2.3.1 으로 업데이트해",
    "대전 갔다가 대구 들러",
    "한강에서 자전거 타자",
    "광주 친구 만나러 가",
    "인천공항 도착",
    "12,345,678원 이나 썼어",
    "20261103123456 은 송장 번호야",
    "수량 1000개 입고",
    "전화 안 받네",
    "오늘 저녁은 삼겹살 어때",
    "이번 주말 가족 모임 장소 정하자",
    "아이 학원 시간표 공유할게",
    "장 보러 가는데 필요한 거 있어?",
    "우리 집 고양이 사진 봐",
    "사진 찍어서 올려줘",
    "고마워 잘 먹었어",
    "생일 축하해",
    "서울 친구랑 부산 구경 갔어",
    "전화번호 0312345 맞아?",
    "회원번호 1234-5678-9012 로 조회",
    "예약번호 2026-1103-0042-7781 이야",
    "송장 1234567890 이야",
    "큰 번호 9876543210123 이래",
    "스코어 3-2 로 이겼어",
    "새벽 02:30 에 일어났어",
    "2026.11.03 에 출발",
    "0.5kg 만 사와",
    "코드 123-456 입력해",
    "5-6-7 다음 주 일정이야",
    "사무실 3동 2층이야",
    "한강로 3가 쪽으로 가",
    "4인분 5000원씩 걷자",
    "그건 마음먹기에 달렸어",
    "이번 달 가스비 82,340원",
    "집에 가면 연락할게",
    "예약하려면 전화해야 해",
    "아이가 2호선에서 내렸대",
    "2번 방에서 자",
    "5분만 기다려",
    "3명 예약했어",
    "10시 20분 도착 예정",
]
for s in NEG:
    add(s)

if __name__ == "__main__":
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for row in ROWS:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pos = sum(1 for r in ROWS if r["spans"])
    print(f"{len(ROWS)}문장 (양성 {pos}, 음성 {len(ROWS) - pos}) -> {OUT}")
