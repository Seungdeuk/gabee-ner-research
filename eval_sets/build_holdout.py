"""독립 보조 평가 세트(pii_eval_holdout.jsonl) 생성기.

이 문장들은 탐지 규칙을 만든 쪽이 아니라 **독립 검토자가 따로 쓴** 가족 대화체 81문장이다(원문은 검토 스크립트의 CASES).
검토 결과를 반영해 규칙을 고쳤으므로 이 세트도 더는 완전한 '처음 보는 데이터'가 아니다(수치는 낙관적일 수 있다).
일부 어려운 사례(한글 숫자, 공백 구분 계좌, 키워드와 떨어진 비밀번호, 사전에 없는 이름 …)는 일부러 남겨 한계를 드러낸다.
모든 값은 합성이며 실존 번호와 우연히 일치할 수 있다. 이메일은 예약 도메인만 쓴다.

사용: ``python app/shared/privacy/eval/build_holdout.py``
"""

import json
import re
from pathlib import Path

MARK = re.compile(r"\{\{([A-Z]+)\|(.*?)\}\}")
OUT = Path(__file__).with_name("pii_eval_holdout.jsonl")
FAM = ["김민수", "이영희", "박지후", "최서연"]
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
        {
            "id": f"holdout-{len(ROWS) + 1:04d}",
            "text": text,
            "spans": spans,
            "known_names": FAM if known is None else known,
        }
    )


# ---- 정상 문장(오탐을 유발할 수 있는 것)
for s in [
    "내일 오후 3시 30분에 치과 가야 해", "2026-11-03 10:30 예약했어", "이번 달 관리비 182,340원 나왔어",
    "아파트 15층이야, 1502호 아니고 1503호", "버스 402번 타고 3정거장", "쿠팡 주문번호 1000123456789 확인해봐",
    "네이버 주문번호 2026102612345671 이야", "우체국 택배 운송장 번호 6012345678901 이야",
    "우체국 등기 1234567890123 조회해봐", "CJ 운송장 612345678901 이래", "엄마랑 통화 23분 12초 했어",
    "비율은 3-1-4 로 섞어", "스코어 2-1, 연장전 3-3", "카카오 선물하기 주문번호 202611030012 로 조회해봐",
    "카카오택시 예약번호 1234567890 이야", "국민은행 3시에 만나자", "농협 앞에서 2026-11-03 10:30에 보자",
    "토스트 사왔어 10개", "비밀번호 바꿨어, 2026-11-03에", "비밀번호 8자 이상이어야 한대",
    "인증번호 6자리 10분 안에 입력", "비번 wifi 랑 같아", "김치찌개랑 박수 치고 이사 준비",
    "하나님 감사합니다", "박사님이 그러는데 괜찮대", "형수님 오신대", "올레길 7코스 걷자", "산책로 2바퀴 돌았어",
    "진동 3단계로 해놔", "서울 대도시라 복잡해", "경기 신도시로 이사 갈까", "3동 2호선 타고 와",
    "1588-1234 로 전화해",  # 대표번호는 개인정보가 아니라고 보고 음성으로 둔다
]:  # fmt: skip
    add(s)
add("하나도 안 먹었어", ["김하나", "이영희"])
add("수정 좀 해줘, 수정이 필요해", ["이수정"])
add("이거 보라고!", ["박보라"])
add("더 나은 방법 없을까", ["이나은"])
add("지원 신청 마감이 내일이야", ["김지원"])
add("엄마가 아빠한테 전화했어", ["엄마", "아빠", "김민수"])

# ---- 전화
add("{{PHONE|010 1234 5678}}")
add("{{PHONE|010-1234-5678}}로 해")
add("연락처 {{PHONE|010–1234–5678}} (엔대시)")
add("{{PHONE|０１０-１２３４-５６７８}} 전각")
add("{{PHONE|010)1234-5678}}")
add("{{PHONE|010/1234/5678}}")
add("{{PHONE|공일공 일이삼사 오육칠팔}}")  # 어려운 사례
add("{{PHONE|02 345 6789}}")
# ---- 주민번호
add("주민번호 {{RRN|900101 - 1234567}}")
add("주민번호 {{RRN|900101-123456*}}")  # 어려운 사례(일부 가림)
add("주민등록번호 앞자리 {{RRN|900101 뒷자리 1234567}}")  # 어려운 사례(앞뒤를 나눠 적음)
add("{{RRN|9001011234567}}")
# ---- 카드
add("카드 {{CARD|4539-1488-0343-6467}}")
add("카드 {{CARD|4539 1488 0343 6467}} 유효기간 08/29 cvc 123")
add("카드번호 {{CARD|4539148803436467}}")
add("카드 {{CARD|4539-1488-0343-6467}} 비번 {{SECRET|1234}}")
# ---- 이메일
add("메일 {{EMAIL|KIM.TEST@Example.COM}}")
add("{{EMAIL|mom1962@example.net}}.")
add("{{EMAIL|kim @ example.net}}")  # 어려운 사례(공백)
add("{{EMAIL|kim@example}}")  # 어려운 사례(최상위 도메인 없음)
# ---- 계좌
add("국민 {{ACCOUNT|123456-01-123456}}")
add("계좌 {{ACCOUNT|110 123 456789}}")  # 어려운 사례(공백 구분)
add("신한 {{ACCOUNT|110123456789}} 로 보내")
add("{{ACCOUNT|1002-345-678901}} (우리)")
add("{{ACCOUNT|110-123-456789}} 로 보내줘")  # 어려운 사례(문맥 없음)
# ---- 비밀 정보
add("현관 비밀번호 {{SECRET|1234*}}")
add("비밀번호: 공동현관 {{SECRET|1234}} 세대 5678")  # 어려운 사례
add("인증번호 왔어? {{SECRET|482913}}")  # 어려운 사례(키워드와 값이 떨어짐)
add("비번은 생일 뒤집은 거 {{SECRET|1010}}")  # 어려운 사례
# ---- 주소
add("우리 집 {{ADDRESS|서울시 중구 세종대로 110}}")
add("{{ADDRESS|부산 동구 좌천동 123-4}}")
add("{{ADDRESS|강남역 3번 출구 앞 빌라 302호}}")  # 어려운 사례(약식 표현)
add("래미안 아파트 {{ADDRESS|103동 1204호}}")
add("{{ADDRESS|테헤란로152}}")
# ---- 이름·복합
add(
    "{{PERSON|민수}} 전화 {{PHONE|010-1234-5678}}, {{PERSON|지후}} 메일 {{EMAIL|jihoo@example.com}}"
)
add("{{PERSON|영희}}씨고 {{PERSON|민수}}는 아직")
add("{{PERSON|최서연}}이 아니라 최서윤이야")
add("{{PERSON|김민수}}님 {{PERSON|박지후}}군 {{PERSON|최서연}}양")
add("{{PERSON|철수}}한테 말해놨어")  # 어려운 사례(사전에 없고 호칭도 없음)
add("{{PERSON|김철수}}선생님 {{PERSON|박영숙}}씨 {{PERSON|이수정}}님")
add("{{PERSON|김 민수}} (띄어쓴 이름)")  # 어려운 사례
add("민수기념관 갔다가 {{PERSON|민수}}네 집")

if __name__ == "__main__":
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for row in ROWS:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pos = sum(1 for r in ROWS if r["spans"])
    print(f"{len(ROWS)}문장 (양성 {pos}, 음성 {len(ROWS) - pos}) -> {OUT}")
