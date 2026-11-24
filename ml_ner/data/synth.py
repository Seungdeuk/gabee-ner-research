"""합성 가족 대화 NER 학습 데이터 생성기(Baseline 과 독립, 모든 값은 합성).

- 문장 틀(TEMPLATES)에 슬롯 {P} {G} {AL} {NK} {ADx} {LC} {ST} {SR} {Ox} 를 채운다. 슬롯 뒤 ``@이/가`` 는 받침에 따라 조사를 고른다.
- 라벨은 LABELING_GUIDE.md 를 따른다: 호칭만 있는 엄마·아빠는 O, 은행 이름은 O(계좌는 규칙 몫), 이름+가족호칭·별명은 PERSON_ALIAS,
  상세 주소(도로명+번호, 아파트 N동 M호, 동 번지)는 ADDRESS, 역·지역·랜드마크는 LOCATION, 병원·학교·회사 이름은 ORGANIZATION
  (병원·약국·치과는 span 의 ``context_sensitive=True`` 로 표시; 학습 라벨은 ORGANIZATION 하나).
- 라벨이 모호한 틀·값(이름+과/의원, ``...근처`` 가 붙은 주소, 집근처마트, 보리·우리왕자 같은 일반명사형 별명)은 뺀다(REMOVED_AMBIGUOUS).
- ``augment=True`` 면 오탐 감소용 보강 틀(가족 호칭·일반 명사·일반 장소 명사의 음성, 어려운 양성)을 더한다.
- 틀(template) 단위로 train/val 을 나누고, 값 풀(이름·도로·단지·기관·지명)도 train/val 이 서로소가 되게 나눈다.
- test 에 나온 이름·기관·주소·지명 값은 풀에서 뺀다(leak_guard).
"""

import random
import re

# --------------------------------------------------------------------------- 값 풀
SURNAMES = list(
    "김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민나진지"
) + ["남궁", "황보", "제갈", "사공", "선우", "독고"]
# GIVEN2 뒤쪽(177번째 이름 "서온" 다음)부터는 로드맵 2단계(메신저체 추가 확인)에서 못 보던 이름 토큰(성경안·
# 주소온·전송유 등, 위쪽 240여개 고정 목록과 음절 조합만 다를 뿐인데도 모델이 놓치는 경우)을 분석해 찾은 일반화
# 문제 — 이 고정 목록 밖의 "이름처럼 생긴 토큰"을 학습 중에 거의 못 보는 게 원인으로 보여, 같은 성·이름 음절을
# 교차 조합해 이름 다양성을 넓혔다(초성·종성 풀 자체는 위 목록에서만 뽑아 완전히 새 음절을 쓰지 않는다). 흔한
# 명사와 같아지는 조합은 걸러냈지만(연결·시민·소주·보지 등 BLOCKLIST, apps/api 쪽 build_evalset.py 와 같은
# 방식) 사전 검증은 아니라 드물게 남을 수 있다. 이 설명은 문자열 밖에 둔다 — exclusions.py 의 GIVEN2 추출
# 정규식이 문자열 전체를 그대로 읽어서, 문자열 안에 주석을 넣으면 주석 글자까지 "이름"으로 잘못 들어간다.
GIVEN2 = (
    "민준 서준 도윤 하준 지호 지훈 준서 현우 건우 우진 승민 재원 태민 시현 은우 이안 유찬 정우 도현 연우 지안 하람 서진 예준 주안 해원 "
    "시율 라온 로운 가온 노아 이든 하진 도겸 규민 세민 상훈 영호 병철 순자 말숙 영희 영수 철수 민수 정수 동현 성민 재훈 기훈 진우 석진 "
    "태준 윤호 용준 창민 대현 형준 범수 인성 종현 광수 경수 서연 서윤 지우 하윤 하은 서아 지유 지아 수아 채원 예은 윤서 다은 소율 가은 "
    "나은 시은 아윤 유나 예린 민서 지민 수빈 하영 예지 은서 보미 유진 소연 미영 정은 현주 은정 경희 미선 순옥 영자 말순 지원 수진 혜진 "
    "소희 다혜 채은 나연 세은 하린 아린 단비 해솔 한결 슬기 푸름 나래 이슬 다정 새별 유빈 소민 윤아 시온 도하 루아 지후 선호 태오 현서 "
    "은찬 시후 민재 준혁 동혁 성훈 상철 경훈 재민 승우 지성 대호 한울 연서 서하 하랑 예서 규리 유림 휘윤 슬우 채하 나윤 예담 도경 "
    "보현 가율 다현 시안 이준 건희 정호 미란 숙자 옥순 봉수 판석 갑순 덕배 해강 이솔 로하 소담 아현 은채 윤재 서온 "
    "가아 가찬 갑겸 갑기 갑란 갑미 갑수 갑옥 갑정 건래 건옥 건운 건혜 경린 경선 경원 경하 광서 광안 규래 규비 규순 규안 규울 규율 규주 규혜 "
    "기미 기옥 기유 나람 나림 나민 나진 나채 나훈 노원 노후 다준 단석 단솔 단수 단슬 대든 대리 대별 대석 대재 덕리 덕원 도성 도채 도희 동안 "
    "동영 동오 동주 라석 라수 라자 라하 로옥 로율 로현 루랑 루래 루리 루미 루연 루영 말람 말원 말철 미강 미결 미별 미윤 미율 미자 민림 민별 "
    "민호 범람 범성 범혁 범훈 병란 병선 병준 보겸 보빈 보숙 보순 보오 상란 상미 상안 상혁 새름 새순 새오 새후 새훈 서경 서선 서율 석랑 석온 "
    "석윤 석정 선선 선운 선희 성겸 성운 성은 성진 성현 성호 세든 세순 소강 소빈 소석 소오 소은 수민 수슬 숙기 숙아 숙철 숙후 순경 순리 순린 "
    "순오 순채 승미 승숙 승오 승온 승율 시강 시우 아유 아율 연철 연훈 영겸 영미 영연 예배 예연 예혁 옥겸 옥나 옥선 옥아 옥울 옥철 옥호 우강 "
    "우래 우림 우성 우정 우현 우후 우희 유강 유기 유유 유현 윤비 윤빈 윤연 윤훈 은결 은나 은슬 이진 이현 이훈 인나 인리 인린 인운 재솔 재하 "
    "정란 정림 정배 정선 종래 종원 주나 주운 주하 준림 준안 준연 준온 준원 준자 지연 지온 지울 지자 지주 진서 진호 창든 창람 창배 창자 창희 "
    "채람 채슬 채옥 채훈 철결 철유 철재 철채 철혁 태윤 태자 태철 태혁 푸민 푸별 푸울 푸원 푸혁 푸훈 하란 하미 하서 하주 하채 하혜 한리 한연 "
    "한채 한혁 한후 해기 해옥 해윤 해훈 현은 형림 형아 형혁 혜채 혜혁 휘든 휘숙 휘우 휘지"
).split()
GIVEN1 = list(
    "현빈린준훈석민"
)  # 한 글자 이름. 조건·한결·오담 처럼 일반 단어/조사와 겹치는 글자(건 결 담 솔 별 율 온 겸 찬 휘)는 뺐다
NICKS = (
    "콩이 밤톨이 쭈니 똘이 찡이 뽀미 몽이 달봉이 쭈쭈 순돌이 까미 삐삐 초롱이 몽글이 뭉이 솜솜이 두둥이 뽀삐 꾸꾸 쫑이 "
    "몽실이 다솜이 새롬이 호야 찰떡이 뽀돌이"
).split()  # 일반명사형(보리·도토리·우리왕자·막내공주 등)은 모호해서 뺐다
KINS = [
    "엄마",
    "아빠",
    "삼촌",
    "이모",
    "고모",
    "할머니",
    "할아버지",
    "오빠",
    "언니",
    "형",
    "누나",
    "이모부",
    "고모부",
]

SIDO = [
    (
        "서울특별시",
        "서울시",
        [
            "노원구",
            "관악구",
            "서초구",
            "강동구",
            "영등포구",
            "동작구",
            "성북구",
            "은평구",
            "용산구",
            "광진구",
        ],
    ),
    (
        "경기도",
        "경기",
        [
            "용인시 기흥구",
            "고양시 일산동구",
            "안양시 동안구",
            "부천시 원미구",
            "화성시",
            "평택시",
            "파주시",
            "의정부시",
            "김포시",
        ],
    ),
    ("부산광역시", "부산시", ["부산진구", "수영구", "동래구", "연제구", "사하구"]),
    ("인천광역시", "인천시", ["연수구", "남동구", "부평구", "계양구"]),
    ("대구광역시", "대구시", ["달서구", "수성구", "북구"]),
    ("대전광역시", "대전시", ["유성구", "서구", "중구"]),
    ("광주광역시", "광주시", ["북구", "광산구"]),
    ("충청남도", "충남", ["천안시 서북구", "아산시", "공주시"]),
    ("전라북도", "전북", ["전주시 덕진구", "익산시", "군산시"]),
]
ROADS = (
    "학동로 반포대로 올림픽로 양재대로 논현로 선릉로 영동대로 봉은사로 청계천로 광나루로 왕산로 한강대로 공항대로 마포대로 월드컵로 신촌로 "
    "성산로 도봉로 동일로 화랑로 진흥로 장안로 송정로 서초대로 효령로 사임당로 남부순환로 개봉로 신길로 가산디지털1로 판교로 정자일로 "
    "분당로 수내로 백현로 광교중앙로 영통로 중부대로 용구대로 일산로 호수로 태평로 연수로 센트럴로 아라로 테헤란로 세종대로 중앙로 "
    "을지로 퇴계로 충무로 율곡로 사직로 새문안로 강남대로 언주로 도곡로 압구정로 청담로 삼성로 역삼로 오금로 문정로 위례성대로"
).split()
DONGS = (
    "연남동 합정동 서교동 망원동 문래동 신림동 봉천동 화곡동 목동 상계동 중계동 공릉동 길음동 수유동 쌍문동 천호동 둔촌동 "
    "잠원동 반포동 서초동 논현동 대치동 도곡동 개포동 일원동 구의동 자양동 행당동 이문동 청량리동 정자동 서현동 야탑동 "
    "수내동 판교동 광안동 우동 센텀동 연산동 구월동 송도동 청라동 만촌동 지산동 둔산동 관저동"
).split()
COMPLEXES = (
    "자이 힐스테이트 아이파크 롯데캐슬 e편한세상 센트럴파크 해든마을 햇살마을 늘푸른 한솔 우성 동아 현대 삼성 대림 "
    "두산위브 쌍용 금호어울림 호반써밋 제일풍경채 우미린 반도유보라 계룡리슈빌 신동아파밀리에 서희스타힐스"
).split()
STATIONS = (
    "강남역 홍대입구역 잠실역 수원역 서울역 신촌역 건대입구역 사당역 판교역 정자역 광교역 송도역 해운대역 서면역 동대구역 대전역 "
    "광화문역 여의도역 종로3가역 명동역 이태원역 합정역 노원역 부평역 일산역 천호역 교대역 선릉역 안양역 영등포역 마곡나루역 "
    "미사역 동탄역 청량리역 수서역 성수역 왕십리역"
).split()
CITIES = (
    "인천 수원 대구 광주 울산 강릉 속초 전주 여수 춘천 경주 포항 청주 천안 평택 목포 통영 부천 성남 판교 제주도 거제 양양 "
    "남해 순천 군산 안동 충주 원주 서귀포 김해 창원 광명 의정부"
).split()
LANDMARKS = (
    "코엑스 롯데월드 남산타워 한강공원 올림픽공원 경복궁 북촌한옥마을 인천공항 김포공항 광안리 해운대해수욕장 에버랜드 서울숲 "
    "월드컵경기장 스타필드 반포한강공원 여의도공원 낙산공원 동대문시장 남대문시장 경포대 성산일출봉 불국사 광장시장"
).split()
HOSP_PREFIX = (
    "늘봄 온누리 참사랑 으뜸 한마음 새한 희망 소망 다나은 해맑은 제일 중앙 열린 맑은숲 튼튼 바른 든든 나눔 은빛 한솔 사랑 "
    "연둣빛 푸른솔 서로 이음 아침 소리 하늘숲 미소"
).split()
HOSP_SUFFIX = ["병원", "의원", "치과", "한의원", "내과", "소아과", "안과", "피부과"]
SCHOOL_PREFIX = (
    "한빛 새솔 푸른들 햇살 대성 동산 샛별 꿈나무 무지개 초록 별빛 은하 솔빛 다람 푸른 아름 가람 마루 새움 나래 한울 "
    "도담 소나무 샘터 은행나무"
).split()
SCHOOL_SUFFIX = ["초등학교", "중학교", "고등학교", "대학교", "어린이집", "유치원", "학원"]
COMPANY_PREFIX = (
    "대양 동림 세림 청우 광휘 태성 우림 신우 영진 도원 성광 미르 하늬 누리 아람 해든 드림 비전 넥스트 에이스 모아 "
    "바로 코어 라온 이레 한길 큐브 소프트 유니온"
).split()
COMPANY_SUFFIX = [
    "전자",
    "물산",
    "건설",
    "제약",
    "푸드",
    "식품",
    "상사",
    "테크",
    "시스템즈",
    "산업",
    "금속",
    "유통",
    "화학",
    "소프트",
    "랩",
]
# 금융기관명은 라벨 없음(O): 계좌는 규칙 탐지기 몫이다(LABELING_GUIDE 개정판). 은행 문장은 음성/어려운 음성 용도.
BANKS = [
    "국민은행",
    "신한은행",
    "우리은행",
    "하나은행",
    "농협",
    "기업은행",
    "카카오뱅크",
    "토스뱅크",
    "새마을금고",
    "부산은행",
    "대구은행",
    "광주은행",
    "전북은행",
    "경남은행",
    "제주은행",
    "SC제일은행",
    "케이뱅크",
    "수협",
    "신협",
    "우체국",
    "씨티은행",
    "산업은행",
    "iM뱅크",
    "수협은행",
]
MEDICAL_SUFFIX = tuple(HOSP_SUFFIX) + ("약국",)
EMOJI = [
    "ㅋㅋ",
    "ㅋㅋㅋㅋ",
    "ㅎㅎ",
    "ㅠㅠ",
    "ㅜㅜ",
    "^^",
    "~",
    "!!",
    "?",
    "😊",
    "🙏",
    "👍",
    "ㅇㅇ",
    "..",
    "ㅠ",
]
LEAD = ["아 ", "근데 ", "그리고 ", "참 ", "ㅇㅋ ", "야 ", "저기 ", "혹시 "]
SHORTEN = {
    "했어": "햇어",
    "있어": "잇어",
    "같아": "같애",
    "왔어": "왔엉",
    "갈게": "갈께",
    "알려줘": "알려줭",
    "봤어": "봤엉",
    "거야": "꺼야",
    "괜찮": "괜춘",
    "맞지": "맞져",
    "할게": "할께",
    "어디야": "어디얌",
    "된다": "된당",
}

# --------------------------------------------------------------------------- 문장 틀
# (카테고리, 틀). 카테고리 안에서 4번째마다 val 로 보낸다(틀 단위 분리).
TEMPLATES = [
    ("person_full", "{P}@이/가 내일 학교에 온다고 하더라"),
    ("person_full", "아까 {P}한테 전화 왔었어"),
    ("person_full", "{P} 선생님이 상담 받으래"),
    ("person_full", "다음 주 {P}님 면접이라 응원해줘야지"),
    ("person_full", "야 {P} 알지? 걔 이사 간대"),
    ("person_full", "{P}@을/를 어디서 봤더라"),
    ("person_full", "회식 장소는 {P} 대리가 정한대"),
    ("person_full", "{P}@이랑/랑 점심 먹기로 했어"),
    ("person_full", "축의금은 {P}에게 보내면 돼"),
    ("person_full", "혹시 {P} 학생 어머님 되세요?"),
    ("person_full", "{P}@이/가 보낸 문자 봤어?"),
    ("person_full", "내일 {P}하고 약속 있음ㅋㅋ"),
    ("person_full", "ㅇㅋ {P} 한테 말해둠"),
    ("person_full", "예약자 성함은 {P} 으로 부탁드려요"),
    ("person_given", "{G}@이/가 숙제 다 했대"),
    ("person_given", "{G}@아/야 밥 먹어"),
    ("person_given", "오늘 {G}@은/는 학원 안 가?"),
    ("person_given", "{G}한테 용돈 줬어?"),
    ("person_given", "{G}@이/가 감기 걸렸나봐 ㅠㅠ"),
    ("person_given", "{G}@이랑/랑 같이 갈게"),
    ("person_given", "형이 {G}@을/를 데리러 갔어"),
    ("person_given", "어제 {G} 열 났었잖아"),
    ("person_given", "{G}도 온대 같이 먹자"),
    ("alias", "{AL}@이/가 요즘 말을 안 들어"),
    ("alias", "우리 {AL} 오늘 일찍 와?"),
    ("alias", "{AL}한테 간식 챙겨줬어"),
    ("alias", "할머니는 {AL}@을/를 제일 예뻐하셔"),
    ("alias", "{AL}@아/야 이리 와봐"),
    ("alias", "{AL} 사진 보내줘 ㅎㅎ"),
    ("alias", "내일 {AL}@이/가 소풍 간대"),
    ("alias", "{AL}@은/는 아직 자나?"),
    ("name_kin", "{NK}@이/가 전화했어"),
    ("name_kin", "{NK} 한테 물어봐"),
    ("name_kin", "내일 {NK}@이랑/랑 시장 가기로 했어"),
    ("name_kin", "{NK} 번호 좀 알려줘"),
    ("name_kin", "아 맞다 {NK}한테 안부 전해줘"),
    ("name_kin", "{NK}@이/가 학부모 모임 간대"),
    ("name_kin", "{NK}@은/는 오늘 야근이래"),
    ("kin_neg", "엄마한테 전화해봐"),
    ("kin_neg", "아빠 오늘 몇 시에 와?"),
    ("kin_neg", "할머니 생신 선물 뭐가 좋을까"),
    ("kin_neg", "누나 어디야 언제 와"),
    ("kin_neg", "삼촌이랑 이모가 온대"),
    ("kin_neg", "큰아들 방 좀 치우라고 해"),
    ("kin_neg", "엄마 아빠한테는 비밀이야"),
    ("kin_neg", "시어머니 모시고 병원 다녀올게"),
    ("address", "택배는 {ADF} 로 보내주세요"),
    ("address", "우리 집 주소 {ADF} 맞지?"),
    ("address", "이사 간 곳이 {ADR} 이래"),
    ("address", "주소 찍어줘. {ADR}"),
    ("address", "지금 우리 {ADA} 앞이야"),
    ("address", "나 {ADA} 에 있어"),
    ("address", "여기 {ADC} 맞아?"),
    ("address", "배달 주소 {ADD} 로 바꿔줘"),
    ("address", "할머니 댁은 {ADD} 이야"),
    ("address", "새 집은 {ADC} 래 ㅋㅋ"),
    ("address", "내 주소는 {ADC} 야 기억해둬"),
    ("address", "우편물 {ADA} 로 와"),
    ("address", "{ADR} 맞음?"),
    ("address", "받는 사람 {P}, 주소 {ADF}"),
    ("address", "택배 {ADS} 으로 와요"),
    ("address", "{ADS} 에 주차했어"),
    ("location", "{ST} 에서 만나자"),
    ("location", "{ST} 에서 기다릴게"),
    ("location", "{SR} 에 있는 카페 갈래?"),
    ("location", "주말에 {LC} 놀러 갈까"),
    ("location", "이번 휴가는 {LC} 로 가자"),
    ("location", "{LC} 이번 주말 날씨 괜찮을까?"),
    ("location", "오늘 {LC}에서 약속 있어"),
    ("location", "{LC} 가는 기차 예매했어"),
    ("location", "방금 {LC} 도착했어ㅋㅋ"),
    ("location", "퇴근하고 {LC} 들렀다 갈게"),
    ("location", "{LC} 맛집 추천해줘"),
    ("location", "ㄱㄱ {ST} 에서 봄"),
    ("location", "넹 {LC} 도착함!!"),
    ("location", "집이 {SR} 이라서 금방 가"),
    ("org", "{OH}에 예약 잡았어"),
    ("org", "{OH} 가서 진료 받고 왔어"),
    ("org", "내일 {OH} 갈 사람?"),
    ("org", "{OS} 앞에서 픽업할게"),
    ("org", "{OS} 가정통신문 봤어?"),
    ("org", "우리 애 {OS} 행사가 다음 주야"),
    ("org", "아빠는 {OC} 다니셔"),
    ("org", "{OC} 회식 때문에 늦을 듯"),
    ("org", "{OC} 면접 합격했대!"),
    ("org", "엄마 {OP}에서 약 받아왔어"),
    ("org", "학원은 {OA} 로 옮겼어"),
    ("org", "{OH} 에서 문자 왔네"),
    ("org", "{OG}@이/가 이번 주에 쉰대"),
    ("org", "{OH} 예약 취소함ㅠ"),
    ("mix", "{P}@은/는 {OC} 다니고 {NK}@은/는 {OH}에 입원했어"),
    ("mix", "{P}@이/가 {LC}에서 {OS} 다닌대"),
    ("mix", "{G} 데리고 {OH} 갔다 올게"),
    ("mix", "{NK}@이랑/랑 {LC} 다녀왔어"),
    ("mix", "{AL}@이/가 {OS} 갔다 왔어"),
    ("mix", "{P}한테 {ADC} 로 택배 보냈대"),
    ("mix", "{LC} 사는 {P} 알아?"),
    ("mix", "내일 {ST} 에서 {G}@이랑/랑 만나"),
    ("mix", "{P} 원장님이 {OH} 원장이래"),
    ("mix", "오늘 {OS} 선생님이 {G}@을/를 칭찬했어"),
    ("mix", "어제 {P}@이랑/랑 {LC}에 갔다가 {OH}에 들렀어. 주소는 {ADD} 이야"),
    ("mix", "{NK}@은/는 {ADR} 에 있는 {OP}에 다녀오셨대"),
    ("neg", "이번 달 생활비 {NUM}만원 나왔어"),
    ("neg", "전화번호는 {PH} 야"),
    ("neg", "오늘 저녁 뭐 먹을까?"),
    ("neg", "병원 예약 시간 다시 확인해줘"),
    ("neg", "학교 끝나고 학원 가야 돼"),
    ("neg", "회사 가기 싫다 ㅠㅠ"),
    ("neg", "{DATE} 에 병원 가기로 했어"),
    ("neg", "동생이랑 같이 있어"),
    ("neg", "주차 위치 3층 B구역이야"),
    ("bank", "오늘 {OB} 앱에서 이체했어"),
    ("bank", "퇴근길에 {OB} 에 들러야 해"),
    ("bank", "{OB} 가서 통장 만들어야 해"),
    ("bank", "{OB} 계좌로 {NUM}만원 보냈어"),
    ("bank", "용돈은 {OB} {AC} 로 넣어줘"),
    ("bank", "{OB} 대출 이자 얼마였지?"),
    ("bank", "송금할 곳은 {AC} ({OB}) 이야"),
    ("bank", "{OB} 에서 온 문자 보이스피싱 아니야?"),
    ("bank", "적금 {OB} 로 옮기자"),
    ("bank", "엄마 {OB} 카드 재발급 받아야 한대"),
]

# --------------------------------------------------------------------------- 오탐 감소용 보강(augment)
# 가족 호칭(음성). Baseline val/test 에 나오는 호칭(외할머니·고모·큰누나·외삼촌·울 엄마·큰아들·막내딸·시어머니·장모님·작은이모)은
# 평가의 일반화 측정을 흐리지 않도록 일부러 뺐다.
KIN_TERMS = [
    "엄마",
    "아빠",
    "할머니",
    "할아버지",
    "삼촌",
    "이모",
    "이모부",
    "고모부",
    "형",
    "누나",
    "언니",
    "오빠",
    "동생",
    "큰형",
    "작은누나",
    "큰언니",
    "막내삼촌",
    "외할아버지",
    "처제",
    "형수",
    "올케",
    "사촌형",
    "시아버지",
    "시누이",
    "사위",
    "며느리",
    "형부",
    "처남",
    "둘째형",
    "셋째누나",
    "큰고모",
    "막내이모",
    "친할머니",
    "작은아빠",
    "큰엄마",
    "새언니",
]
# 이름/기관으로 오탐되던 일반 명사(음성). Baseline val 의 송장·박수·강아지는 뺐다.
GENERIC_NOUNS = [
    "택배",
    "주문",
    "배송",
    "환불",
    "영수증",
    "장바구니",
    "사은품",
    "반품",
    "쿠폰",
    "포인트",
    "결제",
    "배달",
    "운송장",
    "주문서",
    "입금",
    "송금",
    "청구서",
    "고지서",
    "상품권",
    "교환",
    "적립금",
    "택배함",
    "배송지",
    "출고",
    "재고",
]
GENERIC_PLACES = [
    "마트",
    "편의점",
    "시장",
    "병원",
    "학교",
    "약국",
    "은행",
    "회사",
    "학원",
    "카페",
    "식당",
    "도서관",
    "세탁소",
    "미용실",
    "우체국",
    "어린이집",
    "유치원",
    "슈퍼",
    "문구점",
    "빵집",
]
# 일반 단어처럼 보이는 이름(어려운 양성). 성+이름 또는 '이름' 표지가 있는 문맥에서만 쓴다.
WORDLIKE_GIVEN = [
    "하늘",
    "나래",
    "이슬",
    "단비",
    "푸름",
    "한결",
    "슬기",
    "다정",
    "가람",
    "보람",
    "노을",
    "하람",
    "새별",
    "나무",
    "바다",
]
WORDLIKE_GIVEN = [
    x for x in WORDLIKE_GIVEN if x not in GIVEN2
]  # 기존 이름 풀과 겹치면 train/val 서로소가 깨진다

AUG_TEMPLATES = [
    ("kin_neg2", "오늘 {KT} 이랑 저녁 먹었어"),
    ("kin_neg2", "{KT} 생일 선물 뭐가 좋을까"),
    ("kin_neg2", "{KT}한테 전화 왔었어"),
    ("kin_neg2", "{KT} 말로는 {P} 오늘 온대"),
    ("kin_neg2", "{KT} 도 같이 가자"),
    ("kin_neg2", "{KT} 모시고 병원 갈게"),
    ("kin_neg2", "주말에 {KT} 만나러 {LC} 가"),
    ("kin_neg2", "{KT} 주민번호 {RR} 맞아?"),
    ("kin_neg2", "{KT} 보고 싶다 ㅠㅠ"),
    ("kin_neg2", "{KT}@이/가 용돈 {NUM}만원 줬어"),
    ("generic_neg", "{GN} {LN} 도착 예정이래"),
    ("generic_neg", "{GN} 언제 와?"),
    ("generic_neg", "오늘 {GN} 확인해줘"),
    ("generic_neg", "{GN} 번호 알려줘"),
    ("generic_neg", "{GN} 취소했어 ㅠㅠ"),
    ("generic_neg", "{GN} 때문에 짜증나"),
    ("generic_neg", "{GN} 보냈다고 문자 왔네"),
    ("generic_neg", "어제 {GN} 했는데 아직이야"),
    ("generic_neg", "{GN} {LN} 조회해봐"),
    ("place_neg", "{GP} 가서 장 봐올게"),
    ("place_neg", "퇴근하고 {GP} 들를게"),
    ("place_neg", "{GP} 문 닫았대"),
    ("place_neg", "{GP} 어디 있어?"),
    ("place_neg", "오늘 {GP} 쉬는 날이래"),
    ("place_neg", "{GP} 에서 {NUM}만원 썼어"),
    ("person_given2", "{G}@이/가 열이 나서 조퇴했어"),
    ("person_given2", "병원에 {G} 데려갔다 왔어"),
    ("person_given2", "{G} 알림장 확인했어?"),
    ("person_given2", "오늘 {G} 생일이야"),
    ("person_given2", "{G} 이번 주 시험이래"),
    ("person_given2", "{G}@은/는 학교 갔어?"),
    ("name_hard", "딸 이름은 {HG}@이/가 좋겠다"),
    ("name_hard", "{HP} 학생 어머님이세요?"),
    ("name_hard", "우리 반 {HP}@이/가 전학 간대"),
    ("name_hard", "내 친구 이름이 {HG} 이래"),
    ("name_hard", "{HP}@이랑/랑 같이 갈게"),
    ("name_hard", "친구 이름은 {HP} 야"),
    ("alias_hard", "집에서 부르는 별명은 {AL} 이야"),
    ("alias_hard", "{AL}@아/야 밥 먹자"),
    ("alias_hard", "별명이 {AL}@이야/야 ㅋㅋ"),
    ("alias_hard", "우리 {AL}@이/가 오늘 소풍 갔어"),
    ("alias_hard", "{AL}@은/는 지금 학원이야"),
    ("org_hard", "{OS} 앞 문구점에서 샀어"),
    ("org_hard", "{OH} 옆 약국이야"),
    ("org_hard", "{OC} 근처 마트에서 장 봤어"),
    ("org_hard", "{OH} 가는 길에 편의점 들를게"),
    ("org_hard", "{OS} 앞 빵집에서 기다릴게"),
]
# 2차 보강(augment=2): 처음 보는 금융기관명(은행·카드·증권·보험)을 라벨 없이 두는 음성. 1차 보강 결과에서 합성 val 의 처음 보는
# 은행명이 ORGANIZATION 으로 오탐되는 것이 가장 큰 잔여 오류여서 추가했다.
EXTRA_BANKS = [
    "웰컴저축은행",
    "OK저축은행",
    "SBI저축은행",
    "한국씨티은행",
    "DGB대구은행",
    "BNK부산은행",
    "NH농협은행",
    "KB국민은행",
    "IBK기업은행",
    "우리카드",
    "롯데카드",
    "BC카드",
    "삼성증권",
    "NH투자증권",
    "DB손해보험",
    "흥국생명",
    "교보생명",
    "한화생명",
    "토스증권",
    "카카오페이",
    "하나카드",
    "농협카드",
    "광주은행",
    "경남은행",
    "저축은행중앙회",
]
AUG_TEMPLATES_2 = [
    ("bank_neg2", "{OB2} 앱 업데이트 해야 돼"),
    ("bank_neg2", "{OB2} 에서 대출 상담 받았어"),
    ("bank_neg2", "{OB2} 적금 만기가 다음 달이야"),
    ("bank_neg2", "{OB2} 카드 한도 올려달래"),
    ("bank_neg2", "{OB2} 에서 문자 왔는데 이상해"),
    ("bank_neg2", "{OB2} 지점 어디 있어?"),
    ("bank_neg2", "이번 달 {OB2} 이자 {NUM}만원 나왔어"),
    ("bank_neg2", "월급이 {OB2} 로 들어와"),
    ("bank_neg2", "{OB2} {AC} 계좌 확인해줘"),
]
# 로드맵 2단계 오류 분석(apps/api NER_VALIDATION.md 11절): 기사체(직함이 붙은 이름, 나이 괄호+인용) 문장에서
# 완전미탐·기관명 오분류가 많았다. 가족 대화체 학습 틀에는 이런 문장이 전혀 없어서 보강한다. evalset_v4(저장소
# apps/api 쪽)의 기사체 틀과 글자 그대로 같은 문장은 넣지 않는다(학습·평가 오염 방지, 시험으로 확인).
AUG_TEMPLATES_2 += [
    ("news_title", "{P2}@이/가 기자회견을 열었다"),
    ("news_title", "{P2} 사장이 입장을 밝혔다"),
    ("news_title", "{P2} 시장이 현장을 찾았다"),
    ("news_title", "{P2} 교수가 연구 결과를 공개했다"),
    ("news_title", "참고인 {P2}@이/가 조사를 받았다"),
    ("news_title", '증인 {P2}({NUM}세)씨는 "기억이 안 난다"고 말했다'),
    ("news_title", "{P2} 선수가 결승에 올랐다"),
    ("news_title", "{P2} 위원장이 회의를 주재했다"),
    ("news_title", "{P2} 부위원장이 성명을 발표했다"),
    ("news_title", "목격자 {P2}({NUM}대)씨가 상황을 설명했다"),
]
# 로드맵 2단계 보조 확인(apps/api NER_VALIDATION.md 15~16절): 실제 가족 메신저 대화체(합성)로 평가했을 때
# 완전미탐이 기사체 보강 전후로 거의 안 줄었다. 이름 토큰을 더 늘리는 보강(16절)은 효과가 없었다 — 평가셋이
# 애초에 train 과 다른 이름 조합을 쓰도록 만들어졌기 때문에(누출 방지 설계), 이름을 얼마나 늘려도 "안 겹치는
# 이름"은 항상 새로 나온다. 대신 이번엔 문장 구조 쪽을 보강한다: 가족 채팅방은 한 메시지에 사람이 2명 이상
# 섞여 나오는 경우가 많은데(틀 "mix" 가 일부 다루지만 적다), 기존 틀의 대부분은 사람 1명짜리라 모델이 "문장에
# 이름이 여러 개 나오는 맥락"을 상대적으로 적게 본다. evalset_v4/v5/realistic_family(apps/api 쪽, 전부 합성)의
# 문장 틀과 글자 그대로 같은 문장은 넣지 않는다(학습·평가 오염 방지, 스크립트로 확인).
AUG_TEMPLATES_3 = [
    ("family_msg", "{P} 이번 주 알림장에 현장학습 적어왔대"),
    ("family_msg", "{P}@이/가 {G} 손 잡고 유치원 갔어"),
    ("family_msg", "저녁에 {P}랑 {G} 피아노 학원 데려다줄 사람?"),
    ("family_msg", "{P} 머리 잘랐는데 완전 달라보여 ㅋㅋ"),
    ("family_msg", "{G}@은/는 줄넘기 연습 다 했어?"),
    ("family_msg", "{P}@이/가 {G} 숙제 봐주고 있어"),
    ("family_msg", "오늘 {P} 태권도 승급 시험 봤대!!"),
    ("family_msg", "{G} 핸드폰 게임만 해서 걱정이야"),
    ("family_msg", "{P}랑 {G} 둘 다 늦잠 자서 지각했어"),
    ("family_msg", "{P} 짝꿍이 바뀌었다고 좋아하더라"),
    ("family_msg", "{G} 받아쓰기 100점 맞았대 ㅎㅎ"),
    ("family_msg", "{P}한테 예방접종 날짜 확인해달라고 해"),
    ("family_msg", "할머니가 {P} 간식 투표하자셔"),
    ("family_msg", "{G} 사진 찍어서 가족방에 올려줘"),
    ("family_msg", "{P} 방학 계획 짜놨는데 같이 볼래?"),
    ("family_msg", "{NK}@이/가 {G} 데리고 급식 먹으러 갔어"),
    ("family_msg", "{P}@이랑/랑 {G} 체험학습 가방 다 쌌어"),
    ("family_msg", "{AL} 전화 안 받는데 혹시 알아?"),
    ("family_msg", "{P} 알러지 검사 결과 나왔대"),
    ("family_msg", "{G}@이/가 {P}@이랑/랑 또 투닥거렸대 ㅋㅋㅋ"),
]
AUG_CATEGORIES = sorted({c for c, _ in AUG_TEMPLATES + AUG_TEMPLATES_2 + AUG_TEMPLATES_3})

# --------------------------------------------------------------------------- 보조
_SLOT = re.compile(r"\{(\w+)\}(?:@([가-힣]+)/([가-힣]+))?")
LABEL_OF = {
    "P": "PERSON",
    "P2": "PERSON",
    "G": "PERSON",
    "AL": "PERSON_ALIAS",
    "NK": "PERSON_ALIAS",
    "ADF": "ADDRESS",
    "ADR": "ADDRESS",
    "ADA": "ADDRESS",
    "ADC": "ADDRESS",
    "ADD": "ADDRESS",
    "ADS": "ADDRESS",
    "LC": "LOCATION",
    "ST": "LOCATION",
    "SR": "LOCATION",
    "OH": "ORGANIZATION",
    "OS": "ORGANIZATION",
    "OC": "ORGANIZATION",
    "OP": "ORGANIZATION",
    "OA": "ORGANIZATION",
    "OG": "ORGANIZATION",
    "HP": "PERSON",
    "HG": "PERSON",
}


def _has_batchim(ch):
    return "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 != 0


def split_pool(items, seed, val_every=5):
    """값 풀을 train/val 서로소로 나눈다(고정 시드 섞기 후 5번째마다 val).

    @param items: 값 목록
    @param seed: 시드
    @param val_every: val 비율의 역수
    @returns: (train 목록, val 목록)
    """
    items = sorted(set(items))
    random.Random(seed).shuffle(items)
    val = [x for i, x in enumerate(items) if i % val_every == val_every - 1]
    tr = [x for i, x in enumerate(items) if i % val_every != val_every - 1]
    return tr, val


def split_templates(templates=TEMPLATES, augment=False):
    """카테고리별로 4번째 틀마다 val 로 보내 틀 목록을 만든다.

    @param templates: (카테고리, 틀) 목록
    @returns: [{"template_id", "category", "text", "split"}]
    """
    counters, out = {}, []
    extra = (
        (AUG_TEMPLATES if augment else [])
        + (AUG_TEMPLATES_2 if augment and int(augment) >= 2 else [])
        + (AUG_TEMPLATES_3 if augment and int(augment) >= 3 else [])
    )
    for cat, text in list(templates) + extra:
        n = counters.get(cat, 0)
        counters[cat] = n + 1
        out.append(
            {
                "template_id": f"synth-{cat}-{n + 1:02d}",
                "category": cat,
                "text": text,
                "split": "val" if n % 4 == 3 else "train",
            }
        )
    return out


class Pools:
    """한 split 이 쓰는 값 풀(test 차단 목록 반영 후, train/val 서로소)."""

    def __init__(self, split, guard, seed=7):
        val = split == "val"

        def part(items, tag, ok):
            tr, va = split_pool([x for x in items if ok(x)], seed + sum(map(ord, tag)))
            return va if val else tr

        def nm(s):
            return guard is None or not guard.blocked_name(s)

        def tx(label):
            return lambda s: guard is None or not guard.blocked_text_value(s, label)

        self.surnames = SURNAMES
        self.given2 = part(GIVEN2, "g2", nm)
        self.given1 = part(GIVEN1, "g1", nm)
        self.nicks = part(NICKS, "nick", nm)
        self.roads = part(ROADS, "road", tx("ADDRESS"))
        self.dongs = part(DONGS, "dong", tx("ADDRESS"))
        self.complexes = part(COMPLEXES, "cx", tx("ADDRESS"))
        self.stations = part(STATIONS, "st", tx("LOCATION"))
        self.cities = part(CITIES + LANDMARKS, "city", tx("LOCATION"))
        self.hosp = part(HOSP_PREFIX, "hp", tx("ORGANIZATION"))
        self.school = part(SCHOOL_PREFIX, "sp", tx("ORGANIZATION"))
        self.comp = part(COMPANY_PREFIX, "cp", tx("ORGANIZATION"))
        self.banks = part(BANKS, "bank", lambda s: True)
        self.extra_banks = part(EXTRA_BANKS, "xb", lambda s: True)
        self.kin = part(KIN_TERMS, "kt", lambda s: True)
        self.generic = part(GENERIC_NOUNS, "gn", lambda s: True)
        self.places = part(GENERIC_PLACES, "gp", lambda s: True)
        self.wordgiven = part(WORDLIKE_GIVEN, "wg", nm)
        self.guard = guard

    def full_name(self, rng):
        """성+이름(2~4자). test 이름 문자열과 같으면 다시 뽑는다.

        GIVEN1(한 글자 이름) 분기는 성 한 글자 + GIVEN1 한 글자가 우연히 GIVEN2 의 다른 이름과 완전히
        같은 문자열이 될 수 있다(예: 성 "하" + GIVEN1 "준" = "하준", GIVEN2 에도 "하준" 이 있음). 그러면
        train 분할의 이 이름과 val 분할의 "하준"(GIVEN2 몫)이 같은 문자열이 되어 "PERSON 값이 train/val
        에 걸침" 누출이 된다. 전체 GIVEN2(분할 전)와 겹치면 다시 뽑아서 막는다.
        """
        for _ in range(100):
            g = (
                rng.choice(self.given2)
                if rng.random() < 0.85 or not self.given1
                else rng.choice(self.given1)
            )
            v = rng.choice(self.surnames) + g
            if g in GIVEN1 and v in GIVEN2:
                continue
            if self.guard is None or not self.guard.blocked_name(v):
                return v
        raise RuntimeError("name pool exhausted")

    def full_name_g2(self, rng):
        """성+이름(given2 전용, GIVEN1 분기 없음).

        ``full_name`` 의 GIVEN1(한 글자 이름) 분기는 성 한 글자 + GIVEN1 한 글자가 우연히 GIVEN2 의 다른
        이름과 완전히 같은 문자열이 될 수 있다(예: 성 "하" + GIVEN1 "준" = "하준", GIVEN2 에도 "하준" 이
        있음). 이 조합이 train/val 다른 분할에서 뽑히면 "같은 PERSON 값이 train/val 에 걸침" 누출이 된다.
        뽑는 문장이 많아지면(보강 틀 추가 등) 실제로 자주 걸려서, given1 을 안 쓰는 버전을 따로 둔다.
        """
        for _ in range(100):
            v = rng.choice(self.surnames) + rng.choice(self.given2)
            if self.guard is None or not self.guard.blocked_name(v):
                return v
        raise RuntimeError("name pool exhausted")


def make_value(slot, pools, rng):
    """슬롯 이름에 맞는 합성 값을 만든다.

    @param slot: 슬롯 이름(P, G, AL, NK, AD*, LC, ST, SR, O*, NUM, PH, DATE)
    @param pools: Pools
    @param rng: random.Random
    @returns: 문자열
    """
    r = rng
    if slot == "P":
        return pools.full_name(r)
    if slot == "P2":
        return pools.full_name_g2(r)
    if slot == "G":
        return r.choice(pools.given2)
    if slot == "AL":
        return r.choice(pools.nicks)
    if slot == "NK":
        sep = "" if r.random() < 0.2 else " "
        return r.choice(pools.given2) + sep + r.choice(KINS)
    if slot in ("ADF", "ADR", "ADS", "ADD", "ADA", "ADC"):
        num = r.randint(1, 450)
        sub = f"-{r.randint(1, 30)}" if r.random() < 0.3 else ""
        road = f"{r.choice(pools.roads)} {num}"
        apt = f"{r.randint(1, 130)}동 {r.randint(1, 25) * 100 + r.randint(1, 12)}호"
        if slot == "ADR":
            return road
        if slot in ("ADS", "ADD"):
            return f"{r.choice(pools.dongs)} {num}{sub}"
        if slot == "ADA":
            return f"아파트 {apt}"
        if slot == "ADC":
            tail = apt if r.random() < 0.7 else apt.split()[0]
            return f"{r.choice(pools.dongs)} {r.choice(pools.complexes)}아파트 {tail}"
        sido, short, gus = r.choice(SIDO)
        head = (sido if r.random() < 0.6 else short) + " " + r.choice(gus)
        tail = f" {apt}" if r.random() < 0.4 else ""
        return f"{head} {road}{tail}"
    if slot == "LC":
        return r.choice(pools.stations) if r.random() < 0.3 else r.choice(pools.cities)
    if slot == "ST":
        v = f"{r.choice(pools.stations)} {r.randint(1, 9)}번 출구"
        return v + " 앞" if r.random() < 0.4 else v
    if slot == "SR":
        return r.choice(pools.stations) + " " + r.choice(["근처", "뒤쪽 골목", "맞은편", "쪽"])
    if slot == "OH":
        return r.choice(pools.hosp) + r.choice(HOSP_SUFFIX)
    if slot == "OP":
        return r.choice(pools.hosp) + "약국"
    if slot == "OS":
        p, suf = r.choice(pools.school), r.choice(SCHOOL_SUFFIX)
        if r.random() < 0.2 and suf.endswith("학교") and suf[0] in "초중고":
            return p + suf[0]
        return p + suf
    if slot == "OA":
        return r.choice(pools.school) + r.choice(
            ["학원", "영어학원", "수학학원", "태권도장", "피아노학원"]
        )
    if slot == "OC":
        return r.choice(pools.comp) + r.choice(COMPANY_SUFFIX)
    if slot == "OG":
        return make_value(r.choice(["OH", "OS", "OC", "OP"]), pools, r)
    if slot == "OB":
        return r.choice(pools.banks)
    if slot == "OB2":
        return r.choice(pools.extra_banks)
    if slot == "KT":
        return r.choice(pools.kin)
    if slot == "GN":
        return r.choice(pools.generic)
    if slot == "GP":
        return r.choice(pools.places)
    if slot == "HG":
        return r.choice(pools.wordgiven)
    if slot == "HP":
        for _ in range(100):
            v = r.choice(pools.surnames) + r.choice(pools.wordgiven)
            if pools.guard is None or not pools.guard.blocked_name(v):
                return v
        raise RuntimeError("name pool exhausted")
    if slot == "LN":
        return "".join(str(r.randint(0, 9)) for _ in range(12))
    if slot == "RR":
        return (
            "".join(str(r.randint(0, 9)) for _ in range(6))
            + "-"
            + str(r.randint(1, 4))
            + "".join(str(r.randint(0, 9)) for _ in range(6))
        )
    if slot == "AC":
        return f"{r.randint(100, 999)}-{r.randint(10, 99)}-{r.randint(100000, 999999)}"
    if slot == "NUM":
        return str(r.randint(2, 300))
    if slot == "PH":
        return f"010-{r.randint(0, 9999):04d}-{r.randint(0, 9999):04d}"
    if slot == "DATE":
        return f"{r.randint(1, 12)}월 {r.randint(1, 28)}일"
    raise KeyError(slot)


def fill_template(text, pools, rng):
    """틀을 채워 구간 목록을 만든다. 라벨 슬롯만 라벨을 갖고 NUM/PH/DATE/AC(계좌번호)는 일반 글자(O)다(은행명 OB 는 ORGANIZATION).

    @param text: 문장 틀
    @param pools: Pools
    @param rng: random.Random
    @returns: [(문자열, 라벨 또는 None)]
    """
    segs, pos = [], 0
    for m in _SLOT.finditer(text):
        if m.start() > pos:
            segs.append((text[pos : m.start()], None))
        slot = m.group(1)
        val = make_value(slot, pools, rng)
        segs.append((val, LABEL_OF.get(slot)))
        if m.group(2):
            segs.append((m.group(2) if _has_batchim(val[-1]) else m.group(3), None))
        pos = m.end()
    if pos < len(text):
        segs.append((text[pos:], None))
    return segs


def add_noise(segs, rng):
    """리터럴 구간에만 반말 줄임·붙여쓰기·말머리·이모티콘을 섞는다(라벨 구간은 건드리지 않는다).

    @param segs: [(문자열, 라벨 또는 None)]
    @param rng: random.Random
    @returns: 새 segs
    """
    out = []
    nospace = rng.random() < 0.10
    short = rng.random() < 0.15
    for idx, (t, lab) in enumerate(segs):
        if lab is None:
            if short:
                for a, b in SHORTEN.items():
                    t = t.replace(a, b)
            if nospace:  # 라벨 구간과 맞닿은 공백은 남겨 경계를 모호하게 만들지 않는다
                lead = " " if t[:1] == " " and idx > 0 and segs[idx - 1][1] else ""
                trail = (
                    " "
                    if t[-1:] == " " and t.strip() and idx + 1 < len(segs) and segs[idx + 1][1]
                    else ""
                )
                t = lead + t.replace(" ", "") + trail
        out.append((t, lab))
    if rng.random() < 0.15:
        out.insert(0, (rng.choice(LEAD), None))
    if rng.random() < 0.35:
        out.append((rng.choice(EMOJI) if rng.random() < 0.6 else " " + rng.choice(EMOJI), None))
    return out


def segs_to_row(segs):
    """segs 를 text 와 문자 단위 spans({label,start,end}) 로 바꾼다.

    @param segs: [(문자열, 라벨 또는 None)]
    @returns: (text, spans)
    """
    text, spans = "", []
    for t, lab in segs:
        if lab:
            sp = {"label": lab, "start": len(text), "end": len(text) + len(t)}
            if lab == "ORGANIZATION" and t.endswith(MEDICAL_SUFFIX):
                sp["context_sensitive"] = True  # 병원·약국: 학습 라벨은 같고 정책 표시만 다르다
            spans.append(sp)
        text += t
    return text, spans


def generate(split, guard, per_template=40, seed=42, augment=False):
    """한 split(train|val) 의 합성 문장을 만든다.

    @param split: "train" 또는 "val"
    @param guard: leak_guard.TestGuard (None 이면 차단 없음; 시험용)
    @param per_template: 틀당 목표 문장 수
    @param seed: 시드
    @param augment: True 면 AUG_TEMPLATES(오탐 감소 보강 틀)도 만든다
    @returns: [{"id","text","spans","source","template_id","category","split","augment"}]
    """
    import ner_common as nc

    pools = Pools(split, guard)
    rows, seen = [], set()
    for tp in split_templates(augment=augment):
        if tp["split"] != split:
            continue
        rng = random.Random(f"{seed}-{tp['template_id']}")
        n, tries = 0, 0
        while n < per_template and tries < per_template * 20:
            tries += 1
            text, spans = segs_to_row(add_noise(fill_template(tp["text"], pools, rng), rng))
            key = nc.normalize_text(text)
            if key in seen or (
                guard is not None
                and (
                    key in guard.texts_norm
                    or nc.skeleton(text, spans) in guard.skeletons_ner
                    or nc.skeleton(text, spans) in guard.skeletons_all
                )
            ):
                continue
            seen.add(key)
            n += 1
            rows.append(
                {
                    "id": f"synth-{split}-{len(rows) + 1:05d}",
                    "text": text,
                    "spans": spans,
                    "source": "synth",
                    "template_id": tp["template_id"],
                    "category": tp["category"],
                    "split": split,
                    "augment": tp["category"] in AUG_CATEGORIES,
                }
            )
    return rows
