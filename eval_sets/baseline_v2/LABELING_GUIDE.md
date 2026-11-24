# Baseline v2 라벨링 기준 (정책 확정판)

모든 문장과 값은 합성이다. 공개 저장소의 세트는 **안전 픽스처**(실존할 수 없는 식별자만)이고, 유효한 전체 식별자가 있는 원본은 git 이 무시하는 `apps/api/data/private_eval/baseline_v2/` 에만 생성한다(`FIXTURE_SAFETY.md`). 위치는 파이썬 문자열 인덱스(코드포인트), 구간은 `[start, end)` 이다.

## 스키마

문장: `id, text, spans, known_names, tags, template_id, split`.
구간: `kind`(탐지 라벨), `start`, `end`, **`mask_default`**(자동 마스킹 기본값), **`context_sensitive`**(문맥 의존 민감 플래그). **탐지 라벨과 마스킹 정책은 분리한다.** 탐지기는 `kind` 를 맞히고, 가릴지는 정책 필드로 따로 정한다.

| kind | 뜻 | mask_default | context_sensitive | 기존 탐지기 Kind(매핑) |
|---|---|:-:|:-:|---|
| PHONE_PERSONAL | 개인 전화(휴대·집) | true | false | PHONE |
| PHONE_PUBLIC | 기관 대표번호(1588·1577·1599·1644·080 …) | **false** | false | PHONE |
| EMAIL | 이메일 | true | false | EMAIL |
| RRN | 주민·외국인등록번호 | true | false | RRN |
| CARD | 카드번호 | true | false | CARD |
| ACCOUNT | 계좌 숫자열 | true | false | ACCOUNT |
| SECRET | 비밀번호·인증번호 값 | true | false | SECRET |
| ADDRESS | 상세 위치(번지·도로명 주소·아파트 동·호수 등) | true | false | ADDRESS |
| LOCATION | 일반 지역명·랜드마크(`강남역 근처`, `서울`) | **false** | false | 없음(평가 전용) |
| PERSON | 성+이름, 또는 이름만 불리는 경우의 이름 | true | false | PERSON |
| PERSON_ALIAS | 특정 사람을 식별하는 별명·`이름+가족호칭`(`콩이`, `지우 엄마`) | true | false | PERSON(접어서 비교) |
| ORGANIZATION | 학교·병원·학원·약국·회사 이름(**금융기관은 제외**) | **false** | 병원·약국·치과면 **true** | 없음(평가 전용) |

평가 전용 라벨(LOCATION, ORGANIZATION)은 기존 탐지기에 대응하는 Kind 가 없다. PERSON_ALIAS 는 기존 탐지기가 `PERSON` 으로만 낼 수 있어 세분 라벨 평가에서는 항상 FN 이다. 이 셋은 결과표에서 "기존 탐지기에 없음"으로 따로 집계한다(0 으로 보여도 탐지기가 약해서가 아니라 라벨 자체가 없기 때문).

`context_sensitive` 는 "병원·약국 이름이 처방 문맥에 나오면 건강정보가 된다" 같은 정책용 표시다. 현재 세트에서는 기관 이름이 `병원`·`약국`·`치과`로 끝나면 true 이다(은행 이름은 계좌 문맥의 일부로만 쓰이고 따로 라벨하지 않는다).

## 일반 원칙

- 구간은 **값 자체**만 잡는다. 앞뒤 조사·호칭·단위·따옴표·구분 공백은 제외한다.
- 값 안의 공백·하이픈·점·OCR 오인식 문자(`O`, `l`, `S`, `B`)와 한글 숫자(`공일공`)는 값의 일부로 **포함**한다.
- **체크섬·Luhn·달력 검증에 실패해도** 해당 형식으로 쓰였으면 보호 대상으로 라벨한다. 주민번호는 2020년 10월 이후 뒷자리가 임의 번호 체계라 체크섬 실패만으로 제외하지 않는다(형식+날짜(+문맥)가 맞으면 RRN). 월일이 달력에 없는 오타(`rrn_invalid_date`)도 형식이 주민번호이면 RRN 으로 라벨하되, 이 경우는 정책상 약한 증거라 탐지기 규칙 쪽이 다르게 볼 수 있다는 점을 결과에서 따로 본다. 기존 세트(`pii_eval*.jsonl`)도 이미 체크섬 실패 주민번호를 RRN 으로 라벨했다(30개 중 12개가 체크섬 불일치).
- 구간끼리 겹치지 않는다. 일반 가족 호칭(엄마, 큰아들, 울 엄마, 시어머니 …)·주문번호·운송장·날짜·버전·금액·일반 명사는 라벨이 없다(음성).

## 종류별 기준

- **PERSON**: `김철수씨` -> `김철수`. 호칭(씨·님·선생님·교수님·대표님)과 조사(한테·도·에게)는 제외. 성 없는 이름(`지우한테` -> `지우`)도 사람 이름이면 PERSON.
- **PERSON_ALIAS**: 이름이 아닌 별명이 특정 사람을 가리킬 때(`콩이`), 또는 `이름 + 가족호칭`이 한 사람을 가리킬 때(`지우 엄마` 전체). 호칭만 있는 `엄마`·`큰아들`은 라벨 없음.
- **ADDRESS**: 시·도~번지/도로명 번호/동·호수(`서울특별시 강남구 테헤란로 152 101동 1204호`). 약식이라도 번지·도로명+번호·`아파트 N동 M호`·`동이름 아파트 N동`처럼 **상세 위치가 식별되면** ADDRESS. `우리`·`집은` 같은 수식어는 제외.
- **LOCATION**: 상세 위치 없이 지역·랜드마크만(`강남역 근처`, `홍대입구역 3번 출구 앞`, `서울`, `코엑스`). 집 근처라고 말해도 상세 위치가 아니면 LOCATION.
- **PHONE_PERSONAL / PHONE_PUBLIC**: 번호 전체(구분자·공백·한글 숫자·`+82` 포함). 기관 대표번호는 전화번호 개체로는 인정하되 PUBLIC 으로 구분하고 기본 마스킹에서 뺀다. `114` 같은 짧은 안내번호는 번호 개체로 보지 않는다(세트에서 제외).
- **EMAIL**: 주소 전체. **ACCOUNT**: 계좌 숫자열만(은행 이름·`계좌` 단어 제외). 은행 이름은 구간이 아니다(`국민은행 123-45-678901` -> ACCOUNT `123-45-678901` 만). 계좌는 규칙 탐지기(은행명·계좌 문맥+숫자열)가 잡고 NER 대상이 아니다. **CARD**: 번호 숫자열 전체(자릿수 오타·Luhn 실패 포함, 유효기간·CVC 제외). **SECRET**: 값만.
- **ORGANIZATION**: 기관 이름만(`새봄병원`). **금융기관명(은행·`농협`·`카카오뱅크`·`토스뱅크` 등)은 라벨 없음**. 평가 전용, 자동 마스킹 기본 false.

## 애매한 사례의 결정

| 사례 | 결정 | 이유 |
|---|---|---|
| 집 근처 랜드마크(`우리 집은 강남역 근처`) | LOCATION | 상세 위치가 아님 |
| `아파트 3동 502호`(단지명 없음) | ADDRESS | 동·호수 상세 위치 |
| `테헤란로 123` | ADDRESS | 도로명+번호 |
| `1588-1234` | PHONE_PUBLIC | 기관 대표번호 |
| 약속 장소(`내일 강남역에서 만나`), 지역(`서울 날씨`) | LOCATION | 일반 지명도 개체로 인정, 마스킹은 안 함 |
| `콩이`(사람처럼 쓰인 별명) | PERSON_ALIAS | 특정 사람 식별 가능 |
| `지우 엄마` | PERSON_ALIAS(전체) | 이름+호칭이 한 사람을 가리킴 |
| `엄마`, `울 엄마`, `큰아들`, `시어머니` | 라벨 없음 | 일반 가족 호칭 |
| 반려동물 이름 | 이 세트에 없음 | 사람 별명과 구분이 어려워 다음 단계로 미룸 |
| 체크섬 실패·월일 오류 주민번호 | RRN | 임의번호 체계·오타 고려 |
| Luhn 실패·자릿수 오타 카드 | CARD | 오타여도 보호 |
| 은행 이름 | 라벨 없음 | 계좌 문맥의 일부. 계좌는 규칙(ACCOUNT)이 따로 잡는다 |
| 병원·약국 이름 | ORGANIZATION + context_sensitive=true | 처방·건강 문맥에서 민감 |
| 날짜(`2024-03-12`)·버전·금액·주문번호·운송장 | 라벨 없음 | 개인 식별 아님 |

## 태그(`tags`)

`normal_*` 정상 형식 / `nospace` / `odd_space` / `name_len_2·3·4` / `name_no_title` / `name_given_only` / `name_known` / `kin_nickname`(가족 호칭 옆 이름) / `kin_nickname_name`(별명) / `alias_title`(이름+가족호칭) / `addr_abbrev`(약식 위치, 정답은 ADDRESS 또는 LOCATION) / `location_general` / `phone_public` / `hangul_num` / `ocr_noise` / `rrn_checksum_fail` / `rrn_invalid_date` / `card_luhn_fail` / `card_typo` / `org_name` / `secret` / `multi_pii` / `long_text`(500자 이상) / `neg_*`(order, waybill, datetime, version, amount, word, kin) / `negative`(정답 구간 없음).

## 분할

`template_id`(범주-틀 번호) 단위로 train/val/test 약 60/20/20. 같은 틀의 변형은 한 분할에만 있고 test 는 train 과 이름 풀·가족 호칭 풀·도로명 풀·문장 틀이 겹치지 않는다. 평가 스크립트가 template_id·정규화 본문·문장 틀(정답 구간을 `[라벨]`로 바꾼 것) 누출을 검사한다. `known_names` 는 가족 사전(`name_known`)에만 채운다.
