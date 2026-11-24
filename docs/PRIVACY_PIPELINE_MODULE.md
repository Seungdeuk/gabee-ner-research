# 가명화 파이프라인 모듈 (`app/shared/privacy`)

외부 LLM(Claude)으로 보내는 가족 대화에서 개인정보를 줄이는 서버 기준 모듈이다. 표준 라이브러리만 쓰고(선택 NER 보강 `ner.py` 만 이미 선언된 `onnxruntime`·`tokenizers`·`numpy` 사용) DB 모델·기존 기능 코드에 의존하지 않는다(독립 패키지). 가족 채팅 AI(분석·답장·주제·미리보기, `app/features/family_chat/ai_service.py`)에 연결되어 있다.

**하지 않는 말**: 이 모듈은 종단간 암호화(E2EE)가 아니다. 서버는 원문을 읽는다. 탐지는 정규식·검증 규칙이라 한국어 이름·주소를 놓칠 수 있고, 마스킹이 완전한 보호를 보장하지 않는다. 성능은 `docs/security/PII_EVALUATION.md` 를 본다.

## 구성

| 파일 | 역할 |
|---|---|
| `types.py` | `Kind`(PHONE, EMAIL, RRN, CARD, ACCOUNT, SECRET, ADDRESS, PERSON), `Risk`, `Span` |
| `detectors.py` | `detect(text, known_names=()) -> list[Span]` 탐지·겹침 정리(전각·대시 변형은 반각으로 접어서 본다) |
| `policy.py` | `Policy`, `Action`(KEEP/MASK/BLOCK), `ANALYSIS`/`REPLY`/`TOPICS`(가족 채팅 줄 단위), **`ASSISTANT`**(자유 글·JSON, AI 어시스턴트용) |
| `pseudonymizer.py` | 요청 단위 `Pseudonymizer`: `apply`, `restore`, `clear` |
| `gateway.py` | `prepare_transcript`, `restore_value`, `restore_json` (통합 지점, 순수 함수) |
| `ner.py` | 선택: 이름·주소 NER 보강(아래 'NER 보강'), `detect_with_ner`·`batch_scope`·`NerEngine` |
| `freetext.py` | 자유 글·JSON: `prepare_text`, `prepare_json`, `PrivacySession`, `generalize_coordinates`, `contains_placeholders`(아래 '자유 글·JSON API') |
| `eval/` | 합성 평가 데이터 2종과 생성기, 평가 스크립트는 `apps/api/scripts/eval_pii.py` |

## 정책 표

| 종류 | 위험도 | 동작 (ANALYSIS / REPLY / TOPICS 공통) | 응답 복원 |
|---|---|---|---|
| PHONE, EMAIL | 중간 | MASK (`[PHONE_1]`, `[EMAIL_1]`) | 복원하지 않고 `(연락처)`, `(이메일)` 로 대체 |
| ADDRESS | 중간 | MASK (`[ADDR_1]`) | ANALYSIS 만 복원(일정 장소), REPLY·TOPICS 는 `(주소)` |
| PERSON | 낮음 | MASK (`[PERSON_1]`), 발화자 이름도 같은 토큰 | 복원(담당자 연결). 가족 호칭이 표시 이름이면 토큰화하지 않음 |
| ACCOUNT, SECRET, RRN, CARD | 높음 | BLOCK: 점수 >= `block_min_score`(0.7)이면 그 **메시지 전체**를 `[민감정보가 있어 제외된 메시지]` 로 대체. 점수가 낮으면 값만 `[RRN]` 등 복원 불가 토큰으로 가림 | 복원 불가 |

정책은 `Policy.actions`·`block_min_score`·`restore_kinds` 필드라서 값을 바꾸면 동작이 바뀐다. 표에 없는 종류는 KEEP. REPLY(15줄)·TOPICS(30줄)의 `max_context_lines` 는 호출부가 문맥 줄 수를 정할 때 참고하는 값이다.

### 카드번호 탐지 단계 (`_detect_card`)

길이 13~19자리와 카드 발급 대역(3·4·5·6·9 시작, 2221~2720)은 모든 단계의 전제다(연도로 시작하는 주문번호 `20…` 제외). 점수 0.7 미만은 값만 `[CARD]` 로 가리고 메시지는 통째로 빼지 않는다(`block_min_score`).

| 검증 | 형식·문맥 | 점수 | 결과 |
|---|---|---|---|
| Luhn 통과 | 4-4-4-4 / 4-6-5 구분자 | 0.95 | 가림 + 메시지 제외 |
| Luhn 통과 | 끊지 않은 15·16자리 + 카드·결제 문맥 | 0.95 | 가림 + 메시지 제외 |
| Luhn 통과 | 끊지 않은 15·16자리, 문맥 없음 | 0.65 | 값만 가림 |
| Luhn 통과 | 그 밖의 길이(13·14·17~19) | 0.4 | 값만 가림 |
| Luhn 통과 | 끊지 않은 번호 + 운송장·주문번호 문맥 | - | 버림 |
| Luhn 실패 | 15·16자리 + 구분자 | 0.6 | 값만 가림 (오타·시험용 번호) |
| Luhn 실패 | 끊지 않은 15·16자리 + 카드·결제 문맥 | 0.5 | 값만 가림 |
| Luhn 실패 | 문맥도 구분자도 없는 순수 숫자 | - | 버림 (주문번호·운송장 오탐 방지) |
| Luhn 실패 | 운송장·주문번호 문맥(구분자 있어도) | - | 버림 |

한계: Luhn 실패 번호는 15·16자리만 본다. 카드 문맥 단어가 20자 밖이거나 문맥 없이 오타 난 끊지 않은 번호는 못 가린다. 주민등록번호·계좌·비밀번호의 단계와 한글 숫자·OCR 문자·띄어쓴 숫자 처리는 아래 절을 본다.

### 주민번호·계좌·비밀번호 탐지 단계 (`_detect_rrn`·`_detect_account`·`_detect_secret`, 숫자 보조 뷰)

**숫자 보조 뷰(`_digit_view`)**: 한글 숫자(공영일이삼사오육칠팔구)·OCR 혼동 문자(`O o l I S B Z`)·한 자리씩 띄운 숫자(`3 5 6 6 …`)로 쓴 덩어리를 숫자로 바꾼 글자에서 주민번호·카드·계좌 규칙을 한 번 더 돌리고, 찾은 구간을 원문 위치로 되돌린다. 숫자로 볼 글자가 10개 이상이고(한글 숫자는 6개 이상 또는 60% 이상, OCR 문자는 진짜 숫자가 6개 이상이며 60% 이상) 앞뒤가 글자·숫자에 붙지 않은 덩어리에만 적용한다. 그런 덩어리가 없으면 이 경로를 타지 않으므로 일반 문장의 결과는 바뀌지 않는다.

**주민등록번호** (0.7 이상이면 메시지 차단, 미만이면 값만 가림):

| 날짜 | 체크섬 | 근거 | 점수 | 처리 |
|---|---|---|---|---|
| 달력에 있음 | 통과 | - | 0.97 | 차단 |
| 달력에 있음 | 불일치 | 하이픈·공백 구분자 있음 / 없음 | 0.6 / 0.45 | 가림 (구분자 없으면 운송장·주문번호 문맥 제외) |
| 달력에 없음 | 통과 | 주민번호 단서 있음 | 0.75 | 차단 |
| 달력에 없음 | 통과 | 단서 없음, 하이픈·공백 있음 | 0.6 | 가림 |
| 달력에 없음 | 불일치 | 주민번호 단서 있음 | 0.5 | 가림 |
| 달력에 없음 | 불일치 | 단서 없음 | - | 탐지 안 함 |
| 달력에 없음 | 통과 | 구분자도 단서도 없는 13자리 | - | 탐지 안 함 (바코드·일련번호와 구분 불가) |
| (앞 6자리 + 구분 숫자 + 뒤 5자리 + `*`·`X`·`●` 1~3개) | - | 날짜 유효 | 0.75(단서) / 0.6 | 끝자리를 가린 번호 |
| `주민등록번호 … 앞자리 YYMMDD … 뒷자리 NNNNNNN` | - | 단서 필수 | 0.75 | 차단 (앞자리부터 뒷자리 숫자 끝까지 한 구간) |

주민번호 단서는 숫자 앞 16자·뒤 6자 안의 `주민번호`·`주민등록번호`·`주민등록증`·`외국인등록번호`다. `본인확인`·`신분증`만으로는 단서로 보지 않는다.

**계좌번호** (점수 사다리, 하이픈이 있으면 +0.1, 최대 0.95. 0.7 이상이면 메시지 차단):

| 문맥 | 기본 점수 | 하이픈 계좌 모양일 때 | 처리 |
|---|---:|---:|---|
| 은행 이름이 숫자 바로 앞뒤 / `은행·뱅크·금고·계좌·입금·송금·이체·예금` 과 함께 20자 안 | 0.8 | 0.9 | 차단 |
| `계좌`·`통장`·`예금주` 가 20자 안 | 0.7 | 0.8 | 차단 |
| `입금`·`송금`·`이체` 만 있음 | 0.6 | **0.7** | 하이픈 계좌 모양(`110-123-456789`)이면 **차단**, 하이픈 없는 10~14자리면 가림만 |
| 하이픈 계좌 모양 + 돈을 옮기는 행동 말(`보내`·`보낼`·`쏴`·`부쳐`·`넣어`) | 0.5 | 0.6 | 가림만 (제한적 후보) |
| 단서 없음 | - | - | 탐지 안 함 |

- 단서 단어 하나가 모든 숫자를 차단하지는 않는다. 숫자는 10~14자리의 계좌 모양(하이픈 묶음 `2-6 / 2-8 …`, 공백 묶음 3~4묶음, 연속 10~14자리)이어야 한다. 전화번호 모양(`_PHONE_FULL`), 앞 14자·뒤 6자 안의 `주문·운송장·택배·접수·영수증·승인번호 …` 문맥, 공백 묶음 뒤에 붙은 금액(`원·만원·개·명·%`), 숫자 목록의 중간은 계좌로 보지 않는다.
- 위 표의 `입금·송금·이체` 약한 단서 + 하이픈 모양 = 0.7 은 점수 사다리의 결과로 **의도한 차단**이다(과거 주석은 '가리기만'이라고 적었으나 실제 동작과 달랐다). 하이픈 없는 숫자는 같은 단서에서도 가리기만 한다.

**비밀번호·인증번호** (`_detect_secret`): 단서 말(`비밀번호·비번·패스워드·핀번호·인증번호·인증코드·보안코드·OTP·PIN`) 바로 뒤의 값(기존)에 더해, 단서 말 뒤 16자 안(숫자·줄바꿈 없는 구간)에서 **처음 나오는 숫자열이 4~8자리**이면 값으로 본다(`인증번호 왔어? 482913`). 앞이 `-./:` 이거나 뒤에 단위(`년월일시분초원개명%자호세동층번주달회`)가 붙거나 날짜·시간·소수·전화번호의 일부이거나 이미 잡힌 값이면 제외한다. 연도 모양(19xx·20xx 4자리)은 단서와 떨어져 있으면 날짜일 가능성이 커서 **잡지 않는다**(`비밀번호 바꿨어, 2026` 이 가드 임계값 0.45 에서 막히지 않게). 단서 바로 뒤의 값은 기존 규칙이 연도 모양이어도 그대로 본다(`비밀번호 2026` 은 차단). 그 밖의 값은 0.9(차단). 단서 없는 4~8자리 숫자는 비밀번호로 단정하지 않는다.

**카드번호 추가 단계** (`_detect_card_loose`): 4-4-4-4 가 아닌 묶음(8-8, 4-8-4 …)과 `4000 - 0000 - …` 처럼 앞뒤에 공백이 낀 하이픈은 15·16자리이고 발급 대역이며 운송장·주문번호 문맥이 아닐 때, Luhn 통과 + 카드·결제 문맥이면 0.95(차단), Luhn 통과 + 문맥 없음이면 0.65(가림), Luhn 실패 + 문맥이면 0.5(가림), 그 밖은 버린다.

이 단계는 외부 전송 직전 잔류를 줄이려는 합성 데이터 기반 개선이며, 모든 입력에서 누출이 없다는 보증이 아니다(사진 속 숫자, 임의의 은어·암호화 표기는 규칙으로 찾을 수 없다).

복원 범위를 좁힌 이유: 요약 문장에 실제 전화·이메일이 되살아나면 AI 결과 캐시(DB)와 모든 구성원 화면에 그대로 들어간다. 연락처는 필요하면 사용자가 채팅 원문에서 본다.

## 사용법

```python
from app.shared.privacy import ANALYSIS, prepare_transcript, restore_json

rows = [(msg.id, speaker_name, msg.text) for msg, speaker_name in db_rows]   # DB 모델에 의존하지 않는다
with prepare_transcript(rows, member_names, ANALYSIS) as prep:
    for line in prep.lines:            # line.id, line.speaker(토큰), line.text(가명화됨), line.blocked
        ...                            # 이것만 LLM 프롬프트에 넣는다
    llm_result = call_llm(prep.lines)  # dict (구조화 출력)
    restored = restore_json(prep, llm_result)
    data, unknown = restored.value, restored.unknown_tokens
# with 블록이 끝나면(예외 포함) 대응표 참조가 지워진다
```

- `prep.blocked_ids`, `prep.blocked_count`, `prep.counts_by_kind` (예: `{"PHONE": 2, "SECRET": 1}`)는 사용량·화면 안내용이다. 값은 담지 않는다.
- `restore_json` 결과: `.value`(복원된 JSON), `.restored_tokens`, `.generalized_tokens`(정책상 복원하지 않고 `(연락처)` 로 바꾼 것), `.unknown_tokens`(이 요청이 만들지 않은 토큰 모양). `generalize_unknown=True` 를 주면 unknown 자리를 `(알 수 없는 항목)` 으로 바꾼다(기본은 그대로 두고 호출부가 결정).
- `prepare_transcript` 의 `member_names` 원소는 이름 문자열이거나 **같은 사람의 표기 묶음** `(실제 이름, 표시 이름, ...)` 이다(첫 번째가 대표). 묶음의 어느 표기로 불러도 발화자(실제 이름)와 같은 `[PERSON_n]` 이 되고 복원하면 대표 이름이 나온다. 가족 채팅 AI 는 구성원마다 `(User.name, display_name)` 을 넘긴다.
- `restore_json(..., scrub=True)` 는 복원 뒤에도 남은 무력화 토큰(`‹PERSON_2›`, 사용자가 대화에 직접 쓴 토큰 모양)을 `(알 수 없는 항목)` 으로, 복원 불가 고정 토큰(`[RRN]`·`[CARD]`·`[ACCOUNT]`·`[SECRET]`)을 `(민감 정보)` 로 바꾼다(`scrub_leftovers`). 가족 채팅 AI 는 이 옵션을 켠다.
- 낱개 글자는 `Pseudonymizer().apply(text, policy, known_names)` -> `MaskResult(text, counts_by_kind, blocked)`, `restore(text, kinds=None, generalize_unknown=False)` -> `RestoreResult`.

## 실제 연결 방식 (가족 채팅 AI, 구현됨)

- 서버가 DB 에서 읽은 줄 `(id, 발화자 이름, 글자)` 와 그 가족 구성원의 실제 이름·표시 이름(`ai_repository.member_names`)으로 `prepare_transcript` 를 부른다. 글자 수 한도(`AI_ANALYSIS_MAX_CHARS`)는 **가명화된 줄 기준**이고, 오래된 메시지를 뺀 뒤에는 남은 메시지만으로 다시 가명화해 대응표와 개수가 보내는 줄과 맞게 한다.
- 호출 단위 정책: 분석 `ANALYSIS`(최대 200줄), 답장 `REPLY`(15줄), 주제 `TOPICS`(30줄). `max_context_lines` 는 쓰지 않고 호출부가 줄 수를 정한다.
- **캐시에는 토큰 상태의 결과를 저장**한다(이 문서 아래 4번의 '복원된 값을 저장하지 않는다'를 그대로 따른다). 응답 직전에 같은 범위를 다시 `prepare_transcript` 해(결정적: 같은 입력이면 같은 토큰 번호) `restore_json(prep, 결과, generalize_unknown=True)` 로 복원한다. 캐시 키에는 프롬프트 버전, `MODULE_VERSION`(`app/shared/privacy/__init__.py`), 정책 이름·`block_min_score`·`restore_kinds`, 가명화된 줄의 해시가 들어간다. 규칙을 바꾸면 `MODULE_VERSION` 을 올린다.
- 순서: 아래 1번(복원 먼저 후 검증)과 달리 **검증(`clean_analysis`)을 토큰 상태에서 먼저** 하고 캐시에 저장한 뒤 복원한다. 담당자 검증은 '대화에 나온 `[PERSON_n]` 토큰 또는 가명화하지 않은 발화자(엄마·아빠)'인지로 하고, 복원된 이름은 응답에만 들어간다. 이 요청이 만들지 않은 토큰(unknown)은 일정·할 일 후보에서 해당 후보를 버리고, 나머지 문장은 `(알 수 없는 항목)` 으로 바꾼다.
- 차단된 메시지의 id 는 `important_messages`·`source_message_ids` 에서 뺀다. 응답 `meta.privacy = {masked: {종류: 개수}, blocked_messages: 개수}` 에 개수만 담는다(사용 기록 `ai_usage_logs` 에는 컬럼이 없어 남기지 않는다).
- `POST /family-chat/ai/preview` 는 같은 `prepare_transcript(ANALYSIS)` 결과의 줄을 그대로 보여 준다. AI 를 부르지 않는다.
- 차단 조건: 차단되지 않은 메시지가 3개 미만이면(전부 차단된 대화 포함) 분석·답장·주제는 AI 를 부르거나 한도를 쓰지 않고 400 `too_few_messages` 이다. 미리보기는 `sendable` 로 알린다. 차단된 줄은 시각도 보내지 않는다(빈 문자열).
- 알려진 한계: 이름 사전은 구성원의 이름·표시 이름뿐이다(개인별 별명 `member_nicknames` 는 보는 사람마다 달라 쓰지 않는다). 가명화 후에도 가족 호칭·문맥으로 사람을 추측할 수 있다.

## 통합 가이드 (`ai_service.py` 에 연결할 때)

1. **순서**: 가족 채팅 AI 는 `clean_analysis` 를 **토큰 상태로 먼저** 하고(담당자는 '대화에 나온 `[PERSON_n]` 토큰'인지, 날짜·id 검증), 그 결과를 캐시에 토큰 상태로 저장한 뒤, 사용자에게 돌려줄 때 `restore_json` 으로 복원한다. 복원된 값은 캐시에 들어가지 않는다. 복원 뒤에는 담당자 이름이 그 가족 구성원의 이름·표시 이름과 일치하는지 한 번 더 본다(아니면 담당자만 비움). 호출부가 복원된 값 기준으로 검증하려면 `restore_json` 을 먼저 부르면 되지만, 그러면 복원된 값을 저장하지 않도록 따로 신경 써야 한다. 별칭("민수")은 대표 이름("김민수")으로 복원된다.
2. **프롬프트**: 시스템 프롬프트를 바꿔야 한다. "토큰(`[PERSON_1]`, `[ADDR_1]` 등)은 글자 그대로 복사하고 바꾸거나 새로 만들지 않는다. 전화번호·이메일 토큰은 응답에 쓰지 않는다"를 넣고, 기존 "사람 이름은 대화에 나온 이름 그대로" 문구를 토큰 기준으로 고친다. 토큰이 깨지면 복원되지 않는다. `PROMPT_VERSION` 도 올린다.
3. **글자 수 자르기**: `max_chars` 한도는 **가명화 후 토큰화된 줄 기준**으로 자르는 것을 권장한다(실제로 외부에 나가는 길이). 오래된 메시지부터 빼는 규칙은 그대로 둔다. 가명화 전에 자르면 한도를 잘못 센다.
4. **캐시**: `ai_result_cache.result` 에 **복원된 값을 저장하지 않는다**. 복원은 응답 직전에 요청 메모리에서만 하고, 캐시에는 복원 범위 정책을 적용한 결과(전화·이메일은 이미 `(연락처)` 로 일반화됨)만 넣는다. 이름·주소가 들어가는 일정/할 일 후보를 캐시에 둘지는 별도로 결정해야 한다(감사 M3). 캐시 키에는 정책 이름·`block_min_score` 가 영향을 주므로 `PROMPT_VERSION` 으로 구분한다.
5. **차단 메시지**: `prep.blocked_ids` 의 메시지는 AI 가 보지 못한다. `source_message_ids`·`important_messages` 에 그 id 가 들어오면 버릴지 정한다. 화면에 "민감정보가 있어 N개 메시지는 제외했어요"를 안내할 수 있다(`blocked_count`).
6. **`restore_json` 의 범위**: dict 의 **값과 키**, list·tuple·set 의 원소 안 문자열을 복원한다. 문자열이 아닌 값은 건드리지 않는다. 사용자 정의 객체(Pydantic 모델 등)는 `model_dump()` 한 dict 로 넘긴다. 문자열 안에서 토큰이 쪼개져 있으면(`[PERSON_` + `1]`) 복원하지 못한다.
7. **로그**: 모듈은 `logging` 을 쓰지 않는다. 호출부도 `prep`·`MaskResult`·복원 전 응답을 로그에 찍지 않는다(`counts_by_kind` 만 사용량 기록용으로 안전).
8. **두 군데로 갈라진 마스킹 규칙**: `app/features/ai_assistant/ingest/masking.py`(`mask_text`)가 문서 저장용으로 따로 정규식 마스킹을 한다. 이 모듈과 규칙(전화·카드·주민번호·계좌·이메일·주소·라벨 이름)이 겹치지만 목적이 다르다(저장 시 영구 치환 vs 요청 시 가역 가명화). 통합 계획만 적는다: (a) 당장은 코드를 합치지 않고, (b) 중복되는 탐지 규칙(`detect`)을 `masking.mask_text` 가 재사용하도록 다음 단계에서 정리하며, (c) AI 어시스턴트 질문(`/ai-assistant/chat`)에도 같은 `ANALYSIS`/별도 정책을 적용할지는 요구사항 확인 후 정한다. 이번에는 기존 코드를 수정하지 않았다.
9. **성능**: 200줄×2,000자는 1초 안팎이라 요청 경로에서 동기로 돌려도 된다. 반복 입력 같은 최악 조건도 10만 자에 0.1~0.2초 이하다(선형).

## 자유 글·JSON API (`freetext.py`, 추가됨 / 통합은 별도 PR)

가족 채팅 줄 단위(`prepare_transcript`)와 별개로, **줄 구조가 없는 글**(AI 어시스턴트 질문·앞 대화)과 **구조화 JSON**(탭 에이전트 도구 결과)을 가명화하는 API 다. **이번 변경은 공용 모듈 안에서만** 이루어졌고 `ai_assistant`·`billing`·`care`·`schedule`·`family_chat` 코드는 바꾸지 않았다. 아래 '통합 가이드'는 **담당 팀과 협의한 뒤 별도 PR** 에서 진행한다. 탐지기는 `detect` 그대로라 탐지 성능은 `PII_EVALUATION.md` 와 같다(시험 `test_free_text_uses_same_detector_as_evaluation` 이 두 평가 데이터셋으로 탐지 결과가 그대로 반영되는지 확인한다). 기존 `prepare_transcript`·`ANALYSIS`/`REPLY`/`TOPICS` 동작은 바뀌지 않았다.

### 사용법

```python
from app.shared.privacy import ASSISTANT, new_session, prepare_json, prepare_text

# 한 요청 안의 여러 조각이 토큰 번호를 공유하도록 세션 하나를 만든다
with new_session(ASSISTANT, members=[("김민수", "민수"), "이영희"]) as sess:
    q, _ = prepare_text(question, session=sess)                      # 자유 글
    data, _ = prepare_json(tool_result, session=sess,                # 구조화 JSON
                           keys_to_drop=("memo",))
    answer = call_llm(q, data)                                       # 가명화된 것만 나간다
    shown = sess.restore_text(answer, generalize_unknown=True, scrub=True).value
    # sess.blocked_count, sess.counts_by_kind 는 안내·사용량용(값은 담지 않는다)
```

- `prepare_text(text, *, policy=ASSISTANT, members=(), session=None)` -> `(글, 세션)`. `session` 을 안 주면 새로 만들어 돌려주고, 호출부가 `close()`/`with` 로 지운다.
- `prepare_json(obj, *, policy, members, keys_to_drop, coordinate_keys, coordinate_precision, numbers, number_keys, pseudonymize_keys, session)` -> `(JSON, 세션)`.
  - 문자열 **값**만 가명화한다. 구조·키 순서·불리언·null·날짜·숫자는 유지(원본 객체는 건드리지 않는다). dict·list·tuple·set 을 재귀로 본다(깊이 64 제한).
  - 키는 기본으로 그대로 둔다. 키가 사람 이름일 수 있는 dict(`{"김민수": 5000}`)는 `pseudonymize_keys=True`. 서로 다른 키가 같은 키로 바뀌어 값이 덮어써질 상황(같은 사람의 두 표기 등)이면 **`ValueError`** 로 막는다(값을 합치거나 잃지 않는다).
  - `keys_to_drop`: 해당 키(대소문자 무시)를 값과 함께 뺀다.
  - 숫자: `numbers="keep"`(기본, `Policy.number_mode`) 이면 그대로, `"round"` 면 유효숫자 2자리로 뭉갠다(`generalize_number`: 123456 -> 120000). `ASSISTANT` 는 금액이 답에 필요해 keep. **`id`·`ids`·`*_id`·`*_ids`·`*Id`·`*ID`·`*Ids`·`*IDs` 키(`userID`, `member_ids` 등)의 숫자는 `round` 에서도 항상 보존**(값이 리스트여도 안쪽까지)하고, `number_keys=["amount", ...]` 를 주면 그 키(대소문자 무시)의 숫자만 뭉갠다. 정한 방침: 지정한 키 **아래의 중첩 dict 값도** 그 키의 숫자로 보고 뭉갠다(`{"amount": {"value": 12345}}` -> 12000). id 키 아래는 보존하고, 지정하지 않은 키의 하위(`items` 안의 `price`)는 뭉개지 않는다. 기본(None)은 id 가 아닌 숫자 전부다.
  - 지원하지 않는 형식(사용자 정의 객체)은 개인정보를 못 가릴 수 있어 **TypeError(실패 시 막음)**. Pydantic 모델은 `model_dump(mode="json")` 으로 넘긴다(`mode="json"` 이어야 날짜·UUID 같은 값이 문자열이 되어 가명화 대상이 된다). 정수·실수 필드에 든 전화번호 같은 값은 못 찾는다.
- `PrivacySession`: `prepare_text`, `prepare_json`, `restore_text`, `restore_json`, `blocked_count`, `high_risk_count`, `counts_by_kind`, `coordinates_generalized`, `close()`. `repr` 에는 개수만, `__slots__` 라 `vars(sess)`/`__dict__` 가 없고, `pickle`·`copy`·`dataclasses.replace`(대응표를 공유하는 사본) 는 `TypeError`. 기존 `restore_json(sess, ...)` 에도 넘길 수 있다. 속성을 코드가 직접 읽는 것까지는 막지 못한다.
- `contains_placeholders(text_or_obj) -> bool`: 일반화 문구(`(연락처)`·`(이메일)`·`(주소)`·`(사람)`·`(알 수 없는 항목)`·`(민감 정보)`, 공백·전각 괄호 변형 `( 연락처 )` 포함)·고정 토큰(`[RRN]` 등)·번호 토큰 모양(`[PERSON_3]`·`(PERSON_1)`·`<PHONE_1>`·`{PERSON_1}`·`[ADDRESS_1]`·`[NAME_1]`·대소문자·공백 변형·ZWSP 낀 토큰·괄호 없는 `PERSON_1`·`PHONE_1`)·무력화 토큰(`‹PHONE_1›`)이 남았는지 본다(키·값·리스트 원소 전부). 사용자가 실제로 쓴 `회의 (주소) 확인` 도 True 가 되는데, 쓰기 도구 호출만 거부되는 **안전한 방향의 오탐**이라 허용한다. 쓰기 도구 인자 거부용이다(아래 통합 가이드 2번).

### `ASSISTANT` 정책과 고위험 값 처리

| 종류 | 동작 | 복원 |
|---|---|---|
| PHONE, EMAIL | MASK(`[PHONE_1]`, `[EMAIL_1]`) | 하지 않음 -> 응답에서 `(연락처)`, `(이메일)` |
| ADDRESS, PERSON | MASK(`[ADDR_1]`, `[PERSON_1]`) | 복원 |
| RRN, CARD, ACCOUNT, SECRET | **그 값만** `[RRN]`·`[CARD]`·`[ACCOUNT]`·`[SECRET]` 로 대체 | 복원 불가(`scrub=True` 면 `(민감 정보)`) |

**설계 결정(통째 제외 vs 값 대체)**: 줄 단위 대화에는 '그 메시지를 빼도 나머지 대화가 남는다'는 단위가 있지만, 자유 글(질문 하나·JSON 한 덩어리)에는 그런 단위가 없다. 통째로 빼면 질문이 사라지고 도구 결과는 쓸 수 없게 된다. 그래서 자유 글·JSON 은 **점수와 관계없이 고위험 값만 고정 자리표시로 대체**하고, 점수가 `block_min_score`(0.7) 이상인 것의 수를 `blocked_count` 로 알린다(점수가 낮은 후보도 가리지만 세지 않는다). **그래서 `blocked_count` 는 '가린 고위험 값의 수'가 아니다.** 체크섬이 맞지 않는 `주민 900101-1234567`(점수 0.6)이나 라벨 없는 카드는 가려져도 0 일 수 있다. 사용자 안내("민감정보 N개는 가리고 처리했어요")에는 **`session.high_risk_count`**(= `counts_by_kind` 의 RRN·CARD·ACCOUNT·SECRET 합, 점수와 무관)를 쓴다. 정책이 `ANALYSIS` 여도 `prepare_text` 는 항상 값 단위다(줄 단위 BLOCK 은 `prepare_transcript` 에서만).

### 좌표·숫자 일반화

- `generalize_coordinates(lat, lng, *, precision=2)`: 소수점 `precision` 자리 아래를 **절삭**(0 쪽으로 버림, 반올림 아님)한다. precision 0=약 111km, 1=약 11km, **2=약 1km(위도 1.1km, 경도는 한국 위도에서 약 0.9km)**, 3=약 110m. 칸의 모서리 값이 되므로 원래 위치는 그 칸 안 어디든일 수 있다. 정확한 주소는 알 수 없지만 **익명은 아니다**(동네 수준은 추정 가능). 범위(위도 +-90, 경도 +-180) 밖이나 숫자가 아니면 `ValueError`. `-0.0` 은 `0.0` 으로 정리한다.
- 글 속 좌표 쌍(`prepare_text`): 소수점 4자리 이상 두 수가 쉼표·세미콜론·공백으로 붙고(영문자·숫자·점 뒤에서 시작하지 않음) 지구상 유효한 쌍이면서, **(a) 가까이(앞 24자·뒤 12자) 위도·경도 단서**(위치·좌표·위도·경도·GPS·lat·lng·지도·`?q=`·`@` 등)가 있거나 **(b) 단서 없이도 두 수 모두 소수 5자리 이상이고 한 수가 10 이상인 정밀 좌표**일 때만 `37.56, 126.97` 로 절삭한다. 단서 없는 4자리 소수 두 개(환율·금리·체온·`v1.23456 2.34567`·금액·날짜·버전·확률 `0.12345, 0.98765`)는 건드리지 않는다. 경도가 먼저 오는 표기(`126.977969, 37.566535`, GeoJSON 순서)는 앞 수가 |90| 을 넘을 때만 범위로 알아본다(두 수가 모두 90 이하이면 순서와 상관없이 같은 절삭이라 결과는 같다). 단서가 멀면(예: 같은 문단의 '위치' 와 떨어진 숫자 쌍) 4자리 쌍은 못 잡는다. 소수 5자리 미만이면서 단서가 없는 좌표는 한계다.
- JSON 의 좌표 쌍(`[위도, 경도]`·`[경도, 위도]` 실수 두 개 리스트)은 **부모 키가 좌표 단서**(coord·coordinates·location·position·pos·point·geo·latlng·lnglat·pt·loc·center·geometry·gps)이고 소수 4자리 이상이면 바꾸고, 단서 키가 아니면 정밀 좌표 규칙(b)일 때만 바꾼다(`{"probs": [0.12345, 0.98765]}`·`{"weights": [12.345, 56.789]}` 는 그대로).
- JSON 에서는 `coordinate_keys`(기본 `COORDINATE_KEYS` = lat, lng, lon, long, latitude, longitude, 대소문자 무시)의 값에 같은 절삭을 적용한다. 숫자로 해석되지 않으면 None. 좌표 키가 다른 의미로 쓰이면 `coordinate_keys=()` 로 끈다. 아예 빼려면 `keys_to_drop` 이 우선한다.

### 복원 규칙

기존과 같다(`restore_kinds` 안의 종류만, **이 요청이 만든 정확한 토큰만**). `ASSISTANT` 는 이름·주소만 복원하고 연락처는 `(연락처)` 같은 일반 문구가 된다. 지어낸·변형 토큰(`[PHONE_9]`, `[person_1]`, 전각 괄호)은 복원하지 않고 `unknown_tokens` 에 적는다. 복원 결과를 다시 복원하거나 사용자가 직접 쓴 토큰 모양을 복원하는 일은 무력화(`‹PHONE_1›`)로 막는다. 다른 요청의 세션은 서로 토큰을 복원하지 못한다. 번호는 등장 순서라 같은 입력이면 같은 토큰이다.

### 한계

- 문맥 없는 이름(이름 사전·호칭 없는 '민수가 왔어' 속 모르는 사람), 약식 주소('강남역 3번 출구 빌라 302호'), **한글 숫자 표기('공일공 일이삼사 …')**, 공백 구분 계좌, 키워드 없는 비밀번호는 못 잡는다. 독립 보조 세트 기준 놓침 목록은 `PII_EVALUATION.md` 에 있다.
- 가명화 후에도 문맥으로 사람을 추측할 수 있다. 좌표는 거칠게 해도 동네 수준 위치가 남는다. E2EE 가 아니다.
- 못 잡는 것(추가 확인됨): 숫자 사이에 낀 폭 있는 공백이나 한글·영문 글자(`010 일234`), 분해형(NFD) 한글 이름, **붙어 있는 두 번호**(`010-1234-5678010-9999-8888`: 하나로 보거나 못 잡음), 좌표 쌍 규칙에 안 맞는 좌표(소수점 3자리 이하, 한쪽만 적힌 좌표, 좌표 키 밖의 문자열 `37.56°N`). 투명·방향 제어 문자(soft hyphen U+00AD, CGJ U+034F, U+180E, U+200B–200F(ZWSP·ZWNJ·ZWJ·LRM·RLM), bidi 제어 U+202A–202E·2066–2069, U+2060–2064, FEFF)가 낀 번호·이름은 탐지기가 지우고 탐지한 뒤 원문 위치로 되돌려 찾는다(번호 안쪽 투명 문자는 번호와 함께 사라지고 바깥쪽은 남는다). 그 밖의 보이지 않는 문자(예: U+3164 한글 채움 문자)는 처리하지 않는다.
- 주소 뒤 번호가 전화번호 모양(`테헤란로 010-1234-5678`, `역삼동 02-1234-5678`)이면 건물번호·지번으로 보지 않고 전화로 찾는다(탐지기 수정, 가족 채팅 `prepare_transcript` 에도 적용). 이때 도로명·동 이름만 주소로 남기는 것은 **앞에 시·군·구 이름이 바로 붙은 때**(`서울시 강남구 테헤란로 010-…`)로 한정한다. `이 번호로 010-…`·`전화로 010-…`·`학교로 031-…` 같은 일상 말은 주소가 아니라 전화만 잡는다(시·군·구 없는 `테헤란로 010-…` 의 도로명은 주소로 못 잡는다).
- 숫자 필드(금액)와 키는 기본으로 그대로 나간다. 지출 제목·메모 같은 **문자열**은 탐지 범위 안에서만 가려진다.
- 매우 긴 입력도 선형이다(10만 자 반복 입력 시험). 그래도 **통합 시 요청당 입력 길이 상한을 두기를 권장**한다(질문·도구 결과 합계 N자 등: 외부 LLM 비용·지연 때문이기도 하다). 한 요청에 쓰는 세션은 하나로 묶어야 토큰이 일관된다.

### 통합 가이드 (담당 팀 협의 후 별도 PR)

**이번 PR 에서는 아래를 하지 않았다.** 담당 팀과 협의한 뒤 별도 PR 로 한다.

> **2026-11-04 현황 (메인 AI 채팅)**: 2026-11-02 에 메인 채팅에 연결했던 가명화(`rag_privacy.py`, ADR 0013)는 **걷어냈다**(`rag_privacy.py` 와 시험은 삭제됨, ADR 0014).
> 메인 채팅은 질문·앞 대화·도구 결과를 가리지 않고 모델로 보낸다(채팅은 서버 DB 에 저장되지 않고 외부 전송은 공지로 처리). 메인 채팅의 도구 반복 그래프
> (현재 `shared/llm/tool_graph.py`)에도 가리기·되돌리기 노드는 없다. 아래 1~4·6은 **메인 채팅에는 적용되지 않고**, 탭 에이전트(`billing`·`family_chat`)가
> 자기 모델을 부르는 경로에 가명화를 붙일 때의 가이드로 남긴다(`family_chat` 에이전트는 이미 요청당 세션으로 가명화·복원한다). 돌봄 에이전트(`care/agent.py`)는
> 없어져 대상이 아니다(다시 붙일 때 새로 만든다). 아직 안 된 것은 정산 에이전트의 가명화(2의 `agent.py` 지점)와 5(`agent_guard` 규칙 위반 정리)다.
> 메인이 탭 에이전트에 넘기는 질문은 방 대화 문맥이 있으면 이름 토큰이 풀리지 않은 채로 간다: `rag_chat._handle_calls` 가 일정 에이전트(자체 가명화가 있다)에게만 원래 이름으로 되돌려 넘기고,
> 정산·가족채팅 에이전트에는 토큰을 그대로 둔다. 방 대화 문맥이 없으면 질문은 가리지 않은 실제 글이라 그 에이전트들의 규칙 가리기가 막지 못하는 이름은 나간다.

1. **`rag_chat.ask` 의 `build_messages(question, history)` 호출 지점**(`ai_assistant/knowledge/rag_chat.py`, 함수는 `shared/llm/agent.py` 와 공용): 질문과 앞 대화를 만들기 전에 요청당 세션을 만들고(`new_session(ASSISTANT, members=가족 구성원 이름·표시 이름)`), 질문과 각 `turn["text"]` 를 `sess.prepare_text` 로 바꾼다. 같은 세션을 도구 결과와 공유한다. 세션은 `try/finally` 또는 `with` 로 닫는다.
2. **탭 에이전트 도구 결과**(`shared/llm/agent.py` 의 `_result_text`, `json.dumps(data, default=str)` 지점): dumps 하기 전에 `sess.prepare_json(data, keys_to_drop=..., pseudonymize_keys=필요 시)` 를 거친다. 돌봄은 좌표(`lat`/`lng`)가 기본 일반화 대상이다. 정산은 지출 제목·구성원 이름이 문자열 값이므로 가려지고, 금액은 `numbers` 로 선택한다. `MAX_TOOL_RESULT_CHARS` 자르기는 **가명화 뒤 길이 기준**으로 한다. 도구 호출 **입력**(모델이 만든 인자)은 **조회용**이면 서버가 실행 전에 `restore_json` 으로 이름·주소 토큰을 되돌려 쓰고, **쓰기 도구**(일정·지출·돌봄 기록 저장 등)는 복원한 뒤에도 `contains_placeholders(args)` 가 True 이면(`(연락처)` 같은 일반화 문구, `[RRN]` 같은 고정 토큰, 안 풀린 `[PERSON_3]`) **그 호출을 거부**하고 모델에 다시 묻거나 사용자에게 알린다. 그대로 저장하면 자리표시가 DB 에 들어간다.
3. **응답 복원 위치**: 모델의 최종 답 문자열을 사용자에게 돌려주기 직전에 `sess.restore_text(answer, generalize_unknown=True, scrub=True)`. 쓰기 제안(제안 토큰·저장할 값)은 복원된 값이 아닌지, 이름 토큰이 깨졌는지 검증한 뒤 쓴다. 시스템 프롬프트에 "`[PERSON_1]` 같은 토큰은 글자 그대로 복사하고 새로 만들지 않는다"를 넣는다.
4. **캐시·로그**: 에이전트 결과는 캐시하지 않는 현재 규칙(`cache_key=None`)을 유지하고, 캐시를 도입한다면 토큰 상태로만 저장한다. 세션·가명화 전 질문·도구 결과·복원 전 응답을 로그에 남기지 않는다(`counts_by_kind`·`blocked_count` 만 안전). 규칙을 바꾸면 `MODULE_VERSION` 을 올린다.
5. **`agent_guard` 규칙 위반 정리**: `shared/llm/agent_guard.py` 가 `features.ai_assistant.ingest.masking.mask_text` 를 import 한다(`shared` 가 기능 코드에 의존). 이 API 로 대체하면 `shared` 안에서 끝나고 위반이 사라진다. 두 규칙은 겹치고(전화·카드·주민번호·계좌·이메일·주소·이름) 목적이 다르다(`mask_text` 는 문서 저장 시 영구 치환, 이 API 는 요청 시 가역 가명화). PR #87 이 `masking.py` 에 라벨 있는 번호(고객·증권·계약·환자 번호 등), 은행명이 낀 계좌, 줄 중간 라벨 주소, 동·호 주거지 주소를 추가해 **`mask_text` 가 이 모듈보다 넓게 잡는 항목이 생겼다**. 따라서 대체 전에 그 규칙을 `detectors.py` 로 옮길지(이 모듈의 탐지 성능 재평가 필요) 정하고, 문서 수집(`ingest`) 쪽은 `mask_text` 를 유지한다.
6. **시험**: 통합 PR 에서는 가짜 LLM 으로 '원문 이름·전화·좌표 정밀값이 외부 요청 본문에 없다'는 시험과 복원 시험을 추가한다. 실제 외부 호출은 시험에서 하지 않는다.

## NER 보강 (`ner.py`, 선택)

규칙 탐지(`detect`) 위에 경량 한국어 NER(KoELECTRA-small v2b, ONNX)을 **더해** 이름·별명·주소를 보강한다. 규칙을 대체하지 않는다. 호출부(가족 채팅 AI·일정 에이전트 등)는 코드를 바꾸지 않는다: `Pseudonymizer.apply`/`apply_values` 가 `detect` 대신 `ner.detect_with_ner` 를 부른다. `detectors.detect` 자체는 규칙만 쓴다(`shared/high_risk_guard.py` 가 이것을 쓰며 NER 지연·503 영향을 받지 않는다). `features/ai_assistant` 의 `mask_text` 는 `detect` 를 쓰지 않아 이 보강을 받지 않는다.

**구조**: `NerEngine`(`tokenizer.json` + onnxruntime CPU 세션, 지연 로드·프로세스 단위 싱글턴·스레드 안전) -> 입력 정규화(전각·대시는 반각, 보이지 않는 문자 제거, **조합형(NFD) 한글 자모는 완성형으로 합침**)하되 결과는 **원문 위치**로 되돌린다 -> 128 토큰 슬라이딩 윈도우(겹침 32, 겹친 토큰은 창 가운데 쪽 예측 사용) -> BIO 디코딩(조사가 붙은 `정채은이` 는 모델 출력 그대로) -> 문자 단위 `NerSpan(label, start, end, score)`(score = span 토큰 softmax 확률 평균). 한글이 없는 글은 모델에 넣지 않는다. 같은 글은 작은 LRU(512개, 소금 친 해시 키, 프로세스 메모리 안, 스팬만 보관)로 다시 쓴다. 로그에는 글자·값·경로를 남기지 않는다.

**라벨 -> 종류** (`ner.LABEL_TO_KIND`): `PERSON`·`PERSON_ALIAS` -> `Kind.PERSON`, `ADDRESS` -> `Kind.ADDRESS`. `LOCATION`(랜드마크·지역명)·`ORGANIZATION` 은 마스킹 기본 제외 정책이라 내보내지 않는다(표에 줄을 더하면 바뀐다). 가족 호칭(엄마·아빠 등)은 모델이 라벨하지 않으며 혹시 라벨해도 `KINSHIP_TERMS` 와 같으면 버린다.

**병합 우선순위** (`merge_ner`): 규칙 탐지가 항상 우선이고 규칙 구간의 종류·점수는 바뀌지 않는다(PHONE·EMAIL·카드·주민번호 등 높은 신뢰 구간 포함).
- 규칙 구간과 겹치지 않는 NER 구간은 그대로 더한다.
- 같은 종류(ADDRESS·PERSON)이고 NER 구간이 겹친 규칙 구간들을 **모두 감싸며 더 넓으면** 규칙 구간들을 NER 범위로 넓힌다(규칙 점수·출처 유지; 이름 사전 대표 이름이 있는 PERSON 은 넓히지 않는다). 예: 규칙이 `부산 해운대구`·`101동 202호` 만 잡아도 NER 가 `부산 해운대구 우동 1234-5 101동 202호` 전체를 잡으면 하나의 주소로 가려진다.
- 그 밖의 부분 겹침·다른 종류와의 겹침은 규칙 구간이 덮지 못한 나머지 조각만 별도 구간으로 더한다(2글자 미만은 버림). NER 구간 가장자리의 구두점·주소 끝 조사(`202호로` 의 `로`)는 잘라 낸다.
- NER 점수는 `[0.5, 0.69]` 로 제한한다: 0.5 미만은 버리고, 상한 0.69 는 정책 `block_min_score`(0.7) 바로 아래라 NER 탐지만으로는 메시지 통째 제외가 일어나지 않고 **값만 가명화**된다(지금 모든 정책에서 PERSON·ADDRESS 는 MASK 라 영향은 없다).
- 검증(Baseline v2 val 180문장 / synth val 1427문장, 실모델): 가려지다 만 정답 구간 4 -> 0 / 52 -> 9, 남은 평문 글자 19 -> 3 / 306 -> 141, 오탐 구간 2 -> 1 / 50 -> 43(병합 확장 이전 -> 이후).

**설정** (`app/shared/config.py`, `apps/api/.env.example`):

| 환경변수 | 기본값 | 뜻 |
|---|---|---|
| `PRIVACY_NER_MODE` | `auto` | `off` 안 씀 / `auto` 쓸 수 있으면 사용, 없거나 실패하면 규칙만(경고 로그 한 번) / `required` 못 쓰면 fail-closed. 알 수 없는 값은 `required` 로 본다 |
| `PRIVACY_NER_MODEL_DIR` | 비움 -> `<DATA_DIR>/models/ner` | 모델 폴더 |
| `PRIVACY_NER_THREADS` | `2` | onnxruntime CPU 스레드 수(추론 하나당) |
| `PRIVACY_NER_MAX_CONCURRENCY` | `2` | 동시에 도는 추론 수. 넘으면 대기(최대 8초), 그래도 못 얻으면 auto 는 규칙만·required 는 503 |
| `PRIVACY_NER_MODEL_FILE` | `model.onnx` | 폴더 안 모델 파일. 양자화는 `model.int8.onnx` |

**권장 모드**: 운영은 `required`(모델을 못 쓰면 외부로 보내지 않는다), 개발·시험은 `auto`(모델이 없어도 동작). 시험(`apps/api/conftest.py`)은 기본 `off` 로 고정해 모델이 있는 PC 와 없는 CI 의 결과를 같게 한다(NER 시험은 각자 모드를 명시한다).

**mode 별 동작**: `auto` 에서 모델 폴더가 없거나, 로드·무결성·추론이 실패하거나, 시간·예산·입력 상한을 넘으면 규칙만으로 처리한다(초과분만 규칙, 나머지는 NER). 모델이 없는 환경(CI·로컬)의 동작은 NER 이 없던 때와 완전히 같고 불러오기 시도는 60초에 한 번이다. `required` 에서는 모델 없음·로드 실패·무결성 불일치·추론 오류·상한 초과·대기 초과 모두 `PrivacyUnavailable`(`ApiError`, 503 `privacy_unavailable`, "개인정보 보호 엔진을 사용할 수 없어 AI 요청을 보내지 않았어요")을 낸다. `new_session`·`prepare_transcript`·`prepare_text`·`prepare_json` 이 시작할 때 확인하고 추론 중에도 낸다(에이전트는 세션을 먼저 만들므로 **외부 모델을 부르기 전에** 막힌다). `shared/llm/agent.py` 는 도구 안에서 난 `PrivacyUnavailable` 을 도구 오류로 숨기지 않고 다시 던져 503 으로 끝낸다(다른 `ApiError` 는 기존처럼 도구 오류).

**비용 상한**(요청 하나 = `prepare_*` 한 번): 추론이 필요한 글(캐시 적중 제외)의 합이 `BATCH_MAX_CHARS`=30000자·`BATCH_MAX_TEXTS`=3000개를 넘으면 초과분은 추론하지 않는다(대화는 **최신 줄부터** 예산을 쓴다: 줄 순서는 오래된 것 -> 최신). 근거: 실측 긴 줄(2000자) 기준 약 110us/글자(2스레드)라 30000자 ≈ 3.4초(짧은 줄은 더 빠르다)이고, 가족 채팅 분석의 글자 한도(12000자)의 2.5배라 정상적인 대화·도구 결과보다 넉넉하다. 글 하나는 8000자·24윈도우까지, 추론 호출 하나는 10초까지다. 동시 추론은 `PRIVACY_NER_MAX_CONCURRENCY` 로 제한한다. 초과 시 auto 는 규칙만(경고 한 번), required 는 503.

**성능**: `prepare_transcript` 는 줄 전체를, `prepare_json` 은 문자열 값 전체를 `ner.batch_scope` 로 **한 번(배치)** 추론한 뒤 쓴다(길이가 비슷한 창끼리 64개씩 묶는다). `prepare_text` 는 글 하나다. 한글 없는 값은 건너뛰고 같은 글은 캐시한다. off·모델 없음에서는 추가 비용이 없다. 실측(개발 PC, 2스레드, fp32): 문장 하나 약 3~5ms, 서로 다른 값 3000개 약 1.7초.

**시작·로드**: 서버가 시작할 때 `app/main.py` 의 `lifespan` 이 `ner.start_preload()` 로 모델을 백그라운드에서 미리 불러온다(off 면 아무 일도 안 함, 실패해도 서버 기동은 막지 않는다). 로드 중 도착한 요청은 auto 면 기다리지 않고 규칙만, required 면 로드가 끝나길 기다린다. 로드 때 `model_card.json` 의 `sha256`(모델 파일·`tokenizer.json`·`labels.json`)과 실제 파일을 한 번 대조해 다르면 사용 불가(`ModelIntegrityError`)로 본다. 카드나 해시가 없으면 경고만 한다.

**캐시 키**: 가족 채팅 AI 결과 캐시 키(`ai_service._key`)에는 가명화된 줄의 해시가 들어가므로 NER 로 가명화 결과가 달라지면 키도 달라진다. 그래서 `MODULE_VERSION` 은 올리지 않았다. 진단용으로 `ner_status()` 가 모드·사용 여부·모델 지문(모델 파일 sha256 앞 12자)을 돌려준다(글자·경로 없음).

**배포**: 모델 폴더(`model.onnx`, 선택 `model.int8.onnx`, `tokenizer.json`, `labels.json`, `model_card.json`)를 서버의 `data/models/ner/`(또는 `PRIVACY_NER_MODEL_DIR`)에 복사한다. **모델 파일은 Git 에 올리지 않는다**(`apps/api/data/` 는 `.gitignore`).

**운영 확인**(상태 확인용 엔드포인트는 두지 않는다: 공개 라우트는 보안 매트릭스 허용 목록에 영향을 준다). 서버 로그(`app.shared.privacy.ner`)로 본다.
- 정상: `INFO NER 엔진 준비 완료 (mode=required, 모델 파일=model.onnx, 지문=4ce93f7913e7, 스레드=2, 동시 추론=2, 로드 0.7초)` — 지문이 배포한 모델 파일의 `sha256` 앞 12자와 같은지 대조한다(`sha256sum data/models/ner/model.onnx | cut -c1-12`, Windows 는 `Get-FileHash`).
- 실패: `NER 엔진을 불러오지 못했다(종류=FileNotFoundError|ModelIntegrityError|…)`(auto: WARNING "규칙 탐지만 사용한다", required: ERROR "가명화 요청은 503 으로 거부된다"). 종류가 `FileNotFoundError` 면 폴더·파일명, `ModelIntegrityError` 면 파일이 `model_card.json` 과 다른 것(복사 중단·손상·다른 버전).
- 기타 경고: `NER 요청 예산(글자·개수)을 넘은 글은 …`, `NER 추론이 시간 상한을 넘어 …`, `NER 추론 대기가 길어져 …`, `model_card.json 에 sha256 이 없어 …`. 경고는 종류마다 프로세스당 한 번만 나온다.
- 프로세스 안에서 직접: `python -c "from app.shared.privacy import ner_status; print(ner_status())"` -> `{'mode': ..., 'active': True, 'fingerprint': '...'}`.

**모델 교체·롤백**: (1) 새 모델 폴더를 옆 이름(`data/models/ner.new/`)에 복사하고 `sha256` 이 `model_card.json` 과 맞는지 확인, (2) 서버를 내린 뒤(또는 트래픽이 적을 때) `ner` -> `ner.prev`, `ner.new` -> `ner` 로 이름을 바꾸고 서버를 다시 시작, (3) 로그의 `준비 완료` 줄과 지문을 확인. 롤백은 `ner.prev` 를 `ner` 로 되돌리고 재시작. 프로세스는 모델을 시작할 때 한 번 읽으므로 실행 중 폴더를 바꿔도 바로 반영되지 않는다(auto 에서 로드 실패 상태는 60초마다 재시도하므로 모델을 나중에 복사한 경우는 최대 1분 안에 쓰기 시작한다). 모델이 바뀌면 지문이 바뀌어 NER 결과 캐시는 자동으로 분리된다.

**한계**: 모델은 틀릴 수 있다(오탐으로 일반 낱말을 가리거나 미탐). `ORGANIZATION`·`LOCATION` 은 내보내지 않는다. 조사가 붙은 이름은 조사까지 한 토큰으로 가려진다(`정채은이` 와 `정채은` 은 서로 다른 토큰). 외국 이름·한글 없는 글은 보지 않는다. 이모지가 이름 앞뒤에 붙으면 모델 결과가 달라질 수 있다. **조합형(NFD) 한글은 NER 쪽만 완성형으로 합쳐 처리한다**: 규칙 탐지(`detectors`)는 NFD 를 놓치는 기존 결함이 그대로이며 이번 범위 밖이다. 학습 데이터는 합성 중심이라 실제 대화 분포와 다를 수 있다. 모델·기반 모델(`monologg/koelectra-small-v3-discriminator`)과 학습 데이터(KLUE-NER CC BY-SA 4.0)의 **라이선스는 배포 전에 확인이 필요**하다(`model_card.json` 의 `license_status` 참고).

## 약속(코드로 지킨 것, 시험이 있다)

- 대응표(원본 <-> 토큰)는 `Pseudonymizer` 인스턴스의 메모리에만 있다. 이 모듈은 `logging` 을 쓰지 않는다(NER 어댑터의 글자 없는 경고 로그 제외). DB·캐시·응답에 넣지 않는다.
- 같은 원본 값은 같은 토큰, 번호는 등장 순서. 전화번호는 숫자만 비교해 표기가 달라도 같은 토큰이다. 이름 사전의 별칭("민수" -> "김민수")은 같은 토큰이고 복원하면 대표 이름이 나온다.
- 복원은 **이 요청이 만든 토큰만** 한다. `[PHONE_9]`, `[phone_1]`, `[PHONE_ 1]`, `[PHONE_01]`, 전각 괄호 `【PHONE_1】` 같은 지어내거나 변형된 토큰은 그대로 두고 `unknown_tokens` 에 적는다. 사용자가 대화에 직접 쓴 토큰 모양(변형 포함)과 대응표에 넣는 값(발화자 이름 등)이 토큰 모양이면 `‹PHONE_1›` 로 바꿔 복원 대상이 되지 않게 한다(이중 복원 방지).
- BLOCK 대상 값(SECRET·RRN·CARD·ACCOUNT)은 대응표에 넣지 않는다(복원 불가). 정책이 MASK 여도 번호 없는 `[SECRET]` 같은 고정 토큰으로만 바꾼다.
- `repr`/`str` 은 값을 보이지 않고(`Span.canonical` 도 repr 에서 제외), `pickle`·`copy` 는 `TypeError`. `clear()` 와 `__exit__` 는 참조를 지운다(파이썬 객체 참조 정리이며 메모리에서 물리적으로 완전히 지워진다고 보장하지 않는다). 단 `Pseudonymizer` 내부 필드(`vars`/슬롯)를 코드가 직접 읽는 것까지 막지는 못한다.
- 토큰 복원·탐지 정규식은 입력 길이에 선형이다(길이 제한 후보 + 코드 검사, 반복 입력 시험 포함).

## 한계

- 한국어 이름·주소 탐지는 불완전하다(사전에 없는 이름, 호칭 없는 이름, 붙여 쓴 주소 미탐 가능). 가족 호칭(엄마·아빠)은 마스킹하지 않는다.
- 한글 숫자·띄어쓰기 오타 **전화번호**, 키워드 없는 비밀번호, 건강·종교 같은 의미 기반 민감정보는 찾지 못한다. 전각 숫자·대시 변형은 찾는다. 한글 숫자·OCR 문자·띄운 숫자·공백 구분으로 쓴 주민번호·카드·계좌는 숫자 보조 뷰로 찾는다(위 절). 단서 없는 긴 숫자와 단서 없는 4~8자리 숫자는 계좌·비밀번호로 단정하지 않는다.
- 차단은 메시지 단위라 요약 품질이 떨어질 수 있다. 이름 토큰화로 AI 가 관계 추론을 잃을 수 있다. 전화·이메일은 응답에서 일반화되어 요약에 구체 값이 나오지 않는다.
- 토큰 모양이 응답에서 깨지면 복원되지 않는다. 호출부는 `unknown_tokens` 가 있으면 해당 값을 쓰지 않거나 일반화해야 한다.
- 차단된 메시지도 **발화자 토큰은 Claude 로 간다**(시각은 비운다). 누가 민감 정보를 보냈는지의 패턴(차단 줄이 몇 번, 누구에게서)은 전달된다. 설계상 허용했다.
- 이름 + 공백 1칸 + 호칭('홍길동 선생님', '김철수 씨', '박영희 님께')은 이제 찾는다. 공백이 있는 '군'·'양'은 일반 문장과 구분되지 않아 제외하고, '사용자 님'·'모두 님'·'정대표 님' 같은 역할·직함 말(끝이 '자', 직함 불용어)은 거른다. 그 밖의 일반어가 성씨로 시작하면 오탐할 수 있다.
- 법적 판단(국외 이전·동의 등)은 이 모듈의 범위가 아니다.

9. **가족 AI 동의 범위와 안내 문구**: 가족 AI 동의 하나가 가명화를 쓰는 가족 채팅 AI 외에 AI 비서·문서 질문·일정/가계/돌봄 도우미도 켠다(이 기능들은 가명화를 쓰지 않거나 질문만 `mask_text` 로 가린다). 앱의 동의 안내와 켜기 확인은 이 사실대로 쓰며(`apps/mobile/src/features/family-chat/privacyCopy.js`), 가명화를 넓히면 그 상수도 같이 고친다.
   동의 범위 밖 외부 전송도 있다: 파일 자동 분류(`ingest/classify.py`)는 `TYPESAFE_API_KEY` 가 설정된 서버에서 가린 문서 글자·날짜·금액 후보를 외부 분류 서비스로 보내며 가족 AI 동의(`ai_guard`)를 거치지 않는다. 앱 문구는 "끄면 Claude 로는 나가지 않아요" 로 한정하고 이 경우를 따로 적는다.
