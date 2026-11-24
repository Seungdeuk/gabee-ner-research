# 개인정보 가명화 파이프라인 점검 및 최소 안정화 (2026-11-17)

**배경**: KoELECTRA NER 모델 개선 연구(R1~R7b, S0)는 2026-11-17 사용자 지시로 **종료**했다(`docs/security/
NER_VALIDATION.md` 1~69절 참고, 연구 산출물로 보관, 운영 모델·R6+B·R7 계열 변경 없음). 이 문서는 그 이후
"가비의 기존 가명화 기능이 실제로 안전하게 동작하는가"를 점검한 결과와, 점검에서 찾은 문제 1건(A)에 대한
**최소 범위 수정**을 기록한다. 신규 모델 학습·대규모 리팩터링·DB 구조 변경·운영 모델 교체·배포는 하지 않았다.

이 문서는 세 가지를 명확히 구분해서 적는다: **① 직접 테스트로 확인한 사실**, **② 코드·ADR 분석으로 확인한
정책**(실행하지 않고 코드를 읽어서 확인한 것), **③ 아직 확인하지 못한 배포 환경**(이 저장소·개발 환경에서는
알 수 없는 것). 추정은 "추정"이라고 쓴다.

## 1. 가명화 파이프라인 구조(② 코드 분석)

핵심 공통 모듈은 `app/shared/privacy/`다.
- `gateway.py`의 `prepare_transcript()` — 소통(family_chat) 자동분석(요약·일정추출·답장추천·주제추천)이
  쓰는, 줄 단위 대화용 API.
- `freetext.py`의 `new_session()`/`PrivacySession.prepare_text()`/`restore_text()` — 자유 글(질문·
  JSON)용 API. 내부에서 같은 탐지기(`detectors.py`)·같은 `Pseudonymizer`를 쓴다.
- 둘 다 `known_names`(가족 이름 사전) + 정규식 탐지기(전화·이메일·주민번호·카드·계좌·주소) + KoELECTRA
  NER을 합쳐 `PERSON` 등을 찾고, `merge_ner`로 최종 확정한 뒤 `[PERSON_1]` 같은 토큰으로 바꾼다.
- `Pseudonymizer`/`PrivacySession`/`Prepared`는 `__reduce__`/`__copy__`/`__deepcopy__`가 전부
  `TypeError`를 던지도록 막혀 있다(`freetext.py:258-268`, `gateway.py:47`) — 직렬화·복사(= 캐시·로그·
  DB로 새어나가는 흔한 경로)가 코드로 차단돼 있다.

## 2. 호출 경로별 가명화 적용 여부(② 코드 분석, 일부 ① 직접 테스트)

| 경로 | 질문/입력 텍스트 | 도구 결과·대화 내용 | 근거 |
|---|---|---|---|
| 소통 자동분석(`family_chat/ai_service.py`) | — | **완전**(`prepare_transcript`, NER+규칙+가족이름사전) | ADR 0011 |
| **소통 질의응답(`family_chat/agent.py:ask()`)** | **수정 전: 부분(라벨 있는 패턴만) → 수정 후: 완전**(아래 3절) | 완전(같은 세션) | 이번 수정 |
| **가계(billing) 에이전트(`billing/agent.py:ask()`)** | **수정 전: 부분 → 수정 후: 완전**(질문만) | **그대로 비보호**(의도된 정책, 변경 안 함) | `agent_guard.py` docstring, 이번 수정 |
| 메인 AI 채팅(ai-assistant) | **없음(의도된 정책, 변경 안 함)** | — | ADR 0014(2026-11-04, 0013 대체) |
| 일정(schedule) 에이전트 | 완전(게이트웨이 사용, 위험도 ≥0.7 자체 차단) | 완전 | 코드 확인(`import app.shared.privacy`) |
| 돌봄(care) 에이전트 | 기능 삭제됨(`app/features/care` 없음) | — | 코드 확인 |
| 문서/사진 업로드(ingest, OCR) | Claude로 OCR+가리기(`INGEST_LLM_OCR/MASK` 기본 켜짐) | — | ADR 0014 + 2026-11-09·11-10 갱신 |

메인 채팅이 탭 에이전트(`ask_family_chat` 등)로 질문을 넘길 때, 고위험 값(주민번호 등)이 남아 있으면 **그
호출만** 거부한다(ADR 0014) — 이것은 가명화가 아니라 차단이고, 메인 채팅 자신의 답변 생성에는 영향이 없다.

## 3. 수정한 문제(A) — 소통/가계 질문 텍스트의 가족 이름 미보호

### 문제(① 직접 테스트로 확인, 수정 전)

`family_chat/agent.py:ask()`와 `billing/agent.py:ask()`는 사용자의 **질문 원문**을 `agent_guard.prepare()`
→ `mask_text()`(`ai_assistant/ingest/masking.py`)로만 거쳤다. `mask_text`의 이름 탐지(`_NAME`,
`masking.py:48`)는 `"이름:"`류 **라벨이 붙은 경우만** 잡는다(`_NAME_LABELS` 필수). 일반 대화체로 쓴 가족
이름("수진한테 전화해서...")은 라벨이 없어 **그대로 모델로 나갔다** — 반면 같은 가족 채팅 에이전트가
도구로 조회해 오는 실제 대화 내용은 `new_session`/가족 이름 사전으로 완전히 보호돼 있어, 같은 기능 안에
보호 수준이 서로 달랐다.

### 수정 내용

- `apps/api/app/features/family_chat/agent.py` — `ask()`에서 `members = ai_repository.member_names(...)`로
  이름 사전을 가져온 뒤, 도구 결과에 이미 쓰던 `with new_session(ASSISTANT, members) as session:` 블록
  **안에서** `question`·`history`를 `session.prepare_text()`로 먼저 가명화하고 그 결과를 `ask_agent`에
  넘긴다. 기존 `agent_guard.prepare()`(정규식 안전망)는 그대로 유지(순서상 먼저 거친다).
- `apps/api/app/features/billing/repository.py` — `member_names()` 함수 추가(`family_chat/ai_repository.
  py`·`weekly_news/repository.py`와 같은 조회·묶음 규칙. billing 은 다른 기능의 내부 파일을 직접 import
  할 수 없어 작은 복제가 필요했다 — 이 중복은 이미 두 곳에 있던 것이라 셋째 복제는 기존 패턴을 따른
  것이고, `app/shared`로 모으는 건 더 큰 범위라 이번엔 하지 않았다(아래 "추후" 참고)).
- `apps/api/app/features/billing/agent.py` — `ask()`를 같은 방식으로 바꿨다. **도구 결과(`build_tools`)는
  `session`을 넘기지 않아 그대로 비보호 상태를 유지한다**(지시사항: 가계 도구 결과의 의도적 비가명화
  정책 변경 금지). 질문에서 만든 토큰만 `session.restore_text()`로 복원한다(도구 결과는 토큰화되지
  않으므로 복원이 거기에 영향을 주지 않는다).

### 질문과 도구 결과에 같은 세션을 쓸지 검토한 결과

**같은 세션을 쓰도록 했다.** 세션이 다르면 같은 이름("민준")이 질문과 도구 결과에서 서로 다른 토큰
번호를 받을 수 있어(모델이 같은 사람인지 혼동할 위험) 토큰 번호가 중복·충돌할 수 있고, 복원도 두 세션의
대응표를 따로 들고 다녀야 해서 오류 위험이 커진다. 한 세션을 쓰면 번호가 자동으로 일관되고, 복원도
`session.restore_text()` 한 번으로 끝난다.

### 수정 전후 테스트(① 직접 테스트, RED → GREEN 확인)

새 테스트 2개를 추가하고, **수정 전 코드로 되돌려 먼저 실패하는 것을 확인한 뒤(RED)**, 수정을 다시 적용해
통과를 확인했다(GREEN). 실제 Claude API는 호출하지 않는다(`llm_agent._post`/`rag_chat._post`를 가짜로
바꿔 요청 본문만 가로챈다).

- `apps/api/app/features/ai_assistant/tests/test_tab_agents.py::test_registered_family_name_in_the_question_is_masked_before_it_leaves`
  — `/billing/agent/chat`에 "수진한테 전화해서 이번 달 지출 알려줘"를 보내고, 실제 모델 호출 본문에
  "수진"이 없는지 확인. **수정 전: 실패("수진"이 그대로 있었음). 수정 후: 통과.**
- `apps/api/app/features/ai_assistant/tests/test_ask_family_chat.py::test_registered_family_name_in_the_question_is_masked_before_sub_agent_call`
  — 메인 에이전트가 `ask_family_chat`에 "지민한테 전화해서 숙제 확인해줘"를 넘겼다고 가정하고, 탭
  에이전트의 실제 모델 호출 본문에 "지민"이 없고 `[PERSON_n]` 토큰이 있는지 확인. **수정 전: 실패.
  수정 후: 통과.**

### 회귀 테스트 결과

```
pytest app/features/ai_assistant/tests/test_tab_agents.py app/features/ai_assistant/tests/test_ask_family_chat.py app/features/billing/tests/ -q
  -> 1 failed, 183 passed (506.6s)
```

실패 1건(`test_billing_allowance.py::test_ledger_starts_empty_and_tracks_credit_use_and_running_balance`)은
**이번 수정과 무관하다** — 이 시험은 용돈 장부 순서(최신순) 시험이고 `agent.py`/`repository.py`의
`member_names` 를 전혀 참조하지 않는다(확인됨, grep 결과 0건). 단독 실행하면 통과한다(`1 passed in 1.96s`)
— 전체 묶음으로 돌릴 때만 간헐적으로 실패하는, **이번 수정 전부터 있던 것으로 보이는 타이밍 의존(flaky)
시험**이다(새로 만들지 않았고, 고치지도 않았다 — 범위 밖).

더 넓게 `app/shared/privacy/tests/`·`app/features/family_chat/tests/`도 돌렸다:

```
pytest app/shared/privacy/tests/ app/features/family_chat/tests/ -q
  -> 1 failed, 996 passed, 3 skipped (1282s)
```

실패 1건(`test_todos.py::test_done_list_is_most_recently_finished_first_regardless_of_due_date`)도
**이번 수정과 무관하다** — `todo_repository`만 참조하고 `agent`/`privacy`를 전혀 쓰지 않으며(확인됨),
단독 실행하면 통과한다(`1 passed in 1.74s`). 완료 시각 동일성(타이밍)에 의존하는 또 다른 기존
flaky 시험으로 보인다 — 둘 다 새로 만들지도, 고치지도 않았다(범위 밖이라 그대로 뒀다).

`ruff check`는 수정·추가한 파일 전부 통과했다.

## 4. 1글자 등록 이름 보호 — 정책 제안만(코드 미수정)

### 확인된 사실(① 직접 테스트)

1글자로 등록된 가족 이름(예: "훈")은 **사전 경로와 NER 경로 양쪽에서 독립적으로** 탈락한다.
- `detectors.py:573`(`_detect_person_known`): `len(name) < 2` 면 그 이름은 매칭 시도 자체를 안 한다 —
  **등록된 이름이라도** 1글자면 사전이 전혀 보호하지 않는다.
- `ner.py`의 `merge_ner`: NER 모델이 잡은 1글자 `PERSON` 후보도 별도의 `len(value) < 2` 필터로 떨어진다
  (이쪽은 NER 전체에 적용되는 기존 서비스 정책 — `NER_VALIDATION.md` 48절 등에서 이미 문서화됨).

### 제안(실행하지 않음, 승인 필요)

사용자 지시대로 **일반 1글자 단어 탐지를 전부 여는 것은 제안하지 않는다**(조사·흔한 1글자 단어가 전부
`PERSON`으로 오탐될 위험이 큼, 서비스가 이 필터를 둔 원래 이유). 대신 **등록된 이름에 한해서만** 여는
좁은 변경을 제안한다:

- **제안 A(가장 좁음)**: `_detect_person_known`에서만 1글자 이름의 길이 제한을 풀되, 1글자 이름은 반드시
  명시적 조사(이/가/한테/아/야 등)나 호격으로 끝나야 매칭되게 한다(현재도 `_name_end_ok`가 조사를 보지만,
  2글자 이상은 조사가 없어도 단어 경계면 매칭되는 데 비해 1글자는 **조사·호격이 필수**가 되게 조건을
  강화). 바로 뒤가 일반 단어로 이어지는 문맥(예: "별이 반짝인다")은 걸리지 않게 하려는 목적.
- **제안 B**: 가족 구성원 등록 시 1글자 이름을 쓰는 사람에게만 "이 이름은 흔한 낱말과 겹칠 수 있어
  일부 문장에서 과잉 보호(또는 누락)가 생길 수 있다"는 안내를 보여주고, 그 사람에 한해 제안 A의 완화를
  적용하는 선택적(opt-in) 방식. **`GroupMember`를 확인했고, 이 용도의 필드는 현재 없다** — 이 방식을
  쓰려면 작은 Alembic 마이그레이션(boolean 컬럼 1개)이 필요하다(DB 구조 변경이라 이번 단계에서는 하지
  않았고, 승인 시 별도로 진행해야 한다).
- 어느 쪽이든 **NER 쪽(`merge_ner`)의 일반 1글자 필터는 그대로 둔다** — 사전 매칭(이미 특정 등록 이름만
  대상인 좁은 경로)만 손보는 것이 가장 작은 변경이다.

## 5. 운영 환경설정(`PRIVACY_NER_MODE`) 확인 방법 — 배포 이미지 검증 완료, 실제 Cloud Run 런타임 상태 미확인

> **정정(2026-11-18, 사용자 보고)**: 아래는 이 문서를 쓸 때(운영에 접근할 수 없던 시점)의 기록이다. 이후 Google Cloud Run 배포
> 이미지(revision `gabee-00018-bd2`)를 Cloud Shell 의 임시 컨테이너에서 검증한 결과 **NER mode `auto`, active `True`,
> fingerprint `46210943c45f`(서비스 모델)** 로 확인됐다. 따라서 "운영 모델 미확인"은 **"배포 이미지의 모델 검증 완료, 실제 Cloud Run
> 런타임(트래픽을 받는 인스턴스)의 상태·환경변수는 미확인"** 으로 읽는다. 이 검증은 Claude Code 가 직접 한 것이 아니라 사용자 보고다.
> mode `auto` 는 모델을 못 불러오면 조용히 규칙만 쓰는 모드이고, 못 쓸 때 요청을 막는 fail-closed 는 `required` 다.

이 문서를 쓸 때 저장소·개발 환경에서는 실제 배포 서버의 환경변수 값을 **확인할 수 없었다**. 코드 기본값(`config.py:213`)은
`"auto"`지만, 실제 배포에서 다른 값으로 덮어썼을 수 있어 **코드 기본값만으로 실제 값을 추측하지 않았다.**

안전하게 확인하려면(운영 접근 권한이 있는 담당자가 직접):
1. 배포 서버에서 `echo $PRIVACY_NER_MODE`(또는 컨테이너라면 `docker exec <컨테이너> env | grep
   PRIVACY_NER_MODE`)로 직접 확인. 이 값은 `off`/`auto`/`required` 중 하나일 뿐 비밀번호·키가 아니므로
   값 자체를 팀에 공유해도 안전하지만, 혹시 같은 명령으로 다른 비밀 환경변수(API 키 등)가 같이 보이면
   그 출력은 공유하거나 기록하지 않는다.
2. 배포 설정 파일(`docker-compose.yml`, `deploy/` 등)을 담당자가 직접 열람 — Claude Code가 대신 읽거나
   값을 출력하지 않는다(이 문서에도 값을 적지 않는다).
3. (제안, 미구현) `ner.ner_status()`가 이미 모드·지문 등을 돌려주므로, 읽기 전용 관리자 화면(`service_
   admin`)에 "현재 가명화 모드" 한 줄만 보여주는 엔드포인트를 추가하면 다음부터는 코드로 쉽게 확인
   가능하다 — 이번 단계에서는 코드를 바꾸지 않았으므로 제안만 남긴다.

`required`가 아니면(`auto`에서 모델이 없거나 실패하면) **fail-closed가 아니라 규칙+사전만으로 조용히
넘어간다**(`config.py:212` 주석, 경고 로그 1회만) — 이 환경에서 직접 확인한 사실이다(코드 동작이지
운영 서버의 실제 모드값은 아니다).

## 6. R6 모델 — 실험 체크포인트 확인 (② 코드/문서 분석)

**이전 점검(2026-11-17)에서 조건 B("NER 켜짐")로 쓴 모델은 운영 모델이 아니라 R6 연구 체크포인트다.**
`C:\Users\<user>\venvs\ner-runs\export\r6-candidate-s42`에 있고, 이 경로는 저장소 밖(개발자 개인 학습
환경)이다. `NER_VALIDATION.md` 59~67절 전체가 "R6+B는 연구 후보일 뿐 운영 적용은 별도 승인 사항"이라고
반복해서 명시한다 — **운영에 배포된 적이 없다.**

실제 운영 서버가 어떤 NER 모델 파일을(또는 아예 모델 없이 규칙만) 쓰고 있는지는 **이 저장소·개발 환경에서
확인할 수 없다**(③). `privacy_ner_model_dir` 기본값은 `<DATA_DIR>/models/ner`이고, 그 경로의 실제 파일은
운영 서버에만 있다. 2026-11-14 전후 문서(`NER_R1_OPS_CHECKLIST.md`, `NER_VALIDATION.md` 36절)에 R1 배포
절차가 적혀 있어 R1이 배포됐을 가능성이 있지만, 이번 점검에서 그 **배포 완료 여부를 직접 재확인하지는
않았다**(이번 조사 범위 밖, 필요하면 담당자 확인 요청).

## 7. 가족방 문서 Claude OCR·가명화 — 사용자 공지와의 일치 여부(③ 확인 불가, 기능 변경 없음)

**코드 동작(현재, ADR 0014 + 2026-11-09·11-10 갱신 기준)**: 가족 전체방에 올린 문서·사진은 `INGEST_LLM_
OCR`·`INGEST_LLM_MASK` 둘 다 **기본 켜짐**으로 Claude API가 글자를 읽고 가린다. 2026-10-22 결정("처방전은
어떤 모델에도 보내지 않는다")은 **가족 전체방 문서·사진에 한해 명시적으로 대체됨**(2026-11-10). 1:1 방
파일은 그대로 외부로 안 나간다.

**실제 사용자 공지(개인정보처리방침·Play 데이터 안전 양식)와 일치하는지는 이 저장소에서 확인할 수
없다** — 저장소 안에는 그 공지의 실제 문구 파일이 없다(스토어 콘솔·웹사이트 등 외부에 있음). 다만
**저장소 문서 자체가 이미 이 간극을 알고 추적 중이다**: ADR 0014·0015와 `docs/requirements/family-
library.md`가 전부 "개인정보처리방침·스토어 고지 반영은 로드맵 **T31**(사용자 작업)에 남아 있다"고
동일하게 명시한다. 즉 **코드는 2026-11-10부터 가족방 문서(처방전 포함)를 Claude로 보내지만, 공지 문구
갱신은 아직 안 됐다고 문서 자체가 말하고 있다** — 새로 발견한 문제가 아니라 이미 추적되는 할 일이다.
이 점검에서 관련 기능은 변경하지 않았다.

## 8. 남아 있는 개인정보 노출 위험 (종합)

| 위험 | 성격 | 상태 |
|---|---|---|
| 메인 AI 채팅이 입력을 가리지 않음 | **설계된 정책**(ADR 0014) | 변경 안 함(지시사항) |
| 가계 도구 결과(이름·금액) 비보호 | **설계된 정책** | 변경 안 함(지시사항) |
| 소통/가계 질문 텍스트의 라벨 없는 이름 미보호 | 구현 누락 | **이번에 수정함** |
| 등록된 1글자 가족 이름 미보호(사전+NER 둘 다) | 알려진 정책 한계 | 정책 제안만(4절), 미수정 |
| 무공백으로 붙은 두 등록 이름(사전 단독으로는 미탐지) | 알려진 한계, NER이 보완 | 변경 없음(낮은 위험으로 평가) |
| `auto` 모드에서 NER 실패 시 조용히 규칙만 씀(fail-open to rules) | 설계된 완화(진짜 fail-open 아님, 규칙+사전은 유지) | 실제 배포 모드값 확인 필요(③) |
| 가족방 문서 공지 문구 미갱신(T31) | 이미 추적 중인 할 일(사용자 작업) | 코드 변경 대상 아님 |

## 9. 변경 파일

```
수정:
  apps/api/app/features/family_chat/agent.py      ask() 가 질문·앞대화를 세션으로 가명화
  apps/api/app/features/billing/agent.py          ask() 가 질문·앞대화를 세션으로 가명화(도구 결과는 그대로)
  apps/api/app/features/billing/repository.py     member_names() 추가
  apps/api/app/features/ai_assistant/tests/test_tab_agents.py        회귀 시험 추가
  apps/api/app/features/ai_assistant/tests/test_ask_family_chat.py   회귀 시험 추가
신규(읽기 전용 점검 스크립트, 운영 코드 아님):
  apps/api/scripts/audit_pseudonymization_pipeline.py  가상 데이터만 사용, 외부 API 호출 없음
```

운영 모델·`ner.py`·`merge_ner`·DB 구조·배포 설정은 전혀 바꾸지 않았다. PR 생성·병합·배포는 하지 않았다.

## 10. 후속 변경(2026-11-18) — 이 문서는 보안 수정 PR 에 넣지 않고 연구 기록으로만 보존한다

위 3절의 질문 가명화 때문에 **질문 속 이름이 `[PERSON_n]` 토큰이 되면 모델이 그 토큰을 도구 인자(정산 "나눌 사람", 소통 검색어)에 그대로
써서 도구가 구성원·메시지를 찾지 못하는 기능 회귀**가 확인돼(가짜 모델 재현: 정산은 `가족 구성원에 '[PERSON_1]' 이(가) 없어요`, 소통은 빈
검색 결과), 보안 수정 전용 브랜치(`fix/seungdeuk-agent-question-pseudonymize`)에서 다음을 고쳤다.

- 두 에이전트의 도구를 질문을 가린 **같은 세션**에 묶어(`privacy_tools`), **도구 실행 직전에만** 서버 안에서 인자의 토큰을 되돌린다
  (이 세션이 만든 PERSON·ADDRESS 토큰만, 모르는 토큰·사용자가 쓴 토큰 모양은 그대로, 호출 기록에는 토큰이 남음).
- 정산: 도구 결과 전체를 가명화하지 않는 기존 정책(PR #440)은 유지하되, **질문에서 이미 가린 이름만** 결과·오류 글·질문한 사람 이름에서
  같은 토큰으로 맞춘다(새 토큰·다른 개인정보·아직 가리지 않은 이름은 그대로).
- 따라서 이 문서 3절의 "정산 도구 결과는 session 을 넘기지 않아 그대로 비보호"는 "질문에서 이미 가린 이름만 같은 토큰"으로 바뀌었다.
