# 소통 에이전트 전용 NER 모델 분리 설계 (방안 B, 설계만 — 구현·배포 미승인)

작성 2026-11-20. 공통 모듈(`app/shared/privacy`, `app/shared/config.py`)을 건드리는 설계이므로 **구현은 별도 승인 후** 진행한다. 이 문서의 모든 사실은 이번 세션에서 코드를 읽어 확인한 것이며, 모델 파일 측정은 R9 후보 ONNX 로만 했다(서비스 v2b 의 `model.onnx` 는 이 PC 에 없고 `model_card.json`·`tokenizer.json` 만 있다).

## 1. 현재 구조 (확인한 사실)

**엔진**: `app/shared/privacy/ner.py` 의 `NerEngine`(KoELECTRA-small ONNX, CPU). 프로세스 전역 싱글턴 `_state`(설정 `_Config` 하나와 엔진 하나)와 전역 세마포어(`PRIVACY_NER_MAX_CONCURRENCY`, 기본 2), 전역 캐시(키 = 엔진 지문 + 문장 해시, 모델이 달라지면 키가 달라져 **캐시는 모델 간 안전**). 모델 위치는 `settings.privacy_ner_model_dir`(비면 `<DATA_DIR>/models/ner`), 파일 `privacy_ner_model_file`(기본 `model.onnx`), 모드 `privacy_ner_mode`(기본 `auto`: 없거나 실패하면 규칙만, `required`: 못 쓰면 503). 로드 때 `model_card.json` 의 sha256 과 파일을 대조한다. `main.py` 가 기동 때 `start_preload()` 로 미리 올린다.

**가명화 호출 경로(소통 에이전트)**
1. 소통 AI 기능(분석·답장·주제): `family_chat/ai_service.py:268` → `privacy.prepare_transcript(rows, names, policy)`(gateway) → `ner.ensure_ready()` → `ner.batch_scope(줄 전체)`(배치 추론) → `Pseudonymizer.apply` → `detect_with_ner(text, known)`(`pseudonymizer.py:206`).
2. @가비 에이전트: `family_chat/agent.py:214` `new_session(ASSISTANT, members)` → `session.prepare_text`(질문·이전 대화, `agent.py:220-222`)와 `session.prepare_json`(도구 결과, `agent.py:110,115,136`) → `freetext.PrivacySession` → `Pseudonymizer.apply_values` → `detect_with_ner`(`pseudonymizer.py:246`). `prepare_json` 은 `ner.batch_scope`(`freetext.py:385`)를 쓴다.
3. 복원만 하는 곳(`family_chat/assistant.py`)은 NER 을 쓰지 않는다.

**다른 기능과의 공유**: 같은 `ner._state` 를 다음이 함께 쓴다 — 정산 에이전트(`billing/agent.py`, `new_session(ASSISTANT)`), 일정 에이전트(`schedule/agent/chat.py` 의 `prepare_transcript`), 주간 뉴스(`weekly_news/generate.py` 의 `new_session`), 기타 `freetext.prepare_*`. 모델 폴더 하나라서 **파일을 바꾸면 이 기능이 전부 같이 바뀐다.** 정책 객체(`ASSISTANT` 등)는 여러 기능이 같이 써서 정책으로는 소통 에이전트를 구분할 수 없다.

**설정의 한계**: `_config()` 는 호출마다 전역 설정을 읽고 `_fresh_state` 가 `cfg` 가 다르면 엔진을 버린다. 두 모델을 번갈아 쓰면 그대로는 매번 다시 로드하는 **스래싱**이 된다 → 프로필별 상태가 필요하다.

## 2. 설계

**원칙**: 기본값은 지금과 100% 같다(서비스 v2b 가 모든 기능의 기본). 새 모델은 설정을 명시해야만, 그리고 호출부가 프로필을 명시한 곳에서만 쓰인다.

1. **설정(공통 모듈, `config.py` + `.env.example`)**
   - `privacy_ner_chat_model_dir: str = ""` (비어 있으면 chat 프로필도 기본 모델 폴더를 쓴다 = 현재와 동일), `privacy_ner_chat_model_file: str = ""`(비면 `privacy_ner_model_file`).
   - 모드·스레드·동시성은 공유(새 설정 없음).
2. **`ner.py`**: `_state` 를 `dict[프로필, _State]` 로, `_config(profile)`·`_get_engine(cfg)`·`ensure_ready(profile)`·`start_preload()`·`ner_status()` 가 프로필을 받는다(기본 `"default"`). `batch_scope(texts, *, profile="default", ...)` 는 프로필의 엔진을 쓰고, 이미 `_Scope`(ContextVar) 안에 프로필을 담아 `spans_for` 가 같은 엔진 결과를 쓴다. `detect_with_ner(text, known, profile="default")`. chat 폴더가 비어 있거나 기본과 같으면 **같은 엔진 객체**를 재사용(두 번 로드하지 않음). 세마포어·캐시는 공유.
3. **명시적 프로필 전달(ContextVar 전역 전환은 쓰지 않음)**: 스레드풀에서 컨텍스트가 새지 않도록 `PrivacySession(..., ner_profile=)`, `new_session(..., ner_profile=)`, `prepare_transcript(..., ner_profile=)` 인자로 내려보낸다(`pseudonymizer.apply*` 가 프로필을 `detect_with_ner` 에 전달).
4. **호출부 변경(소통 에이전트만)**: `family_chat/ai_service.py:268`, `family_chat/agent.py:214` 두 곳만 `ner_profile="chat"` 을 명시한다. 정산·일정·뉴스는 변경 없음(기본).
5. **장애 처리**: chat 모델 폴더가 지정됐는데 로드·무결성 실패 → `auto` 는 **기본 모델(v2b)로 자동 대체**하고 프로세스당 한 번 경고(가명화가 약해지지 않는 쪽), `required` 는 기존처럼 503.
6. **관측**: `ner_status()` 가 프로필별 지문(fingerprint 12자리)·모델 파일 이름·모드를 낸다. 배포 전후 지문을 비교해 교체를 확인한다(원문·경로는 담지 않는다).

**변경 파일(예상)**: `shared/privacy/ner.py`, `freetext.py`, `gateway.py`, `pseudonymizer.py`, `shared/config.py`, `.env.example`, `family_chat/ai_service.py`, `family_chat/agent.py`, 테스트(`shared/privacy/tests`, `family_chat/tests`). 공통 모듈 4개(+설정 2개)는 다른 담당자 영향이 있으므로 영향 범위 보고 후 승인을 받는다.

## 3. 두 ONNX 모델 로드 비용 (측정: R9 후보 ONNX 를 두 번 로드, CPU 스레드 2, 더미 입력, 평가 데이터 미사용)

| 항목 | 값 |
|---|---|
| 모델 파일 | 56.5 MB(R9 후보, FP32). 서비스 v2b 파일은 이 PC 에 없어 크기 미확인 |
| 첫 번째 로드 | 0.64초, RSS +75.6 MB |
| 두 번째 로드(같은 모델) | 0.43초, RSS +68.4 MB |
| 두 개 모두 올린 프로세스 RSS | 193.8 MB(빈 프로세스 48.3 MB) — 즉 엔진 하나당 약 70 MB |
| 짧은 문장 1건 추론 | 약 4~5 ms |

해석: 엔진을 하나 더 올리면 메모리 약 70 MB, 기동 시 로드 약 0.5초가 늘고, 추론은 세마포어(동시 2)를 공유하므로 CPU 경쟁은 늘지 않는다. 운영 서버의 실제 모델 크기(v2b 가 양자화인지)와 메모리 한도는 확인이 필요하다. 이 측정은 개발 PC 이며 운영 CPU 의 지연은 다를 수 있다.

## 4. 다른 기능 영향과 회귀 테스트 범위

- 기본 동작 불변: chat 폴더 설정이 비면 모든 기능이 현재와 같은 엔진·같은 결과를 낸다(회귀 시험으로 고정).
- 새 시험: (a) 프로필 미지정 호출은 기본 엔진, (b) chat 설정이 비면 같은 엔진 객체, (c) chat 설정이 있으면 소통 에이전트 두 경로만 chat 엔진, 정산·일정·뉴스는 기본 엔진(가짜 엔진으로 호출 기록 확인), (d) chat 폴더 실패 시 auto 는 기본 모델 대체·required 는 503, (e) 캐시 키가 엔진별로 분리, (f) 프로필별 `ner_status` 지문, (g) 스레드 동시 호출에서 프로필 혼선 없음.
- 기존 시험 전부: `shared/privacy`, `family_chat`, `schedule`, `billing`, `weekly_news`, `tests/` 보안 매트릭스, 가명화 회귀(`SECURITY_TEST_REPORT` 의 시험 포함). 모델 파일은 저장소에 없으므로 시험은 가짜 엔진으로 하고, 실제 모델은 서버 폴더 스모크로 확인한다.

## 5. 교체·롤백 절차(구현 승인 후, 배포 승인 후에만)

1. **백업**: 서버의 현재 모델 폴더(`models/ner`: `model.onnx`·`tokenizer.json`·`labels.json`·`model_card.json`)를 날짜 폴더로 복사하고 sha256 을 기록한다(서비스 v2b 지문 `46210943c45f…`).
2. **신규 모델 폴더**: `models/ner-chat-r10/` 에 새 `model.onnx`·`tokenizer.json`·`labels.json`·`model_card.json`(sha256·버전 고정). tokenizer·라벨 매핑이 v2b 와 같은지 확인(`labels.json` 해시 비교, 다르면 호환 시험).
3. **코드 배포(설정 비움)**: 먼저 프로필 코드만 배포하고 `PRIVACY_NER_CHAT_MODEL_DIR` 는 비워 둔다. 이 단계에서 모든 기능이 이전과 같음을 확인한다(지문 동일).
4. **활성화**: `PRIVACY_NER_CHAT_MODEL_DIR=<경로>` 설정 후 재시작. `ner_status` 로 chat 지문이 신규 모델인지, 기본이 v2b 인지 확인. 소통 에이전트 스모크(합성 대화, 실제 개인정보 금지)로 가명화 결과를 점검.
5. **롤백**: 설정을 비우고 재시작하면 즉시 v2b 로 돌아간다(파일 교체 없음). 필요하면 신규 폴더를 지운다(auto 는 기본 모델로 대체).
6. 모든 단계에서 소통 이외 기능의 지문이 v2b 인지 확인한다.

## 6. 영향 범위 보고(다른 담당자)

공통 모듈 변경이 필요하다: `shared/privacy`(ner·freetext·gateway·pseudonymizer), `shared/config.py`, `.env.example`. 정산·일정·뉴스 에이전트의 코드는 바꾸지 않지만 이 모듈을 공유하므로 회귀 시험 범위에 들어간다. **승인 전에는 구현하지 않는다.**

## 7. 확인하지 못한 것

- 서비스 v2b `model.onnx` 의 크기·양자화 여부와 운영 서버의 메모리 한도(이 PC 에 파일 없음).
- 운영 CPU 의 추론 지연. 측정값은 개발 PC 의 더미 입력 기준이다.
- "소통 에이전트"의 범위: 이 설계는 `family_chat` 기능(분석·답장·주제·@가비)으로 정의했다. 일정 에이전트가 가족 채팅 줄을 가명화하는 경로는 포함하지 않았다(확인 필요).
