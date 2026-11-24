# 워크플로 검증 보고서 (NER 연결 서비스)

- 실행: 2026-11-23 14:23:52 ~ 2026-11-23 14:24:49 / 코드 경로 `C:\Users\<user>\Documents\gabee-ner\apps\api\app\__init__.py` / Python 3.12.10
- NER: `s0_writable_copy` 상태 {'mode': 'required', 'active': True, 'fingerprint': '46210943c45f'}
- 방식: 임시 SQLite + FastAPI TestClient + 가짜 Claude(`httpx.post` 가로채기, 요청 본문 캡처). 실제 Claude·.env·공용/개인 DB 미사용, 합성 데이터만.

## 1. 시나리오 x 항목 결과표

| # | 항목 | A | B | C | D | X | ALL | LIFE | 총괄 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 워크플로 동작(복원·정상 응답) | 통과 | 통과 | - | 통과 | 통과 | - | - | **통과** |
| 1 | 개인정보 원문 잔류(이름·전화·이메일·주소) | 부분 | 부분 | 통과 | 부분 | - | - | - | **부분** |
| 2 | 고위험 잔류(주민번호·카드·계좌·비밀번호) | 통과 | 통과 | 통과 | 통과 | - | - | - | **통과** |
| 3 | 이름·주소 가명화 정확도 | 부분 | 통과 | - | - | - | - | - | **부분** |
| 4 | 가족 간 혼합·접근 차단 | 해당없음 | 해당없음 | 해당없음 | 통과 | 통과 | - | - | **통과** |
| 5 | 민감 문서 근거 포함(CHAT_INCLUDE_SENSITIVE) | - | - | 통과 | - | - | - | - | **통과** |
| 6 | OCR 글자 가림 누락 | - | - | 통과 | - | - | - | - | **통과** |
| 7 | 로그·오류 응답 개인정보 | - | - | - | 통과 | - | 통과 | 통과 | **통과** |
| 8 | 요청별 매핑표 외부 전달 | 통과 | 통과 | 통과 | - | 통과 | - | - | **통과** |
| 9 | 승인 미리보기 = 실제 페이로드 | 통과 | 해당없음 | 해당없음 | - | - | - | - | **통과** |
| 10 | 탐지 모델 실패 시 외부 전송 중단 | - | - | - | 부분 | - | - | - | **부분** |

(0번은 시나리오 고유 워크플로 동작 확인이라 10개 항목 밖이다. A 정상 흐름·B·D·X 포함. `ALL` 은 전 시나리오 로그 일괄 검사, `LIFE` 는 서버 시작 로그.)

## 2. 항목별 근거(캡처 본문 기준)

- **[A#1] 부분** — required: 분석·답장·주제 본문 원문 전체 잔류=없음; 이름 일부(조각) 잔류={'analysis': ['박서'], 'replies': ['박서'], 'topics': ['박서']} (값 12개 + 이름 조각 검사) _(비고: off(규칙만) 잔류={'analysis': ['박서연', '최지후'], 'replies': ['박서연', '최지후'], 'topics': ['박서연', '최지후']})_
- **[A#2] 통과** — 고위험(주민번호·카드·계좌·비밀번호) 잔류=없음; 미리보기 차단 메시지=1건
- **[A#3] 부분** — 기대 토큰 일치 8/10 ({'ADDR': {'n': 2, 'replaced': 2, 'right_token': 2}, 'PERSON': {'n': 6, 'replaced': 6, 'right_token': 4}, 'PHONE': {'n': 1, 'replaced': 1, 'right_token': 1}, 'EMAIL': {'n': 1, 'replaced': 1, 'right_token': 1}}); 발화자 이름 노출=없음; 같은 사람 토큰 일치={0: ['[PERSON_2]'], 1: ['[PERSON_1]', '[PERSON_2]']} _(비고: off 일치: {'ADDR': (2, 2), 'PERSON': (3, 6), 'PHONE': (1, 1), 'EMAIL': (1, 1)})_
- **[A#4] 해당없음** — 가족 간 혼합은 시나리오 X 에서 검증
- **[A#8] 통과** — 본문 최상위 키=['max_tokens', 'messages', 'model', 'output_config', 'system'], 토큰=원문 대응 패턴=없음, 미리보기 응답 원문=없음
- **[A#9] 통과** — 미리보기(id·발화자·글자·차단) vs 실제 전송: {'analysis': True, 'replies': True, 'topics': True}. 승인 뒤 새 메시지 도착 시 승인본에 없던 id 가 실제 전송에 포함: [8] _(비고: 미리보기에는 승인 결합 토큰/해시가 없다(재요청 시 다시 계산))_
- **[A#0] 통과** — 사용자 화면 복원: summary='요약: 한가온 서울특별시 송파구 올림픽로 300 연 이모 윤도현 (연락처) (이메일) 서하준 최지후 삼촌 경기도 성남시 분당구 판교역로 235', 후보=[{'title': '한가온 와 저녁', 'date': '2026-11-24', 'time': '18:00', 'place': '서울특별시 송파구 올림픽로 300', 'source_message_ids': []}], 남은 토큰=[] _(비고: 항목 0 = 시나리오 고유(복원))_
- **[B#1] 부분** — 사용자 문장 경로 잔류=없음; 일정 제목(규칙 마스킹) 경로 잔류=['박서연', '최지후', '경기도 성남시 분당구 판교역로 235', '판교역로', '박서', '서연', '최지', '지후']; 추천(suggest) 경로 잔류=['박서연', '최지후', '경기도 성남시 분당구 판교역로 235', '판교역로', '박서', '서연', '최지', '지후'] _(비고: 일정 제목·추천은 NER 미연결(mask_text 규칙))_
- **[B#2] 통과** — 입력 문장 고위험: {'rrn': {'status': 400, 'code': 'sensitive_content', 'calls': 0, 'raw_in_payload': []}, 'card': {'status': 400, 'code': 'sensitive_content', 'calls': 0, 'raw_in_payload': []}, 'account': {'status': 400, 'code': 'sensitive_content', 'calls': 0, 'raw_in_payload': []}, 'password': {'status': 400, 'code': 'sensitive_content', 'calls': 0, 'raw_in_payload': []}}; 일정 제목에 든 고위험의 전송본문 잔류=없음
- **[B#3] 통과** — {'이모 이름': True, '주소': True, '전화': True, '날짜·시간 유지': True}; 전송된 사용자 문장 줄: 'ALPHA7Q 내일 오후 3시에 [PERSON_1]랑 [ADDR_1] 에서 점심, 연락처 [PHONE_1]'
- **[B#4] 해당없음** — 시나리오 X 에서 검증
- **[B#8] 통과** — 대응표 패턴=없음; 본문 키=['max_tokens', 'messages', 'model', 'output_config', 'system']
- **[B#9] 해당없음** — 일정 에이전트에는 미리보기·승인 단계가 없다(등록 확인 카드만 있음 — 전송 본문 미리보기는 제공되지 않음)
- **[B#0] 통과** — 등록 결과 응답='등록했어요.', 저장된 일정=[{'title': '서하준 와 점심', 'place': '서울특별시 송파구 올림픽로 300', 'memo': '010-4321-8765', 'start_time': '15:00:00'}], 남은 토큰=[]; draft 경로 status=200 calls=1 잔류=['경기도 성남시 분당구 판교역로 235', '판교역로', '최지', '지후']; query 경로 status=200 잔류=['박서연', '박서', '서연']
- **[C#1] 통과** — OCR 문서가 문서 QA(채팅 검색 도구)로 갈 때 Claude 본문 원문 잔류=없음; knowledge/ask 잔류=없음; 저장 본문 잔류=없음 _(비고: 문서 경로는 NER 미적용(규칙 mask_text 만). 증명서 OCR 자체는 다른 에이전트 담당)_
- **[C#2] 통과** — OCR 고위험(주민번호·계좌·비밀번호) 전송본문 잔류=없음
- **[C#5] 통과** — CHAT_INCLUDE_SENSITIVE=true: 민감 문서 근거 포함=True(설정상 기대 true), false: 포함=False(기대 false); 포함 시 원문 잔류=['010-9988-7766', 'doyoon.lee@example.com']
- **[C#6] 통과** — 가짜 OCR 글자 10줄(성명·주민번호·전화·주소·이메일·담당자·계좌·수령인·비밀번호) 중 최종 본문 잔류=없음; 보관본문 잔류=없음
- **[C#8] 통과** — 대응표 패턴=없음(문서 경로는 가명화 토큰을 쓰지 않음)
- **[C#9] 해당없음** — 문서 QA 경로에는 미리보기·승인이 없다
- **[C#4] 해당없음** — 시나리오 X 에서 검증
- **[D#10] 통과** — required + 모델 없음/변조/추론오류/시간초과 x NER 연결 입구 ['family_analysis', 'family_replies', 'family_topics', 'family_preview', 'schedule_agent']: 정상 차단(503 privacy_unavailable, Claude 호출 0) 20건, 위반 없음
- **[D#10] 부분** — 메인 채팅 -> 가족 채팅 에이전트 경로(required+NER 실패): {'required/missing': (200, None, 3), 'required/tampered': (200, None, 3), 'required/infer_error': (200, None, 3)}. 도구 단계에서 차단되지만 그 전에 오케스트레이터 호출이 mask_text 규칙만 거친 사용자 질문과 함께 이미 나간다(호출 수 참조)
- **[D#10] 부분** — NER 미연결 입구는 required+모델 없음에서도 Claude 를 호출한다(규칙 마스킹만): {'schedule_suggestions': {'required/missing': (200, 1), 'required/tampered': (200, 1)}, 'assistant_chat': {'required/missing': (200, 1), 'required/tampered': (200, 1)}, 'knowledge_ask': {'required/missing': (200, 1), 'required/tampered': (200, 1)}, 'billing_agent': {'required/missing': (503, 0), 'required/tampered': (503, 0)}, 'care_agent': {'required/missing': (404, 0), 'required/tampered': (404, 0)}} _(비고: 설계상 NER 를 쓰지 않는 경로 — required 가 보호를 보장하지 못하는 범위)_
- **[D#1] 부분** — auto + NER 실패: 연결 입구 동작(200, 호출 1회/미리보기 0회)=모두 정상; 비구성원 이름·조각 잔류=['박서', '박서연', '서연', '지후', '최지', '최지후'](규칙만으로는 사전에 없는 이름을 못 가림); 구성원 이름 잔류=['윤도현'](대상 입구: 일정 에이전트의 일정 목록 제목 경로 — 에이전트가 등록하면 복원된 실명이 제목으로 저장되고 이후 호출에서 규칙 마스킹만 거쳐 나감) _(비고: auto 는 의도된 열화(경고 1회) — 이름 보호 수준이 NER 정상 때보다 낮다)_
- **[D#0] 통과** — off: 연결 입구 정상(이전과 동일 규칙 동작)=True, 비구성원 이름 잔류=['박서', '박서연', '서연', '지후', '최지', '최지후']; required+정상 모델: {'family_analysis': (200, 1), 'family_replies': (200, 1), 'family_topics': (200, 1), 'family_preview': (200, 0), 'schedule_agent': (200, 1)}
- **[D#2] 통과** — 모든 모드·실패 유형에서 고위험(주민번호·카드·계좌·비밀번호) 전송본문 잔류=없음(가족 대화 7번 메시지는 차단 자리표시)
- **[D#7] 통과** — 실패 응답 본문(경로·내부 용어·합성 개인정보) 노출=없음; 예: {"code":"privacy_unavailable","message":"개인정보 보호 엔진을 사용할 수 없어 AI 요청을 보내지 않았어요"}
- **[D#4] 통과** — 권한 실패(가족 없음·잘못된 토큰·토큰 없음) x 5 입구: Claude 호출 0, 상태 {'no_family_user:family_analysis': 400, 'no_family_user:family_preview': 400, 'no_family_user:schedule_agent': 400, 'no_family_user:assistant_chat': 200, 'no_family_user:billing_agent': 400, 'bad_token:family_analysis': 401, 'bad_token:family_preview': 401, 'bad_token:schedule_agent': 401, 'bad_token:assistant_chat': 401, 'bad_token:billing_agent': 401, 'no_token:family_analysis': 401, 'no_token:family_preview': 401, 'no_token:schedule_agent': 401, 'no_token:assistant_chat': 401, 'no_token:billing_agent': 401, 'required_missing:bad_token:family_analysis': 401}; 위반=없음
- **[D#10] 통과** — 모델 파일 한 바이트 변조(sha256 불일치): required 503 차단 / auto 규칙만으로 계속 동작
- **[X#4] 통과** — 동시 120건/16스레드(4.92s) 상태={'200': 120}: 응답 내 타 가족 데이터=없음, 전송본문 혼합=없음, 표식 없는 호출=0, 저장 일정 혼합=없음, 복원 안 된 토큰=없음, 자기 이름 복원 누락=없음; 교차 접근 시도={'A_reads_B_room': (404, False), 'A_posts_B_room': (404, False), 'A_room_list_has_B_room': (200, False), 'A_reads_B_event_sharing': (404, False), 'A_patches_B_event': (404, False), 'A_agent_deletes_B_event': (404, False), 'B_event_still_exists_unchanged': True, 'A_analysis_with_B_message_id_as_after_id': (400, False), 'A_analysis_body_group_id_B': (200, False), 'A_schedule_agent_sees_B_events': (200, False)} _(비고: 토큰 번호가 두 가족에서 같아도([PERSON_1]) 각자 자기 이름으로만 복원됐는지는 '자기 이름 복원 누락/타 가족 데이터' 로 확인)_
- **[X#8] 통과** — 동시 요청 96개 전송본문 전체에서 토큰=원문 대응 패턴 검색
- **[X#0] 통과** — 동시 요청 정상 응답 120/120; 비정상=없음
- **[L#0] 통과** — 지연 측정 완료: analysis 평균 off 29.89ms -> on(warm) 49.54ms / on(cold cache) 123.46ms
- **[LIFE#7] 통과** — lifespan 로그: required/ok: 1줄, health 200, active=True, 금지어 0; auto/ok: 1줄, health 200, active=True, 금지어 0; required/missing: 1줄, health 200, active=False, 금지어 0; auto/missing: 1줄, health 200, active=False, 금지어 0; required/tampered: 1줄, health 200, active=False, 금지어 0; auto/tampered: 1줄, health 200, active=False, 금지어 0; off/ok: 0줄, health 200, active=False, 금지어 0 _(비고: 성공 로그에는 모델 파일 이름(model.onnx)·지문(12자리)·스레드 수만, 실패 로그에는 예외 종류만)_
- **[ALL#7] 통과** — 전 시나리오 로그 1048줄(root DEBUG 핸들러) 중 합성 개인정보(이름·전화·이메일·주소·고위험) 포함=0건  _(비고: 예외 메시지에 합성 이름·전화·경로를 넣은 추론 실패(infer_error)에서도 로그는 예외 종류만 기록)_

## 3. 발견된 문제

### [중간] NER 가 이름 경계를 어긋나게 잡아 이름 일부(성+첫 글자)가 Claude 로 나감
- 재현: 시나리오 A: '박서연 이모랑 저녁 먹자' -> 전송 본문 '박서[PERSON_2]랑'(탐지 구간이 '연 이모'). 이름 전체 검사는 통과하지만 '박서' 가 남는다. 복원 시 '연 이모' 로 돌아가 사용자 화면은 정상. 재현: PRIVACY_NER_MODE=required 로 위 문장을 가족 채팅에 넣고 /family-chat/ai/analysis 호출 후 캡처된 본문 확인(eval_workflows.py A).
- 상세: {"analysis": ["박서"], "replies": ["박서"], "topics": ["박서"]}

### [낮음/설계] 미리보기 승인과 실제 전송이 결합돼 있지 않음
- 재현: POST /family-chat/ai/preview 후 새 메시지를 보내고 POST /family-chat/ai/analysis 호출 -> 승인본에 없는 메시지(id)가 Claude 로 전송됨(새 메시지도 같은 가명화 적용). 미리보기 응답에는 해시·만료 토큰이 없다.

### [중간] 일정 목록 제목(event_line)의 이름·주소는 NER 가 아닌 규칙 마스킹(mask_text)만 적용돼 Claude 로 나감(에이전트가 등록한 일정은 복원된 실명이 제목으로 저장돼 이후 호출마다 실명이 나감)
- 재현: POST /schedule/events title='박서연 이모 병원 010-4321-8765' / '최지후 삼촌 댁 경기도 성남시 분당구 판교역로 235' 등록 후 POST /schedule/agent. 캡처 본문의 일정 목록 줄: ['[일정 목록] 과거·미래 1년 중 지금과 가까운 일정(기간을 알 수 없어 넓게 보냄). id | 날짜 시간 | 제목 [| 공동 [| 공유:id,id]]', '6fd8384cb777 | 2026-11-24(화) 15:00~16:00 | 서하준 와 점심', '6ad45802dab9 | 2026-11-26(목) 10:00~(종료 시간 미정) | 박서연 이모 병원 010-****-****']
- 상세: ["박서연", "최지후", "경기도 성남시 분당구 판교역로 235", "판교역로", "박서", "서연", "최지", "지후"]

### [중간] GET /schedule/suggestions(일정 추천)는 규칙 마스킹만 사용, NER·required 차단 없음
- 재현: 일정 제목에 이름·주소가 든 일정이 있는 상태에서 GET /schedule/suggestions -> 캡처 본문에 이름 잔류
- 상세: ["박서연", "최지후", "경기도 성남시 분당구 판교역로 235", "판교역로", "박서", "서연", "최지", "지후"]

### [중간] POST /ai-assistant/knowledge/items 로 직접 넣은 문서의 개인정보 원문이 채팅 도구 결과로 Claude 에 전송(기존 보안 시험의 알려진 결함과 같은 현상)
- 재현: 문서 입력 API 에 전화·이메일·주민번호가 든 body_md 를 넣고 /ai-assistant/chat 로 검색. (고위험 가드는 /ai-assistant/chat 입력에만 적용)
- 상세: ["010-9988-7766", "doyoon.lee@example.com"]

### [낮음] 메인 AI 채팅 -> 탭 에이전트 도구 호출에서 required 의 503 privacy_unavailable 이 삼켜져 사용자에게 200 으로 보임
- 재현: 모델 폴더를 비우고 PRIVACY_NER_MODE=required 로 POST /ai-assistant/chat(첫 응답이 ask_family_chat 도구 호출). 결과 status=200 Claude 호출=3회. 원인: knowledge/rag_chat.py 의 `except (ValueError, ApiError, TypeError)` 가 PrivacyUnavailable(ApiError 하위)을 도구 오류로 처리. 가족 대화 데이터는 전송되지 않지만(에이전트가 실행되지 않음) 사용자는 보호 엔진 장애를 알 수 없고 오케스트레이터에 질문(규칙 마스킹만)이 이미 나간다.

### [높음] NER(가명화 게이트웨이) 미연결 AI 입구에서 사용자가 입력한 이름·주소·전화 원문이 Claude 본문에 그대로 나감(정상 모델·required 에서도)
- 재현: required + 정상 NER 에서 POST /ai-assistant/chat, /ai-assistant/knowledge/ask, /billing/agent/chat, /care/agent/chat 에 '박서연 이모랑 서울특별시 송파구 올림픽로 300 에서 만나기로 했어. 연락처 010-4321-8765' 류 문장을 보내고 캡처. 입구별 잔류: {'schedule_suggestions': ['박서', '서울특별시 송파구 올림픽로 300'], 'assistant_chat': ['010-4321-8765', '박서', '박서연', '서연', '서울특별시 송파구 올림픽로 300'], 'knowledge_ask': ['박서', '박서연', '서연', '서울특별시 송파구 올림픽로 300']}. 정산·돌봄 에이전트 시스템 프롬프트에는 요청자 실명도 포함(member_names_in_payload). 기존 tests/test_security_matrix.py 의 known_defect 와 같은 현상.

### [중간] PRIVACY_NER_MODE=required 가 일부 AI 입구에서는 fail-closed 가 아님(NER 미연결 경로)
- 재현: 모델 폴더를 비우고 required 로 실행 -> 다음 입구는 503 없이 Claude 호출: ['schedule_suggestions', 'assistant_chat', 'knowledge_ask'] (일정 추천 GET /schedule/suggestions, AI 비서 채팅/문서질문, 정산·돌봄 에이전트). 이름·주소는 규칙 마스킹(mask_text)만 거치고 NER 보호가 없다.


## 4. 종단 지연(가짜 Claude, 같은 시나리오 30회)

| 요청 | 설정 | 평균(ms) | 중앙값 | P95 | 최소 | 최대 | 첫 요청 | 평균 증가(vs off) | P95 증가 |
|---|---|---|---|---|---|---|---|---|---|
| analysis | off(1차) | 30.91 | 28.71 | 35.91 | 25.23 | 84.2 | 28.26 | - |
| analysis | off(2차) | 28.87 | 28.31 | 32.95 | 25.26 | 37.56 | 37.56 | - |
| analysis | required 엔진 로드됨 | 49.54 | 33.16 | 54.4 | 29.3 | 505.16 | 54.4 | +19.65 (1.66배) | +19.97 |
| analysis | required 캐시 비움(매번 추론) | 123.46 | 118.45 | 182.62 | 57.24 | 276.81 | 60.38 | +93.57 (4.13배) | +148.19 |
| preview | off(1차) | 10.48 | 10.71 | 12.15 | 8.81 | 12.7 | 9.69 | - |
| preview | off(2차) | 10.52 | 10.19 | 15.1 | 8.39 | 17.08 | 8.83 | - |
| preview | required 엔진 로드됨 | 10.96 | 11.16 | 13.21 | 8.72 | 14.66 | 8.8 | +0.46 (1.04배) | +-0.41 |
| preview | required 캐시 비움(매번 추론) | 102.43 | 101.51 | 176.1 | 39.26 | 189.25 | 39.26 | +91.93 (9.76배) | +162.48 |
| schedule | off(1차) | 53.11 | 30.65 | 195.86 | 26.6 | 511.08 | 34.3 | - |
| schedule | off(2차) | 35.57 | 31.09 | 61.67 | 27.53 | 142.95 | 142.95 | - |
| schedule | required 엔진 로드됨 | 38.6 | 36.12 | 49.76 | 31.79 | 76.4 | 49.76 | +-5.74 (0.87배) | +-79.01 |
| schedule | required 캐시 비움(매번 추론) | 63.65 | 46.57 | 88.03 | 43.08 | 485.05 | 53.3 | +19.31 (1.44배) | +-40.74 |

첫 요청(가족 분석): 미리 로드 없음 403.5ms (그 다음 같은 요청 29.2ms) / 미리 로드 후 48.2ms (미리 로드 자체 334.6ms). 엔진 로드 시간 355.7ms.

_가짜 Claude(지연 0) 기준: 서버 처리(가명화 포함) 종단 지연. 실제 Claude 응답 시간(수 초)은 포함되지 않는다._


## 5. 서버 시작 시 미리 로드(lifespan)

| 설정 | health | 미리 로드 완료까지(s) | 이후 NER 상태 | 로그 | 금지어(경로·개인정보) |
|---|---|---|---|---|---|
| required/ok | 200 | 0.43 | {'mode': 'required', 'active': True, 'fingerprint': '46210943c45f'} | app.shared.privacy.ner:INFO:NER 엔진 준비 완료 (mode=required, 모델 파일=model.onnx, 지문=46210943c45f, 스레드=2, 동시 추론=2, 로드 0.3초) | 0 |
| auto/ok | 200 | 0.45 | {'mode': 'auto', 'active': True, 'fingerprint': '46210943c45f'} | app.shared.privacy.ner:INFO:NER 엔진 준비 완료 (mode=auto, 모델 파일=model.onnx, 지문=46210943c45f, 스레드=2, 동시 추론=2, 로드 0.3초) | 0 |
| required/missing | 200 | 0.12 | {'mode': 'required', 'active': False, 'fingerprint': ''} | app.shared.privacy.ner:ERROR:NER 엔진을 불러오지 못했다(종류=FileNotFoundError). mode=required 라 가명화 요청은 503 으로 거부된다 | 0 |
| auto/missing | 200 | 0.11 | {'mode': 'auto', 'active': False, 'fingerprint': ''} | app.shared.privacy.ner:WARNING:NER 엔진을 불러오지 못했다(종류=FileNotFoundError). 규칙 탐지만 사용한다 | 0 |
| required/tampered | 200 | 0.24 | {'mode': 'required', 'active': False, 'fingerprint': ''} | app.shared.privacy.ner:ERROR:NER 엔진을 불러오지 못했다(종류=ModelIntegrityError). mode=required 라 가명화 요청은 503 으로 거부된다 | 0 |
| auto/tampered | 200 | 0.28 | {'mode': 'auto', 'active': False, 'fingerprint': ''} | app.shared.privacy.ner:WARNING:NER 엔진을 불러오지 못했다(종류=ModelIntegrityError). 규칙 탐지만 사용한다 | 0 |
| off/ok | 200 | 0.13 | {'mode': 'off', 'active': False, 'fingerprint': ''} | (없음) | 0 |

## 6. 확인하지 못한 것

- 실제 Claude 응답에 따른 동작(토큰을 바꿔 쓰거나 새 토큰을 만드는 경우) — 가짜 응답이 요청 속 토큰을 그대로 되돌려 쓰는 경우만 시험함(위조 토큰 방어는 기존 단위 시험 범위)
- PostgreSQL·운영 환경(Docker/Caddy) — 임시 SQLite + TestClient 만 사용
- 증명서(OCR) 이미지 인식 정확도 — 이번에 제외(다른 에이전트 담당). 가짜 OCR 글자로 가림만 확인함
- NER 동시 추론 대기 초과(QUEUE_WAIT 8s)·시간 상한 10s 실제 초과 — 시간 초과는 NerTimeout 을 직접 일으켜 모의했고 실제 대기 초과는 만들지 않음
- 가족 대화 사진(photos)·위치·돌봄/정산 도구 결과의 이름 가명화 — 탭 에이전트 도구 결과 경로는 입구 호출 수·입력 문장만 확인(기존 보안 시험이 결함으로 표시함)
- 다른 PC·다른 모델 버전에서의 정확도 — koelectra-small-v2b 한 가지만 사용, 합성 이름 몇 개 기준이라 통계적 정확도가 아님
- 앱(모바일) 화면에서의 승인 UX — 서버 API 응답만 확인(미리보기 응답과 실제 전송 본문 비교)

## 7. 재현
```
cd apps/api
PYTHONPATH=. python scripts/eval_workflows.py            # 전체
PYTHONPATH=. python scripts/eval_workflows.py A B C      # 일부 섹션
```
환경: PRIVACY_NER_MODEL_DIR=<모델 폴더>, 결과 폴더는 WV_OUT.
