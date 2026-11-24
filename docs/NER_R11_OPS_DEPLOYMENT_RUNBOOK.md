# R11 운영 교체 런북 (준비본 — 실행 금지: S2b·S4·위험 수용 서명·사용자 최종 승인 전)

2026-11-23. 이 런북은 **실행하지 않았다.** 이미지 빌드·배포·교체 없음. 표시: 자동 검증 기반 · 사람 검수 미실시.

## 1. 변경 파일과 핵심 diff
| 파일 | 변경 |
|---|---|
| `apps/api/Dockerfile` | 패치 `r11_validation/ops_replacement/dockerfile_r11_ops.patch`(**미적용**): NER 모델 폴더 이름 `ner-koelectra-small-v2b` → `ner-koelectra-small-r11b-s21`, 이미지 기본 `PRIVACY_NER_MODE=required` 추가 |
| 모델 파일 | 저장소 밖. `apps/api/data/models/ner-koelectra-small-r11b-s21/` 에 `model.onnx`, `tokenizer.json`, `labels.json`, `model_card.json` 4개(R11 export 폴더 `…/r11/export/r11-B-s21-ep5` 에서 복사) |
| 코드 | 변경 없음(`ner.py` 는 폴더·파일명·모드를 설정에서 읽는다). `model.int8.onnx` 불필요 |
```
-# 압축을 푼 폴더 3개(doc-classifier, e5-small-ft, ner-koelectra-small-v2b)를 두고 빌드한다.
+# 압축을 푼 폴더 3개(doc-classifier, e5-small-ft, ner-koelectra-small-r11b-s21)를 두고 빌드한다.
-    PRIVACY_NER_MODEL_DIR=/srv/data/models/ner-koelectra-small-v2b
+    PRIVACY_NER_MODEL_DIR=/srv/data/models/ner-koelectra-small-r11b-s21 \
+    PRIVACY_NER_MODE=required
```
새 폴더 이름을 쓰는 이유: 폴더 이름만으로 S0/R11 을 구분하고, 옛 폴더를 지우지 않아도 롤백 이미지가 그대로 동작한다. 별도로 `data/models/ner-koelectra-small-v2b`(S0)는 보관 폴더에 남긴다.

## 2. 빌드 절차(비운영 → 운영 순서)
1. 담당자가 현재 운영 이미지 태그와 운영 S0 해시를 기록한다(S4-1): `docker exec <컨테이너> sha256sum /srv/data/models/ner-koelectra-small-v2b/model.onnx` → `46210943c45f…` 이어야 함.
2. 빌드 PC 에서 4개 파일을 새 폴더에 복사하고 해시 확인:
   `sha256sum model.onnx` = `3831601b7ffb6093a8d1b8c832e46ad140712c6e402f17b3054c3ec0f9ae4c94`
3. Dockerfile 패치 적용(`git apply dockerfile_r11_ops.patch`) 후 **새 태그**로 빌드(예: `gabee-api:r11b-s21`). 이전 이미지 태그는 지우지 않는다.
4. 비운영 환경에서 새 이미지를 띄우고 4절 검증을 통과시킨다.
5. 사용자 최종 승인 후에만 운영에 새 태그를 배포한다.

## 3. 환경 변수
`PRIVACY_NER_MODE=required`(필수), `PRIVACY_NER_MODEL_DIR=/srv/data/models/ner-koelectra-small-r11b-s21`, 스레드·동시성은 S2b 측정 때 정한 값(`PRIVACY_NER_THREADS`, `PRIVACY_NER_MAX_CONCURRENCY`; 기본 2/2). 운영 오버라이드로 `auto` 가 들어와 있지 않은지 확인한다.

## 4. 배포 직후 안전성 확인(전부 통과해야 유지)
1. 이미지 안 해시: `docker exec <컨테이너> sha256sum /srv/data/models/ner-koelectra-small-r11b-s21/model.onnx` → `3831601b…`
2. `ner_status()` / health: mode `required`, active true, 지문 `3831601b7ffb`
3. 합성 요청(가짜 이름·전화번호만)으로 가족채팅 분석·일정 에이전트 → 가명화 토큰 생성, 응답 복원 정상
4. 비운영 사본에서만: 모델 폴더를 비운 채 시작 → 연결 입구 503 `privacy_unavailable`, 외부 LLM 호출 0(운영에서 변조 시험 금지)
5. 5분간 p50 지연·RSS·503 비율 관찰(S0 대비 +30% 이내)
실제 사용자 개인정보·대화 원문은 테스트에 쓰거나 로그에 남기지 않는다.

## 5. 즉시 중단·롤백 조건
`docs/security/NER_R11_OPS_RISK_ACCEPTANCE.md` 4절과 같다: 로딩 실패·해시/지문 불일치, 503 증가, 지연·RSS 초과, 노출 사고 보고, 검출 비율 급변.

## 6. 롤백 절차
1. 이전 이미지 태그(S0 포함 이미지)로 재배포. 코드·DB 마이그레이션 변경이 없으므로 데이터 영향 없음.
2. 재시작 후 지문 `46210943c45f` 확인(모델은 시작 시 1회 로드, hot reload 없음이라 재시작 필수).
3. 롤백 소요 시간을 기록하고 사고 원인을 별도 보고. 롤백 후 재배포는 새 승인으로만.

## 7. 예상 호환성·성능 위험
- 모델 크기·tokenizer·라벨 동일, 지연 1.06×·RSS 1.00×(개발 PC) — 운영 CPU/메모리 제한에서의 값은 **S2b 미검증**.
- `required` 는 NER 로딩 실패 시 연결 입구 503 을 유발한다(의도). 이미지에 모델이 없으면 가명화 경로가 막힌다.
- `model_card.json` 이름이 옛 `koelectra-small-v2b` 로 남아 있어 로그만 보고 모델을 오인할 수 있다 → 폴더 이름·지문으로 확인.
- NER 미연결 입구(메인 채팅, 일정 제안, knowledge_ask)는 교체와 무관하게 보호되지 않는다.
- D1P 노출 40%·A3N 오탐 증가 가능성은 위험 수용 결정서 3절 참고.
