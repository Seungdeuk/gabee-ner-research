# R11 Cloud Run 운영 전환 런북 (준비본 — 최종 승인 전 실행 금지)

2026-11-23. 대상: Cloud Run `gabee` (프로젝트 `<GCP_PROJECT>`, 리전 `asia-northeast3`; 이전 준비 자료 기준이며 담당자 확인으로 갱신). **이 런북의 변경 명령은 실행하지 않았다.** 표시: 자동 검증 기반 · 사람 검수 미실시. 사전 등록 P1~P3 FAIL·`KEEP_S0` 는 변경하지 않는다. 일반 원칙: 환경변수 변경은 이미지 교체와 **같은 배포 명령**으로만 하고(이미지와 `required` 를 함께 적용), 서비스에 환경변수만 먼저 바꾸지 않는다. `--set-env-vars` 는 쓰지 않는다(기존 환경변수 삭제).

## 착수 조건(모두 충족해야 단계 1 시작)
1. 운영 담당자 회신표(`NER_R11_OPS_OPERATOR_COMMANDS.md` 3절) 완료, S4-1·S4-2·S4-6 확정
2. S2b 통과(운영 자원 조건과 같은 조건의 S0/R11 교차 측정)
3. 위험 수용 결정서 서명 완료
4. 릴리스 PR(`feature/seungdeuk-ner-r11-image`)이 리뷰·병합됨, 이미지는 병합된 `release` 커밋으로 빌드
5. 사용자 최종 승인(이 단계부터 실제 Cloud Run 변경)

## 1. 기존 운영 리비전·이미지 보존
```bash
P=<GCP_PROJECT>; R=asia-northeast3; S=gabee
PREV_REV=$(gcloud run services describe $S --project $P --region $R --format="value(status.latestReadyRevisionName)")
PREV_IMG=$(gcloud run revisions describe $PREV_REV --project $P --region $R --format="value(spec.containers[0].image)")
echo "$PREV_REV $PREV_IMG" | tee prev_revision.txt          # 기록·보관(삭제 금지). 이미지는 @sha256 digest 형태여야 함
gcloud run services describe $S --project $P --region $R --format="yaml(status.traffic)"
```
이전 리비전·이미지는 어떤 경우에도 삭제하지 않는다(롤백 대상). 트래픽이 "최신 리비전 100%"로 설정돼 있으면 새 리비전이 생길 때 자동 전환될 수 있으므로, 단계 3 전에 이전 리비전에 고정한다:
```bash
gcloud run services update-traffic $S --project $P --region $R --to-revisions $PREV_REV=100
```

## 2. 이미지 빌드·사전 검증(Cloud Run 변경 없음)
1. 병합된 `release/1.4` 커밋으로 컨텍스트 조립: `gabee-deploy-prep/stage_build_context_nonprod.sh r11 origin/release/1.4`(모델 3개 폴더, R11 `model.onnx` SHA256 `3831601b7ffb6093a8d1b8c832e46ad140712c6e402f17b3054c3ec0f9ae4c94` 확인 포함). 운영 반영 이미지는 `…-nonprod` 폴더를 쓰지 말고 같은 스크립트로 새 폴더를 만든다.
2. 빌드·푸시: 레지스트리 경로·인증은 기존 방식대로, **고유 태그**(날짜+커밋 7자)로 푸시하고 **digest** 를 기록(`NEW_IMG=<저장소>/gabee@sha256:<digest>`). `latest` 만 쓰지 않는다.
3. 이미지 안 확인(네트워크 없음): R11 `sha256sum`, `PRIVACY_NER_MODE=required`, `ner_status()` = required·active·지문 `3831601b7ffb`, 모델 폴더 정확히 3개, `doc-classifier`·`e5-small-ft` 가 이전 이미지와 해시 목록 동일(체크리스트 B-4).
4. 실패 차단 시험(비운영, 네트워크 없음): `container_failclosed.py` 로 폴더 비움·1바이트 변조 → `PrivacyUnavailable`, 연결 시도 0.
5. 배포 전 게이트: `python apps/api/scripts/check_ner_deploy_mode.py --service-json <ner_env.json> --image-status-json <image_status.json>` 종료 코드 0 이어야 한다(`auto`·오타·모델 경로 덮어쓰기·지문 불일치면 중단). 서비스 JSON 에는 `PRIVACY_NER_*` 만 담는다(운영 담당자 명령 1절 jq).
어느 하나라도 실패하면 단계 3 으로 가지 않는다.

## 3. 새 리비전 배포 — 트래픽 0%, 이미지와 `required` 를 함께 (승인 후 첫 변경 명령)
```bash
gcloud run deploy $S --project $P --region $R --image $NEW_IMG --no-traffic --tag candidate \
  --update-env-vars PRIVACY_NER_MODE=required
```
이외 옵션(환경변수·시크릿·서비스 계정·자원·스케일)은 넘기지 않아 기존 리비전 설정이 이어진다. 이미지 안 `PRIVACY_NER_MODEL_DIR` 가 새 폴더를 가리키므로 서비스에 `PRIVACY_NER_MODEL_DIR` 환경변수가 덮어쓰여 있으면(1절 확인) 같은 명령에서 `--update-env-vars` 에 올바른 값 `PRIVACY_NER_MODEL_DIR=/srv/data/models/ner-koelectra-small-r11b-s21` 도 함께 지정하거나 환경변수를 지워야 한다(`gate` 가 차단). 사용자 트래픽은 계속 이전 리비전으로 간다.

## 4. 0% 리비전에서 검증
1. **실제 설정 확인:** 새 리비전 JSON 에서 `PRIVACY_NER_MODE=required`(및 모델 경로 환경변수)만 추출해 게이트 재실행(종료 코드 0).
2. **실제 실행 모드·모델 지문(일회성 Job, 서비스·트래픽 무변경):** `/health` 는 NER 상태를 노출하지 않고 성공 시 NER 로그도 기본 레벨에서 보이지 않으므로, **같은 digest·같은 환경변수로 일회성 Cloud Run Job 을 실행**한다.
```bash
gcloud run jobs create gabee-ner-check --project $P --region $R --image $NEW_IMG --set-env-vars PRIVACY_NER_MODE=required \
  --command python --args=-c,"import json;from app.shared.privacy import ner;print(json.dumps(ner.ner_status()))" --max-retries 0 --task-timeout 120s
gcloud run jobs execute gabee-ner-check --project $P --region $R --wait
gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="gabee-ner-check"' --project $P --limit 5 --format="value(textPayload)"
gcloud run jobs delete gabee-ner-check --project $P --region $R --quiet      # 점검 후 삭제(점검용 Job 만 삭제)
```
기대 출력: `{"mode": "required", "active": true, "fingerprint": "3831601b7ffb"}`. (`--set-env-vars` 는 이 **일회성 Job** 에만 쓴다 — 서비스에는 쓰지 않는다.) Job 은 서비스와 환경이 같은 이미지·같은 필수 환경변수를 쓰며, 서비스 전용 환경변수·시크릿은 주지 않는다. 이 방법으로도 서비스 리비전의 실제 프로세스 상태를 직접 보는 것은 아니므로 "digest 기반 실행 상태 검증"으로 표기한다.
3. **후보 URL 스모크**(`candidate` 태그 URL, 시험 가족·가상 이름만, 가족 AI 동의 필요, 실제 LLM 비용 발생): `GET /health` 200, `GET /health/db` 정상, NER 연결 입구(가족채팅 분석 등)에서 가명화 토큰이 생성되고 응답이 복원됨(required 에서 성공하면 엔진 로딩의 간접 증거), 5xx 없음.
4. **실패 시 외부 LLM 차단:** 운영 후보 리비전에서 모델을 일부러 망가뜨리지 않는다. 비운영 이미지 시험(모델 비움·변조 → `PrivacyUnavailable`(503), 연결 시도 0)을 이 digest 에 대해 재실행한 결과와, 후보 리비전 로그에 `NER 엔진을 불러오지 못했다`(ERROR/WARNING) 가 없음을 확인한다. NER 미적용 경로(규칙 마스킹만·가명화 없음)는 required 에서도 호출되므로 이 검증 대상이 아님을 보고서에 명시한다.
5. **자원·동시성 확인:** 후보 리비전의 CPU·메모리 제한·동시성·최소/최대 인스턴스가 이전 리비전과 같은지 JSON 비교(`jq` 로 필드 비교), 후보 URL 에 가상 요청 부하를 주어 p50 지연·메모리 사용량(Cloud Monitoring)이 이전 리비전 대비 +30% 이내인지 확인.
6. 이상 시 → 7단계로 가지 않고 5절(롤백/정리).

## 5. 트래픽 전환(승인된 경우만, 점진)
```bash
gcloud run services update-traffic $S --project $P --region $R --to-revisions <새 리비전>=10,$PREV_REV=90   # 로그·오류율 관찰 후
gcloud run services update-traffic $S --project $P --region $R --to-revisions <새 리비전>=100
```
각 단계에서 15분 이상: 새 리비전 5xx·503 `privacy_unavailable` 비율, p50 지연, 메모리, `NER 엔진을 불러오지 못했다` 로그, 개인정보 노출 사고 신고. 임계값(롤백 조건)은 위험 수용 결정서 4절.

## 6. 롤백
```bash
gcloud run services update-traffic $S --project $P --region $R --to-revisions $PREV_REV=100    # 재빌드 없이 즉시
gcloud run revisions list --service $S --project $P --region $R --limit 10                        # 이전 리비전이 남아 있는지 확인(삭제 금지)
```
새 리비전·이미지는 남겨 두고 원인 확인 후 정리한다. 롤백 후 이전 리비전 이미지를 4절 2번 방식으로 다시 확인(지문 `46210943c45f`). 롤백 시 R11 개선은 사라지고 S0 의 위험(독립 평가에서 확인된 노출률 17.07%)으로 돌아간다. 롤백 소요 시간을 기록한다. 이전 리비전이 삭제됐거나 이미지가 레지스트리에서 사라진 경우에는 이 롤백이 불가능하므로 단계 1 의 보존이 착수 조건이다.

## 7. 알려진 한계·후속 과제(이번 전환으로 해결되지 않음)
- **NER 미적용 경로**(메인 AI 채팅, knowledge_ask, 문서 OCR/PII, 사진 판별: 가명화 없음 / 일정 제안·call_talk·관리자 챗봇: 규칙 마스킹만 / 메인 채팅→가족 채팅 에이전트 도구 구간)는 `required` 여도 NER 실패 때 호출된다. **이 경로까지 보호된다고 보고하지 않는다. 보호 정책 변경은 별도 작업으로 기록한다.**
- 런타임 모드·지문을 서비스에서 직접 볼 방법이 없다. 후속 과제로 관리자 전용 상태 조회(모드·active·지문 앞 12자만)를 별도 PR 로 검토한다(이번 PR 에는 넣지 않음).
- 이미지에 모델이 없거나 깨지면 `required` 로 NER 연결 입구가 503 이다(의도된 fail-closed).
