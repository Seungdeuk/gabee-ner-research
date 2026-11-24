# R11 Cloud Run 운영 전환 준비 현황 (2026-11-23)

표시: 자동 검증 기반 · 사람 검수 미실시. 사전 등록 P1~P3 FAIL·`KEEP_S0`·R11-S 미충족은 그대로다. **Cloud Run 설정·리비전·트래픽은 변경하지 않았다. 이 PC 에서 운영에 접속하지 못했다(`gcloud` 없음).**

| # | 항목 | 상태 |
|---|---|---|
| 1 | 운영 Cloud Run 환경 확인 | **미확인(담당자 회신 필요)** — 읽기 전용 명령 정리 완료(`NER_R11_OPS_OPERATOR_COMMANDS.md`): 프로젝트·리전, 리비전·트래픽, 이미지 digest, `PRIVACY_NER_MODE`, 모델 경로 환경변수, CPU·메모리, 스레드, 동시성, 최소·최대 인스턴스, 레지스트리·배포 방식. 비밀·다른 환경변수는 출력하지 않는 `jq` 필터 사용. 마지막 알려진 값(2026-11-18): 리비전 `gabee-00018-bd2`, NER `auto`, 지문 `46210943c45f`(이후 변경 여부 미확인) |
| 2 | S2b | **미실행(UNVERIFIED)** — 운영 자원 조건을 받지 못해 같은 조건 측정을 할 수 없다. 임의 2CPU/2GB 값은 `PLUMBING-ONLY` 유지. 측정 도구(`s2b_container.sh`)는 준비·동작 확인. 통과 못 하면 배포 중단 |
| 3 | S4-1 운영 S0 해시 | **미검증** — 운영 이미지 digest 로 실행하는 확인 방법을 제시(2절). 직접 확인 전에는 "검증 완료"로 처리하지 않음 |
| | S4-2 운영 `PRIVACY_NER_MODE` | **미확인(마지막 `auto`)** — 배포 전 게이트가 `required` 가 아니면 차단 |
| | S4-3 실패 시 외부 LLM 호출 차단 | 비운영 이미지 PASS(연결 시도 0), **운영 미검증** |
| | S4-4 통합 경로·`ner_status` | 비운영 PASS(required·active·`3831601b7ffb`), 운영은 일회성 Job(digest)·후보 URL 스모크로 간접 확인 예정 |
| | S4-5 이미지 재빌드·롤백 | 비운영 PASS(교체 약 7초), **Cloud Run 리비전 롤백은 미실행** |
| | S4-6 S2b 환경 정보 | **미확인** |
| 4 | 릴리스 PR 준비 | **준비됨(로컬 브랜치, 푸시·PR 생성 안 함)** — 아래 |
| 5 | R11 배포 이미지·모델 해시 | R11 `model.onnx` SHA256 `3831601b7ffb6093a8d1b8c832e46ad140712c6e402f17b3054c3ec0f9ae4c94`(지문 `3831601b7ffb`), tokenizer `d5c7a9a8…`·labels `f5cf8bd5…` 는 S0 와 동일. **운영 이미지 digest 는 병합된 release 로 빌드한 뒤 확정**(비운영 검증 이미지 id `5288b34f…` 는 운영 이미지가 아님) |
| 6 | 0% 트래픽 배포·롤백 계획 | **런북 준비**(`NER_R11_OPS_CLOUDRUN_RUNBOOK.md`): 보존 → 빌드·사전검증·게이트 → 0% 배포(이미지와 `required` 를 한 명령에 `--update-env-vars`) → 후보 검증(일회성 Job 으로 모드·지문) → 점진 전환 → 이전 리비전 롤백 |
| 7 | 위험 수용 승인 | **미승인(서명 없음)** — 결정서에 최소 개선 폭 미달, D1P 40%, 합성 평가셋·사람 검수 미실시, NER 미적용 경로 존재, Cloud Run 장애·롤백 위험 포함 |
| 8 | 최종 운영 교체 가능 여부 | **불가(추가 조치 필요)** |

## release/1.4 전환 (2026-11-23 추가)
- 대상 릴리스를 `release/1.3` 에서 `release/1.4` 로 바꾼다. `release/1.3` 은 `deploy/kang` 으로 병합될 예정이므로 **R11 변경을 1.3 배포에 섞지 않는다.**
- 확인(fetch 후): `origin/release/1.4` **없음**. `origin/release/1.3` 팁 `9b5282f3`, `origin/deploy/kang` 팁 `58be2bf4`(1.3 보다 209커밋 뒤), `origin/main` `3c3832fb`. 그래서 R11 브랜치를 1.4 기준으로 **만들지 않았다**(오래된 1.3 기준 임의 생성 금지).
- 기준 커밋 제안: `release/1.4` 는 **1.3 병합 직전의 `release/1.3` 팁에서 분기**(현재 `9b5282f3`, 이후 1.3 에 커밋이 더 들어오면 그 팁)해야 1.3 의 변경을 이어받는다. `deploy/kang`(209커밋 뒤, `app/shared/privacy` 최신 수정 미포함)이나 `main` 에서 만들면 안 된다. 담당자가 1.4 를 만들면 그 팁을 알려 주세요.
- 기존 `feature/seungdeuk-ner-r11-image`(1.3 기준 `04937642`)는 **PR 로 쓰지 않는다**(로컬 초안, upstream 추적 해제함). R11 변경만 패치 2개로 분리해 보관: `C:\Users\<user>\Documents\gabee-deploy-prep\r11-pr-patches\0001…(Dockerfile)`, `0002…(게이트 스크립트·시험)`. 변경 파일은 정확히 3개이며 현재 `release/1.3` 팁 `9b5282f3` 에 순서대로 깨끗하게 적용됨(`git apply --check`). 1.4 가 생기면 `git am` 으로 새 브랜치에 얹고 **그 기준에서 Dockerfile 재검토·시험·이미지 재빌드**를 다시 한다.
- CI/CD 자동 배포 트리거 확인(저장소): `release/1.3`·`deploy/kang`·`main` 모두 `ci-check.yml` 하나뿐이고 `on:` 은 `workflow_dispatch`(수동)만 켜져 있다. gcloud·docker push·Cloud Run 배포 자동화는 저장소에 없다. **저장소 밖의 Cloud Build 트리거·Cloud Run 지속 배포 연결은 확인하지 못했다**(담당자 확인 필요: `gcloud builds triggers list`, 서비스의 소스 연결).
- 주의: 모델 파일은 저장소에 없으므로(`apps/api/data/` 는 gitignore) 코드만으로 자동 빌드하면 `COPY data/models` 가 실패한다. 그러나 **옛 `ner-koelectra-small-v2b` 폴더가 있는 빌드 컨텍스트에 새 Dockerfile 을 쓰면** 빌드는 성공하고 `PRIVACY_NER_MODEL_DIR` 가 없는 R11 폴더를 가리켜 `required` 에서 NER 연결 입구가 503 이 된다. 이미지 검증(폴더 이름·해시·`ner_status()`)이 필수인 이유다. 조립 스크립트의 ref 인자는 이제 필수이며 기본값이 없다.

## 릴리스 PR 준비 상태
- 브랜치 `feature/seungdeuk-ner-r11-image`(워크트리 `C:\Users\<user>\Documents\gabee-ner-r11-release`), 기준 `origin/release/1.3@04937642`(**1.4 전환으로 초안만 보관, PR 아님**), **로컬 커밋 2개, 변경 3파일(+208/−2)**: `apps/api/Dockerfile`(모델 폴더 `ner-koelectra-small-r11b-s21`, `PRIVACY_NER_MODE=required`, 파일 전체가 아닌 3줄 추가·2줄 삭제), `scripts/check_ner_deploy_mode.py`와 시험. 연구 브랜치(134커밋 차이)의 변경은 섞지 않았다.
- 릴리스 코드 호환성: 비운영 이미지는 이전 기준 `7fd9f708` 로 빌드했고 그 뒤 `04937642` 까지 `Dockerfile`·`app/shared/privacy`·`pyproject.toml` 변경은 없음. 릴리스 코드에서 R11 모델로 `app/shared/privacy/tests` 전체 + 게이트 시험 **502개 통과**, ruff 통과. 이미지는 병합된 커밋으로 다시 빌드·검증한다.
- 유지된 조건: `model.onnx` 사용(`model.int8.onnx` 불필요), tokenizer·labels 동일, 이전 S0 이미지·리비전 보존(런북 1단계).
- 푸시·PR 생성·병합은 하지 않았다(브랜치 푸시와 PR 은 사용자 요청 시). `docs/security` 문서와 증거는 연구 브랜치에만 있다.

## 승인 전 남은 조치
1. 운영 담당자 회신(`OPERATOR_COMMANDS` 3절 표) → S4-1·2·6 확정
2. 회신한 자원 조건으로 S2b 교차 측정 통과
3. 위험 수용 결정서 서명(운영 담당자·승인 책임자·보안 검토자)
4. PR 푸시·리뷰·병합 → 병합 커밋으로 이미지 재빌드·digest 확정
5. 사용자 최종 승인 후 런북 순서대로 0% 배포
