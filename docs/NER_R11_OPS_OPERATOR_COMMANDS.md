# 운영 담당자용 명령 모음 — Cloud Run `gabee` 현재 상태 확인 (읽기 전용)

2026-11-23 개정. 대상(이전 준비 자료 `CLOUD_RUN_DEPLOY_CHECKLIST.md` 기준): 프로젝트 `<GCP_PROJECT>`, 리전 `asia-northeast3`, 서비스 `gabee`(2026-11-18 당시 리비전 `gabee-00018-bd2`, NER `auto`). **이 PC 에는 `gcloud` 가 없고 저는 운영에 접속하지 않았다. 아래는 담당자가 Cloud Shell 에서 실행하는 읽기 전용 명령이다.** 이 문서의 명령은 서비스 설정·리비전·트래픽을 바꾸지 않는다.

## 안전 규칙
- `describe` 를 `--format` 없이 통째로 출력하거나 공유하지 않는다(환경변수·시크릿 참조가 보인다). 아래처럼 `jq` 로 필요한 필드만 추려 파일에 남긴다.
- `PRIVACY_NER_*` 이름의 환경변수 외에는 값을 저장·공유하지 않는다. `ANTHROPIC_API_KEY` 등은 어떤 출력에도 넣지 않는다.
- 이미지를 pull 하는 명령은 로컬 비밀을 쓰지 않는 읽기 전용 실행이며, 컨테이너에는 네트워크를 주지 않는다(`--network none`).

## 1. 필수 확인 항목과 명령
```bash
P=<GCP_PROJECT>; R=asia-northeast3; S=gabee
gcloud run services describe $S --project $P --region $R --format=json > /tmp/svc_full.json   # 파일은 공유·보관하지 않는다(시크릿 참조 포함). 아래 jq 로만 사용 후 삭제
```
| 확인 항목 | 명령(모두 `/tmp/svc_full.json` 에서 필드만 추출) |
|---|---|
| 프로젝트·리전 | 위 변수. `gcloud config get-value project` 로 현재 활성 프로젝트가 같은지도 확인 |
| 활성 리비전·트래픽 배분 | `jq '{latestReady:.status.latestReadyRevisionName, latestCreated:.status.latestCreatedRevisionName, traffic:.status.traffic}' /tmp/svc_full.json` |
| 운영 이미지 정확한 digest | 리비전 이름 `REV` 로 `gcloud run revisions describe $REV --project $P --region $R --format="value(spec.containers[0].image,status.imageDigest)"` — `...@sha256:<digest>` 형태를 기록 |
| 현재 `PRIVACY_NER_MODE` | `jq '[.spec.template.spec.containers[0].env[]?\|select(.name=="PRIVACY_NER_MODE")\|{name,value:(.value // "<secret-ref>")}]' /tmp/svc_full.json` (비어 있으면 "환경변수 없음 = 이미지 기본값") |
| 모델 경로·파일 선택 환경변수 | `jq '[.spec.template.spec.containers[0].env[]?\|select(.name\|test("^PRIVACY_NER_(MODEL_DIR\|MODEL_FILE)$"))\|{name,value}]' /tmp/svc_full.json` |
| 추론 스레드·NER 동시 추론 | `jq '[.spec.template.spec.containers[0].env[]?\|select(.name\|test("^PRIVACY_NER_(THREADS\|MAX_CONCURRENCY)$"))\|{name,value}]' /tmp/svc_full.json` (없으면 기본 2/2) |
| CPU·메모리 제한 | `jq '.spec.template.spec.containers[0].resources.limits' /tmp/svc_full.json` |
| Cloud Run 요청 동시성 | `jq '.spec.template.spec.containerConcurrency' /tmp/svc_full.json` |
| 최소·최대 인스턴스, CPU 상시 할당·부스트 | `jq '.spec.template.metadata.annotations \| with_entries(select(.key\|test("minScale\|maxScale\|cpu-throttling\|startup-cpu-boost\|execution-environment")))' /tmp/svc_full.json` |
| 서비스 이미지 레지스트리 | 위 이미지 URI 의 호스트·저장소 부분(예: `asia-northeast3-docker.pkg.dev/<프로젝트>/<저장소>/gabee`) |
| 배포 방식(소스 배포/이미지 푸시/Cloud Build) | `gcloud builds list --project $P --limit 5`(리전 빌드면 `--region $R`) + `gcloud artifacts docker images list <저장소 경로> --include-tags --limit 10` 로 최근 푸시 이력 확인 |
| ONNX Runtime 버전 | 아래 2절(운영 이미지 digest 로 실행) |
확인 후 `rm /tmp/svc_full.json`.

## 2. 운영 S0 모델 해시·실행 상태를 digest 로 확인 (S4-1, S4-4)
운영 서비스에는 `exec` 가 없고 `/health` 는 NER 상태를 보여 주지 않으며, 성공 시 NER 준비 로그는 기본 로그 레벨에서 보이지 않는다(비운영 이미지 시험으로 확인). 그래서 **운영 이미지와 같은 digest 를 따로 실행**해 확인한다.
```bash
IMG=<위에서 기록한 이미지@sha256:digest>        # 태그가 아니라 digest 로 고정
docker run --rm --network none --entrypoint sh $IMG -c 'sha256sum /srv/data/models/ner-koelectra-small-v2b/model.onnx; python -c "import onnxruntime;print(\"onnxruntime\",onnxruntime.__version__)"'
docker run --rm --network none --entrypoint python $IMG -c "import json;from app.shared.privacy import ner;print(json.dumps(ner.ner_status()))"
```
기대(S0): 해시 `46210943c45f01b6a9fa500894fb9821b1c130be63de8d8f7ef603d2d87ccf8e`, 지문 `46210943c45f`. **이 확인은 "그 digest 이미지의 내용"이며, 실제 Cloud Run 인스턴스의 환경변수 적용은 1절의 설정 확인과 합쳐서만 결론낸다.** 해시를 직접 확인하지 못하면 S4-1 은 "검증 완료"가 아니라 "미검증"으로 기록한다. 이미지를 pull 할 권한이 없으면 담당자가 권한을 받아 실행하거나 Artifact Registry 의 digest·빌드 이력(빌드 로그의 모델 해시 출력)으로 간접 확인하고 "간접 확인"이라고 표기한다.

## 3. 회신할 표 (S4-1, S4-2, S4-6, S2b 조건)
| 항목 | 값 |
|---|---|
| 프로젝트/리전/서비스 | |
| 활성 리비전과 트래픽(%) | |
| 운영 이미지 URI@digest | |
| 운영 S0 model.onnx SHA256 / `ner_status()` | |
| 운영 `PRIVACY_NER_MODE`(값 또는 "없음") | |
| `PRIVACY_NER_MODEL_DIR`·`MODEL_FILE`(값 또는 "없음") | |
| CPU 제한 / 메모리 제한 | |
| Cloud Run 동시성 / 최소·최대 인스턴스 / CPU 상시 할당 | |
| `PRIVACY_NER_THREADS` / `PRIVACY_NER_MAX_CONCURRENCY`(값 또는 "없음=기본 2/2") | |
| 운영 이미지 ONNX Runtime 버전 | |
| 레지스트리 경로 / 배포 방식 | |
| 롤백 대상 리비전 이름 | |

## 4. S2b 측정(회신 값을 받은 뒤)
`ops_replace/s2b_container.sh <라벨> <CPUS> <MEM> <THREADS> <CONCURRENCY>` 에 회신 값을 그대로 넣는다. Cloud Run 의 CPU/메모리 제한은 `docker --cpus/--memory` 로, `PRIVACY_NER_THREADS`·`PRIVACY_NER_MAX_CONCURRENCY` 는 같은 값으로, 요청 동시성은 `<CONCURRENCY>`(동시 요청 수)로 대응시킨다. S0·R11·R11·S0·S0·R11 순서 교차 측정, 이미지의 onnxruntime 은 운영과 같은 빌드. 임의 값으로 돌린 결과는 `PLUMBING-ONLY` 로만 표기한다. 기준: 토큰 일치 ≥99.9%(R11 동일 입력 일치는 개발 환경 G12 에서 100%), p50·크기·RSS 모두 S0 대비 +30% 이내.
