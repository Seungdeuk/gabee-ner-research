# R1 시드7 NER 모델 교체 — 운영 환경 확인 가이드

**운영 서버 담당자용.** 아래 체크리스트를 순서대로 실행하고 결과를 알려주시면, 그 결과에 맞는 교체 절차로 바로 진행할 수 있습니다.

**아직 하지 않는 것**: 모델 파일 교체, `.env`/환경변수 변경, 서버·컨테이너 재시작. 이 문서는 **확인용**입니다.

배경(필요하면 참고, 안 읽어도 체크리스트 수행엔 문제없음): `docs/security/NER_VALIDATION.md` 27~39절.

---

## 0. 참고값(이미 검증 완료 — 다시 계산할 필요 없음)

| | 서비스 모델 | R1 시드7 |
|---|---|---|
| `model.onnx` 지문(sha256 앞 12자) | `46210943c45f` | `c536eadeb5a1` |
| `model.onnx` 전체 sha256 | `46210943c45f01b6a9fa500894fb9821b1c130be63de8d8f7ef603d2d87ccf8e` | `c536eadeb5a15c2c00755c10934d5862f0e9e53c2bd85ec8ab0b5891854c31f9` |
| 필요 파일 | `model.onnx`·`tokenizer.json`·`labels.json`·`model_card.json` | 동일 4종(`tokenizer.json`·`labels.json`은 두 모델이 완전히 같은 파일) |

---

## 1. 실제 API가 어떻게 떠 있는지 확인

### Windows PowerShell
```powershell
# Docker 컨테이너가 있는지(이름만 나옴, 값 노출 없음)
docker ps

# 없다면 직접 프로세스로 떠 있는지
Get-Process | Where-Object { $_.ProcessName -like "*python*" -or $_.ProcessName -like "*uvicorn*" }
```

### Linux 서버
```bash
# Docker 컨테이너가 있는지
docker ps

# 없다면 직접 프로세스로 떠 있는지
ps aux | grep -i uvicorn

# systemd 서비스로 등록돼 있는지
systemctl list-units --type=service | grep -i gabee
```

**결과에 따라**: 컨테이너 이름이 보이면 2·3·4절에서 `docker exec <컨테이너 이름>`을 쓰고, 프로세스로만 보이면 그 프로세스가 도는 환경(가상환경 등)에서 `python` 명령을 직접 실행합니다.

---

## 2. 실제 로딩된 NER 모델의 경로·SHA256·보호 모드 확인

**중요**: 환경변수 자체(`echo $PRIVACY_NER_MODE`)가 아니라, 애플리케이션이 실제로 계산해서 쓰는 **최종 적용값**을 봐야 합니다. 아래 명령은 그 값만 출력하고, 비밀번호·API 키 등 다른 설정은 전혀 노출하지 않습니다.

### 2-1. 설정값·지문 확인(가장 먼저, 가장 정확한 방법)

**Docker 컨테이너**:
```bash
docker exec <컨테이너 이름> python -c "from app.shared.config import settings; print('MODE:', settings.privacy_ner_mode, '| MODEL_FILE:', settings.privacy_ner_model_file, '| MODEL_DIR:', settings.privacy_ner_model_dir)"
docker exec <컨테이너 이름> python -c "from app.shared.privacy import ner; print(ner.ner_status())"
```

**로컬 프로세스(Windows PowerShell)**:
```powershell
# 그 프로세스가 쓰는 가상환경의 python으로 실행(예: apps/api/.venv)
& "<venv 경로>\Scripts\python.exe" -c "from app.shared.config import settings; print('MODE:', settings.privacy_ner_mode, '| MODEL_FILE:', settings.privacy_ner_model_file, '| MODEL_DIR:', settings.privacy_ner_model_dir)"
& "<venv 경로>\Scripts\python.exe" -c "from app.shared.privacy import ner; print(ner.ner_status())"
```

**로컬 프로세스(Linux, 직접 또는 systemd)**:
```bash
python -c "from app.shared.config import settings; print('MODE:', settings.privacy_ner_mode, '| MODEL_FILE:', settings.privacy_ner_model_file, '| MODEL_DIR:', settings.privacy_ner_model_dir)"
python -c "from app.shared.privacy import ner; print(ner.ner_status())"
```

`ner_status()`의 `fingerprint` 값이 **0절 표의 지문과 비교하는 기준**입니다 — 경로를 몰라도 지금 실제로 어떤 모델이 로딩돼 있는지 이걸로 바로 알 수 있습니다.

### 2-2. 그 경로의 파일 자체를 직접 재확인

2-1에서 얻은 `MODEL_DIR` 값을 아래 `<모델폴더>`에 그대로 넣습니다.

**Docker 컨테이너**:
```bash
docker exec <컨테이너 이름> sha256sum <모델폴더>/model.onnx
```

**Windows PowerShell**:
```powershell
Get-FileHash "<모델폴더>\model.onnx" -Algorithm SHA256
```

**Linux(로컬 프로세스)**:
```bash
sha256sum <모델폴더>/model.onnx
```

결과 해시의 앞 12자가 2-1의 `fingerprint`와 같아야 합니다(다르면 로딩 이후 파일이 바뀌었다는 뜻 — 재시작 전까지 적용 안 됨).

---

## 3. 이미지 COPY 방식인지 볼륨 마운트 방식인지 구별(Docker인 경우만)

```bash
docker inspect <컨테이너 이름> --format '{{json .Mounts}}'
```

- **결과가 `[]`이거나 모델 경로(2-1의 `MODEL_DIR`)와 무관한 항목만 있다면** → **이미지 COPY 방식**입니다. 호스트의 어떤 폴더를 지금 바꿔도 이미 떠 있는 컨테이너에는 반영되지 않습니다 — **이미지를 다시 빌드해야 교체됩니다.**
- **`MODEL_DIR` 경로가 `Mounts` 목록에 호스트 경로와 함께 나온다면** → **볼륨 마운트 방식**입니다. 호스트의 그 폴더를 바꾸고 **컨테이너만 재시작**하면 반영됩니다(이미지 재빌드 불필요).

Docker가 아니라 로컬 프로세스로 뜬 경우는 이 구분이 필요 없습니다 — 폴더를 바꾸고 프로세스를 재시작하면 바로 반영됩니다.

---

## 4. 확인 결과에 따른 R1 시드7 교체·롤백 절차

### 분기 A — 이미지 COPY 방식(3절에서 `Mounts` 비어 있음)

1. R1 모델 파일 4종을 `apps/api/data/models/<2-1에서 확인한 폴더명>/`에 놓고 **이미지를 새로 빌드**합니다.
2. 새 이미지로 컨테이너를 **재생성**합니다(기존 컨테이너를 내리고 새 이미지로 올림 — 단순 재시작이 아닙니다).
3. 2절을 다시 실행해 `fingerprint`가 `c536eadeb5a1`로 바뀌었는지, `MODE=required`가 유지됐는지 확인합니다.
4. **롤백**: 이전 이미지로 되돌립니다(이전 이미지 태그 보존 또는 이전 모델 폴더로 재빌드) → 2절로 `46210943c45f` 복원 확인.

### 분기 B — 볼륨 마운트 방식(3절에서 호스트 경로가 보임)

1. 호스트의 `<모델폴더>` 옆에 `<모델폴더>.new`로 R1 파일 4종을 준비하고 해시를 한 번 더 대조합니다.
2. 서버(또는 컨테이너)를 내린 뒤, `<모델폴더>` → `<모델폴더>.prev`, `<모델폴더>.new` → `<모델폴더>`로 이름을 바꿉니다.
3. 재시작 → 2절로 `fingerprint=c536eadeb5a1`·`MODE=required` 확인.
4. **롤백**: `<모델폴더>` → `<모델폴더>.failed`, `<모델폴더>.prev` → `<모델폴더>`로 되돌리고 재시작 → 2절로 `46210943c45f` 복원 확인.

### 분기 C — Docker 없이 로컬/서버에서 직접 실행 중

분기 B와 같은 방식(폴더 이름 맞바꾸기)이고, "서버를 내린다"가 "그 프로세스를 재시작한다"로 바뀔 뿐입니다. `--reload` 옵션은 코드 변경에만 반응하고 모델은 시작 시 1회만 읽으므로, 모델 교체는 **반드시 수동 재시작**이 필요합니다.

### 공통 — 교체 직후마다 실행

- 2절(지문·`MODE`·`MODEL_FILE`) 재확인 — 매 재시작마다, 교체든 롤백이든 **항상** 수행합니다.
- 소통·일정·주간뉴스 가명화↔복원 왕복 테스트(`apps/api/scripts/diag_roundtrip.py <모델폴더>`), CPU 지연 스팟체크(`apps/api/scripts/diag_prod_env_bench.py <모델폴더> --n 100`) — 둘 다 합성 데이터만 쓰는 기존 검증 스크립트입니다.
- 수 시간~하루 모니터링(503 `privacy_unavailable` 비율, 응답 지연) 후 문제 없으면 확정.

---

## 주의사항

- 위 명령은 모두 **지정한 변수 하나만** 출력합니다. `printenv`·`env`·`.env` 파일 전체를 출력하거나 공유하지 마십시오.
- 이 문서의 절차를 실행하기 전, **반드시 사용자(요청자)의 최종 승인**을 받으십시오.
