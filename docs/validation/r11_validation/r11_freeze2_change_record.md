# R11 코드 재고정 r11-freeze-2 변경 기록

상태: 자동 검증 기반 연구용 · 사람 검수 미완료(r11-2.0). 이번 변경은 **실행 오류 복구**이며 평가 결과에 맞춘 사후 기준 변경이 아니다.

## 오류 원인
B군 시드 7 학습 뒤 후보 ONNX 내보내기(`export_ep1`)가 `FileNotFoundError: cand/ep1/eval_full.json` 으로 중단됐다.
`ml/ner/export/export_onnx.py` 의 `metrics_from_eval()` 이 **모델 카드에 그 모델의 baseline_val·synth_val Precision·Recall·F1(겹침·정확, 문장 수, 정답 span 수, 음성 문장 FPR)을 넣기 위해** 런 폴더의 `eval_full.json` 을 읽는데, 파생 후보 폴더(`cand/ep<k>`, `cand/swa`)에는 그 파일이 없었다.
`eval_full.json` 은 모델별 성능값이므로 best 모델의 파일을 복사하지 않는다. 후보마다 기존 고정 평가 절차(`evaluate_full.py`, `--run-dir cand/<name>`)를 그대로 실행해 후보 자신의 값을 만든다. 평가 데이터·지표 계산·`export_onnx.py` 는 바꾸지 않았다.

## 수정 내용(허용 범위: r11_derive.py, run_r11_B.py, 시험)
- `r11_derive.py`: `verify_cands()` 추가(읽기 전용 검증). 에폭 체크포인트 SHA256 이 `train_log.json` 기록과 같은지, `best/` 가 `epoch_<best>/` 와 같은지, 후보 가중치·`train_log.json` 사본이 `cand_info.json` 과 같은지 확인한다. `cands`·`select` 동작과 선정 규칙은 그대로다.
- `run_r11_B.py`:
  - 후보마다 `evalfull_<name>` 단계(기존 `evaluate_full.py`)를 `export_<name>` 앞에 추가. 평가 뒤 `eval_full.json` 의 run_id 가 후보 이름과 같은지, 가중치 해시가 `cand_info.json` 과 같은지, best 에폭 후보의 TP/FP/FN 이 런 전체 평가와 같은지 확인한다(다르면 중단).
  - `--resume` 옵션(기본 실행은 기존과 같음). 재개 시 train·evalfull·export_best·parity·cands 산출물을 해시·내용으로 확인(학습 입력 데이터 해시, 시드·행 수, 체크포인트 SHA256, parity 기준, `best==epoch_best`)하고 통과해야만 재사용한다. 하나라도 어긋나면 종료 코드 5 로 중단.
  - 이후 단계는 정상 종료 때만 `stages/<name>.done.json`(출력 해시)을 남기고, 재개 때 표식과 해시가 모두 맞는 단계만 건너뛴다. 실패한 `export_*` 의 부분 산출물 폴더는 그 단계 재실행 전에만 지운다.
  - `--resume` 없이 런 폴더가 이미 있으면 종료 코드 4 로 중단(덮어써 재학습하는 일 방지). 이미 `pipeline_done.json` 이 있는 런은 두 모드 모두 건너뛴다.
  - `pipeline_done.json` 은 모든 단계가 정상 종료된 뒤에만 쓰며, `resumed`, `reused_stages`, 후보별 `eval_full.json` 해시를 기록한다.

## 바뀌지 않은 것
모델 구조·하이퍼파라미터, 학습 데이터, Validation·Development 평가셋, B-new 체크포인트 선정 규칙, G1~G12·R1~R4·R11-S, 종합 후보 선정 규칙, `train_ner.py`, 공용 NER 서비스 코드.

## 영향 범위와 시험
- 시드 7: 재학습 없음(체크포인트·train_log·eval_full·parity·cand_info 재사용). 시드 13·21·42·99: 새 단계 `evalfull_<name>` 이 추가된 같은 파이프라인.
- `ml/ner/tests/test_r11_runner_resume.py` 13건, `test_r11_derive.py` 기존 포함 통과. 전체 `ml/ner/tests`: 3건 실패는 `known_test_failures.md` 의 기존 실패와 동일.

## 코드 버전 이력(실험 이력 명시)
- 시드 7의 학습·evalfull·export_best·parity·cands 산출물: 코드 `r11-freeze-1`(`c8ab3ad1`)로 생성.
- 시드 7의 후보 평가·내보내기·Validation 평가·B-new 선정·개발 평가와 시드 13·21·42·99 전체: 코드 `r11-freeze-2`로 생성.
- 두 버전에서 `train_ner.py`, 평가·선정·게이트 코드, 데이터, 기준선 해시는 같고 실행기(`run_r11_B.py`)와 `r11_derive.py` 만 다르다.
- freeze-2 추가 변경: 시드마다 디스크(6GB)·GPU 게이트 재확인(기존에는 시작 전 한 번과 RAM 만 시드별 확인).
