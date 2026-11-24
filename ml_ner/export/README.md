# NER ONNX 내보내기 (서버 CPU 추론용)

서비스용 NER 기본 모델 KoELECTRA-small(보강 v2b, 시드 7)을 `onnxruntime` + `tokenizers` + `numpy` 만으로 돌리게 내보낸다.
서버에는 torch·transformers 가 없다. **모델 가중치(`*.onnx`)는 Git 에 올리지 않는다**(저장소 밖 산출 폴더).

## 산출 폴더 (`<VENVS>\ner-runs\export\koelectra-small-v2b\`)

| 파일 | 내용 |
|---|---|
| `model.onnx` | fp32, opset 17, 입력 `input_ids`·`attention_mask`(int64, `[batch, sequence]`), 출력 `logits`(`[batch, sequence, 11]`) |
| `model.int8.onnx` | 동적 INT8 양자화본(선택) — 권장하지 않음(아래) |
| `tokenizer.json` | HF fast tokenizer 저장본(`tokenizers.Tokenizer.from_file` 단독 로드, offset 반환) |
| `labels.json` | `id2label`(태그 11개), `labels`(NER 5종), `max_len`(128), `pad/cls/sep_token_id` |
| `model_card.json` | run_id·시드·체크포인트·데이터 버전(v2b 해시)·지표·라이선스 상태·sha256·생성 시각·onnx/ort 버전·시드 선택 근거 |

`token_type_ids` 는 입력에 없다(항상 0 이라 그래프에 고정). 라벨: PERSON, PERSON_ALIAS, ADDRESS, LOCATION, ORGANIZATION (BIO 태그 11개).

## 서버 배포

산출 폴더를 서버의 `data/models/ner/` 로 복사한다(필요한 파일: `model.onnx`, `tokenizer.json`, `labels.json`, `model_card.json`).
복사 뒤 `model_card.json` 의 `sha256` 과 파일을 대조한다. 서버 의존성은 이미 선언된 `onnxruntime`, `tokenizers`, `numpy` 뿐이다.

## 재현 명령 (저장소 루트)

```powershell
$env:HF_HOME='<VENVS>\hf-cache'; $NER='<VENVS>\gabee-ner\Scripts\python.exe'
$PRJ='<REPO>\apps\api\.venv\Scripts\python.exe'   # tokenizers+onnxruntime+numpy 만 있는 프로젝트 venv
$R='<VENVS>\ner-runs'; $E="$R\export"
# 1) 내보내기(+INT8, 모델 카드)                     — 학습 venv
& $NER ml/ner/export/export_onnx.py --run-dir $R\v2b-koelectra-small-klue10-s7 --out $E\koelectra-small-v2b
# 2) PyTorch·HF 참조값(val 분할만)                  — 학습 venv
& $NER ml/ner/export/dump_torch_reference.py --run-dir $R\v2b-koelectra-small-klue10-s7 --data-dir $R\data_v2b --export-dir $E\koelectra-small-v2b --out $E\verify-ref
# 3) 검증(토크나이저·로짓·span 일치율·지표)          — 프로젝트 venv (torch 없음)
& $PRJ -I -X utf8 ml/ner/export/verify_onnx.py --export-dir $E\koelectra-small-v2b --ref-dir $E\verify-ref --model-file model.onnx --out $E\verify_model.onnx.json
# 4) 속도·메모리(스레드 1/2/4 각각 별도 프로세스)
& $PRJ -I -X utf8 ml/ner/export/bench_onnx.py --backend onnx --export-dir $E\koelectra-small-v2b --threads 2 --ref $E\verify-ref\reference.json --out $E\bench\onnx_t2.json
& $NER ml/ner/export/bench_onnx.py --backend torch --run-dir $R\v2b-koelectra-small-klue10-s7 --threads 4 --ref $E\verify-ref\reference.json --out $E\bench\torch_t4.json
# 5) 시험(학습 venv; pytest shim 은 ml/ner/README.md 참고)
$env:PYTHONPATH="$R\pytest_shim"; & $NER -m pytest ml/ner -q -p no:cacheprovider
```

## 런타임 인터페이스 계약

참조 구현: `onnx_ner.py` 의 `OnnxNer` (서버 코드는 이 로직을 옮겨 쓰면 된다).

1. `tokenizers.Tokenizer.from_file("tokenizer.json")`; `no_truncation()`·`no_padding()` 으로 쓴다. `encode(text)` 의 `ids`·`offsets`·`special_tokens_mask` 를 쓴다.
   offset 은 **문자(유니코드 코드포인트) 단위**이고 end 는 배타적이다(파이썬 슬라이스와 같음). HF `AutoTokenizer` 와 id·offset 이 100% 일치한다.
2. `[CLS]`(id 2) 내용 토큰 `[SEP]`(id 3) 형태로 `input_ids`, 같은 길이의 1 로 된 `attention_mask`(패딩은 id 0, mask 0)를 int64 로 넣는다. **최대 길이 128**(특수 토큰 포함, 내용 토큰 126).
3. 출력 `logits[b, t, 11]` 의 argmax 가 태그 id. `labels.json` 의 `id2label["0".."10"]` 로 `O`/`B-X`/`I-X` 를 얻는다.
4. 디코딩: `B-X` 또는 앞 토큰과 라벨이 다른 `I-X` 에서 새 span 시작, 같은 라벨의 `I-X` 가 이어지면 확장, `O` 에서 종료. span = (첫 토큰 start, 마지막 토큰 end). 특수 토큰·`end<=start` 토큰은 건너뛴다.
5. 긴 글: 내용 토큰이 126 개를 넘으면 **슬라이딩 윈도우** — 크기 126, 이웃 겹침(overlap) 32 토큰(보폭 94). 겹치는 토큰은 윈도우 중앙에 가까운 쪽 로짓을 쓴다. 126 이하면 윈도우 1개(학습 입력과 동일).
   문장 분리는 런타임 몫이 아니다(문장 단위로 부르는 것이 가장 정확하다 — 긴 글 하나로 부를 때의 정확도는 verify 결과 참고).
6. 스레드: `SessionOptions.intra_op_num_threads`(2~4 권장), `inter_op_num_threads=1`, `CPUExecutionProvider`. 세션은 프로세스당 1개를 재사용하고 시작 시 한 번 워밍업한다(첫 호출이 느릴 수 있음).
7. 배치: 길이가 비슷한 문장끼리 묶어 패딩하면 처리량이 오른다(`predict_many`).

## 라이선스 주의 (서비스 투입 전 확인 필요)

- 기반 모델 `monologg/koelectra-small-v3-discriminator`: **가중치 라이선스 미확정**.
- 학습 데이터 일부 KLUE-NER: **CC BY-SA 4.0**(출처 표시, 동일조건변경허락) — 파생 모델 배포 조건 확인 필요.
- 합성 데이터·Baseline v2 는 자체 제작.
- 산출물에는 개인정보가 없지만(학습 문장은 가짜 데이터) 모델 파일은 저장소에 커밋하지 않는다.

## 시드·양자화 결정 근거

`model_card.json` 의 `seed_selection`(s7 vs s42, 같은 val)과 검증 결과 파일 `verify_*.json`, `bench/*.json`(산출 폴더) 참고.
