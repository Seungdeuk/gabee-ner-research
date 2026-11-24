# ml/ner - 개인정보 NER(PERSON · PERSON_ALIAS · ADDRESS · LOCATION · ORGANIZATION) 학습 코드

서비스 코드(`apps/`)와 분리된 학습용 코드다. 서비스는 이 폴더를 import 하지 않고, 이 폴더도 `apps/` 를 import 하지 않는다.
전화·이메일·카드·주민번호·계좌·비밀번호는 규칙 탐지기 몫이라 NER 대상이 아니다. 라벨 정의는
`apps/api/app/shared/privacy/eval/baseline_v2/LABELING_GUIDE.md` 를 그대로 따른다(엄마·아빠 같은 호칭만은 라벨 없음, **금융기관명(은행)은 ORGANIZATION + `context_sensitive=true`**,
상세 주소는 ADDRESS, 랜드마크·지역은 LOCATION, `이름+가족호칭`·별명은 PERSON_ALIAS).

## 환경

- 학습용 venv 만 쓴다: `<VENVS>\gabee-ner\Scripts\python.exe` (프로젝트 `.venv`/`node_modules` 는 건드리지 않는다. 새 패키지 설치 금지).
- `HF_HOME=<VENVS>\hf-cache` (모델·KLUE 캐시). 오프라인이면 `HF_HUB_OFFLINE=1`.
- 모델 가중치·체크포인트·산출 데이터는 **저장소 밖** `<VENVS>\ner-runs\` (`NER_RUNS_DIR` 로 변경). 저장소에는 코드·작은 설정만 둔다.
- 모델: 1차 `monologg/koelectra-small-v3-discriminator`, 비교 `klue/roberta-small`.

## 실행 순서 (저장소 루트에서)

```bash
export HF_HOME='<VENVS>\hf-cache'
PY=<VENVS>/gabee-ner/Scripts/python.exe
$PY ml/ner/data/build_ner_data.py                       # 데이터 변환·합성·누출 검사 -> ner-runs/data/*.jsonl, stats.json, leak_report.json
$PY ml/ner/train/tokenize_align.py --out align.json     # 정렬 오류율·잘림 비율 (max_len 128/256)
$PY ml/ner/train/train_ner.py --train-files synth_train:350 baseline_train:150 --val-files synth_val baseline_val --epochs 3   # 파이프라인 확인용
$PY ml/ner/train/evaluate_ner.py --model-dir <run>\best --data baseline_val                # val 평가(기본)
$PY ml/ner/train/evaluate_ner.py --model-dir <run>\best --data baseline_test --final       # 최종 1회만
```

시험: `$PY -m pytest ml/ner -q` (이 venv 에는 pytest 가 없어 `PYTHONPATH` 로 pytest 만 복사해 둔 shim 폴더를 지정해야 한다. 아래 "시험" 참고).

## 데이터 출처·라이선스

| 출처 | 내용 | 라이선스/비고 |
|---|---|---|
| KLUE-NER (`klue/klue`, ner) | 뉴스 문장. PS->PERSON, LC->LOCATION, OG->ORGANIZATION, DT/TI/QT->O | CC BY-SA 4.0. **원문은 저장소에 두지 않는다**(HF 캐시에서 매번 변환, 산출 jsonl 은 저장소 밖). 뉴스 도메인이라 대화체와 다르고 PS 에 유명인 이름이 있다(학습용 외부 데이터) |
| Baseline v2 train/val | `baseline_v2/{train,val}.jsonl` 의 NER 라벨 5종만 사용(나머지 span 은 O) | 합성 안전 픽스처(저장소 내) |
| 합성 가족 대화 | `data/synth.py` 의 문장 틀 116종 x 값 풀. Baseline 과 독립 생성 | 전부 합성. 풀의 이름은 흔한/드문 이름 조합이며 실존 인물과 우연히 같을 수 있다 |

## 누출 방지 규칙

1. **test.jsonl 은 학습·검증에 쓰지 않는다.** 읽는 곳은 `data/leak_guard.load_test_guard` 하나뿐이고, 결과는 (a) 합성 값 풀에서 뺄 이름·기관·주소·지명 차단 목록과 (b) 정규화 문장·문장 틀 비교 집합으로만 쓴다. test 문장은 어떤 산출물에도 들어가지 않는다. 평가 스크립트는 `--final` 없이는 test 를 거부한다. 시험이 이를 보장한다.
2. 합성 train/val 은 **틀(template) 단위로 분리**(카테고리별 4번째 틀이 val)하고, 이름·별명·도로·동·단지·역·지명·기관 접두 값 풀도 train/val **서로소**로 나눈다(성씨는 공유).
3. 합성 문장은 test 와 정규화 문장 중복 0, 문장 틀 중복 0(NER 라벨 기준과 전체 라벨 기준 둘 다)이어야 한다. test 에 나온 이름 문자열·이름 부분(성 뗀 부분)·기관/주소/지명 값(서로 포함 관계 포함)은 값 풀에서 제외하고, 생성된 span 중 같은 값이 있으면 실패한다.
4. 하나라도 걸리면 `build_ner_data.py` 가 종료 코드 1 로 실패한다. 보고서는 `leak_report.json`.
5. val 지표를 보고 하이퍼파라미터를 바꿔도 되지만 test 는 최종 1회(`--final`)만 쓴다.

## 정렬 정책 (`train/tokenize_align.py`)

토큰이 span 과 한 글자라도 겹치면 그 span 의 토큰: span 첫 토큰 B, 이후 I, 나머지 O, 특수 토큰은 -100. 기본 `policy=all` 은 span 안 모든 서브워드에 B/I 를 준다.
`policy=first` 는 단어 중간 서브워드(`##`)를 -100 으로 둔다. 예측 복원은 첫 토큰 시작~마지막 토큰 끝(offset_mapping)이다.
한계: 한국어는 이름에 조사가 한 토큰으로 붙는 경우(`구예은이`)가 있어 토큰 경계가 정답 경계와 다르면 exact 복원이 불가능하다(정렬 상한 보고서 참고).

## 시험

```bash
PYTHONPATH='<VENVS>\ner-runs\pytest_shim' $PY -m pytest ml/ner -q -p no:cacheprovider
```

`pytest_shim` 은 프로젝트 `.venv` 의 pytest 관련 순수 파이썬 패키지를 읽기 전용으로 복사한 폴더다(저장소 밖). `apps/api` 의 pytest 와는 별개이며 서로 영향이 없다.

## 학습 실험 설정 (KLUE 혼합 비율 x 모델 비교)

실행: `python ml/ner/train/run_matrix.py --seeds 42 7` (6 run x 시드, 순차) -> `evaluate_full.py --run-dir <run>` -> `aggregate_results.py`.
규칙 탐지기 비교용 예측은 프로젝트 `.venv` 로 `rule_baseline_preds.py` 를 실행해 만든다(ml/ner 에서 apps 를 읽는 유일한 파일, 비교용).

| 항목 | 값 | 근거 |
|---|---|---|
| max_len | 128 | Baseline 잘림 2.2%(train 12문장, 손실 span 2, val 4문장), 합성·KLUE 는 잘림 0. 256 은 Baseline val 에 보조 평가만 한다 |
| 정밀도 | fp16 autocast(GradScaler) | RTX 3050 6GB(가용 약 5GB) |
| 시드 | 42 (분산 확인용 7) | 재현성. 학습 순서·KLUE 부분 샘플·초기화 모두 시드 고정 |
| 옵티마이저 | AdamW lr 5e-5, 가중치 감쇠 0.01(bias·LayerNorm 제외), 선형 감소, warmup 10%, grad clip 1.0 | small 모델 표준값 |
| 배치 | 32 | VRAM 여유 확인(최대 사용량은 결과 JSON) |
| 에폭 | 최대 8, early stopping patience 2 | 기준 = **Baseline val** char-span overlap micro F1(`--select-on baseline_val --select-metric overlap`), 같은 값이면 앞 에폭 유지 |
| 정렬 | policy=all | 정렬 보고서 참고 |

**KLUE 혼합 비율 정의**: 학습 집합 = Baseline train(540) + 합성 train + KLUE train 의 시드 고정 무작위 부분 샘플.
N = Baseline train + 합성 train 문장 수, 비율 r 에 대해 KLUE 문장 수 K = round(N * r / (1 - r)). 그러면 K / (N + K) = r(전체 학습 **문장** 수 대비,
span·토큰 비율이 아니다). r = 0 / 0.1 / 0.3. 검증 집합에는 KLUE 를 쓰지 않는다. 다른 비율의 KLUE 표본은 같은 시드지만 `random.sample(k)` 의 k 가 달라 서로 부분집합이 아니다.

**평가는 val 만**: Baseline val(주 지표), 합성 val(보조). Baseline test 는 이번 실험 어디에도 쓰지 않았다(`--final` 미사용). 주의: best 에폭을 Baseline val 로 고르므로 Baseline val 수치는 약간 낙관적이다(선택 편향).

## 라벨 정책 (2차 개정: 금융기관명은 NER 대상 아님)

1차 실험에서 은행명을 ORGANIZATION(+context_sensitive)으로 라벨했다가 **되돌렸다**: 은행명은 라벨 없음(O), 계좌는 규칙 탐지기(ACCOUNT: 은행명·계좌 문맥+숫자열)가 잡는다.
금융 외 기관(병원·학교·회사 등)은 ORGANIZATION, `context_sensitive` 는 병원·약국·치과만 true. 은행 문장은 합성에서 음성(어려운 음성)으로만 쓴다.

## 모호 사례 제거 (Baseline 생성기 + 합성 틀, train/val/test 일관)

| 대상 | 제거/변경 | 사유 |
|---|---|---|
| 합성 틀 `내일 {P}과/와 약속` | `{P}하고` 로 변경 | 이름+`과`(양지안과, 오담과)가 의원/조사로 모호 |
| 합성 틀 `{ADS} 근처에 주차했어`, `{ADR} 근처 {OP}에...` | `에 주차했어`, `에 있는 {OP}에` 로 변경 | `...근처` 포함 주소의 경계 불명확 |
| 합성 틀 `집 근처 마트에서 장 봤어` | 삭제 | 붙여쓰면 `집근처마트` 가 기관인지 모호 |
| 합성 별명 풀 | 보리·도토리·땅콩이·모찌·감귤이·우리왕자·막내공주·아기곰·복덩이·새싹이·꼬맹이 등 일반명사형 삭제 | 별명/이름/일반 명사 모호 |
| 합성 한 글자 이름 | 건·결·담·솔·별·율·온·겸·찬·휘 삭제 | `조건`, `한결`, `오담과` 처럼 일반 단어/조사와 겹침 |
| 붙여쓰기 노이즈 | 라벨 구간과 맞닿은 공백은 유지 | `배달주소연산동 292` 처럼 경계가 사라져 span 이 모호해짐 |
| Baseline 이름 풀 | 하늘·나비·태양·세아 -> 서하·하라·태윤·서린 | 일반 명사/기업명과 겹치는 이름 |
| Baseline 별명 풀 | 보리·초코·별이·달이·봄이·하늘이·꿀이·하루·두리·마루·새롬·다솜 등 -> 밤톨이·똘이·찡이·뽀미·쭈니·쭈쭈·달봉이·새롬이·다솜이·꾸꾸·몽실이·쫑이 | 일반 명사형 별명 모호 |

## 오탐 감소 보강 (augment)

`build_ner_data.py --augment 1`: 새 틀 47종(가족 호칭 음성 `kin_neg2`, 일반 명사 음성 `generic_neg`, 일반 장소 명사 음성 `place_neg`, 어려운 양성 `person_given2`·`name_hard`·`alias_hard`·`org_hard`). `--augment 2`: 처음 보는 금융기관명 음성(`bank_neg2`, 9종)과 기사체(`news_title`, 10종, apps/api NER_VALIDATION.md 12절)를 더한다. `--augment 3`: 가족 메신저 다인칭 문장(`family_msg`, 20종, 한 문장에 사람 2명 이상 섞인 틀) — 메신저체 완전미탐 보강 2차 시도(NER_VALIDATION.md 17절).
train/val 틀·값 분리와 test 누출 검사는 그대로(보강 틀도 test 의 문장 틀과 같으면 건너뛴다). Baseline val/test 에 나오는 가족 호칭·송장·박수·강아지는 평가의 일반화 측정을 흐리지 않도록 보강 어휘에서 일부러 뺐다.
합성 val 은 보강 전후를 같은 틀로 비교하도록 `synth_val_core.jsonl`(보강 전 틀)과 `synth_val_aug.jsonl`(보강 틀)로 나눠 쓴다.
