"""S0 파일럿(이어학습 Warm Start + 낮은 LR, 신규 데이터 없음, seed7) 평가 — 읽기 전용.

S0 = R6(s42) 체크포인트에서 이어학습, 데이터는 R6과 완전히 동일(data_r6_phrase, 신규 보강 없음),
lr만 5e-5 -> 1e-5, seed7 단일 파일럿(patience=2로 best_epoch=1에서 조기 종료, train_ner.py 자체
학습 루프로 1회만 학습, optimizer는 매 실행 새로 초기화됨 - 모멘텀 등 옵티마이저 상태는 R6에서
이어받지 않음, 모델 가중치만 이어받음).

세 갈래로 비교한다(전부 R6(그대로)/R6+B 기준선과 동일 경로):
  1) v13_locked + 회귀 평가셋 전체(55절 8개 표준지표, evaluate_r6_candidate.py/evaluate_final_
     r6b_candidate.py 재사용) - "기존 평가셋"은 개발·회귀 진단 목적으로만 쓰고 독립 최종 검증
     으로 주장하지 않는다(v13은 63절에서 이미 1회 소비됨).
  2) style_r7b_dev 4영역(실사용/무공백인접/중의성/스트레스) + 1글자 + 과잉병합 28건
     (evaluate_r7b_dev.py 재사용)
  3) 48절 ambiguity_probe 특정 3사례(보라색/미래/지성미) 직접 확인(evaluate_r7_ablation.py 재사용)

ner.py/merge_ner/운영 모델/DB/설정 미수정, 재학습 없음(이미 끝난 S0 체크포인트만 읽음), 외부 LLM
미호출.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate_r6_candidate as ev6  # noqa: E402
import evaluate_final_r6b_candidate as ev6b  # noqa: E402
import evaluate_r7b_dev as ev7b  # noqa: E402
import evaluate_r7_ablation as abl  # noqa: E402

R6_DIR = r"C:\Users\<user>\venvs\ner-runs\export\r6-candidate-s42"
R6_FP = "c59b174ba126"
S0_DIR = r"C:\Users\<user>\venvs\ner-runs\export\s0-warmstart-lowlr-s7"
S0_FP = "48256d3a88fa"

S0_META = {
    "시작 체크포인트": r"C:\Users\<user>\venvs\ner-runs\r6-koelectra-small-klue10-s42\best (R6 s42, fp12 c59b174ba126)",
    "학습 데이터": "data_r6_phrase (baseline_train+synth_train, klue_ratio 0.1) - R6과 완전히 동일, 신규 보강 0건",
    "학습량": "lr 1e-5(R6의 5e-5에서 1/5로 낮춤), batch 32, max_len 128, warmup 0.1, weight_decay 0.01, patience 2, seed 7(단일 파일럿) - 그 외 전부 R6과 동일",
    "실제 진행": "epoch 1(val 0.9560=R6 best_val_score와 소수점까지 동일)이 best로 선정, epoch 2(0.9325)·3(0.9383) 연속 악화로 조기 종료(stopped_epoch=3), 총 학습시간 30.6초",
    "optimizer 초기화 여부": "매 실행 AdamW를 새로 생성(train_ner.py 구조상 옵티마이저 상태 이어받기 미지원) - 모델 가중치만 R6에서 이어받고 모멘텀 등 옵티마이저 상태는 처음부터 다시 시작",
    "모델 구조·라벨·토크나이저 일치": "num_encoder_layers=12(R6과 동일), id2label/label2id 동일(nc.TAGS 고정), 토크나이저는 R6 best 디렉터리에 저장된 것을 그대로 로드(수정 없음) - 로드 시 classifier 재초기화 경고 없음 확인(68절 스모크테스트와 동일 방식)",
}


def run_v13_and_regression():
    v13_rows = ev6.load_rows(ev6b.V13)
    ev6.REGRESSION_DATASETS = dict(ev6b.REGRESSION_DATASETS)
    out = {
        "R6+기존병합": ev6.run_model("R6+기존병합", R6_DIR, R6_FP, v13_rows),
        "R6+B": ev6b.run_r6b(R6_DIR, R6_FP, v13_rows),
        "S0+기존병합": ev6.run_model("S0+기존병합", S0_DIR, S0_FP, v13_rows),
        "S0+B": ev6b.run_r6b(S0_DIR, S0_FP, v13_rows),
    }
    return out


def run_style_r7b_dev():
    bucket_rows = {b: ev7b.load_rows(p) for b, p in ev7b.BUCKETS.items()}
    one_char_rows = ev7b.load_rows(ev7b.ONE_CHAR)
    combos = [
        ("R6+기존병합", R6_DIR, R6_FP, "normal"),
        ("R6+B", R6_DIR, R6_FP, "b"),
        ("S0+기존병합", S0_DIR, S0_FP, "normal"),
        ("S0+B", S0_DIR, S0_FP, "b"),
    ]
    return {key: ev7b.run_combo(key, model_dir, fp12, mode, bucket_rows, one_char_rows)
            for key, model_dir, fp12, mode in combos}


def run_specific_48():
    amb_rows = [json.loads(l) for l in abl.AMB.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for key, model_dir, fp12 in [("R6", R6_DIR, R6_FP), ("S0", S0_DIR, S0_FP)]:
        abl.ner.reset()
        abl.settings.privacy_ner_mode = "required"
        abl.settings.privacy_ner_model_dir = model_dir
        abl.settings.privacy_ner_threads = 2
        status = abl.ner.ner_status()
        assert status["fingerprint"][:12] == fp12, (key, status["fingerprint"][:12])
        out[key] = abl.check_specific_ambiguity(amb_rows)
    return out


def main():
    out = {
        "s0_meta": S0_META,
        "v13_and_regression": run_v13_and_regression(),
        "style_r7b_dev": run_style_r7b_dev(),
        "specific_48_cases": run_specific_48(),
    }

    v = out["v13_and_regression"]
    print("=== v13(개발·회귀 진단용) + 48절 ===")
    for key in ["R6+기존병합", "R6+B", "S0+기존병합", "S0+B"]:
        o = v[key].get("v13_overall") or v[key]["v12_overall"]  # ev6.run_model은 "v12_overall" 이름을 그대로 쓴다(내용은 v13)
        amb = v[key]["ambiguity_probe_48"]
        print(f"{key}: exposure={o['exposure']['total_exposure_rate']} full={o['exposure']['full_exposure_rate']} "
              f"partial={o['exposure']['partial_exposure_rate']} masked={o['exposure']['fully_masked_rate']} "
              f"P/R/F1={o['final_prf']['precision']}/{o['final_prf']['recall']}/{o['final_prf']['f1']} "
              f"fpr(v13)={o['nonperson_fpr']} fpr(48절)={amb['nonperson_fpr']}")

    print("\n=== style_r7b_dev(실사용/무공백인접/중의성/스트레스 매크로 + 과잉병합) ===")
    for key, r in out["style_r7b_dev"].items():
        c = r["통합(매크로평균)"]
        print(f"{key}: 전체노출(macro)={c['exposure']['total_rate']} 오탐(macro)={c['nonperson_fpr_macro']} "
              f"과잉병합={r['total_overmerge']}/28 1글자노출={r['1글자이름(정책, 통합 미포함)']['full_exposure_rate']}")

    print("\n=== 48절 특정 3사례(보라색/미래/지성미) ===")
    for key, cases in out["specific_48_cases"].items():
        print(f"{key}: " + ", ".join(f"{rid.split('-')[1]}={'오탐' if v['flagged'] else '정상'}" for rid, v in cases.items()))

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "s0_pilot_eval.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
