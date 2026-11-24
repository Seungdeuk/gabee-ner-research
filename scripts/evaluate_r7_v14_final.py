"""R7(+B) 최종 후보 v14 1회 검증 — 읽기 전용. evaluate_r7_4way.py 의 로직을 그대로 재사용하고
BUCKETS/ONE_CHAR_BUCKET 만 v14 파일로 바꾼다. 55절 8개 표준 지표를 실사용·중의성·스트레스로 나눠
보고하고, 표본수 가중 없이 세 영역 동일 비중(1/3) 매크로 평균을 통합 지표로 쓴다(사전에 고정한
기준). 1글자 이름은 통합에 넣지 않고 정책 지표로만 별도 보고한다. `ner.py`/`merge_ner`/모델/DB/
설정 미수정, v14 미수정, 재학습 없음, 외부 LLM 미호출.
"""

import argparse
import json
from pathlib import Path

import evaluate_r7_4way as ev4

V14 = ev4.EVAL_ROOT / "style_v14_locked"
ev4.BUCKETS = {
    "실사용": [V14 / "v14_realistic.jsonl"],
    "중의성": [V14 / "v14_ambiguity.jsonl"],
    "스트레스": [V14 / "v14_stress.jsonl"],
}
ev4.ONE_CHAR_BUCKET = V14 / "v14_one_char.jsonl"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--r6-candidate", required=True)
    ap.add_argument("--r7-candidate", required=True)
    ap.add_argument("--r6-fp12", default="c59b174ba126")
    ap.add_argument("--r7-fp12", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    bucket_rows = {b: ev4.load_rows(paths) for b, paths in ev4.BUCKETS.items()}
    one_char_rows = ev4.load_rows([ev4.ONE_CHAR_BUCKET])
    for b, rows in bucket_rows.items():
        print(f"[{b}] {len(rows)}행 로드")
    print(f"[1글자] {len(one_char_rows)}행 로드")

    combos = [
        ("R6+기존병합", a.r6_candidate, a.r6_fp12, "normal"),
        ("R6+B", a.r6_candidate, a.r6_fp12, "b"),
        ("R7+기존병합", a.r7_candidate, a.r7_fp12, "normal"),
        ("R7+B", a.r7_candidate, a.r7_fp12, "b"),
    ]
    out = {}
    for key, model_dir, fp12, mode in combos:
        out[key] = ev4.run_combo(key, model_dir, fp12, mode, bucket_rows, one_char_rows)
        c = out[key]["통합(매크로평균)"]
        oc = out[key]["1글자이름(정책, 통합 미포함)"]
        print(f"{key}: 통합 F1={c['final_prf']['f1']} 전체노출={c['exposure']['total_rate']} "
              f"| 실사용 전체노출={out[key]['실사용']['exposure']['total_rate']} "
              f"중의성 FPR={out[key]['중의성']['nonperson_fpr']} "
              f"스트레스 전체노출={out[key]['스트레스']['exposure']['total_rate']} "
              f"| 1글자 완전노출={oc['full_exposure_rate']}(정책, 별도)")

    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
