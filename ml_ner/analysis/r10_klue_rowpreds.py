"""KLUE 검증(보고 전용, 5,000행) 행 단위 예측 덤프. 학습 없음, 봉인셋 아님.

r10_rowpreds 의 모델 목록·후크를 재사용해 모델을 한 번에 하나씩 추론한다.
행마다: 정답 PERSON 구간, 최종 PERSON 구간, 원시 span(라벨 포함), 정답 구간 PERSON 확률(최대).
KLUE 본문은 저장하지 않고 id·구간·길이 정보만 남긴다(분석은 id 로 원본과 조인).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r10_rowpreds as R  # noqa: E402

KLUE_VAL = R.E.KLUE_VAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = R.E.rows_of(KLUE_VAL)
    for label, d in R.models():
        if a.only and label not in a.only:
            continue
        outp = a.out_dir / f"klue_rowpreds_{label}.jsonl"
        if outp.exists():
            print("SKIP(이미 있음)", label, flush=True)
            continue
        fp = R.lib.set_model(d, None)
        recs = R.run_rows(rows, "klue_val")
        tmp = outp.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps({"label": label, "model_dir": str(d), "fingerprint12": fp}, ensure_ascii=False) + "\n")
            for x in recs:
                f.write(json.dumps({k: v for k, v in x.items() if k != "stress"}, ensure_ascii=False) + "\n")
        tmp.replace(outp)
        print("DONE", label, fp, len(recs), flush=True)


if __name__ == "__main__":
    main()
