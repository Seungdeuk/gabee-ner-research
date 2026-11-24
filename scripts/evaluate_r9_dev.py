"""R9 개발 평가: R8 개발 하네스(`evaluate_r8_dev.py`)와 같은 지표·게이트에 **1글자 PERSON 이름 문장 제외** 규칙을 더한 래퍼.

사전 등록 §1: 1글자 이름 문장은 평가·학습·검증에서 뺀다(음성으로 바꾸지 않는다). 정답 구간 중 하나라도 1글자인 문장을 모든 개발 평가셋에서 제외하고,
제외한 문장 수를 세트별로 출력한다. 게이트(G1~G4, 개선 요건)와 지표 정의는 R8 과 같다(사전 등록 §6, 완화 없음).
봉인 평가셋은 열지 않는다(이 스크립트는 개발 평가셋 경로만 가진다).

사용: python evaluate_r9_dev.py --model LABEL DIR [FP12] ... --out result.json   /   --select result.json
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate_r8_dev as D  # noqa: E402
import r8_eval_lib as lib  # noqa: E402

EXCLUDED = {}
_orig_load = D.load


def load_excluding_one_char(p):
    rows = _orig_load(p)
    keep = [r for r in rows if not any(e - s == 1 for s, e in lib.gold_person(r))]
    EXCLUDED[Path(p).name] = len(rows) - len(keep)
    return keep


D.load = load_excluding_one_char

if __name__ == "__main__":
    D.main()
    print(
        json.dumps(
            {"excluded_one_char_rows_by_file": {k: v for k, v in EXCLUDED.items() if v}},
            ensure_ascii=False,
        )
    )
