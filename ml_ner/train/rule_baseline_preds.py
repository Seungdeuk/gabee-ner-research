"""규칙 탐지기(apps/api detectors.detect)가 Baseline val 에 내는 NER 라벨(PERSON·ADDRESS) 예측을 JSON 으로 덤프한다.

ml/ner 에서 apps 를 import 하는 **유일한 비교용 파일**이다(학습·평가 코드는 이 파일을 쓰지 않는다). 프로젝트 가상환경 python 으로 실행한다::

    <프로젝트 .venv>/Scripts/python ml/ner/train/rule_baseline_preds.py --out <ner-runs 폴더>/rule_preds_baseline_val.json

known_names 를 준 예측(탐지기 평소 사용 방식)과 주지 않은 예측(NER 과 같은 입력 조건)을 둘 다 낸다.
탐지기는 PERSON·ADDRESS 만 낼 수 있어 PERSON_ALIAS·LOCATION·ORGANIZATION 은 항상 미탐이다.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "apps" / "api"))

from app.shared.privacy.detectors import detect  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val", choices=["val"])  # test 는 이 스크립트로 읽지 않는다
    a = ap.parse_args()
    path = REPO / "apps" / "api" / "app" / "shared" / "privacy" / "eval" / "baseline_v2" / f"{a.split}.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        entry = {}
        for mode, names in (("with_known_names", tuple(r["known_names"])), ("no_known_names", ())):
            entry[mode] = [{"label": p.kind.value, "start": p.start, "end": p.end}
                           for p in detect(r["text"], names) if p.kind.value in ("PERSON", "ADDRESS")]
        out[r["id"]] = entry
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} 문장 -> {a.out}")


if __name__ == "__main__":
    main()
