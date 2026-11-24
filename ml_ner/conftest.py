"""pytest 설정: ml/ner 아래 모듈(ner_common, data/*, train/*)을 import 경로에 올린다. apps/api 의 pytest 와는 무관하다."""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE / "data", _HERE / "train", _HERE / "export"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
