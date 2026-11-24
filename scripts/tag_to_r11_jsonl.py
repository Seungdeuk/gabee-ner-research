"""R11 독립 평가셋 입력 변환기: `tag_to_r9_jsonl.py` 에 R11 선택 필드(pair_id, role, pair_word, mix_kind)를 그대로 넘기도록 확장한 얇은 래퍼.

사용: python tag_to_r11_jsonl.py --in marked.jsonl --out data.jsonl [--source synthetic_gpt]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tag_to_r9_jsonl as base  # noqa: E402

base.PASS_THROUGH = tuple(base.PASS_THROUGH) + ("pair_id", "role", "pair_word", "mix_kind")

if __name__ == "__main__":
    sys.exit(base.main())
