"""R11 단어 목록 진단(건수만): 형식·가드·기존 데이터 필터에서 몇 개가 왜 빠지는지."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_r9_data as b9
import build_r11_data as m

b9.EXISTING_DIR = m.b10.R9_DIR
g = m.Guards11()
for role in ("train", "val", "dev"):
    raw = json.loads((m.FR / "words" / f"{role}.json").read_text(encoding="utf-8"))
    c = Counter()
    for w in dict.fromkeys(x.strip() for x in raw):
        if not (2 <= len(w) <= 3):
            c["len"] += 1
        elif not re.match(r"^[가-힣]+$", w):
            c["not_hangul"] += 1
        elif not g.name_ok(w, allow_common=True):
            c["guard_name_ok_false"] += 1
            if w in g.dev_exact or w in g.dev_words:
                c["  dev_blocks"] += 1
        elif role != "train" and (w in g.seen_words or w in g.r9_given):
            c["seen_in_train(R9/R10)"] += 1
        else:
            c["ok"] += 1
    print(role, len(raw), dict(c))
