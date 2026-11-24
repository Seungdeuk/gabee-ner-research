"""인라인 태그 문장 -> R9 공통 JSONL 변환기(표준 라이브러리만 사용, 평가셋 작성자가 오프셋을 직접 계산하지 않게 한다).

입력(JSONL 한 줄 = 한 문장): ``{"id": "...", "category": "...", "marked": "[[PERSON|서아]]가 방금 왔어", ...선택 필드}``
태그: ``[[라벨|값]]``. 보호 라벨 PERSON, PERSON_ALIAS, ADDRESS, PHONE, EMAIL, ACCOUNT 는 ``spans`` 로, 비보호 문맥 개체 LOCATION, ORGANIZATION 은
``context_entities`` 로 나간다. ``mapped_kind`` 는 PERSON_ALIAS -> PERSON, 나머지는 같은 이름. ``is_negative`` 는 보호 span 이 없으면 true.
선택 필드(split_set, stress_types, primary_stress_type, template_id, style, scenario, batch, exclude, exclude_reason, source)는 그대로 넘긴다.
규격: docs/security/NER_R9_COMMON_SPEC.md

사용: ``python tag_to_r9_jsonl.py --in marked.jsonl --out data.jsonl [--source synthetic_gpt]``
"""

import argparse
import json
import re
import sys
from pathlib import Path

TAG = re.compile(r"\[\[([A-Z_]+)\|([^\[\]|]+)\]\]")
PROTECTED = {"PERSON": "PERSON", "PERSON_ALIAS": "PERSON", "ADDRESS": "ADDRESS", "PHONE": "PHONE", "EMAIL": "EMAIL", "ACCOUNT": "ACCOUNT"}
CONTEXT = {"LOCATION", "ORGANIZATION"}
PASS_THROUGH = ("split_set", "stress_types", "primary_stress_type", "template_id", "style", "scenario", "batch", "exclude", "exclude_reason")


def convert(row, default_source="synthetic"):
    marked = row["marked"]
    if "[[" in TAG.sub("", marked) or "]]" in TAG.sub("", marked):
        raise ValueError(f"{row.get('id')}: 짝이 맞지 않는 태그")
    text, spans, ctx, pos = [], [], [], 0
    cursor = 0
    for m in TAG.finditer(marked):
        text.append(marked[pos:m.start()])
        cursor += m.start() - pos
        label, value = m.group(1), m.group(2)
        start = cursor
        text.append(value)
        cursor += len(value)
        pos = m.end()
        if label in PROTECTED:
            spans.append({"start": start, "end": cursor, "label": label, "mapped_kind": PROTECTED[label], "value": value})
        elif label in CONTEXT:
            ctx.append({"start": start, "end": cursor, "label": label, "value": value})
        else:
            raise ValueError(f"{row.get('id')}: 알 수 없는 라벨 {label}")
    text.append(marked[pos:])
    text = "".join(text)
    out = {"id": row["id"], "category": row["category"], "text": text, "spans": spans, "is_negative": not spans, "source": row.get("source", default_source)}
    if ctx:
        out["context_entities"] = ctx
    for k in PASS_THROUGH:
        if k in row:
            out[k] = row[k]
    for s in spans:
        assert text[s["start"]:s["end"]] == s["value"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source", default="synthetic")
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in a.inp.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = [convert(r, a.source) for r in rows]
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(json.dumps({"in": len(rows), "out": len(out)}))


if __name__ == "__main__":
    sys.exit(main())
