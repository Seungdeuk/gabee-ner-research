"""relabel_alias_r9p.py 결과 독립 검증: 원문·id 불변, 변경은 PERSON_ALIAS 구간이 앞부분(이름)으로 줄어든 것뿐, 호칭 잔존·1글자 이름 없음.

사용: python verify_relabel_r9p.py --old-dir <data_r8> --new-dir <data_r9>
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import relabel_alias_r9p as R  # noqa: E402


def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def lab(s):
    return s.get("label") or s.get("kind")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-dir", required=True, type=Path)
    ap.add_argument("--new-dir", required=True, type=Path)
    a = ap.parse_args()
    kins = R.kin_list()
    tot = Counter()
    problems = []
    for f in R.RELABEL_FILES:
        old, new = rows(a.old_dir / f"{f}.jsonl"), rows(a.new_dir / f"{f}.jsonl")
        assert len(old) == len(new), f
        for ro, rn in zip(old, new):
            if ro["text"] != rn["text"] or ro["id"] != rn["id"]:
                problems.append((f, ro["id"], "text_or_id_changed"))
            if len(ro["spans"]) != len(rn["spans"]):
                problems.append((f, ro["id"], "span_count_changed"))
                continue
            for so, sn in zip(ro["spans"], rn["spans"]):
                if so == sn:
                    continue
                t = ro["text"]
                v, nv = t[so["start"]:so["end"]], t[sn["start"]:sn["end"]]
                tail = v[len(nv):].lstrip(" ")
                if not (lab(so) == "PERSON_ALIAS" and sn["start"] == so["start"] and sn["end"] < so["end"] and v.startswith(nv) and tail in kins
                        and lab(sn) in ("PERSON", "PERSON_ALIAS") and " " not in nv and len(nv) >= 2 and t[sn["end"]:so["end"]].lstrip(" ") == tail):
                    problems.append((f, ro["id"], "unexpected_span_change"))
                else:
                    tot[f"{f}:converted"] += 1
            for s in rn["spans"]:
                v = rn["text"][s["start"]:s["end"]]
                if lab(s) in ("PERSON", "PERSON_ALIAS") and any(v.endswith(k) and len(v) > len(k) and (" " not in k or v[len(v) - len(k) - 1] == " ") for k in kins):
                    problems.append((f, rn["id"], "kin_still_inside_" + lab(s)))
                if lab(s) == "PERSON" and len(v) == 1 and lab(s) != "":
                    pass
            if {k for k in rn} - {k for k in ro} - {"relabel_r9p"}:
                problems.append((f, ro["id"], "unexpected_field"))
    print(json.dumps({"converted_by_file": dict(tot), "problems": len(problems), "problem_examples": problems[:10]}, ensure_ascii=False))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
