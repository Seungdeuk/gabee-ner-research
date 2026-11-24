"""R12b 데이터 작성 결과 검증기. 사전 등록 NER_R12B_PREREGISTRATION.md §3.

check_r12_marked.py 의 모든 검사(태그 문법·수량·쌍 구조·이름 풀·중복·직접 단서 비율)를 먼저 실행하고, 대조쌍 배치에는 R12 실패 원인이던
'낱말 뒤 형태 단서 비대칭'을 막는 검사를 더한다(배치 단위).
  ① 사람·명사 각 역할 문장에서 낱말 뒤 형태 묶음 하나가 60% 를 넘지 않는다
  ② 쌍 k1(같은 단어의 첫 쌍)의 사람·명사 문장은 같은 묶음을 쓴다(90% 이상)
  ③ 명사 문장의 10% 이상이 ⓕ(한테·에게·께)·ⓖ(아·야) 형태, 사람 문장의 20% 이상이 ⓐ(조사 없음) 형태
사용: python check_r12b_marked.py --spec SPEC.json --data OUT.jsonl
"""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_r12_marked as base  # noqa: E402

PARTICLE_CH = set("가는은를을랑한에도만의야아하와과로께")


def after_word(marked, pw, role):
    """낱말(pair_word) 바로 뒤 텍스트. 사람 문장은 PERSON 태그 안의 낱말 위치, 명사 문장은 첫 등장 위치."""
    out, pos, hit = [], 0, None
    for m in base.TAG.finditer(marked):
        out.append(marked[pos:m.start()])
        start = sum(len(x) for x in out)
        val = m.group(2)
        if role == "person" and m.group(1) in ("PERSON", "PERSON_ALIAS") and pw in val and hit is None:
            hit = start + val.index(pw) + len(pw)
        out.append(val)
        pos = m.end()
    out.append(marked[pos:])
    p = "".join(out)
    if hit is None:
        i = p.find(pw)
        if i < 0:
            return None
        hit = i + len(pw)
    return p[hit:]


def form_class(a):
    if a is None:
        return None
    if a[:1] == "이" and a[1:2] in PARTICLE_CH:  # 애칭 이 + 조사
        a = a[1:]
    if a[:2] in ("한테", "에게") or a[:1] == "께":
        return "f"
    if a[:1] in ("아", "야") and (len(a) == 1 or not a[1:2].isalpha() or a[1:2] in "야아"):
        return "g"
    if a[:2] in ("랑", "이랑") or a[:1] in ("랑", "와", "과") or a[:2] == "하고":
        return "e"
    if a[:1] in ("는", "은"):
        return "c"
    if a[:1] in ("를", "을"):
        return "d"
    if a[:1] in ("가", "이"):
        return "b"
    if a == "" or a[:1] in " ,.?!~ㅋㅠ…)\"'」』":
        return "a"
    return "h"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    a = ap.parse_args(argv)
    rc = base.main(["--spec", str(a.spec), "--data", str(a.data)])
    spec = json.loads(a.spec.read_text(encoding="utf-8"))
    if not spec.get("words"):
        return rc
    rows = [json.loads(x) for x in a.data.read_text(encoding="utf-8").splitlines() if x.strip()]
    errs = []
    cls = {"person": collections.Counter(), "noun": collections.Counter()}
    by_pair = collections.defaultdict(dict)
    for r in rows:
        role = r.get("role")
        if role not in ("person", "noun"):
            continue
        c = form_class(after_word(r["marked"], r["pair_word"], role))
        if c is None:
            continue
        cls[role][c] += 1
        by_pair[r["pair_id"]][role] = c
    stats = {}
    for role in ("person", "noun"):
        n = sum(cls[role].values())
        if not n:
            continue
        top, topc = cls[role].most_common(1)[0]
        stats[role] = {"n": n, "top": top, "top_share": round(topc / n, 3), "dist": dict(cls[role])}
        if topc / n > 0.60:
            errs.append(f"{role} 문장의 낱말 뒤 형태 '{top}' 가 {topc}/{n}={topc / n:.0%} (60% 이하여야 함)")
    n_noun, n_per = sum(cls["noun"].values()), sum(cls["person"].values())
    if n_noun and (cls["noun"]["f"] + cls["noun"]["g"]) / n_noun < 0.10:
        errs.append(f"명사 문장 중 한테·에게·아·야 형태 {(cls['noun']['f'] + cls['noun']['g'])}/{n_noun} < 10%")
    if n_per and cls["person"]["a"] / n_per < 0.20:
        errs.append(f"사람 문장 중 조사 없음 형태 {cls['person']['a']}/{n_per} < 20%")
    k1 = [(d.get("person"), d.get("noun")) for pid, d in by_pair.items() if pid.endswith("-k1") and len(d) == 2]
    if k1:
        match = sum(1 for p, nn in k1 if p == nn)
        stats["k1_matched"] = f"{match}/{len(k1)}"
        if match / len(k1) < 0.90:
            errs.append(f"쌍 k1 의 사람·명사 형태 일치 {match}/{len(k1)} < 90%")
    print(json.dumps({"form_balance": stats}, ensure_ascii=False))
    for e in errs:
        print("ERROR", e)
    return 1 if (rc or errs) else 0


if __name__ == "__main__":
    sys.exit(main())
