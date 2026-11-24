"""R12 데이터 작성 결과(인라인 태그 JSONL) 검증기. 사전 등록 NER_R12_PREREGISTRATION.md §3.

사용: python check_r12_marked.py --spec SPEC.json --data OUT.jsonl
- 줄마다 JSON, 필수 필드, 태그 문법(`[[라벨|값]]`), 허용 라벨, 길이 8~150, 1글자 이름 금지, 같은 문장 중복·같은 틀 과다 반복 금지
- 정확한 문장 수, 유형별 수량(spec.mix)
- 대조쌍: pair_id 마다 person 1 + noun 1, pair_word 포함, 사람 문장에 단어를 포함한 PERSON 태그, 명사 문장에 PERSON 태그 없음, 단어별 수량, 배정 단어만 사용
- 이름: PERSON 값이 배정 풀(전체 이름·성 없는 이름·4자 이름)에 있어야 하고 같은 이름은 최대 2회(별명 문장은 PERSON_ALIAS 값 자유)
오류가 0 이면 종료 코드 0. 문장 원문은 오류 보고에 id 와 짧은 부분만 쓴다.
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

TAG = re.compile(r"\[\[([A-Z_]+)\|([^\[\]|]+?)\]\]")
ALLOWED = {"PERSON", "PERSON_ALIAS", "LOCATION", "ORGANIZATION", "ADDRESS", "PHONE", "EMAIL", "ACCOUNT"}
REQ = ("id", "marked", "primary_stress_type", "batch", "group")


def plain(marked):
    return TAG.sub(lambda m: m.group(2), marked)


def frame(marked):
    t = plain(marked).split()
    return " ".join(t[:2]) + " … " + (t[-1] if t else "")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    a = ap.parse_args(argv)
    spec = json.loads(a.spec.read_text(encoding="utf-8"))
    errs, warns = [], []
    rows = []
    for ln, line in enumerate(a.data.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as e:
            errs.append(f"줄 {ln}: JSON 오류 {e}")
    if len(rows) != spec["count"]:
        errs.append(f"문장 수 {len(rows)} != {spec['count']}")
    ids = collections.Counter(r.get("id") for r in rows)
    errs += [f"id 중복: {i}" for i, c in ids.items() if c > 1]
    pool = set(spec.get("full_names", [])) | set(spec.get("bare_names", [])) | set(spec.get("long_names", []))
    words = set(spec.get("words", []))
    type_cnt = collections.Counter(r.get("primary_stress_type") for r in rows)
    for t, n in spec.get("mix", {}).items():
        if type_cnt.get(t, 0) != n:
            errs.append(f"유형 {t}: {type_cnt.get(t, 0)} != {n}")
    for t in type_cnt:
        if t not in spec.get("mix", {}):
            errs.append(f"허용되지 않은 유형 {t}")
    seen_plain = collections.Counter()
    frames = collections.Counter()
    name_use = collections.Counter()
    pairs = collections.defaultdict(dict)
    word_roles = collections.defaultdict(collections.Counter)
    for r in rows:
        rid = r.get("id", "?")
        for k in REQ:
            if k not in r:
                errs.append(f"{rid}: 필드 {k} 없음")
        m = r.get("marked", "")
        if r.get("group") != spec["group"] or r.get("batch") != spec["batch"]:
            errs.append(f"{rid}: group/batch 가 사양과 다름")
        tags = TAG.findall(m)
        rest = TAG.sub("", m)
        if "[[" in rest or "]]" in rest:
            errs.append(f"{rid}: 태그 괄호가 맞지 않음")
        p = plain(m)
        if not 8 <= len(p) <= 150:
            errs.append(f"{rid}: 길이 {len(p)} (8~150)")
        seen_plain[p] += 1
        frames[frame(m)] += 1
        persons = [(lab, val.strip()) for lab, val in tags if lab in ("PERSON", "PERSON_ALIAS")]
        for lab, val in tags:
            if lab not in ALLOWED:
                errs.append(f"{rid}: 허용되지 않은 라벨 {lab}")
        for lab, val in persons:
            if len(val) < 2:
                errs.append(f"{rid}: 1글자 이름 {val!r}")
        role = r.get("role")
        if role in ("person", "noun"):
            pid = r.get("pair_id")
            pw = r.get("pair_word")
            if not pid or not pw:
                errs.append(f"{rid}: 쌍 필드(pair_id/pair_word) 없음")
                continue
            if pid in pairs and role in pairs[pid]:
                errs.append(f"{rid}: pair_id {pid} 의 {role} 가 중복")
            pairs[pid][role] = r
            word_roles[pw][role] += 1
            if words and pw not in words:
                errs.append(f"{rid}: 배정되지 않은 단어 {pw}")
            if pw not in p:
                errs.append(f"{rid}: pair_word {pw} 가 문장에 없음")
            if role == "person":
                if not any(val == pw or val.endswith(pw) or val.startswith(pw) for _, val in persons):
                    errs.append(f"{rid}: 사람 문장에 {pw} 를 포함한 PERSON 태그가 없음")
                if r.get("primary_stress_type") != "D1P":
                    errs.append(f"{rid}: person 인데 유형이 D1P 가 아님")
            else:
                if persons:
                    errs.append(f"{rid}: 명사 문장에 PERSON 태그가 있음")
                if r.get("primary_stress_type") != "D1":
                    errs.append(f"{rid}: noun 인데 유형이 D1 이 아님")
        else:
            if r.get("pair_id") or r.get("pair_word"):
                errs.append(f"{rid}: 쌍이 아닌 문장에 쌍 필드")
            if r.get("primary_stress_type") in ("FAM_NEG",) and persons:
                errs.append(f"{rid}: 음성 문장에 PERSON 태그")
            if r.get("primary_stress_type") != "FAM_NEG" and not persons and r.get("primary_stress_type") != "D1":
                errs.append(f"{rid}: 양성 유형인데 PERSON 태그가 없음")
            for lab, val in persons:
                if lab == "PERSON_ALIAS":
                    continue
                base = val
                if pool and base not in pool:
                    # 이름 뒤 '이' 애칭이 포함된 값이면 기본 이름을 찾아본다
                    if not (base.endswith("이") and base[:-1] in pool):
                        errs.append(f"{rid}: 배정 풀에 없는 이름 {val}")
                name_use[base] += 1
    for pid, d in pairs.items():
        if set(d) != {"person", "noun"}:
            errs.append(f"pair_id {pid}: person/noun 이 짝이 아님")
        else:
            a1, b1 = d["person"], d["noun"]
            if a1.get("pair_word") != b1.get("pair_word"):
                errs.append(f"pair_id {pid}: 두 문장의 pair_word 가 다름")
    if spec.get("per_word"):
        for w in words:
            for role in ("person", "noun"):
                if word_roles[w][role] != spec["per_word"][role]:
                    errs.append(f"단어 {w}: {role} {word_roles[w][role]} != {spec['per_word'][role]}")
    for p, c in seen_plain.items():
        if c > 1:
            errs.append(f"같은 문장이 {c}번: {p[:20]}…")
    for fr, c in frames.items():
        if c > 3:
            warns.append(f"같은 틀 {c}번: {fr}")
    for nme, c in name_use.items():
        if c > 2:
            errs.append(f"이름 {nme} 가 {c}번(최대 2)")
    if len(warns) > max(4, spec["count"] // 25):
        errs.append(f"같은 틀 반복 경고가 너무 많음({len(warns)})")
    direct_cue = re.compile(r"이름표|별명|이름이|이름은|불러|부르|애칭|닉네임|이름으로")
    pp = [r for r in rows if r.get("role") == "person"]
    if pp:
        weak = sum(1 for r in pp if not direct_cue.search(plain(r["marked"])))
        if weak / len(pp) < 0.30:
            errs.append(f"사람 문장 중 직접 단서 없는 문장 {weak}/{len(pp)} < 30%")
    print(json.dumps({"batch": spec["batch"], "rows": len(rows), "types": dict(type_cnt), "pairs": len(pairs), "errors": len(errs), "warnings": len(warns)}, ensure_ascii=False))
    for e in errs[:60]:
        print("ERROR", e)
    for w in warns[:10]:
        print("WARN", w)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
