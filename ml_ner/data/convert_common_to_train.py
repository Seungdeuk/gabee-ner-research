"""공통 JSONL(R9 규격) -> 기존 학습 파이프라인 행 변환기. 운영 서비스 코드는 건드리지 않는다.

공통 원본은 그대로 두고 이 변환기만 학습 형식을 만든다. 규칙:
- 제외 표시(``exclude: true``) 문장은 건너뛴다(1글자 이름 문장 등). 1글자 PERSON 이 라벨된 문장도 건너뛴다(음성으로 바꾸지 않는다).
- PERSON, PERSON_ALIAS, ADDRESS 는 NER 라벨 그대로. PHONE, EMAIL, ACCOUNT 는 규칙 탐지기 몫이라 NER 라벨은 O 로 두고 ``rule_spans`` 에 기록한다.
- ``context_entities`` 의 LOCATION, ORGANIZATION 은 NER 학습 라벨로 쓴다(서비스는 예측을 버리지만 학습에서는 PERSON 과의 대비 신호).
- 정책 r9-1.1: 이름+가족호칭은 이름만 PERSON, 호칭은 라벨 없음, PERSON_ALIAS 는 별명 자체만. 위반 행은 변환하지 않고 오류를 낸다.
- 토큰 단위 BIO 는 기존 ``ml/ner/train/tokenize_align.py`` 가 만든다. 여기서는 검증용 글자 단위 BIO(``char_bio``)와 역변환(``bio_to_spans``)을 제공한다.
"""

import argparse
import json
import sys
from pathlib import Path

NER_LABELS = ("PERSON", "PERSON_ALIAS", "ADDRESS", "LOCATION", "ORGANIZATION")
RULE_KIND = {"PHONE": "PHONE", "EMAIL": "EMAIL", "ACCOUNT": "ACCOUNT"}


KIN_TERMS = frozenset("엄마 아빠 어머니 아버지 어머님 아버님 할머니 할아버지 외할머니 외할아버지 누나 언니 오빠 형 동생 삼촌 이모 고모 외삼촌 이모부 고모부 큰아들 작은아들 큰딸 막내딸 막내 아들 딸 시어머니 장모님 큰누나 작은이모 큰엄마 작은엄마 외숙모".split())
KIN_MULTI = frozenset(k for k in KIN_TERMS if len(k) >= 2)


def kin_inside(value):
    if value in KIN_TERMS or any(t in KIN_TERMS for t in value.split()):
        return True
    return any(value.endswith(k) and len(value) > len(k) for k in KIN_MULTI)


def policy_issues(r):
    """정책 r9-1.1 위반(별칭·이름 구간에 가족 호칭이 들어 있음) 목록. 있으면 변환하지 않고 호출 쪽이 오류로 처리한다."""
    return [s["label"] for s in r.get("spans", []) if s.get("label") in ("PERSON", "PERSON_ALIAS") and kin_inside(s.get("value", ""))]


def is_one_char_person(r):
    return any(s.get("label") in ("PERSON", "PERSON_ALIAS") and len(s.get("value", "")) == 1 for s in r.get("spans", []))


def convert_row(r, split="train"):
    """공통 행 하나를 학습 행으로. 건너뛰면 None."""
    if r.get("exclude") or is_one_char_person(r):
        return None
    if policy_issues(r):
        raise ValueError(f"{r.get('id')}: PERSON_ALIAS/PERSON 구간에 가족 호칭이 포함됨(정책 r9-1.1 위반)")
    spans, rule = [], []
    for s in r.get("spans", []):
        if s["label"] in NER_LABELS:
            spans.append({"label": s["label"], "start": s["start"], "end": s["end"]})
        elif s["label"] in RULE_KIND:
            rule.append({"kind": RULE_KIND[s["label"]], "start": s["start"], "end": s["end"]})
    for ce in r.get("context_entities") or []:
        if ce["label"] in NER_LABELS:
            spans.append({"label": ce["label"], "start": ce["start"], "end": ce["end"]})
    spans.sort(key=lambda x: x["start"])
    row = {"id": r["id"], "text": r["text"], "spans": spans, "source": "synth", "template_id": r.get("template_id") or f"r9-{r['category']}",
           "category": r["category"], "split": split, "augment": True, "r9": True}
    if rule:
        row["rule_spans"] = rule
    return row


def char_bio(text, spans):
    """글자 단위 BIO 태그 목록(겹치는 span 은 오류)."""
    tags = ["O"] * len(text)
    for s in sorted(spans, key=lambda x: x["start"]):
        if any(t != "O" for t in tags[s["start"]:s["end"]]):
            raise ValueError("겹치는 span")
        tags[s["start"]] = f"B-{s['label']}"
        for i in range(s["start"] + 1, s["end"]):
            tags[i] = f"I-{s['label']}"
    return tags


def bio_to_spans(tags):
    spans, cur = [], None
    for i, t in enumerate(tags + ["O"]):
        if t.startswith("B-") or t == "O" or (t.startswith("I-") and (cur is None or cur["label"] != t[2:])):
            if cur:
                spans.append({"label": cur["label"], "start": cur["start"], "end": i})
                cur = None
            if t.startswith("B-"):
                cur = {"label": t[2:], "start": i}
            elif t.startswith("I-"):
                cur = {"label": t[2:], "start": i}
    return spans


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", default="train")
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in a.inp.read_text(encoding="utf-8").splitlines() if l.strip()]
    out, skipped = [], 0
    for r in rows:
        c = convert_row(r, a.split)
        if c is None:
            skipped += 1
        else:
            out.append(c)
    a.out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    print(json.dumps({"in": len(rows), "out": len(out), "skipped": skipped}))


if __name__ == "__main__":
    sys.exit(main())
