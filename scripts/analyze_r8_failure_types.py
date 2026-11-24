"""R8: 서비스 모델과 후보를 A1~D 실패 유형별로 대응 비교한다(개발 평가셋 전용, 봉인 new180 은 열지 않는다).

유형(실패 유형 분석 아티팩트와 같은 정의):
  A1 후보 미생성 / A2 1글자 길이필터 탈락 / A3 ORG·LOC 오분류 / A4 신뢰도 부족 / A5 경계 충돌  (완전 노출 = 정답 글자가 하나도 안 가려짐)
  B1 끝 글자 누락(앞부분만 가림) / B2 앞 글자 누락(뒤쪽만 가림·밀림)  (부분 노출)
  C1 조사 흡수(정답을 다 가렸지만 뒤 조사까지 가림) / C2 그 밖의 과확장
  D  이름 없는 문장에서 PERSON 을 낸 문장(오탐) / Dfp 정답 있는 문장 안에서 어떤 정답과도 안 겹치는 PERSON 구간(순수 오탐)
정답이 여러 개인 문장은 모든 구간을 본다(r8_eval_lib 와 같은 채점).
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r8_eval_lib as lib  # noqa: E402
from r8_eval_lib import ner  # noqa: E402
import evaluate_r8_dev as dev  # noqa: E402
from evaluate_r6_candidate import diagnose_cause  # noqa: E402

JOSA = set("이가은는을를도만과와랑한에야아의로")
CAUSE_KEY = {"후보 미생성": "A1", "1자 길이필터 탈락": "A2", "신뢰도 부족": "A4"}
SETS = dict(dev.MAIN5)
SETS["ambiguity_probe"] = dev.SAFETY_FP["ambiguity_probe"]


def classify(rows):
    """{(set, id, gold_idx): type} 와 문장 단위 오탐 목록을 돌려준다."""
    out, fps = {}, []
    return out, fps


def run_model(label, model_dir, fp12):
    lib.set_model(model_dir, fp12)
    cases = {}
    neg_words = Counter()
    for sname, path in SETS.items():
        for r in dev.load(path):
            text = r["text"]
            golds = lib.gold_person(r)
            final = lib.final_person_spans(text)
            raw = ner.spans_for(text) or []
            if not golds:
                if final:
                    cases[(sname, r["id"], "neg")] = "D"
                    for a, b in final:
                        neg_words[text[a:b]] += 1
                else:
                    cases[(sname, r["id"], "neg")] = "ok"
                continue
            for gi, g in enumerate(golds):
                cov = set()
                for a, b in final:
                    cov.update(range(max(a, g[0]), min(b, g[1])))
                n = g[1] - g[0]
                leaked = n - len(cov)
                key = (sname, r["id"], gi)
                if leaked >= n:
                    cause = diagnose_cause(text, raw, g)
                    cases[key] = CAUSE_KEY.get(cause, "A3" if cause.startswith("ORG/LOC") else "A5")
                elif leaked > 0:
                    m = "".join("C" if (g[0] + i) in cov else "." for i in range(n))
                    cases[key] = "B1" if (m.startswith("C") and m.endswith(".")) else "B2"
                else:
                    ov = [(a, b) for a, b in final if max(a, g[0]) < min(b, g[1])]
                    exact = (g in final)
                    if exact:
                        cases[key] = "ok"
                    else:
                        right = text[g[1]:max(b for _, b in ov)] if ov else ""
                        left = text[min(a for a, _ in ov):g[0]] if ov else ""
                        if right and not left and right[0] in JOSA:
                            cases[key] = "C1"
                        else:
                            cases[key] = "C2"
            gs = set(golds)
            for p in final:
                if p in gs:
                    continue
                if not any(max(p[0], g[0]) < min(p[1], g[1]) for g in golds):
                    cases[(sname, r["id"], f"fp{p[0]}")] = "Dfp"
    return cases, neg_words


def compare(a, b):
    """a=서비스, b=후보. 유형별 건수와 대응 전이."""
    types = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "C1", "C2", "D", "Dfp"]
    ca = Counter(v for v in a.values() if v != "ok")
    cb = Counter(v for v in b.values() if v != "ok")
    fixed, broke, same = Counter(), Counter(), Counter()
    keys = set(a) | set(b)
    for k in keys:
        va, vb = a.get(k, "ok"), b.get(k, "ok")
        if va != "ok" and vb == "ok":
            fixed[va] += 1
        elif va == "ok" and vb != "ok":
            broke[vb] += 1
        elif va != "ok" and vb != "ok":
            same[(va, vb)] += 1
    return {"service": {t: ca.get(t, 0) for t in types}, "candidate": {t: cb.get(t, 0) for t in types},
            "fixed_by_candidate": {t: fixed.get(t, 0) for t in types}, "newly_broken_by_candidate": {t: broke.get(t, 0) for t in types},
            "still_failing_changed_type": {f"{x}->{y}": n for (x, y), n in same.items() if x != y}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--service", nargs="+", required=True, metavar=("DIR", "FP12"))
    ap.add_argument("--candidate", nargs="+", action="append", required=True, metavar=("LABEL", "DIR"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    svc, svc_neg = run_model("service", a.service[0], a.service[1] if len(a.service) > 1 else None)
    res = {"service_neg_words": dict(svc_neg.most_common(30))}
    for spec in a.candidate:
        label, d = spec[0], spec[1]
        cand, neg = run_model(label, d, None)
        res[label] = compare(svc, cand)
        res[label]["candidate_neg_words"] = dict(neg.most_common(30))
        print(label, json.dumps(res[label]["candidate"], ensure_ascii=False), flush=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
