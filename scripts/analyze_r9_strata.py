"""R9 층별·유형별 분석(개발 평가셋 전용, 봉인 평가셋 미사용). 모델 하나에 대해 정답 구간마다 결과 유형(A1~D)과 층(kin_adjacent / 정책 중립)을 계산한다.

kin_adjacent: 정답 PERSON 구간 바로 뒤에서 (선택적 '이', 공백을 건너뛰고) 가족 호칭이 이어지는 구간. 그 밖은 정책 중립.
1글자 이름 문장은 제외(evaluate_r9_dev 와 같은 규칙). 출력: 세트별·층별 건수와 비율, 유형별 건수.
사용: python analyze_r9_strata.py --model LABEL DIR [FP12] ... --out strata.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_r8_failure_types as A  # noqa: E402
import evaluate_r9_dev as D9  # noqa: E402
import r8_eval_lib as lib  # noqa: E402

KIN = (
    "엄마",
    "아빠",
    "할머니",
    "할아버지",
    "이모",
    "고모",
    "삼촌",
    "언니",
    "오빠",
    "누나",
    "동생",
    "외할머니",
    "큰이모",
    "외삼촌",
    "큰누나",
    "막내",
    "형",
    "누님",
    "어머니",
    "아버지",
)
A.dev.load = D9.load_excluding_one_char  # 1글자 이름 문장 제외
MONITOR = Path(r"C:\Users\<user>\venvs\ner-runs\data_r9\r9_val_extra.jsonl")
# 모니터링 전용 층별 보고(844행): 후보·체크포인트 선택에 쓰지 않는다. 개발셋에는 kin_adjacent 가 없어 이 세트로만 층별을 본다.
A.SETS["r9_val_monitor"] = MONITOR


def kin_adjacent(text, end):
    rest = text[end : end + 12].lstrip(" ")
    if rest.startswith("이") and not rest.startswith(("이모", "이름")):
        rest = rest[1:].lstrip(" ")
    return rest.startswith(KIN)


def strata_for(label, model_dir, fp12):
    cases, neg_words = A.run_model(label, model_dir, fp12)
    out = {"label": label, "sets": {}, "overall": {}}
    meta = {}
    for sname, path in A.SETS.items():
        for r in D9.load_excluding_one_char(path):
            for gi, g in enumerate(lib.gold_person(r)):
                meta[(sname, r["id"], gi)] = (
                    "kin_adjacent" if kin_adjacent(r["text"], g[1]) else "policy_neutral"
                )
    agg = defaultdict(Counter)
    for key, typ in cases.items():
        if not isinstance(key[2], int):
            continue
        stratum = meta.get(key, "policy_neutral")
        for scope in (key[0], "ALL"):
            for st in (stratum, "ALL"):
                c = agg[(scope, st)]
                c["n"] += 1
                c["exact_ok"] += typ == "ok"
                c["full_exposed"] += typ in ("A1", "A2", "A3", "A4", "A5")
                c["partial_exposed"] += typ in ("B1", "B2")
                c["boundary_wrong"] += typ in ("C1", "C2")
                c["type_" + typ] += 1
    for (scope, st), c in agg.items():
        n = c["n"]
        out["sets"].setdefault(scope, {})[st] = {
            "n": n,
            "exact_rate": round(c["exact_ok"] / n, 4),
            "full_exposure": round(c["full_exposed"] / n, 4),
            "partial_exposure": round(c["partial_exposed"] / n, 4),
            "total_exposure": round((c["full_exposed"] + c["partial_exposed"]) / n, 4),
            "types": {k[5:]: v for k, v in c.items() if k.startswith("type_")},
        }
    negs = Counter(v for k, v in cases.items() if k[2] == "neg")
    fps = sum(1 for k, v in cases.items() if v == "Dfp")
    out["overall"] = {
        "negative_sentences_flagged_D": negs.get("D", 0),
        "negative_sentences_ok": negs.get("ok", 0),
        "pure_fp_spans_Dfp": fps,
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", nargs="+", action="append", required=True, metavar=("LABEL", "DIR"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {}
    for m in a.model:
        res[m[0]] = strata_for(m[0], m[1], m[2] if len(m) > 2 else None)
        print("done", m[0], flush=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
