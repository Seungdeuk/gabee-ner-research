"""R9 신규 학습·검증 데이터 최종 누수 점검(독립 재검사). 봉인 평가셋은 프로그램 안에서만 읽고 건수만 출력한다.

점검: (1) 신규 학습 F-full(3,126) vs 검증(844) 문장·틀·이름 겹침, (2) 신규 전체 vs 개발 평가셋·기존 학습·봉인·new180 의 문장·골격·구간 값·이름 겹침,
(3) 이름 재사용 분포(최대 재사용 횟수 기록).
사용: python check_r9_train_leak.py --data-dir data_r9 --out report.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_r8_data as r8
import build_r9_data as B


def read(p):
    return [
        json.loads(x)
        for x in Path(p).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def names(rows):
    return [
        r["text"][s["start"] : s["end"]]
        for r in rows
        for s in r["spans"]
        if s["label"] in ("PERSON", "PERSON_ALIAS")
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tr, va = (
        read(a.data_dir / "r9_Ffull_train.jsonl"),
        read(a.data_dir / "r9_val_extra.jsonl"),
    )
    g = B.Guards()
    rep = {"train_rows": len(tr), "val_rows": len(va)}
    tn, vn = {r8.norm(r["text"]) for r in tr}, {r8.norm(r["text"]) for r in va}
    rep["train_val_sentence_overlap"] = len(tn & vn)
    rep["train_val_frame_overlap"] = len(
        {r["template_id"] for r in tr} & {r["template_id"] for r in va}
    )
    ts, vs = set(names(tr)), set(names(va))
    rep["train_val_person_value_overlap"] = len(ts & vs)
    # 이름 부분(성 제외) 겹침: 3자 이상 이름의 뒤 두 글자
    given = lambda s: {x[1:] if len(x) >= 3 else x for x in s}
    rep["train_val_given_name_overlap"] = len(given(ts) & given(vs))
    # 신규 전체 vs 기존 학습·개발·봉인
    allnew = (
        tr
        + va
        + [
            r
            for f in (
                "r9_errA_train",
                "r9_errB_train",
                "r9_errD_train",
                "r9_msg_train",
                "r9_mixed_train",
            )
            for r in read(a.data_dir / f"{f}.jsonl")
        ]
    )
    g.rejected.clear()
    bad = sum(0 if g.text_ok(r["text"], r["spans"]) else 1 for r in allnew)
    rep["rows_checked_against_dev_existing_sealed_new180"] = len(allnew)
    rep["rows_rejected_by_text_guards"] = bad
    rep["rejected_by_reason"] = dict(g.rejected)
    nm = set(names(allnew))
    rep["names_in_dev_sets"] = sum(
        1 for n in nm if n in g.dev_exact or n in g.dev_words
    )
    rep["names_in_sealed_or_new180(count only)"] = sum(
        1 for n in nm if n in g.sealed_names or n in g.sealed_vals
    )
    cnt = Counter(names(tr))
    rep["person_value_reuse_train"] = {
        "max": max(cnt.values()),
        "top5_counts": [c for _, c in cnt.most_common(5)],
        "values_used_over_10_times": sum(1 for c in cnt.values() if c > 10),
        "values_used_once": sum(1 for c in cnt.values() if c == 1),
        "distinct": len(cnt),
        "share_of_name_spans_by_top20_values_pct": round(
            100 * sum(c for _, c in cnt.most_common(20)) / sum(cnt.values()), 1
        ),
    }
    rep["ok"] = all(
        rep[k] == 0
        for k in (
            "train_val_sentence_overlap",
            "train_val_frame_overlap",
            "train_val_person_value_overlap",
            "rows_rejected_by_text_guards",
            "names_in_dev_sets",
            "names_in_sealed_or_new180(count only)",
        )
    )
    a.out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
