"""R7 ablation 전용 — 순수 "데이터 양" 대조군(Vol). R7의 실사용 단일개체 블록(R, train 114/val
100행)과 거의 같은 크기로, PERSON 계열 라벨이 전혀 없는 평범한 일상 명사 문장만 채운다. 이 블록을
추가했을 때도 노출·오탐·과잉병합이 바뀐다면, R7의 효과(또는 부작용)가 "내용"이 아니라 "학습량 증가
그 자체" 때문일 수 있다는 뜻이다 — 이 스크립트는 그 가능성을 분리하기 위한 대조군이다.
`ner.py`/`merge_ner`는 전혀 안 건드린다.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import synth  # noqa: E402
import build_r7_realistic_data as r7data  # noqa: E402

WORDS = synth.GENERIC_NOUNS + synth.GENERIC_PLACES  # 45개, PERSON 아님(leak_check 대상 아님)
TEMPLATES = [
    "{w} 얘기가 나왔다",
    "{w}을 알아봤다",
    "{w}이 필요했다",
    "{w} 관련해서 연락이 왔다",
    "{w}을 준비해야 했다",
]


def make_row(rid, text, split):
    return {"id": rid, "text": text, "spans": [], "source": "synth",
            "template_id": "vol-neg", "category": "vol_neg", "split": split, "augment": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_texts = []
    rid = 0
    for w in WORDS:
        for tpl in TEMPLATES:
            text = tpl.format(w=w)
            rid += 1
            all_texts.append(text)
            all_rows.append((f"vol-{rid:04d}", text))

    r7data.leak_check(all_texts)  # R7/R3~R7a 동결 데이터·check_adjacent_entity_merge.py 와 겹침 검사(0건이어야 함)

    train_rows, val_rows = [], []
    for i, (rid_s, text) in enumerate(all_rows):
        split = "train" if i % 20 < 11 else "val"  # 대략 114/100 비율(block R과 비슷한 크기)
        row = make_row(rid_s, text, split)
        (train_rows if split == "train" else val_rows).append(row)

    def write(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

    write(out / "vol_train_extra.jsonl", train_rows)
    write(out / "vol_val_extra.jsonl", val_rows)
    print(f"Vol(양적 대조군) train: {len(train_rows)}행, val: {len(val_rows)}행 -> {out}")


if __name__ == "__main__":
    main()
