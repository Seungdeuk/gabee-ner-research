"""AI Hub 「대규모 웹데이터 기반 한국어 말뭉치」 라벨링 데이터(TL*.zip)를 NER 학습 행(jsonl)으로 바꾼다.

- 이 데이터에는 사람(PS)·장소(LC) 태그가 없고(사람 이름은 원문에서 "(이름)" 으로 가려져 있다) 기관(OG) 등만 있다.
  그래서 기관(OG) -> ORGANIZATION 만 라벨로 쓰고, 나머지 태그(CV, TM, AF …)는 라벨 없음(O)으로 둔다.
  이름이 없는 일반 뉴스 문장은 일반 명사를 사람으로 오인하는 오탐을 줄이는 음성 예시로 쓴다.
- 원문은 저장소에 두지 않는다. 산출 jsonl 은 저장소 밖(--out)에만 쓴다. AI Hub 이용정책: 데이터 재배포·제3자 제공 금지,
  학습모델 학습용으로만 사용, 이를 이용한 결과물에는 한국지능정보사회진흥원(NIA) 사업결과임을 표시한다.
- 압축은 풀지 않고(zipfile 로 읽기만), 파일 일부를 시드 고정으로 뽑는다.

실행::

    python ml/ner/data/build_aihub_ner.py --zip <TL1.zip> --base-data <data_v2b> --out <새 데이터 폴더> [--n 21000 --files 900 --seed 42]

산출: <out>/aihub_train.jsonl, aihub_stats.json(개수만, 문장 없음), 그리고 <base-data> 의 baseline_*/synth_* 파일 복사본.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import shutil
import zipfile
from pathlib import Path

PLACEHOLDER = re.compile(
    r"\((?:이름|주소|전화번호|계좌번호|이메일|아이디|비밀번호|주민번호|생년월일|카드번호|번호)\)"
)
FINANCE = re.compile(r"은행|증권|금융|카드|보험|캐피탈|저축|신협|새마을금고|농협|수협|뱅크")
COPY_FILES = [
    "baseline_train.jsonl",
    "baseline_val.jsonl",
    "synth_train.jsonl",
    "synth_val.jsonl",
    "synth_val_core.jsonl",
    "synth_val_aug.jsonl",
]
MIN_LEN, MAX_LEN = 10, 200


def sentence_to_row(sentence: str, labels: list[dict]) -> dict | None:
    """문장과 라벨 목록을 학습 행으로 바꾼다. 쓸 수 없으면 None.

    @param sentence: 문장 글자
    @param labels: [{"text", "tag", ...}] (문장 안 등장 순서)
    @returns: {"text", "spans"} 또는 None
    """
    text = sentence.strip()
    if not (MIN_LEN <= len(text) <= MAX_LEN) or PLACEHOLDER.search(text):
        return None
    spans, pos = [], 0
    for lab in labels:
        tag, value = lab.get("tag"), lab.get("text") or ""
        if tag == "OG":
            if FINANCE.search(
                value
            ):  # 금융기관명은 라벨 정책상 O 인데 문장에 남기면 라벨 없는 기관이 생긴다 -> 문장을 버린다
                return None
            at = text.find(value, pos)
            if at < 0:
                at = text.find(value)
            if at < 0 or not value:
                return None
            spans.append({"label": "ORGANIZATION", "start": at, "end": at + len(value)})
            pos = at + len(value)
    spans.sort(key=lambda s: s["start"])
    for a, b in zip(spans, spans[1:], strict=False):  # 겹치면 버린다
        if b["start"] < a["end"]:
            return None
    return {"text": text, "spans": spans}


def iter_sentences(doc: dict):
    for blk in doc.get("named_entity", []):
        for part in ("title", "subtitle", "content"):
            for s in blk.get(part, []):
                yield s.get("sentence", ""), s.get("labels", [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument(
        "--base-data", required=True, help="baseline_*/synth_* 가 있는 폴더(복사해 온다)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=21000, help="뽑을 문장 수")
    ap.add_argument("--files", type=int, default=900, help="읽을 JSON 파일 수")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(a.base_data)
    for name in COPY_FILES:
        shutil.copy2(base / name, out / name)

    rng = random.Random(a.seed)
    z = zipfile.ZipFile(a.zip)
    names = sorted(i.filename for i in z.infolist() if i.filename.endswith(".json"))
    pick = rng.sample(names, min(a.files, len(names)))

    # 평가 데이터(Baseline)와 같은 문장이 들어가지 않게 검사한다(train/val 만 읽는다. test 는 읽지 않는다)
    seen = set()
    for name in ("baseline_train.jsonl", "baseline_val.jsonl"):
        for line in (base / name).read_text(encoding="utf-8").splitlines():
            seen.add(json.loads(line)["text"])

    rows, dropped = [], collections.Counter()
    for n in pick:
        doc = json.loads(z.read(n).decode("utf-8"))
        for sentence, labels in iter_sentences(doc):
            row = sentence_to_row(sentence, labels)
            if row is None:
                dropped["unusable"] += 1
                continue
            if row["text"] in seen:
                dropped["duplicate_or_leak"] += 1
                continue
            seen.add(row["text"])
            rows.append(row)
    rng.shuffle(rows)
    rows = rows[: a.n]
    with open(out / "aihub_train.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(rows):
            r.update(
                {
                    "id": f"aihub-train-{i:05d}",
                    "source": "aihub",
                    "template_id": None,
                    "split": "train",
                }
            )
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {
        "source": "AI Hub 대규모 웹데이터 기반 한국어 말뭉치(라벨링 데이터)",
        "files_read": len(pick),
        "sentences": len(rows),
        "sentences_with_organization": sum(1 for r in rows if r["spans"]),
        "organization_spans": sum(len(r["spans"]) for r in rows),
        "dropped": dict(dropped),
        "seed": a.seed,
    }
    (out / "aihub_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
