"""평가셋의 미탐(none/partial) 이름을 완전미탐/라벨매핑/경계오류로 분류한다(로드맵 2단계: 오류 분석).

``app.shared.privacy.ner.spans_for``(라벨 필터링·규칙 병합 이전의 원본 예측, LOCATION·ORGANIZATION 포함)를
직접 불러 ``error_analysis.classify_miss`` 로 분류한다. 외부 호출은 없다.

사용: python analyze_errors.py <evalset.jsonl> <모델 폴더> [--category 기사체 ...]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from app.shared.config import settings  # noqa: E402
from app.shared.privacy import ner  # noqa: E402
from app.shared.privacy.eval.style_v4.error_analysis import classify_miss  # noqa: E402

PERSON_NER_LABELS = ("PERSON", "PERSON_ALIAS")


def analyze(
    rows: list[dict], categories: set[str] | None = None
) -> tuple[Counter, dict[tuple[str, str], list[dict]]]:
    """카테고리·원인별 건수와 예시(최대 4개)를 돌려준다. ``categories`` 가 없으면 전체 문체를 본다."""
    counts: Counter = Counter()
    examples: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if categories and r["category"] not in categories:
            continue
        if not r["spans"]:
            continue
        raw = ner.spans_for(r["text"]) or []
        for g in r["spans"]:
            if g["kind"] != "PERSON":
                continue
            cov: set[int] = set()
            for s in raw:
                if s.label in PERSON_NER_LABELS:
                    overlap = range(max(s.start, g["start"]), min(s.end, g["end"]))
                    if overlap:
                        cov.update(overlap)
            if len(cov) >= g["end"] - g["start"]:
                continue  # 전부 가려짐, 오류 분석 대상 아님
            cause = classify_miss(g["start"], g["end"], raw)
            key = (r["category"], cause)
            counts[key] += 1
            if len(examples[key]) < 4:
                examples[key].append(
                    {
                        "text": r["text"],
                        "gold": r["text"][g["start"] : g["end"]],
                        "raw": [[s.label, r["text"][s.start : s.end]] for s in raw],
                    }
                )
    return counts, examples


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("evalset")
    ap.add_argument("model_dir")
    ap.add_argument("--category", action="append", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = a.model_dir
    settings.privacy_ner_threads = 2
    status = ner.ner_status()
    if not status["active"]:
        raise SystemExit(f"NER 엔진을 쓸 수 없다(모델 폴더 확인 필요): {a.model_dir}")

    rows = [
        json.loads(line)
        for line in Path(a.evalset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts, examples = analyze(rows, set(a.category) if a.category else None)

    out: dict[str, Any] = {
        "model_dir": Path(a.model_dir).name,
        "ner_status": status,
        "counts": {f"{cat}/{cause}": n for (cat, cause), n in sorted(counts.items())},
        "examples": {f"{cat}/{cause}": ex for (cat, cause), ex in examples.items()},
    }
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["counts"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
