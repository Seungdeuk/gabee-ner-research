"""메신저체(또는 같은 스키마의 다른) 평가셋에서 완전미탐(none)만 골라 세 원인으로 가른다(Codex 2순위 제안).

최종 서비스 경로(``detect_with_ner``, 규칙+NER)로 먼저 full/partial/none 을 가르고(``eval_style_set.py`` 와
같은 기준), none 인 정답 구간만 ``miss_decomposition.classify_complete_miss`` 로 "후보자체미생성 / 점수탈락 /
후처리변형"을 가른다. 외부 호출은 없다.

사용: python analyze_miss_decomposition.py <evalset.jsonl> <모델 폴더> [--out 결과.json]
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
from app.shared.privacy.eval.style_family_realistic.miss_decomposition import (  # noqa: E402
    classify_complete_miss,
)
from app.shared.privacy.types import Kind  # noqa: E402

KIND = {"PERSON": Kind.PERSON}


def analyze(rows: list[dict]) -> tuple[Counter, dict[str, list[dict]]]:
    counts: Counter = Counter()
    examples: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r["spans"]:
            continue
        text = r["text"]
        final = ner.detect_with_ner(text, tuple(r.get("known_names", ())))
        for g in r["spans"]:
            if g["kind"] != "PERSON":
                continue
            cov: set[int] = set()
            for s in final:
                if s.kind is KIND[g["kind"]]:
                    overlap = range(max(s.start, g["start"]), min(s.end, g["end"]))
                    if overlap:
                        cov.update(overlap)
            n = g["end"] - g["start"]
            if len(cov) >= n:
                continue  # full, 분석 대상 아님
            if cov:
                continue  # partial(경계오류 쪽) 은 기존 error_analysis.py 범위, 여기선 완전미탐만
            raw = ner.spans_for(text) or []
            cause = classify_complete_miss(text, g["start"], g["end"], raw)
            counts[cause] += 1
            if len(examples[cause]) < 8:
                examples[cause].append(
                    {
                        "text": text,
                        "gold": text[g["start"] : g["end"]],
                        "raw_person": [
                            [s.label, round(s.score, 3), text[s.start : s.end]]
                            for s in raw
                            if s.label in ("PERSON", "PERSON_ALIAS")
                        ],
                    }
                )
    return counts, examples


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("evalset")
    ap.add_argument("model_dir")
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
    counts, examples = analyze(rows)

    out: dict[str, Any] = {
        "model_dir": Path(a.model_dir).name,
        "ner_status": status,
        "counts": dict(sorted(counts.items())),
        "examples": dict(examples),
    }
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["counts"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
