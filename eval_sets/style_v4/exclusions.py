"""평가셋 v4 가 재사용하면 안 되는 이름을 모은다(학습 풀 + baseline_v2 전체 분할 + 옛 v3).

``build_evalset_v3.py`` 의 ``training_given_names`` 와 같은 방식(``synth.py`` 의 GIVEN1/GIVEN2 정규식 추출 +
jsonl 의 PERSON/PERSON_ALIAS span 문자열)을 여러 폴더·여러 파일에 걸쳐 합친다.
"""

import json
import re
from pathlib import Path

_PERSON_KINDS = ("PERSON", "PERSON_ALIAS")


def _given_pool_from_synth(synth_path: Path) -> set[str]:
    text = synth_path.read_text(encoding="utf-8")
    pool: set[str] = set()
    for var in ("GIVEN2", "GIVEN1"):
        m = re.search(var + r'\s*=\s*(?:list\()?\(?\s*"(.*?)"\s*\)', text, re.S)
        if m:
            pool.update(
                re.sub(r'"\s*"', " ", m.group(1)).split() if var == "GIVEN2" else list(m.group(1))
            )
    if not pool:
        raise ValueError(f"{synth_path} 에서 GIVEN1/GIVEN2 를 찾지 못했다(조용히 건너뛰지 않는다)")
    return pool


def _person_values_from_jsonl(path: Path) -> set[str]:
    values: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for span in row.get("spans", []):
            if span.get("kind") in _PERSON_KINDS:
                values.add(row["text"][span["start"] : span["end"]])
    return values


def collect_excluded_names(
    synth_path: Path, baseline_dirs: list[Path], extra_eval_paths: list[Path]
) -> set[str]:
    """학습 풀 + ``baseline_dirs`` 의 모든 ``*.jsonl`` + ``extra_eval_paths`` 의 PERSON 이름을 합친다.

    ``extra_eval_paths`` 의 경로가 없으면 조용히 건너뛰지 않고 ``FileNotFoundError`` 를 그대로 올린다.
    """
    excluded = _given_pool_from_synth(synth_path)
    for d in baseline_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"baseline 폴더가 없다(조용히 건너뛰지 않는다): {d}")
        files = sorted(d.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"baseline 폴더에 jsonl 이 없다: {d}")
        for f in files:
            excluded.update(_person_values_from_jsonl(f))
    for p in extra_eval_paths:
        excluded.update(_person_values_from_jsonl(p))
    return excluded
