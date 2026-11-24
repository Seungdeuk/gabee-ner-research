"""레포 전체에서 "이미 쓰인 이름"을 전수 수집한다(v8 독립 평가셋의 신규성 확인용, 읽기 전용).

수집 대상:
- ml/ner/data/synth.py 의 GIVEN1·GIVEN2·NICKS(학습 데이터 생성에 쓴 이름·별명 토큰)
- 레포의 모든 평가 jsonl(PERSON·PERSON_ALIAS 정답 구간의 실제 문자열, 성 추정 제거 포함)
- v7_namefreq.jsonl 의 name 필드(이미 명시돼 있어 바로 뽑힘)

성(姓)은 SURNAMES 목록에 있으면 앞에서 떼고 나머지(주어진 이름)만 모은다 — 대략적인 분리라
과도하게 제외하는 쪽으로 보수적이다(그래도 좋다: 교집합 걱정보다 과잉 배제가 안전).
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]  # repo root
API_ROOT = Path(__file__).resolve().parents[5]  # apps/api
sys.path.insert(0, str(ROOT / "ml" / "ner" / "data"))

EVAL_JSONL_FILES = [
    API_ROOT / "app/shared/privacy/eval/baseline_v2/train.jsonl",
    API_ROOT / "app/shared/privacy/eval/baseline_v2/val.jsonl",
    API_ROOT / "app/shared/privacy/eval/baseline_v2/test.jsonl",
    API_ROOT / "app/shared/privacy/eval/style_v4/evalset_v4.jsonl",
    API_ROOT / "app/shared/privacy/eval/style_v5/locked_test_v5.jsonl",
    API_ROOT / "app/shared/privacy/eval/style_v6_blind/blind_v6.jsonl",
    API_ROOT / "app/shared/privacy/eval/style_v7_namefreq/v7_namefreq.jsonl",
    API_ROOT / "app/shared/privacy/eval/style_family_realistic/realistic_family.jsonl",
    API_ROOT / "app/shared/privacy/eval/style_family_realistic/diag_structures.jsonl",
]

SURNAMES = list("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민나진지") + [
    "남궁", "황보", "제갈", "사공", "선우", "독고",
]
SURNAMES_SORTED = sorted(SURNAMES, key=len, reverse=True)


def strip_surname(name: str) -> str:
    for s in SURNAMES_SORTED:
        if name.startswith(s) and len(name) > len(s):
            return name[len(s):]
    return name


def from_synth() -> set[str]:
    import synth  # type: ignore

    out: set[str] = set()
    out.update(synth.GIVEN1)
    out.update(synth.GIVEN2)
    out.update(synth.NICKS)
    return out


def from_eval_jsonls() -> set[str]:
    out: set[str] = set()
    for path in EVAL_JSONL_FILES:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "")
            for g in row.get("spans", []):
                if g.get("kind") in ("PERSON", "PERSON_ALIAS"):
                    raw = text[g["start"] : g["end"]]
                    raw = re.sub(r"[이가은는을를의도만]$", "", raw) if len(raw) > 2 else raw
                    out.add(raw)
                    out.add(strip_surname(raw))
            if "name" in row:
                out.add(row["name"])
    return out


def build() -> set[str]:
    return from_synth() | from_eval_jsonls()


if __name__ == "__main__":
    names = build()
    print(f"제외 대상 이름(학습+전체 평가셋 합산): {len(names)}개")
    out_path = Path(__file__).with_name("used_names_exclusion.json")
    out_path.write_text(json.dumps(sorted(names), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {out_path}")
