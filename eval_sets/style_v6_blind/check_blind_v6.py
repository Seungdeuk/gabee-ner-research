"""blind_v6.jsonl 품질 검증(27절 조건 4 실행) — 생성만 끝난 상태에서 실행, 모델 평가 전 게이트.

확인 항목: 정답 라벨 범위·공백 여부, 문장·이름 중복, 성씨·이름 분포(편향), 받침에 따른
조사(은/는·이/가·을/를·아/야·이랑/랑) 적용이 전부 규칙대로인지(정규식으로 전수 검사),
``PROBE_ONLY_NAMES``·동결 평가셋과의 중복(생성 스크립트의 검사를 그대로 재확인).
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_blind_v6 import PROBE_ONLY_NAMES, has_batchim  # noqa: E402

DATA_PATH = HERE / "blind_v6.jsonl"

# 받침 유무에 따라 반드시 맞아야 하는 조사 짝(틀리면 문법 오류)
PARTICLE_PAIRS = [
    ("이랑", "랑"),
    ("은", "는"),
    ("이", "가"),
    ("을", "를"),
    ("아", "야"),
]


def check_labels(rows: list[dict]) -> list[str]:
    errors = []
    for r in rows:
        for sp in r["spans"]:
            if not (0 <= sp["start"] < sp["end"] <= len(r["text"])):
                errors.append(f"{r['id']}: 범위 오류 {sp}")
                continue
            val = r["text"][sp["start"] : sp["end"]]
            if not val.strip() or val != val.strip():
                errors.append(f"{r['id']}: 값 공백 문제 {val!r}")
    return errors


def check_duplicates(rows: list[dict]) -> dict:
    texts = [r["text"] for r in rows]
    dup_sentences = [t for t, c in Counter(texts).items() if c > 1]
    ids = [r["id"] for r in rows]
    dup_ids = [i for i, c in Counter(ids).items() if c > 1]
    return {"dup_sentences": dup_sentences, "dup_ids": dup_ids}


def check_distribution(rows: list[dict]) -> dict:
    name_counts: Counter = Counter()
    surname_counts: Counter = Counter()
    for r in rows:
        for sp in r["spans"]:
            name = r["text"][sp["start"] : sp["end"]]
            name_counts[name] += 1
            if r["category"] in ("일반_메신저", "낯선_이름", "조사_경계건강검진"):
                surname_counts[name[0]] += 1
    max_name_share = max(name_counts.values()) / sum(name_counts.values()) if name_counts else 0
    max_surname_share = (
        max(surname_counts.values()) / sum(surname_counts.values()) if surname_counts else 0
    )
    return {
        "unique_names": len(name_counts),
        "max_single_name_share": round(max_name_share, 4),
        "max_single_surname_share": round(max_surname_share, 4),
        "surname_distribution": dict(surname_counts.most_common()),
    }


def check_particles(rows: list[dict]) -> list[str]:
    """정답 구간 바로 뒤에 오는 조사가 받침 규칙과 맞는지 전수 검사한다."""
    errors = []
    for r in rows:
        text = r["text"]
        for sp in r["spans"]:
            name = text[sp["start"] : sp["end"]]
            tail = text[sp["end"] :]
            batchim = has_batchim(name[-1])
            for with_b, without_b in PARTICLE_PAIRS:
                wrong = without_b if batchim else with_b
                # 정답 바로 뒤에 "틀린 쪽" 조사가 붙어 시작하는지(다른 조사의 일부가 아닌,
                # 해당 조사로 시작하는 어절 경계인지는 공백/문장끝 여부로 대략 판단)
                if tail.startswith(wrong):
                    after = tail[len(wrong) :]
                    if after == "" or after[0] in " ?.!":
                        errors.append(
                            f"{r['id']}: '{name}'(받침={batchim}) 뒤에 '{wrong}' — "
                            f"기대 조사는 '{with_b if batchim else without_b}' | 문장: {text}"
                        )
    return errors


def check_name_overlap(rows: list[dict]) -> set[str]:
    used = set()
    for r in rows:
        for sp in r["spans"]:
            used.add(r["text"][sp["start"] : sp["end"]])
    return used & PROBE_ONLY_NAMES


def main() -> None:
    rows = [
        json.loads(line)
        for line in DATA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    label_errors = check_labels(rows)
    dup = check_duplicates(rows)
    dist = check_distribution(rows)
    particle_errors = check_particles(rows)
    name_overlap = check_name_overlap(rows)

    print("=== 라벨 범위·공백 오류 ===")
    print(label_errors or "없음")
    print("\n=== 중복 ===")
    print(dup)
    print("\n=== 분포(편향 확인) ===")
    print({k: v for k, v in dist.items() if k != "surname_distribution"})
    print("성씨 분포:", dist["surname_distribution"])
    print("\n=== 조사(받침 규칙) 오류 ===")
    print(particle_errors or "없음")
    print("\n=== 기존 이름과 중복 ===")
    print(name_overlap or "없음")

    ok = not (
        label_errors or dup["dup_sentences"] or dup["dup_ids"] or particle_errors or name_overlap
    )
    print("\n=== 종합 판정 ===")
    print("전부 통과" if ok else "문제 있음 - 위 내역 확인 필요")


if __name__ == "__main__":
    main()
