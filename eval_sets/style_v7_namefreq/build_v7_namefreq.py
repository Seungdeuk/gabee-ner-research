"""v7 평가셋: 공공 이름 빈도 통계 기반 대규모 PERSON 평가셋(전부 합성 문장).

blind_v6(440개, 사용자 세션에서 "정확히 1회"로 이미 소진)보다 훨씬 큰 표본으로, 이름 빈도(흔함/생소함)·
성씨 유무·조사 유무를 독립 변수로 통제한 실험을 하기 위해 새로 만든다. v6 는 이 평가셋에서 쓰지 않는다.

**데이터 출처**: 사용자가 제공한 출생신고 이름 빈도 집계 2개(여아·남아, 이름+건수만 있는 공개 통계, 실제 개인
식별정보 없음)를 합쳐 중복 제거한 뒤, 건수 상위/하위로 "흔한 이름"/"생소한 이름" 두 집단을 뽑는다. **문장은
전부 새로 지은 합성 문장**(가족 메신저체 새 템플릿)이고, 실제 가족 대화·실명 인물 기록은 전혀 쓰지 않는다
(프로젝트 데이터 정책 준수).

변수 설계(사용자 요청 지표에 맞춤):
- 빈도: 상위 300개(흔한 이름) vs 하위 300개(건수가 가장 적은, 생소한 이름)
- 성씨: 없음(이름만) vs 있음(흔한 성 10개 중 하나 + 이름, 같은 이름으로 쌍을 이룸)
- 조사: 없음(이름만 띄어쓰기, 메신저체에서 흔함) vs 있음(이/가 — 받침 유무로 자동 선택)
- 비인명: 사람 이름이 아닌 일반 명사로 된 새 문장(과마스킹 확인용)

사용: ``python build_v7_namefreq.py <여아CSV> <남아CSV> <출력.jsonl>``
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

HANGUL_ONLY = re.compile(r"^[가-힣]{1,4}$")
SURNAMES = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]

TEMPLATES = [
    "{subj} 오늘 학원 끝나고 전화한대",
    "{subj} 저녁에 치킨 먹고 싶대",
    "{subj} 숙제 다 하고 게임한대",
    "{subj} 내일 소풍 간다고 신났어",
    "{subj} 아까 전화 와서 늦는다고 했어",
    "{subj} 시험 잘 봤다고 자랑하더라",
]

NEGATIVE_TEMPLATES = [
    "{w} 오늘 학원 끝나고 전화한대",
    "{w} 저녁에 치킨 먹고 싶대",
    "{w} 숙제 다 하고 게임한대",
]
NEGATIVE_WORDS = [
    "바다", "노을", "하늘", "구름", "시냇물", "소나기", "무지개", "들판", "안개", "이슬",
    "파도", "폭포", "단풍", "눈송이", "별빛", "햇살", "바람", "새벽", "노래방", "운동장",
    "놀이터", "도서관", "미술관", "박물관", "수족관", "놀이공원",
]


def has_batchim(syllable: str) -> bool:
    code = ord(syllable) - 0xAC00
    return code % 28 != 0


def particle_for(name: str) -> str:
    return "이" if has_batchim(name[-1]) else "가"


def load_names(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name, cnt = (row.get("이름") or "").strip(), (row.get("개수") or "").strip()
            if not name or not cnt.isdigit() or not HANGUL_ONLY.match(name):
                continue
            out[name] = out.get(name, 0) + int(cnt)
    return out


def pick_band(pool: list[tuple[str, int]], n: int, negative_words: set[str], *, from_top: bool) -> list[str]:
    ordered = sorted(pool, key=lambda kv: kv[1], reverse=from_top)
    names = [name for name, _ in ordered if name not in negative_words]
    return names[:n]


def deterministic_pick(name: str, options: list[str]) -> str:
    h = int(hashlib.sha256(name.encode("utf-8")).hexdigest(), 16)
    return options[h % len(options)]


def build_rows(names: list[str], band: str) -> list[dict]:
    rows = []
    for name in names:
        surname = deterministic_pick(name, SURNAMES)
        for has_surname in (False, True):
            display = (surname + name) if has_surname else name
            for has_particle in (False, True):
                subj = display + particle_for(display) if has_particle else display
                tmpl_idx = int(hashlib.sha256((name + str(has_surname) + str(has_particle)).encode()).hexdigest(), 16) % len(TEMPLATES)
                text = TEMPLATES[tmpl_idx].format(subj=subj)
                rows.append(
                    {
                        "id": f"v7-{band}-{name}-{'s1' if has_surname else 's0'}-{'p1' if has_particle else 'p0'}",
                        "category": f"{band}_{'성있음' if has_surname else '성없음'}_{'조사있음' if has_particle else '조사없음'}",
                        "freq_band": band,
                        "has_surname": has_surname,
                        "has_particle": has_particle,
                        "name": name,
                        "text": text,
                        "spans": [{"kind": "PERSON", "start": 0, "end": len(display)}],
                        "known_names": [],
                    }
                )
    return rows


def build_negative_rows(words: list[str]) -> list[dict]:
    rows = []
    for w in words:
        for ti, tmpl in enumerate(NEGATIVE_TEMPLATES):
            rows.append(
                {
                    "id": f"v7-neg-{w}-{ti}",
                    "category": "비인명",
                    "freq_band": "neg",
                    "has_surname": None,
                    "has_particle": None,
                    "name": w,
                    "text": tmpl.format(w=w),
                    "spans": [],
                    "known_names": [],
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("girls_csv", type=Path)
    ap.add_argument("boys_csv", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--band-size", type=int, default=300)
    args = ap.parse_args()

    girls = load_names(args.girls_csv)
    boys = load_names(args.boys_csv)
    merged: dict[str, int] = dict(girls)
    overlap = 0
    for name, cnt in boys.items():
        if name in merged:
            overlap += 1
            merged[name] = max(merged[name], cnt)
        else:
            merged[name] = cnt
    print(f"여아 고유 이름 {len(girls)}개, 남아 고유 이름 {len(boys)}개, 중복(양쪽 다 있음) {overlap}개")
    print(f"중복 제거 후 전체 고유 이름 {len(merged)}개")

    neg_set = set(NEGATIVE_WORDS)
    pool = list(merged.items())
    high = pick_band(pool, args.band_size, neg_set, from_top=True)
    low = pick_band(pool, args.band_size, neg_set, from_top=False)
    overlap_hl = set(high) & set(low)
    if overlap_hl:
        raise SystemExit(f"고빈도/저빈도 밴드가 겹친다(표본이 너무 작음): {overlap_hl}")
    print(f"고빈도 밴드: {len(high)}개(건수 {merged[high[0]]}~{merged[high[-1]]})")
    print(f"저빈도 밴드: {len(low)}개(건수 {merged[low[-1]]}~{merged[low[0]]})")

    rows = build_rows(high, "고빈도") + build_rows(low, "저빈도") + build_negative_rows(NEGATIVE_WORDS)
    texts = [r["text"] for r in rows]
    if len(texts) != len(set(texts)):
        dupes = len(texts) - len(set(texts))
        raise SystemExit(f"문장 중복 {dupes}건 발견 — 템플릿/이름 조합을 확인하라")

    person_spans = sum(len(r["spans"]) for r in rows)
    print(f"전체 문장 {len(rows)}개 (PERSON 정답 {person_spans}개, 비인명 문장 {len(NEGATIVE_WORDS) * len(NEGATIVE_TEMPLATES)}개)")

    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
