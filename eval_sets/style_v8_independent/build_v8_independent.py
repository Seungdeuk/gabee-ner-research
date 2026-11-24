"""v8 — v7·학습 데이터와 전혀 안 겹치는 독립 평가셋(약 900문장, 전부 합성 문장).

**독립성**: 이름 데이터는 v7과 같은 공개 출생신고 이름 빈도 통계 2개(여아·남아, 사용자 제공, "이름,개수"만
있는 집계 — 정확한 1차 출처·수집기관·라이선스 조건은 CSV 파일 자체에 기재돼 있지 않아 이 세션에서는 확인할
수 없다. 이력서·포트폴리오에 "공공데이터"라고 못박아 쓰기 전에 사용자가 원 출처를 직접 확인해야 한다)에서
가져오되, 다음을 전부 뺀 나머지만 쓴다:
- v7_namefreq.jsonl 에 이미 쓴 이름 600개
- `ml/ner/data/synth.py` 의 학습용 이름 토큰(GIVEN1·GIVEN2·NICKS)
- 레포의 모든 평가 jsonl(baseline_v2·style_v4·v5·v6·family_realistic)에 등장한 PERSON 정답 문자열
(`build_exclusion_set.py`가 전부 모아 `used_names_exclusion.json`에 저장 — 3,922개)

**문장**: 전부 새로 지은 템플릿 3개(시작·중간·끝 위치), v6/v7의 템플릿과 겹치지 않는다.

**구성**(목표 약 900문장, 정확한 수보다 독립성·품질 우선):
- 신규 고빈도 100개 / 신규 저빈도 100개 / "이"·"가"로 끝나는 이름 60개 — 각 3문장(시작·중간·끝) = 780
- 신규 애칭(새 음절 조합, 블라인드v6·synth.py와 겹치지 않는 조어) 15개 × 3문장 = 45
- 일반 단어이자 실제 등록 이름인 단어(v7과 다른 10개) — "이름 문맥" 3문장 + "사물 문맥" 3문장 = 60
- 순수 비인명(이름과 안 겹치는 일반 명사 15개) × 2문장 = 30
합계 약 915문장.

사용: ``python build_v8_independent.py <여아CSV> <남아CSV> <출력.jsonl>``
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HANGUL_ONLY = re.compile(r"^[가-힣]{1,4}$")

# v7과 다른 "일반 단어=실제 이름" 후보(실제 등록 건수가 있는지는 build() 에서 CSV 로 재확인한다)
AMBIGUOUS_NAME_CANDIDATES = ["봄", "사랑", "여름", "믿음", "소망", "별", "기쁨", "늘", "아름", "나래", "온유", "다솜"]
# v8 전용 순수 비인명(이름과 안 겹치는 명사, v7의 26개와 다른 세트)
PURE_NEGATIVE_WORDS = [
    "책상", "컴퓨터", "자전거", "우산", "냄비", "모니터", "택배", "신호등", "가로등", "정류장",
    "세탁기", "청소기", "냉장고", "선풍기", "보일러",
]
# v8 전용 신규 애칭 조어 음절(blind_v6 의 강·찬·곤·담과 겹치지 않음) + "이"
NICK_SYLLABLES = ["윤", "온", "루", "린", "하", "소", "별", "결", "솔", "해", "담", "유", "다", "새", "노"]


def has_batchim(ch: str) -> bool:
    return (ord(ch) - 0xAC00) % 28 != 0


def josa_ga_i(name: str) -> str:
    return name + ("이" if has_batchim(name[-1]) else "가")


def copula_ending(name: str) -> str:
    return name + ("이었어" if has_batchim(name[-1]) else "였어")


def load_names(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name, cnt = (row.get("이름") or "").strip(), (row.get("개수") or "").strip()
            if not name or not cnt.isdigit() or not HANGUL_ONLY.match(name):
                continue
            out[name] = out.get(name, 0) + int(cnt)
    return out


NAME_TEMPLATES = {
    "시작": lambda n: f"{josa_ga_i(n)} 아침부터 기분이 좋아 보이더라",
    "중간": lambda n: f"어제 {n}한테 문자 보냈는데 아직 답이 없어",
    "끝": lambda n: f"방금 통화한 사람이 {copula_ending(n)}",
}
OBJECT_TEMPLATES = {
    "시작": lambda w: f"{w} 아침부터 어디 있는지 안 보여",
    "중간": lambda w: f"어제 {w} 찾는다고 한참 뒤졌어",
    "끝": lambda w: f"방금 본 게 {copula_ending(w)}",
}


def span_for(text: str, token: str, occurrence_hint: str) -> tuple[int, int]:
    """occurrence_hint 가 가리키는 위치(시작/중간/끝)의 token 을 찾는다 — 문장에 같은 문자열이
    두 번 나올 일은 이 템플릿들엔 없지만, 혹시를 대비해 위치 힌트로 올바른 occurrence 를 고른다."""
    if occurrence_hint == "끝":
        idx = text.rfind(token)
    else:
        idx = text.find(token)
    if idx < 0:
        raise ValueError(f"토큰을 못 찾음: {token!r} in {text!r}")
    return idx, idx + len(token)


def make_name_rows(names: list[str], band: str) -> list[dict]:
    rows = []
    for name in names:
        for pos, tmpl in NAME_TEMPLATES.items():
            text = tmpl(name)
            s, e = span_for(text, name, pos)
            rows.append(
                {
                    "id": f"v8-{band}-{name}-{pos}",
                    "category": f"{band}_{pos}",
                    "name": name,
                    "position": pos,
                    "text": text,
                    "spans": [{"kind": "PERSON", "start": s, "end": e}],
                    "known_names": [],
                }
            )
    return rows


def make_object_rows(words: list[str], band: str) -> list[dict]:
    rows = []
    for w in words:
        for pos, tmpl in OBJECT_TEMPLATES.items():
            text = tmpl(w)
            rows.append(
                {
                    "id": f"v8-{band}-{w}-{pos}",
                    "category": f"{band}_{pos}",
                    "name": w,
                    "position": pos,
                    "text": text,
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
    args = ap.parse_args()

    exclusion_path = HERE / "used_names_exclusion.json"
    if not exclusion_path.exists():
        raise SystemExit("먼저 build_exclusion_set.py 를 실행하라")
    excluded = set(json.loads(exclusion_path.read_text(encoding="utf-8")))
    excluded |= set(PURE_NEGATIVE_WORDS)  # 혹시 모를 교차오염 방지

    girls = load_names(args.girls_csv)
    boys = load_names(args.boys_csv)
    merged: dict[str, int] = dict(girls)
    for k, v in boys.items():
        merged[k] = max(merged.get(k, 0), v)

    pool = {k: v for k, v in merged.items() if k not in excluded}
    print(f"원본 고유 이름 {len(merged)}개 -> 제외 후 사용 가능 {len(pool)}개")

    ordered = sorted(pool.items(), key=lambda kv: kv[1], reverse=True)
    high = [n for n, _ in ordered[:100]]
    low = [n for n, _ in ordered[-100:]]
    used = set(high) | set(low)

    ending_ga_i = [n for n, c in ordered if n not in used and (n.endswith("이") or n.endswith("가"))][:60]
    used |= set(ending_ga_i)

    print(f"신규 고빈도 {len(high)}개(건수 {pool[high[-1]]}~{pool[high[0]]})")
    print(f"신규 저빈도 {len(low)}개(건수 {pool[low[0]]}~{pool[low[-1]]})")
    print(f"'이'/'가'로 끝나는 신규 이름 {len(ending_ga_i)}개: {ending_ga_i[:10]}...")

    nicks = []
    for syl in NICK_SYLLABLES:
        cand = syl + "이"
        if cand not in excluded and cand not in used:
            nicks.append(cand)
    nicks = nicks[:15]
    print(f"신규 애칭 {len(nicks)}개: {nicks}")

    ambiguous = []
    for w in AMBIGUOUS_NAME_CANDIDATES:
        if w in merged and w not in excluded:
            ambiguous.append((w, merged[w]))
    ambiguous = ambiguous[:10]
    print(f"일반단어=실제이름(v8 전용) {len(ambiguous)}개: {ambiguous}")
    amb_words = [w for w, _ in ambiguous]

    rows = (
        make_name_rows(high, "신규고빈도")
        + make_name_rows(low, "신규저빈도")
        + make_name_rows(ending_ga_i, "이가로끝남")
        + make_name_rows(nicks, "신규애칭")
        + make_name_rows(amb_words, "이름문맥_일반단어")
        + make_object_rows(amb_words, "사물문맥_일반단어")
        + make_object_rows(PURE_NEGATIVE_WORDS, "순수비인명")
    )

    for r in rows:
        if r["spans"]:
            g = r["spans"][0]
            if r["text"][g["start"] : g["end"]] != r["name"]:
                raise SystemExit(f"구간 검증 실패: {r['id']} -> {r['text'][g['start']:g['end']]!r} != {r['name']!r}")

    texts = [r["text"] for r in rows]
    dupes = len(texts) - len(set(texts))
    if dupes:
        raise SystemExit(f"문장 중복 {dupes}건 발견")
    cross = set(high) & set(low) | set(high) & set(ending_ga_i) | set(low) & set(ending_ga_i)
    if cross:
        raise SystemExit(f"범주 간 이름 중복: {cross}")

    person_total = sum(len(r["spans"]) for r in rows)
    neg_total = sum(1 for r in rows if not r["spans"])
    print(f"\n총 {len(rows)}문장 (PERSON 정답 {person_total}개, 비인명 {neg_total}개)")

    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"저장: {args.out}")

    meta = {
        "merged_pool_size": len(merged),
        "excluded_names_count": len(excluded),
        "available_after_exclusion": len(pool),
        "ambiguous_words_used": ambiguous,
        "note": "이름 데이터 출처: 사용자 제공 출생신고 이름 빈도 집계 2개(여아·남아). 정확한 1차 출처·수집기관·라이선스는 CSV 자체에 없어 이 세션에서 확인 불가 — 재사용 전 사용자가 원 출처를 확인할 것.",
    }
    (args.out.parent / "v8_build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
