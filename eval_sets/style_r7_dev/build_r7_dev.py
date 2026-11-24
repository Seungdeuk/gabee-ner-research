"""R7 개발셋 — "실사용" 평가 전용(+1글자 이름 정책 지표). R7 학습(`ml/ner/data/build_r7_realistic_data.py`)
에서 쓰지 않은 홀드아웃 이름/별명(`HOLDOUT_NAMES`/`HOLDOUT_NICKS`, R7 학습 데이터와는 전혀 다른
새 문형)으로 R6/R6+B/R7/R7+B 4-way 비교의 "실사용" 축을 만든다. 모델 선정에만 쓰는 **개발셋**이고,
v14(최종 독립 평가셋)와는 다르다 — v14는 이 개발셋 결과를 본 뒤 완전히 새 이름 풀로 따로 만든다.

- 중의성 평가는 이 디렉터리가 아니라 48절 `style_ambiguity_probe/ambiguity_probe.jsonl`을 그대로
  재사용한다(학습에 전혀 쓰이지 않은, 이미 검증된 읽기 전용 진단셋).
- 스트레스 평가는 `style_v12_locked`/`style_v13_locked`(복합명사형)과
  `apps/api/scripts/check_adjacent_entity_merge.py`(인접 엔티티)를 재사용한다.
- 1글자 이름은 이 파일이 만드는 `r7_dev_1char.jsonl`로 별도 보고한다(정책상 미보호 — merge_ner의
  `len(value)<2` 필터 때문에 2글자 이상과 분리해서 봐야 한다).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[6]  # .../apps/api/app/shared/privacy/eval/style_r7_dev -> repo root
sys.path.insert(0, str(ROOT / "ml" / "ner" / "data"))

import synth  # noqa: E402
import build_r7_realistic_data as r7data  # noqa: E402

HOLDOUT_NAMES = r7data.HOLDOUT_NAMES  # 지우 하윤 채원 연우 시현 건우 승민 정우 도현 해원
HOLDOUT_NICKS = r7data.HOLDOUT_NICKS  # 뽀미 달봉이


def has_batchim(ch: str) -> bool:
    return (ord(ch) - 0xAC00) % 28 != 0


def obj(w: str) -> str:
    return w + ("을" if has_batchim(w[-1]) else "를")


def subj(w: str) -> str:
    return w + ("이" if has_batchim(w[-1]) else "가")


def voc(w: str) -> str:
    return w + ("아" if has_batchim(w[-1]) else "야")


def dat(w: str) -> str:
    return w + "한테"


def fill(tpl: str, n: str) -> str:
    return tpl.format(N=n, Nobj=obj(n), Nsubj=subj(n), Nvoc=voc(n), Ndat=dat(n))


def find_span(text: str, token: str) -> tuple[int, int]:
    idx = text.find(token)
    if idx < 0:
        raise ValueError(f"토큰을 못 찾음: {token!r} in {text!r}")
    return idx, idx + len(token)


# R7 학습(SCHEDULE_TEMPLATES/SHORT_MSG_TEMPLATES/r7_heldout_eval)과 전혀 다른 새 문형.
SINGLE_TEMPLATES = [
    ("기본", "{Nsubj} 요즘 바빠 보이더라"),
    ("목적격", "엄마가 {Nobj} 데리러 가신대"),
    ("비교서술", "{Nsubj} 반에서 제일 꼼꼼해"),
    ("여격", "{Ndat} 숙제 끝났는지 카톡 보내봐"),
    ("호격", "{Nvoc}! 밥 먹으러 나와"),
]
PAIR_TEMPLATES = [  # 무공백 인접(R7 학습 템플릿과 다른 문구)
    "{AB} 둘이 같이 하교했대",
    "{AB}한테 준비물 알려줬어",
]


def build_single_rows():
    rows = []
    rid = 0
    for cat, tpl in SINGLE_TEMPLATES:
        for w in HOLDOUT_NAMES + HOLDOUT_NICKS:
            label = "PERSON" if w in HOLDOUT_NAMES else "PERSON_ALIAS"
            text = fill(tpl, w)
            rid += 1
            s, e = find_span(text, w)
            rows.append({"id": f"r7dev-{rid:04d}", "category": cat, "style": "실사용", "text": text,
                         "spans": [{"kind": label, "start": s, "end": e}], "known_names": [], "template_id": cat})
    return rows, rid


def build_pair_rows(rid):
    rows = []
    pairs = [(HOLDOUT_NAMES[i], HOLDOUT_NAMES[i + 1]) for i in range(0, 8, 2)]
    pairs += [(HOLDOUT_NAMES[8], HOLDOUT_NICKS[0]), (HOLDOUT_NAMES[9], HOLDOUT_NICKS[1])]
    for i, (a, b) in enumerate(pairs):
        tpl = PAIR_TEMPLATES[i % len(PAIR_TEMPLATES)]
        ab = a + b
        text = tpl.format(AB=ab)
        label_b = "PERSON" if b in HOLDOUT_NAMES else "PERSON_ALIAS"
        rid += 1
        rows.append({"id": f"r7dev-{rid:04d}", "category": "무공백인접", "style": "실사용", "text": text,
                     "spans": [{"kind": "PERSON", "start": 0, "end": len(a)},
                               {"kind": label_b, "start": len(a), "end": len(a) + len(b)}],
                     "known_names": [], "template_id": "무공백인접"})
    return rows, rid


# 비인명 부정 문장(기존 GENERIC_NOUNS/GENERIC_PLACES 재사용, 새 문구)
NEG_TEMPLATES = ["{w}이 도착했다", "{w} 때문에 늦었어", "{w}을 확인해봤어"]


def build_neg_rows(rid):
    rows = []
    words = synth.GENERIC_NOUNS[:10] + synth.GENERIC_PLACES[:10]
    for i, w in enumerate(words):
        tpl = NEG_TEMPLATES[i % len(NEG_TEMPLATES)]
        text = tpl.format(w=w)
        rid += 1
        rows.append({"id": f"r7dev-{rid:04d}", "category": "background", "style": "실사용", "text": text,
                     "spans": [], "known_names": [], "template_id": "background"})
    return rows, rid


def build_1char_rows():
    rows = []
    rid = 0
    templates_1c = ["{Nvoc}, 밥 먹었어?", "{N} 오늘 학교 잘 다녀왔대", "{Ndat} 전화 좀 해줘"]
    for i, n in enumerate(synth.GIVEN1):
        tpl = templates_1c[i % len(templates_1c)]
        text = fill(tpl, n)
        rid += 1
        s, e = find_span(text, n)
        rows.append({"id": f"r7dev1c-{rid:04d}", "category": "1글자이름", "style": "실사용", "text": text,
                     "spans": [{"kind": "PERSON", "start": s, "end": e}], "known_names": [], "template_id": "1글자"})
    return rows


def leak_check(rows_text: list[str]) -> None:
    r7_extra = Path(r"C:\Users\<user>\venvs\ner-runs\data_r7_phrase\r7_train_extra.jsonl")
    r7_val = Path(r"C:\Users\<user>\venvs\ner-runs\data_r7_phrase\r7_val_extra.jsonl")
    r7_ho = Path(r"C:\Users\<user>\venvs\ner-runs\data_r7_phrase\r7_heldout_eval.jsonl")
    frozen = set()
    for p in (r7_extra, r7_val, r7_ho):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                frozen.add(json.loads(line)["text"].replace(" ", ""))
    new = {t.replace(" ", "") for t in rows_text}
    overlap = new & frozen
    print(f"R7 학습/홀드아웃 데이터와 문장 겹침: {len(overlap)}개", overlap or "(없음, OK)")
    assert not overlap

    cae_path = ROOT / "apps/api/scripts/check_adjacent_entity_merge.py"
    cae_text = cae_path.read_text(encoding="utf-8").replace(" ", "")
    cae_hits = [t for t in rows_text if t.replace(" ", "") in cae_text]
    print(f"check_adjacent_entity_merge.py 문구와 겹침: {len(cae_hits)}개", cae_hits or "(없음, OK)")
    assert not cae_hits

    dup = len(rows_text) - len(set(rows_text))
    print(f"내부 중복 문장: {dup}개")
    assert dup == 0


def write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main():
    out = HERE
    rows1, rid = build_single_rows()
    rows2, rid = build_pair_rows(rid)
    rows3, rid = build_neg_rows(rid)
    all_rows = rows1 + rows2 + rows3
    leak_check([r["text"] for r in all_rows])
    write(out / "r7_dev_realistic.jsonl", all_rows)
    print(f"실사용 dev: {len(all_rows)}행 (단일 {len(rows1)} + 무공백인접 {len(rows2)} + background {len(rows3)})")

    rows_1c = build_1char_rows()
    leak_check([r["text"] for r in rows_1c])
    write(out / "r7_dev_1char.jsonl", rows_1c)
    print(f"1글자 이름 dev(정책상 미보호 지표 전용): {len(rows_1c)}행")


if __name__ == "__main__":
    main()
