"""R1b-가설A(장소·기관 접미 음절 역보강) 학습 데이터 생성 + 중복 검사(25절 가설 실행).

원인 재검증(25절 "소" 가설을 더 넓게 확인, NER_VALIDATION.md 26절에 기록):
"소" 하나가 아니라 **실제 의미가 있는 한국어 단어(접두부) + 기관/장소를 만드는 접미
음절(소·원·관·실·장)**이 결합될 때 PERSON 읽기가 깨진다. 접두부가 이름처럼 생긴
무의미 음절(합성 이름 스타일)이면 같은 접미사를 붙여도 대체로 PERSON으로 잘 읽힌다
(관만 예외적으로 이름형 접두에도 약하게 영향). 그래서 "소로 끝나는 새 이름"이 아니라
**"실제 단어 + 기관 접미 음절"** 조합을 학습 긍정 예시로, **진짜 기관·장소 단어**(주유소·
도서관 등)를 학습 부정(비인명) 예시로 넣는다 - 과마스킹 방지(조건 2).

R1(공통단어 충돌 역보강) 위에 쌓는다(사용자 지시: "기존 R1 모델을 기준으로"). R0의
`data_v5_msg`가 아니라 R1의 `data_R1_pilot` 위에 추가한다.

사용: python build_r1b_location_suffix_data.py --out-dir <폴더>
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# ---- 학습용(6개): 실제 단어 접두 + 기관 접미 음절, 전부 ORG/LOC 로 쏠리는 것을 25절·26절
#      검증 탐색에서 확인한 조합. 기존 실패 사례("고가소"·"유다성")는 그대로 넣지 않는다.
TRAIN_PAIRS = [
    ("환승소", False),
    ("환승관", True),
    ("계산관", True),
    ("출입관", True),
    ("검역장", True),
    ("대기관", True),
]  # (이름, 받침여부)

POS_TEMPLATES = [
    "{N}{i_ga} 오늘 체육대회에서 1등 했대",
    "{N} 생일이 다음주래",
    "{N}{i_ga} 숙제 다 했다고 그러더라",
    "삼촌이 {N} 사진 보내달래",
    "{N} 요즘 축구 배운다며",
    "{N}{i_ga} 오늘 늦게 끝난대",
    "{N} 데리러 몇시에 가면 될까",
    "{N}{i_ga} 저녁 먹었는지 물어봐줄래",
]  # 마지막 2개 틀은 val 전용

# 과마스킹 방지: 진짜 기관·장소 단어(비인명, span 없음)
NEG_TRAIN = [
    "주유소에서 기름 넣고 왔어",
    "버스 정류소에 사람 많더라",
    "도서관에 책 반납하러 가야 돼",
    "체육관에서 줄넘기 연습했어",
    "교실 청소 당번 바뀌었어",
    "사무실에 서류 두고 왔나봐",
    "시장에서 반찬 좀 사왔어",
    "운동장이 공사 중이라 못 들어가",
]

# ---- 일반화 검증용(학습 미사용): 다른 실제 단어 접두 + 다른 기관 접미 음절 ----
HOLDOUT_PAIRS = [
    ("발급소", False),
    ("상담관", True),
    ("주문소", False),
    ("포장관", True),
    ("환불소", False),
]
HOLDOUT_TEMPLATES = [
    "{N} 요즘 키가 많이 컸더라",
    "{N}{i_ga} 어제 놀이터에서 넘어졌대",
    "할머니가 {N} 간식 사주셨대",
    "{N} 사진 가족방에 올라왔어",
    "{N}{i_ga} 오늘 발표를 잘했대",
]
NEG_HOLDOUT = [
    "대기실에 사람이 꽉 찼더라",
    "세차장 다녀왔어",
    "매표소 앞에서 기다려",
    "박물관 관람 시간 확인해줘",
    "수영장 다녀와서 머리 젖었어",
]


def particles(has_batchim: bool) -> dict[str, str]:
    return {
        "i_ga": "이" if has_batchim else "가",
        "eun_neun": "은" if has_batchim else "는",
    }


def make_span_row(rid: str, text: str, name: str, category: str, split: str) -> dict:
    start = text.find(name)
    assert start >= 0, (text, name)
    return {
        "id": rid,
        "text": text,
        "spans": [{"label": "PERSON", "start": start, "end": start + len(name)}],
        "source": "synth",
        "template_id": f"r1b-{category}",
        "category": category,
        "split": split,
        "augment": True,
    }


def make_neg_row(rid: str, text: str, category: str, split: str) -> dict:
    return {
        "id": rid,
        "text": text,
        "spans": [],
        "source": "synth",
        "template_id": f"r1b-{category}-neg",
        "category": category,
        "split": split,
        "augment": True,
    }


def make_eval_row(rid: str, text: str, name: str | None, category: str) -> dict:
    spans = []
    if name:
        start = text.find(name)
        assert start >= 0, (text, name)
        spans.append({"kind": "PERSON", "start": start, "end": start + len(name)})
    return {
        "id": rid,
        "category": category,
        "text": text,
        "spans": spans,
        "known_names": [],
        "template_id": category,
    }


def build(
    pairs: list[tuple[str, bool]], templates: list[str], val_from_idx: int
) -> list:
    rows = []
    for ti, tpl in enumerate(templates):
        split = "val" if ti >= val_from_idx else "train"
        for name, batchim in pairs:
            text = tpl.format(N=name, **particles(batchim))
            rows.append((text, name, split))
    return rows


def skeleton_from_spans(text: str, spans: list[dict]) -> str:
    out, pos = [], 0
    for sp in sorted(spans, key=lambda s: s["start"]):
        out.append(text[pos : sp["start"]])
        pos = sp["end"]
    out.append(text[pos:])
    return "".join(out)


def leak_check(train_pos: list, holdout_pos: list) -> None:
    sys.path.insert(0, str(ROOT / "apps/api/app/shared/privacy/eval/style_v4"))

    frozen_skel: set[str] = set()
    jsonl_paths = [
        ROOT
        / "apps/api/app/shared/privacy/eval/style_family_realistic/realistic_family.jsonl",
        ROOT
        / "apps/api/app/shared/privacy/eval/style_family_realistic/diag_structures.jsonl",
        ROOT
        / "apps/api/app/shared/privacy/eval/style_family_realistic/r1_heldout_eval.jsonl",
        ROOT
        / "apps/api/app/shared/privacy/eval/style_family_realistic/r2_heldout_eval.jsonl",
        ROOT / "apps/api/app/shared/privacy/eval/style_v4/evalset_v4.jsonl",
        ROOT / "apps/api/app/shared/privacy/eval/style_v5/locked_test_v5.jsonl",
        ROOT / "apps/api/app/shared/privacy/eval/baseline_v2/train.jsonl",
        ROOT / "apps/api/app/shared/privacy/eval/baseline_v2/val.jsonl",
        ROOT / "apps/api/app/shared/privacy/eval/baseline_v2/test.jsonl",
    ]
    for p in jsonl_paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            frozen_skel.add(skeleton_from_spans(row["text"], row.get("spans", [])))

    new_skel: set[str] = set()
    for text, name, _split in train_pos + holdout_pos:
        new_skel.add(
            skeleton_from_spans(
                text, [{"start": text.find(name), "end": text.find(name) + len(name)}]
            )
        )
    for text in NEG_TRAIN + NEG_HOLDOUT:
        new_skel.add(text)

    overlap_frozen = new_skel & frozen_skel
    print(
        f"동결 평가셋과 스켈레톤 겹침: {len(overlap_frozen)}개",
        overlap_frozen or "(없음, OK)",
    )
    assert not overlap_frozen, "동결 평가셋과 스켈레톤이 겹친다"

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_synth_check", ROOT / "ml/ner/data/synth.py"
    )
    synth_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(synth_mod)
    new_tpl_skel = {
        tpl.replace("{N}", "").replace("{i_ga}", "").replace("{eun_neun}", "")
        for tpl in POS_TEMPLATES + HOLDOUT_TEMPLATES
    }
    existing_tpl_skel = {
        t.replace("{P}", "").replace("{G}", "").replace("{NK}", "").replace("{AL}", "")
        for _c, t in synth_mod.AUG_TEMPLATES
        + synth_mod.AUG_TEMPLATES_2
        + synth_mod.AUG_TEMPLATES_3
    }
    overlap_tpl = new_tpl_skel & existing_tpl_skel
    print(
        f"기존 보강 템플릿과 뼈대 겹침(대략 검사): {len(overlap_tpl)}개",
        overlap_tpl or "(없음, OK)",
    )
    assert not overlap_tpl, "기존 보강 템플릿과 뼈대가 겹친다"

    prior_names = {
        # R0 완전미탐 36건 교차재현에서 다룬 14개(기존 실패 사례 그대로 넣지 않기)
        "서명소",
        "고한라",
        "성경안",
        "전송유",
        "주소온",
        "문건환",
        "고가소",
        "고가철",
        "우송소",
        "강한소",
        "배송우",
        "민소솔",
        "나호유",
        "유다성",
        # R1 학습·held-out
        "신청호",
        "신청민",
        "송금호",
        "송금아",
        "임대호",
        "임대준",
        "이사호",
        "이사영",
        "진단호",
        "진단영",
        "추천호",
        "추천아",
        "조회민",
        "조회호",
        "정산호",
        "정산준",
        "강좌영",
        "강좌아",
        "차단호",
        "차단민",
        "공유호",
        "공유준",
        # R2 학습·held-out
        "하람",
        "람이",
        "하율",
        "율이",
        "하민",
        "민이",
        "서현",
        "현이",
        "하솔",
        "솔이",
        "지완",
        "완이",
        "하윤",
        "윤이",
        "지환",
        "환이",
        "도겸",
        "겸이",
        "도혁",
        "혁이",
    }
    all_new_names = {n for n, _ in TRAIN_PAIRS + HOLDOUT_PAIRS}
    print(
        f"학습/held-out 이름이 R0·R1·R2 이름과 겹침: {all_new_names & prior_names or '없음, OK'}"
    )
    assert not (all_new_names & prior_names)
    overlap_tv = {n for n, _ in TRAIN_PAIRS} & {n for n, _ in HOLDOUT_PAIRS}
    print(f"학습 이름과 held-out 이름 겹침: {overlap_tv or '없음, OK'}")
    assert not overlap_tv
    overlap_neg = set(NEG_TRAIN) & set(NEG_HOLDOUT)
    print(f"학습 비인명과 held-out 비인명 겹침: {overlap_neg or '없음, OK'}")
    assert not overlap_neg


def write(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_pos = build(TRAIN_PAIRS, POS_TEMPLATES, val_from_idx=6)
    holdout_pos = build(
        HOLDOUT_PAIRS, HOLDOUT_TEMPLATES, val_from_idx=99
    )  # 전부 held-out

    leak_check(train_pos, holdout_pos)

    rid = 0
    train_rows, val_rows = [], []
    for text, name, split in train_pos:
        rid += 1
        row = make_span_row(f"r1b-{rid:04d}", text, name, "locsuffix_person", split)
        (train_rows if split == "train" else val_rows).append(row)
    for i, text in enumerate(NEG_TRAIN):
        rid += 1
        split = "val" if i >= len(NEG_TRAIN) - 2 else "train"
        row = make_neg_row(f"r1b-{rid:04d}", text, "locsuffix_neg", split)
        (train_rows if split == "train" else val_rows).append(row)
    write(out / "r1b_train_extra.jsonl", train_rows)
    write(out / "r1b_val_extra.jsonl", val_rows)
    print(f"학습 추가 train: {len(train_rows)}행, val: {len(val_rows)}행 -> {out}")

    holdout_rows = []
    rid = 0
    for text, name, _split in holdout_pos:
        rid += 1
        holdout_rows.append(
            make_eval_row(f"r1b-ho-{rid:04d}", text, name, "locsuffix_holdout")
        )
    for text in NEG_HOLDOUT:
        rid += 1
        holdout_rows.append(
            make_eval_row(f"r1b-ho-{rid:04d}", text, None, "locsuffix_holdout_neg")
        )
    write(out / "r1b_heldout_eval.jsonl", holdout_rows)
    print(f"held-out 평가 전용: {len(holdout_rows)}행 -> {out}")


if __name__ == "__main__":
    main()
