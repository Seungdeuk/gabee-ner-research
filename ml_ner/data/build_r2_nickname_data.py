"""R2(애칭·줄임말 보강) 학습 데이터 생성 + 동결 평가셋과의 중복(스켈레톤) 검사(21절 가설 실행).

원인 분석(이 스크립트를 쓰기 전 교차 재현으로 확인, NER_VALIDATION.md 23절 뒤에 기록):
한국어에서 받침 있는 음절로 끝나는 전체 이름 뒤에 "이"가 붙으면 거의 항상 **주격조사**다
("민준이 밥 먹었어" = "민준"(이름)+"이"(조사)) - 그래서 모델은 "받침음절+이"를 보면 "이"를
조사로 떼어내고 앞쪽만 PERSON으로 거는 습관이 아주 강하게 든다. 그런데 애칭("윤이"=하윤의
애칭, "람이"=하람의 애칭)에서는 "이"가 조사가 아니라 **애칭 자체의 일부**다 - 모델이 평소
습관대로 "이"를 떼어내면 PERSON 전체 구간을 못 덮어 부분노출/완전미탐이 된다.

R0(5시드) 교차 재현으로 애칭 음절별로 이미 문제가 있는 것(윤·람·율·민·현·솔·완·환·겸·혁)과
문제없는 것(준·안·원·진·영·린·우)이 나뉘는 것을 확인했다 - 전부가 깨지는 게 아니라 특정
음절에서만 "이=조사"라는 습관이 특히 강하다.

설계:
- 학습용 애칭 6개(람,율,민,현,솔,완 음절): 이미 문제로 확인된 음절 중, 21절의 원래 사례("윤")와
  더 확인용으로 남겨둔 음절(환,겸,혁)은 **학습에 넣지 않고** held-out 평가에만 쓴다.
- R1 보강 데이터는 전혀 섞지 않는다(기존 ``data_v5_msg`` 위에만 추가) - R0 대비 R2 단독 효과를
  보려는 것이다.
- 과마스킹 방지: "받침음절+이"가 **진짜 조사**인 평범한 명사 문장(방이·밥이·눈이·꿈이 등)을
  긍정 예시와 같은 비율로 넣는다 - 애칭을 가르치다가 모든 "받침+이"를 PERSON으로 과확신하게
  되는 것을 막기 위해서다.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# ---- 학습용 애칭(6개: 전체이름, 애칭) - 전부 R0 5시드에서 부분노출(P)로 확인된 음절 ----
TRAIN_PAIRS = [
    ("하람", "람이"),
    ("하율", "율이"),
    ("하민", "민이"),
    ("서현", "현이"),
    ("하솔", "솔이"),
    ("지완", "완이"),
]

POS_TEMPLATES = [
    "{N}{i_ga} 오늘 유치원에서 상 받았대",
    "{N} 사진 좀 보내줘",
    "{N}{i_ga} 밥 잘 먹었는지 물어봐줄래",
    "할머니가 {N} 많이 보고싶어하셨어",
    "{N} 요즘 부쩍 컸더라",
    "{N}{i_ga} 어제 열이 좀 있었대",
    "{N} 데리러 누가 갈까",
    "{N}{i_ga} 숙제 다 했는지 확인해줘",
]  # 마지막 2개 틀은 val 전용

# 과마스킹 방지: "받침음절+이"가 진짜 주격조사인 평범한 문장(이름 아님, span 없음)
NEG_TRAIN = [
    "방이 또 엉망이야",
    "밥이 다 식었어",
    "눈이 많이 왔대",
    "꿈이 생생했어",
    "길이 많이 막히더라",
    "잠이 안 와서 늦게 잤어",
    "봄이 왔나봐 꽃이 폈더라",
    "폭이 너무 좁아서 못 지나가",
    "짐이 생각보다 많더라",
    "힘이 다 빠졌어",
]

# ---- 일반화 검증용(학습 미사용): 21절 원래 사례("하윤→윤이") 포함 ----
HOLDOUT_PAIRS = [
    ("하윤", "윤이"),
    ("지환", "환이"),
    ("도겸", "겸이"),
    ("도혁", "혁이"),
]
HOLDOUT_TEMPLATES = [
    "{N} 요즘 많이 먹어서 그런지 쑥쑥 크더라",
    "{N}{i_ga} 어린이집 선생님이 칭찬했대",
    "삼촌이 {N} 데리고 공원 갔대",
    "{N} 밤에 잠투정 심하게 했대",
    "{N}{i_ga} 요즘 말이 많이 늘었어",
]
NEG_HOLDOUT = [
    "숨이 차올랐어",
    "달이 밝더라",
    "손이 차가워",
    "술이 좀 과했나봐",
    "상이 비뚤어졌어",
]


def make_span_row(rid, text, name, category, split):
    start = text.find(name)
    assert start >= 0, (text, name)
    return {
        "id": rid,
        "text": text,
        "spans": [{"label": "PERSON", "start": start, "end": start + len(name)}],
        "source": "synth",
        "template_id": f"r2-{category}",
        "category": category,
        "split": split,
        "augment": True,
    }


def make_neg_row(rid, text, category, split):
    return {
        "id": rid,
        "text": text,
        "spans": [],
        "source": "synth",
        "template_id": f"r2-{category}-neg",
        "category": category,
        "split": split,
        "augment": True,
    }


def make_eval_row(rid, text, name, category):
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


def build(pairs, templates, val_from_idx):
    rows = []
    for ti, tpl in enumerate(templates):
        split = "val" if ti >= val_from_idx else "train"
        for _full, nick in pairs:
            # 애칭은 전부 "이"로 끝나(받침 없음) -> 가/는 만 쓴다
            text = tpl.format(N=nick, i_ga="가", eun_neun="는")
            rows.append((text, nick, split))
    return rows


def skeleton_from_spans(text, spans):
    out, pos = [], 0
    for sp in sorted(spans, key=lambda s: s["start"]):
        out.append(text[pos : sp["start"]])
        pos = sp["end"]
    out.append(text[pos:])
    return "".join(out)


def leak_check(train_pos, holdout_pos):
    sys.path.insert(0, str(ROOT / "apps/api/app/shared/privacy/eval/style_v4"))

    frozen_skel = set()
    jsonl_paths = [
        ROOT
        / "apps/api/app/shared/privacy/eval/style_family_realistic/realistic_family.jsonl",
        ROOT
        / "apps/api/app/shared/privacy/eval/style_family_realistic/diag_structures.jsonl",
        ROOT
        / "apps/api/app/shared/privacy/eval/style_family_realistic/r1_heldout_eval.jsonl",
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

    new_skel = set()
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
    assert not overlap_frozen, "동결 평가셋과 스켈레톤이 겹친다 - 틀을 고쳐야 한다"

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

    r1_names = {
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
    }
    all_new_names = {n for _f, n in TRAIN_PAIRS + HOLDOUT_PAIRS}
    print(
        f"학습/held-out 애칭이 R1 이름과 겹침: {all_new_names & r1_names or '없음, OK'}"
    )
    assert not (all_new_names & r1_names)
    overlap_tv = {n for _f, n in TRAIN_PAIRS} & {n for _f, n in HOLDOUT_PAIRS}
    print(f"학습 애칭과 held-out 애칭 겹침: {overlap_tv or '없음, OK'}")
    assert not overlap_tv


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main():
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
        row = make_span_row(f"r2-{rid:04d}", text, name, "nickname_person", split)
        (train_rows if split == "train" else val_rows).append(row)
    for i, text in enumerate(NEG_TRAIN):
        rid += 1
        split = "val" if i >= len(NEG_TRAIN) - 2 else "train"
        row = make_neg_row(f"r2-{rid:04d}", text, "nickname_neg", split)
        (train_rows if split == "train" else val_rows).append(row)
    write(out / "r2_train_extra.jsonl", train_rows)
    write(out / "r2_val_extra.jsonl", val_rows)
    print(f"학습 추가 train: {len(train_rows)}행, val: {len(val_rows)}행 -> {out}")

    holdout_rows = []
    rid = 0
    for text, name, _split in holdout_pos:
        rid += 1
        holdout_rows.append(
            make_eval_row(f"r2-ho-{rid:04d}", text, name, "nickname_holdout")
        )
    for text in NEG_HOLDOUT:
        rid += 1
        holdout_rows.append(
            make_eval_row(f"r2-ho-{rid:04d}", text, None, "nickname_holdout_neg")
        )
    write(out / "r2_heldout_eval.jsonl", holdout_rows)
    print(f"held-out 평가 전용: {len(holdout_rows)}행 -> {out}")


if __name__ == "__main__":
    main()
