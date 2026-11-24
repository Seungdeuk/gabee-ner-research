"""R1(공통단어 충돌 역보강) 학습 데이터 생성 + 동결 평가셋과의 중복(스켈레톤) 검사(22절 분석 실행).

설계:
- 학습용 충돌 이름 6개(신청,송금,임대,이사,진단,추천): 전부 16/19/22절에서 실제로 발견된
  충돌 단어(고가,배송,성경,전송,주소,문건,한라,서명)와 겹치지 않는 새 단어다 - 평가셋에 이미
  나온 정답을 그대로 외우게 하지 않기 위해서다.
- 일반화 검증용 충돌 이름 5개(조회,정산,강좌,차단,공유): 학습에는 전혀 쓰지 않고
  별도 held-out 평가에만 쓴다(학습 단어와도, 평가셋에 이미 나온 단어와도 겹치지 않음).
- 각 충돌 단어마다 PERSON 긍정 문장(가족 메신저 톤)과, 같은 단어를 평범한 명사로 쓴
  비인명(과마스킹 방지) 문장을 함께 만든다.

사용: python build_r1_collision_data.py --out-dir <폴더>
  <폴더>/r1_train_extra.jsonl, r1_val_extra.jsonl 을 기존 data_v5_msg 의
  synth_train.jsonl/synth_val.jsonl 뒤에 이어붙여 학습 데이터를 만든다.
  <폴더>/r1_heldout_eval.jsonl 은 학습에 쓰지 않는 일반화 전용 평가셋이다.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# ---- 학습용 충돌 이름(6개, 평가셋의 기존 14개와 겹치지 않음) ----
TRAIN_NAMES = [
    ("신청호", False),
    ("신청민", True),
    ("송금호", False),
    ("송금아", False),
    ("임대호", False),
    ("임대준", True),
    ("이사호", False),
    ("이사영", True),
    ("진단호", False),
    ("진단영", True),
    ("추천호", False),
    ("추천아", False),
]  # (이름, 받침여부)

POS_TEMPLATES = [
    "{N}{eun_neun} 이번 주말에 할머니 댁 간대",
    "{N}{i_ga} 오늘 숙제 다 했는지 물어봐줘",
    "엄마가 {N} 간식 챙겨놨대",
    "{N} 생일선물 뭐가 좋을까",
    "{N}{i_ga} 오늘 줄넘기 연습 했대",
    "{N}랑 같이 놀이터 가기로 했어",
    "오늘 {N} 열나서 조퇴시켰어",
    "{N} 사진 가족방에 올려줘",
]  # 마지막 2개 틀은 val 전용(아래서 idx>=6 은 val)

NEG_TRAIN = [
    "신청 다음 주까지 해야 된대",
    "신청서 작성 다 끝냈어?",
    "송금 보냈는데 확인해줄래",
    "송금 내역 캡처해서 보내줘",
    "임대 계약 만기가 다음 달이래",
    "임대료 또 올랐더라",
    "이사 비용이 생각보다 많이 들었어",
    "이사 날짜 아직 안 정했어",
    "진단 결과 나왔는데 괜찮대",
    "건강검진 진단서 메일로 왔어",
    "추천 받은 책 재밌더라",
    "추천 메뉴로 시켰어",
]

# ---- 일반화 검증용(학습 미사용) ----
HOLDOUT_NAMES = [
    ("조회민", True),
    ("조회호", False),
    ("정산호", False),
    ("정산준", True),
    ("강좌영", True),
    ("강좌아", False),
    ("차단호", False),
    ("차단민", True),
    ("공유호", False),
    ("공유준", True),
]
HOLDOUT_TEMPLATES = [  # 학습 긍정 틀과도 다르게 써서 틀 자체의 일반화도 같이 본다
    "{N}{i_ga} 어제부터 기침을 하더라",
    "{N}한테 저녁에 안부전화 좀 넣어줄래",
    "할머니가 {N} 보고싶다고 하셨어",
    "{N} 오늘 늦게 온다고 문자왔어",
    "{N}{eun_neun} 요즘 바빠보이더라",
]
NEG_HOLDOUT = [
    "조회 결과 좀 보내줘",
    "정산 내역 확인해줘",
    "강좌 신청 마감이래",
    "차단 설정 어떻게 하는거야",
    "공유 링크 좀 보내줘",
]


def particles(has_batchim):
    return {
        "i_ga": "이" if has_batchim else "가",
        "eun_neun": "은" if has_batchim else "는",
    }


def make_span_row(rid, text, name, category, split):
    start = text.find(name)
    assert start >= 0, (text, name)
    return {
        "id": rid,
        "text": text,
        "spans": [{"label": "PERSON", "start": start, "end": start + len(name)}],
        "source": "synth",
        "template_id": f"r1-{category}",
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
        "template_id": f"r1-{category}-neg",
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


def build(names, templates, val_from_idx):
    rows = []
    for ti, tpl in enumerate(templates):
        split = "val" if ti >= val_from_idx else "train"
        for name, batchim in names:
            p = particles(batchim)
            text = tpl.format(N=name, **p)
            rows.append((text, name, split))
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

    known_14 = {
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
    }
    all_new_names = {n for n, _ in TRAIN_NAMES + HOLDOUT_NAMES}
    print(
        f"학습/held-out 이름이 기존 14개와 겹침: {all_new_names & known_14 or '없음, OK'}"
    )
    assert not (all_new_names & known_14)
    overlap_tv = {n for n, _ in TRAIN_NAMES} & {n for n, _ in HOLDOUT_NAMES}
    print(f"학습 이름과 held-out 이름 겹침: {overlap_tv or '없음, OK'}")
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

    train_pos = build(TRAIN_NAMES, POS_TEMPLATES, val_from_idx=6)
    holdout_pos = build(
        HOLDOUT_NAMES, HOLDOUT_TEMPLATES, val_from_idx=99
    )  # 전부 held-out

    leak_check(train_pos, holdout_pos)

    rid = 0
    train_rows, val_rows = [], []
    for text, name, split in train_pos:
        rid += 1
        row = make_span_row(f"r1-{rid:04d}", text, name, "collision_person", split)
        (train_rows if split == "train" else val_rows).append(row)
    for i, text in enumerate(NEG_TRAIN):
        rid += 1
        split = "val" if i >= len(NEG_TRAIN) - 3 else "train"
        row = make_neg_row(f"r1-{rid:04d}", text, "collision_neg", split)
        (train_rows if split == "train" else val_rows).append(row)
    write(out / "r1_train_extra.jsonl", train_rows)
    write(out / "r1_val_extra.jsonl", val_rows)
    print(f"학습 추가 train: {len(train_rows)}행, val: {len(val_rows)}행 -> {out}")

    holdout_rows = []
    rid = 0
    for text, name, _split in holdout_pos:
        rid += 1
        holdout_rows.append(
            make_eval_row(f"r1-ho-{rid:04d}", text, name, "collision_holdout")
        )
    for text in NEG_HOLDOUT:
        rid += 1
        holdout_rows.append(
            make_eval_row(f"r1-ho-{rid:04d}", text, None, "collision_holdout_neg")
        )
    write(out / "r1_heldout_eval.jsonl", holdout_rows)
    print(f"held-out 평가 전용: {len(holdout_rows)}행 -> {out}")


if __name__ == "__main__":
    main()
