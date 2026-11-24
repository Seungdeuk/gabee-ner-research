"""NER 학습 데이터 생성: KLUE-NER 변환 + Baseline(train/val) + 합성 가족 대화.

산출물은 저장소 밖(기본 <VENVS>\\ner-runs\\data)에 쓴다. KLUE 원문은 저장소에 두지 않는다(매번 HF 캐시에서 읽어 변환).
test.jsonl 은 학습·검증 데이터로 읽지 않는다. leak_guard.load_test_guard 만 차단 목록·누출 검사용으로 읽는다.

실행(저장소 루트에서)::

    set HF_HOME=<VENVS>\\hf-cache
    <VENVS>\\gabee-ner\\Scripts\\python.exe ml/ner/data/build_ner_data.py [--per-template 40] [--out DIR]
"""

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ner_common as nc  # noqa: E402
import synth  # noqa: E402
from leak_guard import load_test_guard  # noqa: E402

# KLUE-NER 정수 태그(데이터셋 카드의 순서). 0 B-DT 1 I-DT 2 B-LC 3 I-LC 4 B-OG 5 I-OG 6 B-PS 7 I-PS 8 B-QT 9 I-QT 10 B-TI 11 I-TI 12 O
KLUE_TAGS = ["B-DT", "I-DT", "B-LC", "I-LC", "B-OG", "I-OG", "B-PS", "I-PS", "B-QT", "I-QT", "B-TI", "I-TI", "O"]
KLUE_MAP = {"PS": "PERSON", "LC": "LOCATION", "OG": "ORGANIZATION"}  # DT/TI/QT 는 O


def klue_to_row(tokens, tags, idx, split):
    """KLUE 문자 단위 태그를 문자 span 행으로 바꾼다. 문장은 문자를 이어 붙여 만든다.

    @param tokens: 문자 목록(공백 포함)
    @param tags: 정수 태그 목록
    @param idx: 행 번호
    @param split: "train" 또는 "val"
    @returns: 행 dict
    """
    text = "".join(tokens)
    spans, cur = [], None
    for i, t in enumerate(tags):
        name = KLUE_TAGS[int(t)]
        if name == "O":
            kind, begin = None, False
        else:
            bi, kind = name.split("-")
            begin = bi == "B"
        if kind is None:
            if cur:
                spans.append(cur)
                cur = None
            continue
        if cur and not begin and cur["kind"] == kind:
            cur["end"] = i + 1
            continue
        if cur:
            spans.append(cur)
        cur = {"kind": kind, "start": i, "end": i + 1}
    if cur:
        spans.append(cur)
    out = [{"label": KLUE_MAP[s["kind"]], "start": s["start"], "end": s["end"]} for s in spans if s["kind"] in KLUE_MAP]
    return {"id": f"klue-{split}-{idx:05d}", "text": text, "spans": out, "source": "klue", "template_id": None, "split": split}


def load_klue(split):
    """HF_HOME 아래 klue/ner parquet 를 찾아 변환한다.

    @param split: "train" 또는 "validation"
    @returns: 행 목록(없으면 FileNotFoundError)
    """
    import pandas as pd

    home = os.environ.get("HF_HOME", str(Path.home() / "venvs" / "hf-cache"))
    found = glob.glob(os.path.join(home, "hub", "datasets--klue--klue", "snapshots", "*", "ner", f"{split}-*.parquet"))
    if not found:
        raise FileNotFoundError(f"KLUE-NER {split} parquet 를 HF_HOME={home} 에서 찾지 못했다")
    df = pd.read_parquet(found[0])
    name = "val" if split == "validation" else "train"
    return [klue_to_row(list(r.tokens), list(r.ner_tags), i, name) for i, r in enumerate(df.itertuples(index=False))]


def load_baseline(split):
    """Baseline train/val 을 NER 라벨만 남겨 읽는다(test 는 이 함수로 읽을 수 없다).

    @param split: "train" 또는 "val"
    @returns: 행 목록
    """
    if split not in ("train", "val"):
        raise ValueError("Baseline 은 train/val 만 학습 데이터로 쓴다")
    rows = nc.read_jsonl(nc.BASELINE_DIR / f"{split}.jsonl")
    return [{"id": r["id"], "text": r["text"], "spans": nc.to_ner_spans(r), "source": "baseline",
             "template_id": r["template_id"], "split": split} for r in rows]


def label_counts(rows):
    """행 목록의 라벨별 span 수와 span 없는 문장 수.

    @param rows: 행 목록
    @returns: dict
    """
    c = collections.Counter(sp["label"] for r in rows for sp in r["spans"])
    return {"sentences": len(rows), "negative_sentences": sum(1 for r in rows if not r["spans"]),
            "spans": dict(c), "chars": sum(len(r["text"]) for r in rows)}


def span_values(rows):
    """label -> span 문자열 집합."""
    out = collections.defaultdict(set)
    for r in rows:
        for sp in r["spans"]:
            out[sp["label"]].add(r["text"][sp["start"] : sp["end"]])
    return out


def leak_checks(sets, guard):
    """누출 검사. 모든 항목이 비어 있어야(0) 통과.

    @param sets: {"synth_train","synth_val","baseline_train","baseline_val","klue_train","klue_val"} -> 행 목록
    @param guard: TestGuard
    @returns: {검사 이름: 위반 목록}
    """
    res = {}
    syn = sets["synth_train"] + sets["synth_val"]
    allr = {k: v for k, v in sets.items()}
    for name, rows in allr.items():
        res[f"{name}: test 정규화 문장 중복"] = [r["id"] for r in rows if nc.normalize_text(r["text"]) in guard.texts_norm]
        if name.startswith("klue"):
            continue
        res[f"{name}: test 문장 틀 중복(NER 라벨 기준)"] = sorted(
            {r["id"] for r in rows if nc.skeleton(r["text"], r["spans"]) in guard.skeletons_ner})
    res["synth: test 문장 틀 중복(전체 라벨 기준)"] = sorted(
        {r["id"] for r in syn if nc.skeleton(r["text"], r["spans"]) in guard.skeletons_all})
    for name in ("synth_train", "synth_val"):
        vals = span_values(sets[name])
        for lab in nc.NER_LABELS:
            hit = sorted(v for v in vals.get(lab, ()) if v in guard.values.get(lab, set()))
            res[f"{name}: test 의 {lab} 값과 같은 span 문자열"] = hit
        names = vals.get("PERSON", set()) | vals.get("PERSON_ALIAS", set())
        res[f"{name}: test 이름·별명 문자열 또는 이름 부분을 포함하는 span"] = sorted(
            v for v in names if any(g in v.split() or g == v for g in guard.given_parts if len(g) >= 2) or guard.blocked_name(v))
    st, sv = sets["synth_train"], sets["synth_val"]
    res["synth: template_id 가 train/val 에 걸침"] = sorted({r["template_id"] for r in st} & {r["template_id"] for r in sv})
    res["synth: 정규화 문장이 train/val 에 걸침"] = sorted(
        {nc.normalize_text(r["text"]) for r in st} & {nc.normalize_text(r["text"]) for r in sv})
    res["synth: 문장 틀이 train/val 에 걸침"] = sorted(
        {nc.skeleton(r["text"], r["spans"]) for r in st} & {nc.skeleton(r["text"], r["spans"]) for r in sv})
    vt, vv = span_values(st), span_values(sv)
    for lab in nc.NER_LABELS:
        res[f"synth: {lab} 값이 train/val 에 걸침"] = sorted(vt.get(lab, set()) & vv.get(lab, set()))
    return res


def info_overlap(sets, guard):
    """KLUE·Baseline span 값이 test span 값과 겹치는 개수(정보용; 통과/실패 기준 아님)."""
    out = {}
    for name in ("klue_train", "klue_val", "baseline_train", "baseline_val"):
        vals = span_values(sets[name])
        out[name] = {lab: len(vals.get(lab, set()) & guard.values.get(lab, set())) for lab in nc.NER_LABELS}
    return out


def build_all(out_dir, per_template=40, seed=42, use_klue=True, augment=False):
    """모든 산출 jsonl·통계·누출 보고서를 만든다.

    @param out_dir: 산출 폴더(저장소 밖)
    @param per_template: 합성 틀당 문장 수
    @param seed: 시드
    @param use_klue: False 면 KLUE 를 건너뛴다(시험용)
    @param augment: True 면 오탐 감소 보강 틀(synth.AUG_TEMPLATES)을 포함한다
    @returns: {"stats", "leak", "violations"}
    """
    out_dir = Path(out_dir)
    guard = load_test_guard()
    sets = {
        "baseline_train": load_baseline("train"),
        "baseline_val": load_baseline("val"),
        "synth_train": synth.generate("train", guard, per_template, seed, augment),
        "synth_val": synth.generate("val", guard, per_template, seed, augment),
        "klue_train": load_klue("train") if use_klue else [],
        "klue_val": load_klue("validation") if use_klue else [],
    }
    leak = leak_checks(sets, guard)
    violations = {k: v for k, v in leak.items() if v}
    stats = {k: label_counts(v) for k, v in sets.items()}
    tpls = synth.split_templates(augment=augment)
    stats["synth_templates"] = {
        "total": len(tpls),
        "train": sum(1 for t in tpls if t["split"] == "train"),
        "val": sum(1 for t in tpls if t["split"] == "val")}
    stats["synth_categories"] = dict(collections.Counter(r["category"] for r in sets["synth_train"] + sets["synth_val"]))
    for k, v in sets.items():
        nc.write_jsonl(out_dir / f"{k}.jsonl", v)
    # 보강 전후를 같은 검증 틀로 비교하려고 합성 val 을 core(보강 전 틀)와 aug(보강 틀)로 나눠 따로 쓴다
    nc.write_jsonl(out_dir / "synth_val_core.jsonl", [r for r in sets["synth_val"] if not r.get("augment")])
    nc.write_jsonl(out_dir / "synth_val_aug.jsonl", [r for r in sets["synth_val"] if r.get("augment")])
    report = {"leak_checks": {k: {"violations": len(v), "examples": v[:5]} for k, v in leak.items()},
              "info_value_overlap_with_test": info_overlap(sets, guard), "seed": seed, "per_template": per_template, "augment": augment}
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "leak_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"stats": stats, "leak": report, "violations": violations}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(nc.DATA_OUT))
    ap.add_argument("--per-template", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-klue", action="store_true")
    ap.add_argument(
        "--augment",
        type=int,
        default=0,
        help="오탐 감소 보강 단계: 0 없음, 1 보강 틀, 2 +처음 보는 금융기관 음성(+기사체), 3 +메신저 다인칭 문장",
    )
    a = ap.parse_args()
    res = build_all(a.out, a.per_template, a.seed, not a.no_klue, a.augment)
    print(json.dumps({k: v["sentences"] if isinstance(v, dict) and "sentences" in v else v for k, v in res["stats"].items()},
                     ensure_ascii=False))
    if res["violations"]:
        print("누출 검사 실패:", json.dumps({k: len(v) for k, v in res["violations"].items()}, ensure_ascii=False))
        return 1
    print("누출 검사 통과(모든 항목 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
