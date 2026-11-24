"""데이터 생성·누출 방지·test 미사용 보장 시험."""

import ast
import inspect
import random
import re
from pathlib import Path

import build_ner_data as b
import evaluate_ner as ev
import leak_guard
import ner_common as nc
import pytest
import synth

ML = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def guard():
    return leak_guard.load_test_guard()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("ner_data")
    calls = []
    orig = nc.read_jsonl

    def spy(path):
        calls.append((Path(path).name, inspect.stack()[1].function))
        return orig(path)

    nc.read_jsonl = spy
    try:
        res = b.build_all(out, per_template=6, seed=3, use_klue=False)
    finally:
        nc.read_jsonl = orig
    return out, res, calls


def test_template_and_pool_sizes():
    assert len(synth.TEMPLATES) >= 40
    assert len(synth.SURNAMES) >= 40 and len(set(synth.GIVEN2)) >= 100
    assert len(synth.split_templates()) == len(synth.TEMPLATES)


def test_template_split_is_disjoint_and_covers_both():
    sp = synth.split_templates()
    assert {t["split"] for t in sp} == {"train", "val"}
    assert len({t["template_id"] for t in sp}) == len(sp)


def test_value_pools_disjoint_between_splits(guard):
    tr, va = synth.Pools("train", guard), synth.Pools("val", guard)
    for attr in ("given2", "given1", "nicks", "roads", "dongs", "complexes", "stations", "cities", "hosp", "school", "comp"):
        assert not set(getattr(tr, attr)) & set(getattr(va, attr)), attr
        assert getattr(va, attr), attr


def test_pools_exclude_test_values(guard):
    for split in ("train", "val"):
        p = synth.Pools(split, guard)
        for g in p.given2 + p.given1 + p.nicks:
            assert not guard.blocked_name(g), g
        for r in p.roads + p.dongs + p.complexes:
            assert not guard.blocked_text_value(r, "ADDRESS"), r
        for o in p.hosp + p.school + p.comp:
            assert not guard.blocked_text_value(o, "ORGANIZATION"), o


def test_synth_spans_are_exact_values_without_padding(built):
    out, _, _ = built
    for f in ("synth_train", "synth_val"):
        for r in nc.read_jsonl(out / f"{f}.jsonl"):
            last = -1
            for sp in sorted(r["spans"], key=lambda s: s["start"]):
                v = r["text"][sp["start"] : sp["end"]]
                assert v and v == v.strip() and sp["start"] >= last
                last = sp["end"]
                assert sp["label"] in nc.NER_LABELS


def test_no_leak_violations_and_all_labels_present(built):
    _, res, _ = built
    assert res["violations"] == {}, res["violations"]
    assert set(res["stats"]["synth_train"]["spans"]) == set(nc.NER_LABELS)


def test_kin_terms_and_banks_have_no_label(built):
    out, _, _ = built
    bank_rows = 0
    for r in nc.read_jsonl(out / "synth_train.jsonl"):
        for sp in r["spans"]:
            v = r["text"][sp["start"] : sp["end"]]
            assert v not in {"엄마", "아빠", "누나", "형", "할머니"}
            assert v not in synth.BANKS  # 금융기관명은 라벨 없음(계좌는 규칙 몫)
        bank_rows += any(bk in r["text"] for bk in synth.BANKS)
    assert bank_rows > 0  # 은행 문장은 음성으로 들어 있다


def test_ambiguous_templates_and_values_removed():
    texts = [t for _, t in synth.TEMPLATES]
    assert not any("근처" in t and "{AD" in t for t in texts)  # 주소 뒤 '근처' 틀 제거
    assert not any("@과/와" in t and "{P}" in t for t in texts)  # 이름+과 모호
    assert not any("집 근처 마트" in t for t in texts)
    for bad in ("보리", "우리왕자", "도토리", "막내공주"):
        assert bad not in synth.NICKS
    for ch in "건결담":
        assert ch not in synth.GIVEN1


def test_augment_adds_categories_and_stays_leak_free(tmp_path):
    res = b.build_all(tmp_path, per_template=4, seed=5, use_klue=False, augment=True)
    assert res["violations"] == {}, res["violations"]
    cats = set(res["stats"]["synth_categories"])
    assert {c for c, _ in synth.AUG_TEMPLATES} <= cats
    assert len(synth.AUG_TEMPLATES) >= 20
    assert (tmp_path / "synth_val_core.jsonl").exists()
    core = nc.read_jsonl(tmp_path / "synth_val_core.jsonl")
    assert all(not r["augment"] for r in core)


# 로드맵 2단계 오류 분석에서 AI-Hub 교체 모델이 특히 못 맞힌 "직함+이름" 기사체 문장(저장소 apps/api 쪽
# evalset_v4 의 기사체 틀). 학습에 이 틀이 그대로 섞여 들어가면(=평가셋과 겹침) 오염이라, 글자 그대로 같은 틀을
# 여기서 다시 만들지 않는다는 걸 명시적으로 검사한다.
EVALSET_V4_기사체_틀 = {
    "{N} 대표가 사업 계획을 공개했다",
    "{N} 의원이 개정안을 제출했다",
    '목격자 {N}(37)씨는 "놀랐다"고 전했다',
    "{N} 감독이 인터뷰에서 소감을 밝혔다",
    "검찰은 {N|를/를} 참고인으로 소환했다",
    "{N} 연구원이 논문을 공개했다",
    "{N} 선수가 역전 홈런을 쳤다",
    "{N} 군수는 피해 지역을 점검했다",
}


def test_news_title_augment_adds_person_exposure_without_leak(tmp_path, guard):
    """news_title 보강 틀(기사체 노출 늘리기, 2단계 오류 분석 대응)이 2단계 증강에 포함되고, PERSON 라벨이
    실제로 나오며, evalset_v4 기사체 틀과 글자 그대로 겹치지 않는지 확인한다."""
    news_templates = [t for c, t in synth.AUG_TEMPLATES_2 if c == "news_title"]
    assert len(news_templates) >= 8
    for t in news_templates:
        assert t not in EVALSET_V4_기사체_틀

    res = b.build_all(tmp_path, per_template=4, seed=5, use_klue=False, augment=2)
    assert res["violations"] == {}, res["violations"]
    cats = set(res["stats"]["synth_categories"])
    assert "news_title" in cats

    rows = nc.read_jsonl(tmp_path / "synth_train.jsonl") + nc.read_jsonl(tmp_path / "synth_val.jsonl")
    news_rows = [r for r in rows if r["category"] == "news_title"]
    assert news_rows
    for r in news_rows:
        person_spans = [s for s in r["spans"] if s["kind"] == "PERSON"]
        assert person_spans, r
        for s in person_spans:
            assert r["text"][s["start"] : s["end"]].strip() == r["text"][s["start"] : s["end"]]
            assert s["end"] > s["start"]


def test_test_file_only_read_by_guard_and_no_test_rows_in_outputs(built):
    out, _, calls = built
    test_reads = [c for c in calls if c[0] == "test.jsonl"]
    assert test_reads and all(fn == "load_test_guard" for _, fn in test_reads)
    test_rows = nc.read_jsonl(nc.BASELINE_DIR / "test.jsonl")
    ids = {r["id"] for r in test_rows}
    texts = {nc.normalize_text(r["text"]) for r in test_rows}
    for f in out.glob("*.jsonl"):
        for r in nc.read_jsonl(f):
            assert r["id"] not in ids
            assert nc.normalize_text(r["text"]) not in texts


def test_load_baseline_refuses_test():
    with pytest.raises(ValueError):
        b.load_baseline("test")


def test_only_leak_guard_and_evaluate_reference_test_file():
    """data/, train/ 코드에서 test 파일명 문자열(독스트링 제외)을 쓰는 곳은 evaluate_ner 뿐이고, leak_guard 는 조각 결합으로만 만든다."""
    bad = []
    for path in [*(ML / "data").glob("*.py"), *(ML / "train").glob("*.py"), ML / "ner_common.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        doc_ids = {
            id(n.body[0].value)
            for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
            and n.body
            and isinstance(n.body[0], ast.Expr)
            and isinstance(getattr(n.body[0], "value", None), ast.Constant)
        }
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc_ids:
                if re.search(r"test\.jsonl|^test$|baseline_test", n.value):
                    bad.append((path.name, n.value))
    assert all(name in ("evaluate_ner.py", "leak_guard.py") for name, _ in bad), bad
    # 학습·데이터 생성 스크립트는 평가 데이터 지정자(test)도 받지 않는다
    for name in ("train_ner.py", "build_ner_data.py"):
        assert "--final" not in (ML / ("train" if name == "train_ner.py" else "data") / name).read_text(encoding="utf-8")


def test_evaluate_refuses_test_without_final():
    with pytest.raises(ev.TestRefused):
        ev.load_eval_rows("baseline_test", final=False)
    with pytest.raises(ev.TestRefused):
        ev.load_eval_rows("synth_test", final=True)
    assert len(ev.load_eval_rows("baseline_test", final=True)) == 180
    assert len(ev.load_eval_rows("baseline_val")) == 180


def test_leak_check_detects_copied_test_sentence_and_name(guard):
    test_rows = nc.read_jsonl(nc.BASELINE_DIR / "test.jsonl")
    r = next(x for x in test_rows if any(s["kind"] == "PERSON" for s in x["spans"]))
    row = {"id": "x", "text": r["text"], "spans": nc.to_ner_spans(r), "source": "synth", "template_id": "t", "split": "train"}
    empty = {k: [] for k in ("synth_val", "baseline_train", "baseline_val", "klue_train", "klue_val")}
    res = b.leak_checks({**empty, "synth_train": [row]}, guard)
    viol = {k for k, v in res.items() if v}
    assert any("정규화 문장" in k for k in viol)
    assert any("문장 틀" in k for k in viol)
    assert any("PERSON 값" in k or "이름" in k for k in viol)


def test_leak_check_detects_cross_split_template_sharing(guard):
    row = {"id": "a", "text": "민준이 왔어", "spans": [{"label": "PERSON", "start": 0, "end": 2}], "source": "synth",
           "template_id": "t1", "split": "train"}
    row2 = dict(row, id="b", split="val", text="서연이 왔어")
    empty = {k: [] for k in ("baseline_train", "baseline_val", "klue_train", "klue_val")}
    res = b.leak_checks({**empty, "synth_train": [row], "synth_val": [row2]}, guard)
    assert res["synth: template_id 가 train/val 에 걸침"] == ["t1"]
    assert res["synth: 문장 틀이 train/val 에 걸침"]


def test_generation_is_deterministic(guard):
    a = synth.generate("val", guard, 3, seed=5)
    c = synth.generate("val", guard, 3, seed=5)
    assert [r["text"] for r in a] == [r["text"] for r in c]


def test_josa_selection():
    class P:
        given2 = ["철수"]

    segs = synth.fill_template("{G}@이/가 {G}@은/는", P(), random.Random(0))
    assert "".join(t for t, _ in segs) == "철수가 철수는"
    P.given2 = ["민준"]
    assert "".join(t for t, _ in synth.fill_template("{G}@이/가", P(), random.Random(0))) == "민준이"
