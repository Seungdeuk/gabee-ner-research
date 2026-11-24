import pytest

from app.shared.privacy.eval.style_v4.exclusions import collect_excluded_names


def test_collect_excluded_names_scans_all_baseline_splits_and_extra(tmp_path):
    synth = tmp_path / "synth.py"
    synth.write_text('GIVEN2 = ("가나다 라마바")\nGIVEN1 = list("사")\n', encoding="utf-8")
    base_dir = tmp_path / "baseline_v2"
    base_dir.mkdir()
    (base_dir / "train.jsonl").write_text(
        '{"text": "김철수님", "spans": [{"kind": "PERSON", "start": 0, "end": 3}]}\n',
        encoding="utf-8",
    )
    (base_dir / "val.jsonl").write_text(
        '{"text": "이영희씨", "spans": [{"kind": "PERSON", "start": 0, "end": 3}]}\n',
        encoding="utf-8",
    )
    (base_dir / "test.jsonl").write_text(
        '{"text": "박민수가", "spans": [{"kind": "PERSON", "start": 0, "end": 3}]}\n',
        encoding="utf-8",
    )
    v3 = tmp_path / "evalset_v3.jsonl"
    v3.write_text(
        '{"text": "최지은 대표", "spans": [{"kind": "PERSON", "start": 0, "end": 3}]}\n',
        encoding="utf-8",
    )

    excluded = collect_excluded_names(synth, [base_dir], [v3])

    assert "가나다" in excluded or "라마바" in excluded  # GIVEN2 토큰
    assert "사" in excluded  # GIVEN1
    assert "김철수" in excluded  # train
    assert "이영희" in excluded  # val — 빠지면 실패해야 하는 지점
    assert "박민수" in excluded  # test
    assert "최지은" in excluded  # v3


def test_collect_excluded_names_raises_on_missing_baseline_dir(tmp_path):
    synth = tmp_path / "synth.py"
    synth.write_text('GIVEN2 = ("가나다")\nGIVEN1 = list("사")\n', encoding="utf-8")
    v3 = tmp_path / "evalset_v3.jsonl"
    v3.write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        collect_excluded_names(synth, [tmp_path / "no-such-baseline-dir"], [v3])


def test_collect_excluded_names_raises_on_empty_baseline_dir(tmp_path):
    synth = tmp_path / "synth.py"
    synth.write_text('GIVEN2 = ("가나다")\nGIVEN1 = list("사")\n', encoding="utf-8")
    empty_dir = tmp_path / "empty_baseline"
    empty_dir.mkdir()
    v3 = tmp_path / "evalset_v3.jsonl"
    v3.write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        collect_excluded_names(synth, [empty_dir], [v3])


def test_collect_excluded_names_raises_when_synth_pool_empty(tmp_path):
    synth = tmp_path / "synth.py"
    synth.write_text("# GIVEN1/GIVEN2 없음\n", encoding="utf-8")
    base_dir = tmp_path / "baseline_v2"
    base_dir.mkdir()
    (base_dir / "train.jsonl").write_text(
        '{"text": "김철수님", "spans": [{"kind": "PERSON", "start": 0, "end": 3}]}\n',
        encoding="utf-8",
    )
    v3 = tmp_path / "evalset_v3.jsonl"
    v3.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        collect_excluded_names(synth, [base_dir], [v3])
