"""R11 파생 모델 생성 시험(가중치 평균 규칙, B-new 선정 규칙, 후보 폴더 생성과 정합성 중단 조건). 학습 없이 작은 텐서로 확인한다."""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import r11_derive as d  # noqa: E402


def sd(v, ids=0):
    return {"w": torch.full((2, 3), float(v)), "b": torch.tensor([float(v)]), "position_ids": torch.tensor([ids])}


def test_average_is_exact_uniform_mean_and_copies_integer_from_last():
    out = d.average_state_dicts([sd(1, 1), sd(2, 2), sd(6, 3)])
    assert torch.allclose(out["w"], torch.full((2, 3), 3.0)) and torch.allclose(out["b"], torch.tensor([3.0]))
    assert out["position_ids"].item() == 3  # 정수형은 마지막 에폭 값


def test_swa_window_is_last_three_trained_epochs_or_fewer():
    assert d.swa_epochs(6, {1, 2, 3, 4, 5, 6}) == [4, 5, 6]
    assert d.swa_epochs(4, {1, 2, 3, 4}) == [2, 3, 4]
    assert d.swa_epochs(2, {1, 2}) == [1, 2]


def test_select_epoch_pair_accuracy_then_recall_then_exposure_then_f1_then_early():
    v = {1: {"pair_context_accuracy": 0.2, "exact_recall": 0.7, "person_total_exposure": 0.20},
         2: {"pair_context_accuracy": 0.3, "exact_recall": 0.7, "person_total_exposure": 0.20},
         3: {"pair_context_accuracy": 0.3, "exact_recall": 0.8, "person_total_exposure": 0.20}}
    assert d.select_epoch(v, {1: 0.9, 2: 0.9, 3: 0.9})[0] == 3  # 쌍 정확률 동률 -> Recall
    v[3]["exact_recall"] = 0.7
    v[3]["person_total_exposure"] = 0.19
    assert d.select_epoch(v, {1: 0.9, 2: 0.9, 3: 0.9})[0] == 3  # Recall 동률 -> 노출 낮은 쪽
    v[3]["person_total_exposure"] = 0.20
    assert d.select_epoch(v, {1: 0.9, 2: 0.95, 3: 0.9})[0] == 2  # 노출 동률 -> baseline_val F1
    assert d.select_epoch(v, {1: 0.9, 2: 0.9, 3: 0.9})[0] == 2  # 전부 동률 -> 더 이른 에폭


def test_select_excludes_epoch_with_exposure_worse_than_best_by_more_than_1pp():
    v = {1: {"pair_context_accuracy": 0.9, "exact_recall": 0.9, "person_total_exposure": 0.30},
         2: {"pair_context_accuracy": 0.3, "exact_recall": 0.5, "person_total_exposure": 0.10}}
    pick, excluded = d.select_epoch(v, {1: 0.9, 2: 0.9})
    assert pick == 2 and excluded == [1]


def make_run(tmp_path, epochs=(1, 2, 3, 4), best=2, corrupt_best=False, drop=None):
    run = tmp_path / "run"
    (run / "best").mkdir(parents=True)
    for f in d.TOKENIZER_FILES:
        (run / "best" / f).write_text("{}", encoding="utf-8")
    for e in epochs:
        if e == drop:
            continue
        (run / f"epoch_{e}").mkdir()
        st.save_file(sd(e), str(run / f"epoch_{e}" / "model.safetensors"))
    st.save_file(sd(best + (99 if corrupt_best else 0)), str(run / "best" / "model.safetensors"))
    (run / "train_log.json").write_text(json.dumps({"stopped_epoch": epochs[-1], "best_epoch": best, "epochs": [{"epoch": e, "val_score": 0.9} for e in epochs]}), encoding="utf-8")
    return run


def test_cands_builds_epoch_dirs_and_swa_mean(tmp_path):
    run = make_run(tmp_path)
    d.cmd_cands(str(run))
    assert (run / "cand" / "ep3" / "best" / "model.safetensors").exists() and (run / "cand" / "swa" / "train_log.json").exists()
    swa = st.load_file(str(run / "cand" / "swa" / "best" / "model.safetensors"))
    assert torch.allclose(swa["w"], torch.full((2, 3), 3.0))  # (2+3+4)/3
    info = json.loads((run / "cand_info.json").read_text(encoding="utf-8"))
    assert info["candidates"]["swa"]["window_epochs"] == [2, 3, 4]


def test_cands_stops_when_best_differs_from_epoch_best(tmp_path):
    with pytest.raises(SystemExit):
        d.cmd_cands(str(make_run(tmp_path, corrupt_best=True)))


def test_cands_stops_when_epoch_checkpoint_missing(tmp_path):
    with pytest.raises(SystemExit):
        d.cmd_cands(str(make_run(tmp_path, drop=3)))
