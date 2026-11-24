"""R11 실행기 재개·후보별 평가 시험(학습·GPU 없음). 가짜 런 폴더와 가짜 단계 실행기로 재개 규칙을 확인한다.

확인: 후보별 eval_full.json 생성 단계와 메타데이터 정합, 후보 간 값 복사 방지, 기존 학습 산출물 유지(재학습 없음),
완료 단계 재실행 방지, 해시 불일치 중단, 부분 산출물 누락·단계 실패 중단, 완료된 런 중복 실행 방지.
"""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")
pytest.importorskip("psutil")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import r11_derive as d  # noqa: E402
import run_r11_B as r  # noqa: E402

SEED, EPOCHS, BEST = 7, (1, 2, 3), 2
RID = f"r11-B-s{SEED}"
MICRO = {"tp": 72, "fp": 8, "fn": 4, "precision": 0.9, "recall": 0.947, "f1": 0.923}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_eval(path, run_id, tp=72):
    m = {**MICRO, "tp": tp}
    path.write_text(json.dumps({"run_id": run_id, "sets": {"baseline_val": {"overlap": {"micro": m}, "exact": {"micro": m}}}}), encoding="utf-8")


@pytest.fixture
def env(tmp_path, monkeypatch):
    data, runs, export, dev = tmp_path / "data", tmp_path / "runs", tmp_path / "export", tmp_path / "dev"
    for p in (data, runs, export):
        p.mkdir()
    for name in r.BASE + [r.EXTRA]:
        (data / (name.split(":")[0] + ".jsonl")).write_text(name, encoding="utf-8")
    monkeypatch.setattr(r, "DATA", data)
    monkeypatch.setattr(r, "RUNS", runs)
    monkeypatch.setattr(r, "EXPORT", export)
    monkeypatch.setattr(r, "DEVOUT", dev)
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=8 * 2**30))
    monkeypatch.setattr(r, "gpu_info", lambda: None)
    run = runs / RID
    (run / "best").mkdir(parents=True)
    for f in d.TOKENIZER_FILES:
        (run / "best" / f).write_text("{}", encoding="utf-8")
    ep_sha = {}
    for e in EPOCHS:
        (run / f"epoch_{e}").mkdir()
        st.save_file({"w": torch.full((2, 2), float(e))}, str(run / f"epoch_{e}" / "model.safetensors"))
        ep_sha[e] = sha(run / f"epoch_{e}" / "model.safetensors")
    st.save_file({"w": torch.full((2, 2), float(BEST))}, str(run / "best" / "model.safetensors"))
    tl = {"config": {"seed": SEED, "run_id": RID, "save_every_epoch": True}, "train_sentences": r.EXPECT_ROWS, "finished": "x", "stopped_epoch": EPOCHS[-1], "best_epoch": BEST,
          "total_train_time_s": 1.0, "max_vram_mb": 1.0, "data_files": {n: {"sha256_12": sha(data / (n.split(":")[0] + ".jsonl"))[:12]} for n in r.BASE + [r.EXTRA]},
          "epochs": [{"epoch": e, "val_score": 0.9, "checkpoint_sha256": ep_sha[e]} for e in EPOCHS]}
    (run / "train_log.json").write_text(json.dumps(tl), encoding="utf-8")
    write_eval(run / "eval_full.json", RID)
    (run / "parity.json").write_text(json.dumps({"checkpoint_reload_identical_predictions": True, "onnx_vs_torch_token_agreement_pct": 100.0}), encoding="utf-8")
    (export / RID).mkdir()
    for f in ("model.onnx", "model_card.json", "labels.json", "tokenizer.json"):
        (export / RID / f).write_text("x", encoding="utf-8")
    d.cmd_cands(str(run))
    return SimpleNamespace(run=run, data=data, export=export, dev=dev, calls=[])


def fake_step(env, fail_on=None):
    def _step(name, cmd, run_id, timing, monitor=False, cwd=None):
        env.calls.append(name)
        if name == fail_on:
            sys.exit(1)
        if name.startswith("evalfull_"):
            cand = name[len("evalfull_"):]
            write_eval(env.run / "cand" / cand / "eval_full.json", cand)
        elif name.startswith("export_"):
            out = env.export / f"{RID}-{name[len('export_'):]}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "model.onnx").write_text(name, encoding="utf-8")
            (out / "model_card.json").write_text("{}", encoding="utf-8")
        elif name == "valEval":
            (env.run / "val_by_epoch.json").write_text("{}", encoding="utf-8")
        elif name == "select":
            (env.run / "selection.json").write_text(json.dumps({"B_old_epoch": BEST, "B_new_epoch": 3, "same_epoch": False, "excluded_by_safety": []}), encoding="utf-8")
        elif name == "devEval":
            env.dev.mkdir(exist_ok=True)
            (env.dev / f"{RID}.json").write_text("{}", encoding="utf-8")
        else:
            raise AssertionError(f"재개 시 호출되면 안 되는 단계: {name}")

    return _step


def snapshot(run):
    return {str(p.relative_to(run)): sha(p) for p in sorted(run.rglob("*")) if p.is_file() and "stages" not in p.parts and "cand" not in p.parts and not p.name.startswith(("val_by", "selection", "pipeline"))}


def test_resume_creates_per_candidate_eval_and_keeps_training_artifacts(env, monkeypatch):
    before = snapshot(env.run)
    monkeypatch.setattr(r, "step", fake_step(env))
    r.one(SEED, resume=True)
    assert snapshot(env.run) == before  # 기존 학습 산출물(체크포인트·train_log·eval_full·parity·cand_info) 불변
    for forbidden in ("train", "evalfull", "export_best", "parity", "cands"):
        assert forbidden not in env.calls  # 재학습·완료 단계 재실행 없음
    for c in ("ep1", "ep2", "ep3", "swa"):
        assert f"evalfull_{c}" in env.calls and f"export_{c}" in env.calls
        assert json.loads((env.run / "cand" / c / "eval_full.json").read_text(encoding="utf-8"))["run_id"] == c  # 후보 자기 평가
    done = json.loads((env.run / "pipeline_done.json").read_text(encoding="utf-8"))
    assert done["resumed"] and set(done["reused_stages"]) >= {"train", "evalfull", "export_best", "parity", "cands"}
    assert set(done["sha256"]["cand_eval_full"]) == {"ep1", "ep2", "ep3", "swa"}


def test_candidate_eval_is_not_copied_from_best(env):
    meta = json.loads((env.run / "cand_info.json").read_text(encoding="utf-8"))["candidates"]
    write_eval(env.run / "cand" / "ep3" / "eval_full.json", RID)  # best 평가를 그대로 복사한 상황(run_id 가 후보와 다름)
    with pytest.raises(SystemExit):
        r.check_cand_eval(env.run, "ep3", meta["ep3"]["weights_sha256"], BEST)
    write_eval(env.run / "cand" / "ep2" / "eval_full.json", "ep2", tp=71)  # best 에폭 후보인데 런 평가와 다름
    with pytest.raises(SystemExit):
        r.check_cand_eval(env.run, "ep2", meta["ep2"]["weights_sha256"], BEST)
    write_eval(env.run / "cand" / "ep3" / "eval_full.json", "ep3", tp=70)  # 다른 에폭은 값이 달라도 정상(복사하지 않음)
    r.check_cand_eval(env.run, "ep3", meta["ep3"]["weights_sha256"], BEST)


def test_completed_stages_are_skipped_on_second_resume(env, monkeypatch):
    monkeypatch.setattr(r, "step", fake_step(env, fail_on="valEval"))
    with pytest.raises(SystemExit):
        r.one(SEED, resume=True)  # 후보 단계 완료 뒤 valEval 에서 실패
    assert not (env.run / "pipeline_done.json").exists()
    first = list(env.calls)
    env.calls.clear()
    monkeypatch.setattr(r, "step", fake_step(env))
    r.one(SEED, resume=True)
    assert env.calls == ["valEval", "select", "devEval"]  # 이미 끝난 후보 평가·내보내기는 건너뜀
    assert len([c for c in first if c.startswith("export_")]) == 4


def test_partial_export_is_cleaned_not_trusted(env, monkeypatch):
    (env.export / f"{RID}-ep1").mkdir()
    (env.export / f"{RID}-ep1" / "model.onnx").write_text("partial", encoding="utf-8")  # export_ep1 실패 잔여물(model_card 없음)
    monkeypatch.setattr(r, "step", fake_step(env))
    r.one(SEED, resume=True)
    assert (env.export / f"{RID}-ep1" / "model.onnx").read_text(encoding="utf-8") == "export_ep1"
    assert "export_ep1" in env.calls


def test_resume_stops_on_checkpoint_hash_mismatch(env, monkeypatch):
    st.save_file({"w": torch.full((2, 2), 99.0)}, str(env.run / "epoch_1" / "model.safetensors"))
    monkeypatch.setattr(r, "step", fake_step(env))
    with pytest.raises(SystemExit) as e:
        r.one(SEED, resume=True)
    assert e.value.code == 5 and env.calls == []


def test_resume_stops_on_changed_data_hash(env, monkeypatch):
    (env.data / "r11_B_extra.jsonl").write_text("changed", encoding="utf-8")
    monkeypatch.setattr(r, "step", fake_step(env))
    with pytest.raises(SystemExit):
        r.one(SEED, resume=True)
    assert env.calls == []


@pytest.mark.parametrize("victim", ["parity.json", "eval_full.json", "cand_info.json"])
def test_resume_stops_when_artifact_missing(env, monkeypatch, victim):
    (env.run / victim).unlink()
    monkeypatch.setattr(r, "step", fake_step(env))
    with pytest.raises(SystemExit):
        r.one(SEED, resume=True)
    assert env.calls == [] and not (env.run / "pipeline_done.json").exists()


def test_failed_stage_stops_and_writes_no_done_marker(env, monkeypatch):
    monkeypatch.setattr(r, "step", fake_step(env, fail_on="export_ep2"))
    with pytest.raises(SystemExit):
        r.one(SEED, resume=True)
    assert not (env.run / "pipeline_done.json").exists()
    assert not (env.run / "stages" / "export_ep2.done.json").exists() and (env.run / "stages" / "evalfull_ep1.done.json").exists()
    assert "export_ep3" not in env.calls  # 실패 즉시 중단


def test_completed_run_is_not_executed_again(env, monkeypatch):
    (env.run / "pipeline_done.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(r, "step", fake_step(env))
    r.one(SEED, resume=True)
    r.one(SEED, resume=False)
    assert env.calls == []


def test_existing_run_dir_without_resume_is_never_overwritten(env, monkeypatch):
    monkeypatch.setattr(r, "step", fake_step(env))
    with pytest.raises(SystemExit) as e:
        r.one(SEED, resume=False)
    assert e.value.code == 4 and env.calls == []


def test_per_seed_disk_gate_stops_without_running_any_stage(env, monkeypatch):
    monkeypatch.setattr(r, "step", fake_step(env))
    monkeypatch.setattr(r.shutil, "disk_usage", lambda p: SimpleNamespace(free=int(2 * 2**30)))
    with pytest.raises(SystemExit) as e:
        r.one(SEED, resume=True)
    assert e.value.code == 3 and env.calls == []


def test_verify_cands_detects_tampered_swa(env):
    assert d.verify_cands(env.run) == []
    st.save_file({"w": torch.full((2, 2), 5.0)}, str(env.run / "cand" / "swa" / "best" / "model.safetensors"))
    assert any("swa" in p for p in d.verify_cands(env.run))
