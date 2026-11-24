"""`train_ner.py --save-every-epoch` 회귀 시험: 켜도 기존 학습 결과(best 가중치, 에폭별 손실·검증 점수)가 달라지지 않고,
꺼진 기본 경로는 에폭 저장을 만들지 않는다. 작은 무작위 BERT·문자 단위 어휘로 CPU 에서 수 초 안에 돈다(모델 다운로드 없음)."""

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("seqeval")

import train_ner  # noqa: E402

NAMES = ["김철수", "이영희", "박민준", "최서연", "정하늘", "강다은"]


def _rows(n, offset):
    rows = []
    for i in range(n):
        name = NAMES[(i + offset) % len(NAMES)]
        text = f"{name}가 내일 시장에 간다" if i % 2 == 0 else "오늘 날씨가 맑다"
        spans = [{"label": "PERSON", "start": 0, "end": len(name)}] if i % 2 == 0 else []
        rows.append({"id": f"r{offset}_{i}", "text": text, "spans": spans})
    return rows


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny_train")
    data = root / "data"
    data.mkdir()
    train, val = _rows(24, 0), _rows(8, 3)
    for name, rows in (("tr", train), ("va", val)):
        (data / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    chars = sorted({c for r in train + val for c in r["text"] if not c.isspace()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *chars, *[f"##{c}" for c in chars]]
    model_dir = root / "model"
    model_dir.mkdir()
    (model_dir / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tok = transformers.BertTokenizerFast(vocab_file=str(model_dir / "vocab.txt"))
    tok.save_pretrained(model_dir)
    cfg = transformers.BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                  intermediate_size=32, max_position_embeddings=64)
    torch.manual_seed(0)
    transformers.BertModel(cfg).save_pretrained(model_dir)
    return {"root": root, "data": data, "model": model_dir}


def _run(tiny, name, *extra):
    out = tiny["root"] / name
    argv = ["--model", str(tiny["model"]), "--train-files", "tr", "--val-files", "va", "--select-on", "va", "--epochs", "3",
            "--batch-size", "8", "--max-len", "32", "--seed", "7", "--no-fp16", "--out-root", str(out), "--data-dir", str(tiny["data"]),
            "--run-id", "run", *extra]
    log = train_ner.main(argv)
    return out / "run", log


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_flag_does_not_change_training_result(tiny):
    d0, log0 = _run(tiny, "plain")
    d1, log1 = _run(tiny, "flag", "--save-every-epoch")
    assert _sha(d0 / "best" / "model.safetensors") == _sha(d1 / "best" / "model.safetensors")
    assert [e["train_loss"] for e in log0["epochs"]] == [e["train_loss"] for e in log1["epochs"]]
    assert [e["val_score"] for e in log0["epochs"]] == [e["val_score"] for e in log1["epochs"]]
    assert log0["best_epoch"] == log1["best_epoch"] and log0["stopped_epoch"] == log1["stopped_epoch"]


def test_default_path_writes_no_epoch_dirs(tiny):
    d0, log0 = _run(tiny, "plain2")
    assert not list(d0.glob("epoch_*"))
    assert all("checkpoint_sha256" not in e for e in log0["epochs"])


def test_flag_saves_every_epoch_with_hashes_and_matches_best(tiny):
    d1, log1 = _run(tiny, "flag2", "--save-every-epoch")
    n = len(log1["epochs"])
    assert sorted(p.name for p in d1.glob("epoch_*")) == [f"epoch_{k}" for k in range(1, n + 1)]
    for e in log1["epochs"]:
        assert e["checkpoint_sha256"] == _sha(d1 / f"epoch_{e['epoch']}" / "model.safetensors")
    # 같은 런에서 best/ 는 best_epoch 의 에폭 저장본과 같다(저장 정합성 점검, 추가 학습 없음)
    assert _sha(d1 / "best" / "model.safetensors") == _sha(d1 / f"epoch_{log1['best_epoch']}" / "model.safetensors")


def test_no_save_model_ignores_flag(tiny):
    d, _ = _run(tiny, "nosave", "--save-every-epoch", "--no-save-model")
    assert not list(d.glob("epoch_*")) and not (d / "best").exists()
