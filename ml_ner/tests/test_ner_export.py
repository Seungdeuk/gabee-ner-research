"""ONNX 내보내기 산출물 형식·런타임 보조 함수 시험. 산출 폴더(NER_EXPORT_DIR)가 없으면 폴더 의존 시험은 건너뛴다."""

import hashlib
import json
import os
from pathlib import Path

import ner_common as nc
import onnx_ner
import pytest

EXPORT_DIR = Path(os.environ.get("NER_EXPORT_DIR", nc.RUNS_ROOT / "export" / "koelectra-small-v2b"))
needs_export = pytest.mark.skipif(not (EXPORT_DIR / "model_card.json").exists(), reason="내보내기 산출 폴더 없음")


def test_plan_windows():
    assert onnx_ner.plan_windows(0, 126, 32) == []
    assert onnx_ner.plan_windows(126, 126, 32) == [(0, 126)]
    w = onnx_ner.plan_windows(300, 126, 32)
    assert w[0] == (0, 126) and w[-1][1] == 300
    assert all(b[0] < a[1] for a, b in zip(w, w[1:], strict=False))  # 이웃 윈도우는 겹친다
    covered = {i for s, e in w for i in range(s, e)}
    assert covered == set(range(300))


def test_decode_bio_matches_training_rule():
    id2 = dict(enumerate(nc.TAGS))
    t = nc.TAG2ID
    tags = [t["O"], t["B-PERSON"], t["I-PERSON"], t["O"], t["I-ADDRESS"], t["I-ADDRESS"], t["B-PERSON"]]
    offs = [(0, 1), (2, 3), (3, 4), (5, 6), (7, 8), (8, 9), (10, 12)]
    got = onnx_ner.decode_bio(tags, offs, id2)
    assert got == [{"label": "PERSON", "start": 2, "end": 4}, {"label": "ADDRESS", "start": 7, "end": 9},
                   {"label": "PERSON", "start": 10, "end": 12}]


@needs_export
def test_files_exist_and_sha256():
    card = json.loads((EXPORT_DIR / "model_card.json").read_text(encoding="utf-8"))
    for f in ("model.onnx", "tokenizer.json", "labels.json", "model_card.json"):
        assert (EXPORT_DIR / f).exists(), f
    for name, digest in card["sha256"].items():
        assert hashlib.sha256((EXPORT_DIR / name).read_bytes()).hexdigest() == digest, name
    for key in ("run_id", "seed", "data_version", "labels", "max_len", "metrics", "license_status", "created_at", "versions"):
        assert key in card, key
    assert card["max_len"] == 128 and card["opset"] >= 17
    assert "baseline_val" in card["metrics"] and "synth_val" in card["metrics"]


@needs_export
def test_labels_consistent():
    lab = json.loads((EXPORT_DIR / "labels.json").read_text(encoding="utf-8"))
    card = json.loads((EXPORT_DIR / "model_card.json").read_text(encoding="utf-8"))
    assert [lab["id2label"][str(i)] for i in range(len(lab["id2label"]))] == nc.TAGS
    assert lab["labels"] == list(nc.NER_LABELS) == card["labels"]
    assert lab["max_len"] == 128


@needs_export
def test_tokenizer_loads_standalone_with_offsets():
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(EXPORT_DIR / "tokenizer.json"))
    text = "김철수 씨가 서울 강남구에 간다"
    enc = tk.encode(text)
    assert enc.tokens[0] == "[CLS]" and enc.tokens[-1] == "[SEP]"
    content = [(s, e) for (s, e), m in zip(enc.offsets, enc.special_tokens_mask, strict=False) if not m]
    assert content and all(0 <= s < e <= len(text) for s, e in content)


@needs_export
def test_onnx_signature():
    import onnx

    for f in ("model.onnx", "model.int8.onnx"):
        p = EXPORT_DIR / f
        if not p.exists():
            continue
        m = onnx.load(str(p), load_external_data=False)
        assert [i.name for i in m.graph.input] == ["input_ids", "attention_mask"]
        assert [o.name for o in m.graph.output] == ["logits"]
        assert max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")) >= 17
        dims = m.graph.input[0].type.tensor_type.shape.dim
        assert dims[0].dim_param and dims[1].dim_param  # batch·sequence 동적 축
