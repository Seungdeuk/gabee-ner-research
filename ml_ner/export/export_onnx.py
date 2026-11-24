"""NER 체크포인트를 서버 CPU 추론용 ONNX 로 내보낸다(학습용 venv: torch·transformers·onnx·onnxruntime 필요).

산출 폴더(저장소 밖; 모델 가중치는 Git 에 올리지 않는다):
    model.onnx(fp32)  model.int8.onnx(선택)  tokenizer.json  labels.json  model_card.json

    python ml/ner/export/export_onnx.py --run-dir <VENVS>\\ner-runs\\v2b-koelectra-small-klue10-s7 \\
        --out <VENVS>\\ner-runs\\export\\koelectra-small-v2b [--no-int8]

서버 런타임은 onnxruntime·tokenizers·numpy 뿐이므로 tokenizer.json 은 ``tokenizers`` 단독으로 로드·offset 반환이 가능해야 한다.
"""

import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ML_DIR = HERE.parent
sys.path.insert(0, str(ML_DIR))

import ner_common as nc  # noqa: E402

OPSET = 17
MAX_LEN = 128
LICENSE_BASE = {
    "base_model": "monologg/koelectra-small-v3-discriminator — 코드 저장소는 Apache-2.0(원 저작권 고지·라이선스 링크 필요), 가중치 라이선스는 미확정이라 서비스 투입 전 확인 필요",
    "synthetic_and_baseline": "자체 생성 합성 데이터 + Baseline v2(자체)",
}
LICENSE_KLUE = "KLUE-NER 데이터 CC BY-SA 4.0(출처 표시·동일조건변경허락) — 학습 데이터 일부로 사용, 서비스 투입 전 확인 필요"
LICENSE_AIHUB = (
    "AI Hub 「대규모 웹데이터 기반 한국어 말뭉치 데이터」(한국지능정보사회진흥원) 일부로 학습 — 결과물에 데이터셋 정식 명칭과 "
    "AI 허브(aihub.or.kr)를 출처로 표기해야 하고, 원본·가공본을 제3자에게 줄 수 없다(모델 자체의 활용·판매는 가능, 이용정책 확인 필요)"
)


def license_status(cfg):
    """학습 설정에서 외부 데이터 사용 여부를 읽어 라이선스 상태 문구를 만든다.

    @param cfg: train_log.json 의 config
    @returns: dict
    """
    status = dict(LICENSE_BASE)
    if float(cfg.get("klue_ratio") or 0) > 0:
        name = str(cfg.get("klue_file") or "")
        if name.startswith("aihub"):
            status["aihub_corpus"] = LICENSE_AIHUB
        else:
            status["klue_ner"] = LICENSE_KLUE
    else:
        status["external_data"] = "외부 말뭉치 없음(합성 + Baseline v2 만 사용)"
    return status


def sha256(path):
    """파일 sha256(16진 문자열).

    @param path: 파일 경로
    @returns: str
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_fp32(best_dir, out_path):
    """체크포인트를 ONNX(opset 17, batch·sequence 동적 축)로 내보낸다.

    @param best_dir: HF 체크포인트 폴더
    @param out_path: 저장할 .onnx 경로
    @returns: 모델에 token_type_ids 입력을 넣었는지(항상 False; 0 으로 고정)
    """
    import torch
    from transformers import AutoModelForTokenClassification

    model = AutoModelForTokenClassification.from_pretrained(
        str(best_dir), attn_implementation="eager"
    ).eval()

    class Wrap(torch.nn.Module):
        """token_type_ids 는 항상 0 이라 입력에서 뺀다(임베딩에 0 번 행이 고정 반영)."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    ids = torch.randint(5, 30000, (2, 24))
    att = torch.ones_like(ids)
    att[1, 18:] = 0
    with torch.no_grad():
        torch.onnx.export(
            Wrap(model),
            (ids, att),
            str(out_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    return False


def quantize_int8(src, dst):
    """MatMul 가중치를 INT8 로 동적 양자화한다.

    @param src: fp32 .onnx
    @param dst: 저장할 .int8.onnx
    @returns: 없음
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def metrics_from_eval(run_dir):
    """run 의 eval_full.json 에서 모델 카드용 지표를 뽑는다(Baseline val·합성 val).

    @param run_dir: run 폴더
    @returns: dict
    """
    d = json.loads((run_dir / "eval_full.json").read_text(encoding="utf-8"))
    out = {"source": "eval_full.json(PyTorch, val 분할만 사용; test 미사용)"}
    for name in ("baseline_val", "synth_val"):
        s = d["sets"][name]
        out[name] = {
            "sentences": s["sentences"],
            "gold_spans": s["gold_spans"],
            "fpr_negative_sentences": s["fpr"],
        }
        for mode in ("overlap", "exact"):
            m = s[mode]["micro"]
            out[name][mode] = {k: round(m[k], 4) for k in ("precision", "recall", "f1")}
    return out


def build_model_card(run_dir, out_dir, files, ner_labels, extra=None):
    """model_card.json 내용을 만든다.

    @param run_dir: 체크포인트가 든 run 폴더
    @param out_dir: 산출 폴더
    @param files: 산출물 파일 이름 목록
    @param ner_labels: 라벨 목록
    @param extra: 덧붙일 dict(선택 근거 등)
    @returns: dict
    """
    import onnx
    import onnxruntime

    log = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))
    cfg = log["config"]
    card = {
        "name": "koelectra-small-v2b",
        "run_id": cfg["run_id"],
        "checkpoint": f"{run_dir.name}/best",
        "base_model": cfg["model"],
        "seed": cfg["seed"],
        "git_head_at_training": log.get("git_head"),
        "data_version": {
            "build": log["data_build"],
            "files": log["data_files"],
            "klue_ratio": cfg["klue_ratio"],
            "external_file": cfg.get("klue_file")
            if float(cfg.get("klue_ratio") or 0) > 0
            else None,
        },
        "labels": list(ner_labels),
        "tags": nc.TAGS,
        "max_len": MAX_LEN,
        "inputs": ["input_ids", "attention_mask"],
        "outputs": ["logits"],
        "opset": OPSET,
        "metrics": metrics_from_eval(run_dir),
        "license_status": license_status(cfg),
        "sha256": {f: sha256(out_dir / f) for f in files},
        "created_at": datetime.datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),
        "versions": {
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
            "python": sys.version.split()[0],
        },
    }
    try:
        import torch
        import transformers

        card["versions"].update(
            {"torch": torch.__version__, "transformers": transformers.__version__}
        )
    except ImportError:
        pass
    card.update(extra or {})
    return card


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument(
        "--selection-note",
        default=None,
        help="모델 카드에 넣을 시드 선택 근거(JSON 파일 경로 또는 문자열)",
    )
    a = ap.parse_args(argv)
    run, out = Path(a.run_dir), Path(a.out)
    best = run / "best"
    out.mkdir(parents=True, exist_ok=True)
    export_fp32(best, out / "model.onnx")
    files = ["model.onnx"]
    if not a.no_int8:
        quantize_int8(out / "model.onnx", out / "model.int8.onnx")
        files.append("model.int8.onnx")
    # tokenizer.json: HF fast tokenizer 가 저장한 파일. tokenizers 단독으로 로드되고 offset 을 돌려준다.
    shutil.copyfile(best / "tokenizer.json", out / "tokenizer.json")
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(out / "tokenizer.json"))
    tk.no_truncation()
    cfg = json.loads((best / "config.json").read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    if [id2label[i] for i in range(len(id2label))] != nc.TAGS:
        raise SystemExit("config.json 의 id2label 이 ner_common.TAGS 와 다르다")
    labels = {
        "id2label": {str(i): t for i, t in sorted(id2label.items())},
        "labels": list(nc.NER_LABELS),
        "max_len": MAX_LEN,
        "pad_token_id": cfg.get("pad_token_id", 0),
        # BERT 계열은 [CLS]/[SEP], XLM-R 계열(MiniLM·e5)은 <s>/</s>
        "cls_token_id": next((i for i in (tk.token_to_id("[CLS]"), tk.token_to_id("<s>")) if i is not None), None),
        "sep_token_id": next((i for i in (tk.token_to_id("[SEP]"), tk.token_to_id("</s>")) if i is not None), None),
    }
    (out / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    files += ["tokenizer.json", "labels.json"]
    extra = {}
    if a.selection_note:
        p = Path(a.selection_note)
        extra["seed_selection"] = (
            json.loads(p.read_text(encoding="utf-8-sig"))
            if p.exists()
            else a.selection_note
        )
    card = build_model_card(run, out, files, nc.NER_LABELS, extra)
    (out / "model_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for f in [*files, "model_card.json"]:
        print(f"{f}: {(out / f).stat().st_size / 2**20:.2f} MB")


if __name__ == "__main__":
    main()
