"""verify_onnx.py 가 비교할 PyTorch·HF 참조값을 만든다(학습용 venv: torch·transformers 필요; test 분할은 읽지 않는다).

산출(--out 폴더): reference.json(문장·HF 토큰화·PyTorch span), logits_<set>.npz(문장별 로짓), torch_latency.json(--latency 일 때)

    python ml/ner/export/dump_torch_reference.py --run-dir <run> --data-dir <ner-runs/data_v2b> --out <폴더>
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ML_DIR = HERE.parent
for p in (ML_DIR, ML_DIR / "train", HERE):
    sys.path.insert(0, str(p))

import evaluate_ner as ev  # noqa: E402
import ner_infer as ni  # noqa: E402
import numpy as np  # noqa: E402
import onnx_ner  # noqa: E402
import tokenize_align as ta  # noqa: E402

EDGE_TEXTS = [
    "김철수 씨가 서울특별시 강남구 테헤란로 152에 있는 삼성전자에 다닌다.",
    "😀 이영희 😀 대리님 ㅋㅋ 부산 해운대구 우동 1234-5번지로 와주세요!!",
    "ＡＢＣ 박민수　ｗｉｔｈ　전각 공백, 탭\t과 줄바꿈\n그리고 제로폭​문자.",
    "각 NFD 한글 자모 분해: 최지우한 광주광역시",
    "Hello John Smith, meet 김영수 at Google Seoul office.",
    "",
    "   ",
    "ㅎㅎ",
    "엄마가 아빠랑 이모네 집에 갔다",
    "국민은행 앞에서 만나자 -민수-",
]


class TorchNer(onnx_ner.OnnxNer):
    """OnnxNer 와 같은 토큰화·윈도우·디코딩을 쓰되 모델 호출만 PyTorch 로 한다."""

    def __init__(self, model_dir, torch_model, overlap=onnx_ner.DEFAULT_STRIDE_OVERLAP):
        super().__init__(model_dir, session=False, overlap=overlap)
        self.model = torch_model

    def run_batch(self, seqs):
        import torch

        m = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), m), self.pad_id, dtype=torch.long)
        att = torch.zeros((len(seqs), m), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            att[i, : len(s)] = 1
        with torch.no_grad():
            lg = self.model(input_ids=ids, attention_mask=att).logits.numpy()
        return [lg[i, : len(s)] for i, s in enumerate(seqs)]


def long_texts(rows, n, min_tokens, tk):
    """연속한 문장을 이어 붙여 min_tokens 이상(윈도우가 필요한) 긴 글과 이동한 정답 span 을 만든다.

    @param rows: [{text, spans}]
    @param n: 만들 개수
    @param min_tokens: 최소 토큰 수
    @param tk: tokenizers.Tokenizer
    @returns: [{text, spans}]
    """
    out, j = [], 0
    for _ in range(n):
        text, spans = "", []
        while len(tk.encode(text).ids) < min_tokens:
            r = rows[j % len(rows)]
            j += 11
            off = len(text)
            spans += [{"label": s["label"], "start": s["start"] + off, "end": s["end"] + off} for s in r["spans"]]
            text += r["text"] + " "
        out.append({"text": text.rstrip(), "spans": spans})
    return out


def main(argv=None):
    import torch
    from transformers import AutoModelForTokenClassification

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--export-dir", required=True, help="tokenizer.json·labels.json 이 있는 산출 폴더")
    ap.add_argument("--latency", action="store_true")
    a = ap.parse_args(argv)
    run, out, dd = Path(a.run_dir), Path(a.out), Path(a.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    best = run / "best"
    tok = ta.load_tokenizer(str(best))
    model = AutoModelForTokenClassification.from_pretrained(str(best)).eval()
    sets = {"baseline_val": ev.load_eval_rows("baseline_val"), "synth_val": ni.load_rows("synth_val", dd)}
    sets["edge"] = [{"id": f"edge{i}", "text": t, "spans": []} for i, t in enumerate(EDGE_TEXTS)]
    tw = TorchNer(a.export_dir, model)
    sets["long"] = [{"id": f"long{i}", **r} for i, r in enumerate(long_texts(sets["baseline_val"] + sets["synth_val"], 150, 300, tw.tk))]
    ref = {"run_id": run.name, "sets": {}}
    for name, rows in sets.items():
        recs, lens, chunks = [], [], []
        for r in rows:
            enc = tok(r["text"], truncation=True, max_length=128, return_offsets_mapping=True, return_special_tokens_mask=True)
            full = tok(r["text"], truncation=False, return_offsets_mapping=True)
            with torch.no_grad():
                lg = model(input_ids=torch.tensor([enc["input_ids"]]), attention_mask=torch.tensor([enc["attention_mask"]])).logits[0].numpy()
            tags = [int(i) for i in lg.argmax(-1)]
            spans = ta.decode_spans(tags, [tuple(o) for o in enc["offset_mapping"]], enc["special_tokens_mask"])
            rec = {"id": r["id"], "text": r["text"], "gold": r["spans"], "hf_ids": enc["input_ids"],
                   "hf_offsets": [list(o) for o in enc["offset_mapping"]], "hf_full_ids": full["input_ids"],
                   "hf_full_offsets": [list(o) for o in full["offset_mapping"]], "torch_spans": spans,
                   "torch_windowed_spans": tw.predict(r["text"])}
            recs.append(rec)
            lens.append(lg.shape[0])
            chunks.append(lg)
        ref["sets"][name] = recs
        np.savez_compressed(out / f"logits_{name}.npz", logits=np.concatenate(chunks), lens=np.array(lens))
        print(name, len(recs), "문장")
    (out / "reference.json").write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
    if a.latency:
        torch.set_num_threads(4)
        res = {}
        texts_all = [r["text"] for r in sets["baseline_val"] + sets["synth_val"]]
        for length in (30, 100, 300):
            texts = []
            for i in range(105):
                s, j = "", i
                while len(s) < length:
                    s += texts_all[j % len(texts_all)] + " "
                    j += 7
                texts.append(s[:length])
            ts = []
            for i, t in enumerate(texts):
                enc = tok(t, return_tensors="pt", truncation=True, max_length=512)
                t0 = time.perf_counter()
                with torch.no_grad():
                    model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.argmax(-1)
                if i >= 5:
                    ts.append(1000 * (time.perf_counter() - t0))
            ts.sort()
            res[f"{length}chars"] = {"median_ms": statistics.median(ts), "p95_ms": ts[int(0.95 * len(ts))]}
        (out / "torch_latency.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(res)


if __name__ == "__main__":
    main()
