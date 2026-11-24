"""R11-G12 ONNX 일치: 같은 가중치의 PyTorch(CPU fp32)와 ONNX(CPU fp32) 토큰 argmax 예측 일치율. 사전 등록 NER_R11_PREREGISTRATION.md §7-2 G12.

- 고정 문장 1,000개: `r11_val` 600 + `baseline_val` 180 + `synth_val` 앞 220(원문은 출력하지 않고 문장 목록 해시만 기록). 문장 하나씩(배치 1)으로 패딩 영향을 없앤다.
- 서비스 공용 NER 코드·운영 ONNX 를 수정하지 않는다(읽기 전용). GPU 를 쓰지 않는다.
사용: python g12_parity.py --model-dir <cand\\epN\\best 또는 HF 폴더> --onnx-dir <ONNX 폴더> --out parity.json
"""

import argparse
import hashlib
import json
from pathlib import Path

DATA11 = Path(r"C:\Users\<user>\venvs\ner-runs\data_r11")
N_SYNTH = 220


def rd(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def agreement(torch_ids, onnx_ids):
    """문장별 토큰 id 목록 두 개에서 (일치 토큰, 전체 토큰, 완전 일치 문장 수). 길이가 다르면 짧은 쪽에 맞추고 길이 불일치를 센다."""
    agree = total = same = len_mismatch = 0
    for a, b in zip(torch_ids, onnx_ids, strict=True):
        n = min(len(a), len(b))
        len_mismatch += len(a) != len(b)
        agree += sum(1 for x, y in zip(a[:n], b[:n], strict=True) if x == y)
        total += n
        same += a[:n] == b[:n] and len(a) == len(b)
    return agree, total, same, len_mismatch


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    texts = [r["text"] for r in rd(DATA11 / "r11_val.jsonl")] + [r["text"] for r in rd(DATA11 / "baseline_val.jsonl")] + [r["text"] for r in rd(DATA11 / "synth_val.jsonl")[:N_SYNTH]]
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(a.model_dir).eval()
    sess = ort.InferenceSession(str(a.onnx_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    t_ids, o_ids = [], []
    with torch.no_grad():
        for t in texts:
            enc = tok([t], return_tensors="pt", truncation=True, max_length=128)
            t_ids.append(model(**enc).logits.argmax(-1)[0].tolist())
            enc_np = tok([t], return_tensors="np", truncation=True, max_length=128)
            feed = {k: enc_np[k].astype(np.int64) for k in names if k in enc_np}
            o_ids.append(sess.run(None, feed)[0].argmax(-1)[0].tolist())
    agree, total, same, lm = agreement(t_ids, o_ids)
    res = {"sentences": len(texts), "sentence_list_sha256": hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest(), "tokens": total,
           "token_agreement_pct": round(100 * agree / total, 4), "identical_sentences": f"{same}/{len(texts)}", "length_mismatch_sentences": lm,
           "torch": torch.__version__, "onnxruntime": ort.__version__, "device": "cpu", "onnx_sha256": hashlib.sha256((a.onnx_dir / "model.onnx").read_bytes()).hexdigest()}
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
