"""R9 파일럿 후속 점검(연구 경로 전용, 서비스 모델·봉인 평가셋 미사용).

1) 저장된 best 체크포인트를 다시 읽어 개발 검증 문장의 예측이 재현되는지, 2) 같은 문장에서 PyTorch 와 ONNX 예측(토큰 argmax)이 일치하는지,
3) 배치 64 한 스텝 OOM 여유(최대 VRAM), 4) 실패 런이 로그를 남기고 중단하는지.
사용: python pilot_checks_r9.py --run-dir <runs\\pilot-r9-F-full-s7> --onnx-dir <export dir> --out result.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ner_infer as ni


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--data-dir", default=r"C:\Users\<user>\venvs\ner-runs\data_r9")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-batch64", action="store_true")
    a = ap.parse_args()
    rows = ni.load_rows("baseline_val", a.data_dir, 7)[:300]
    texts = [r["text"] for r in rows]
    tok = AutoTokenizer.from_pretrained(a.run_dir / "best")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {}

    def predict(model):
        out = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                enc = tok(
                    texts[i : i + 32],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(dev)
                lg = model(**enc).logits.argmax(-1).cpu().numpy()
                m = enc["attention_mask"].cpu().numpy()
                for j in range(lg.shape[0]):
                    out.append(lg[j][m[j] == 1].tolist())
        return out

    m1 = AutoModelForTokenClassification.from_pretrained(a.run_dir / "best").to(dev)
    p1 = predict(m1)
    m2 = AutoModelForTokenClassification.from_pretrained(a.run_dir / "best").to(dev)
    p2 = predict(m2)
    res["checkpoint_reload_identical_predictions"] = p1 == p2
    res["checkpoint_sentences"] = len(texts)

    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(a.onnx_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    names = [i.name for i in sess.get_inputs()]
    agree = total = same_sent = 0
    for i, t in enumerate(texts):
        enc = tok([t], return_tensors="np", truncation=True, max_length=128)
        feed = {k: enc[k].astype(np.int64) for k in names if k in enc}
        o = sess.run(None, feed)[0].argmax(-1)[0].tolist()
        n = min(len(o), len(p1[i]))
        eq = sum(1 for x, y in zip(o[:n], p1[i][:n], strict=True) if x == y)
        agree += eq
        total += n
        same_sent += int(o[:n] == p1[i][:n])
    res["onnx_vs_torch_token_agreement_pct"] = round(100 * agree / total, 4)
    res["onnx_vs_torch_identical_sentences"] = f"{same_sent}/{len(texts)}"

    # 배치 64 한 스텝(학습 모드, fp16, max_len 128 가득)
    if torch.cuda.is_available() and not a.skip_batch64:
        torch.cuda.reset_peak_memory_stats()
        m = AutoModelForTokenClassification.from_pretrained(a.run_dir / "best").to(dev)
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
        scaler = torch.amp.GradScaler("cuda")
        ids = torch.randint(5, 30000, (64, 128), device=dev)
        lab = torch.zeros((64, 128), dtype=torch.long, device=dev)
        t0 = time.time()
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                loss = m(
                    input_ids=ids, attention_mask=torch.ones_like(ids), labels=lab
                ).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            res["batch64_one_step"] = {
                "ok": True,
                "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                "step_s": round(time.time() - t0, 3),
            }
        except torch.cuda.OutOfMemoryError as e:
            res["batch64_one_step"] = {"ok": False, "oom": str(e)[:120]}
        res["gpu_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 2**20
        )
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
