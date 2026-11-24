"""학습된 NER 모델을 문자 span 으로 평가한다(Baseline eval_pii_v2 와 같은 규칙: exact/overlap P/R/F1, 라벨별 FNR, 음성 문장 FPR).

기본은 Baseline val 이다. **test 는 ``--final`` 을 줄 때만 허용**하고 기본은 거부한다(반복 개선에 쓰지 않기 위해).

예::

    python ml/ner/train/evaluate_ner.py --model-dir <VENVS>\\ner-runs\\<run_id>\\best --data baseline_val
    python ml/ner/train/evaluate_ner.py --model-dir ...\\best --data baseline_test --final     # 최종 1회

``--data``: baseline_val | baseline_test(--final 필요) | 변환된 jsonl 이름(synth_val, klue_val ...) 또는 ``이름:N``.
평가 대상 라벨은 PERSON·PERSON_ALIAS·ADDRESS·LOCATION·ORGANIZATION 뿐이다(전화·이메일·카드 등은 규칙 탐지기 몫이라 정답에서 제외).
"""

import argparse
import json
import sys
import time
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ner_common as nc  # noqa: E402
import ner_infer as ni  # noqa: E402
import tokenize_align as ta  # noqa: E402


class TestRefused(RuntimeError):
    """--final 없이 test 를 요청했을 때."""


def load_eval_rows(data, final=False):
    """평가 행을 읽는다. baseline_test 는 final=True 일 때만 읽는다.

    @param data: baseline_val | baseline_test | <jsonl 이름>[:N]
    @param final: --final 플래그
    @returns: [{id, text, spans(label/start/end)}]
    @raises TestRefused: final 없이 test 요청
    """
    name = data.partition(":")[0]
    if "test" in name:
        if not final:
            raise TestRefused("test 평가는 --final 을 줘야 한다(최종 평가 전용; 반복 개선에 쓰지 않는다)")
        if name != "baseline_test":
            raise TestRefused("test 로 인정하는 이름은 baseline_test 뿐이다")
    if name in ("baseline_val", "baseline_test"):
        split = name.split("_")[1]
        rows = nc.read_jsonl(nc.BASELINE_DIR / f"{split}.jsonl")
        return [{"id": r["id"], "text": r["text"], "spans": nc.to_ner_spans(r)} for r in rows]
    return ni.load_rows(data)


def evaluate_rows(model, tok, rows, max_len, batch_size, device, policy="all", use_amp=True):
    """행들을 예측해 exact/overlap 지표와 FPR·처리시간을 낸다.

    @param model: 모델
    @param tok: tokenizer
    @param rows: 평가 행
    @param max_len: 최대 토큰 길이
    @param batch_size: 배치 크기
    @param device: torch.device
    @param policy: 정렬 정책
    @param use_amp: fp16 autocast
    @returns: dict
    """
    t0 = time.time()
    preds, _ = ni.predict_spans(model, tok, rows, max_len, batch_size, device, policy, use_amp)
    dt = time.time() - t0
    golds = [r["spans"] for r in rows]
    neg = [i for i, g in enumerate(golds) if not g]
    return {"sentences": len(rows), "gold_spans": sum(len(g) for g in golds),
            "exact": nc.span_metrics(golds, preds, "exact"), "overlap": nc.span_metrics(golds, preds, "overlap"),
            "negative_sentences": len(neg), "false_positive_sentences": sum(1 for i in neg if preds[i]),
            "fpr": (sum(1 for i in neg if preds[i]) / len(neg)) if neg else 0.0,
            "predict_seconds": dt, "ms_per_sentence": 1000 * dt / max(len(rows), 1)}


def format_table(res):
    """사람이 읽을 표 문자열."""
    lines = []
    for mode in ("exact", "overlap"):
        lines.append(f"[{mode}]  P / R / F1 / FNR   (tp fp fn)")
        for lab in (*nc.NER_LABELS, "micro"):
            m = res[mode]["micro"] if lab == "micro" else res[mode]["per_label"][lab]
            lines.append(f"  {lab:13s} {m['precision']:.3f} / {m['recall']:.3f} / {m['f1']:.3f} / {m['fnr']:.3f}"
                         f"   ({m['tp']} {m['fp']} {m['fn']})")
    lines.append(f"FPR(음성 문장 중 오탐): {res['false_positive_sentences']}/{res['negative_sentences']} = {res['fpr']:.3f}")
    lines.append(f"{res['ms_per_sentence']:.1f} ms/문장")
    return "\n".join(lines)


def main(argv=None):
    import torch
    from transformers import AutoModelForTokenClassification

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", default="baseline_val")
    ap.add_argument("--final", action="store_true", help="test 평가 허용(최종 평가 전용)")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--policy", default="all", choices=["all", "first"])
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--out", default=None, help="결과 JSON 경로(기본: <model-dir>/../eval_<data>.json)")
    a = ap.parse_args(argv)
    rows = load_eval_rows(a.data, a.final)  # test 거부는 모델을 올리기 전에 일어난다
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = ta.load_tokenizer(a.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(a.model_dir).to(device)
    res = evaluate_rows(model, tok, rows, a.max_len, a.batch_size, device, a.policy, not a.no_fp16)
    res.update({"data": a.data, "final": a.final, "model_dir": a.model_dir})
    out = Path(a.out) if a.out else Path(a.model_dir).parent / f"eval_{a.data.replace(':', '_')}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_table(res))
    print(f"결과: {out}")
    return res


if __name__ == "__main__":
    try:
        main()
    except TestRefused as e:
        print(f"거부: {e}", file=sys.stderr)
        sys.exit(2)
