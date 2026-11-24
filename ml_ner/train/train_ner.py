"""NER 토큰 분류 학습(간단한 PyTorch 루프). 체크포인트·로그는 저장소 밖(기본 <VENVS>\\ner-runs\\<run_id>\\).

예)  파이프라인 확인용 소량 학습::

    set HF_HOME=<VENVS>\\hf-cache
    python ml/ner/train/train_ner.py --train-files synth_train:500 --val-files synth_val baseline_val --epochs 3 --batch-size 16 --max-len 128

데이터 지정자: ``이름`` 또는 ``이름:N``(무작위 N 문장). 이름은 build_ner_data.py 가 만든 jsonl(synth_train, baseline_train, klue_train ...).
test 는 이 스크립트가 읽지 않는다(옵션도 없다).
"""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ner_common as nc  # noqa: E402
import ner_infer as ni  # noqa: E402
import tokenize_align as ta  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="monologg/koelectra-small-v3-discriminator")
    ap.add_argument("--train-files", nargs="+", default=["synth_train", "baseline_train"])
    ap.add_argument("--val-files", nargs="+", default=["synth_val", "baseline_val"])
    ap.add_argument("--klue-ratio", type=float, default=0.0,
                    help="KLUE 문장이 전체 학습 문장에서 차지할 비율 r(0~0.9). --train-files 의 합이 N 이면 KLUE 문장 수 K=round(N*r/(1-r)), "
                         "KLUE 는 --klue-file 에서 시드 고정 무작위 부분 샘플")
    ap.add_argument("--klue-file", default="klue_train")
    ap.add_argument("--select-on", default="baseline_val",
                    help="best/early stopping 기준 검증 집합(--val-files 중 하나). 'mean' 이면 모든 검증 집합 평균")
    ap.add_argument("--select-metric", default="overlap", choices=["overlap", "exact"], help="기준 지표(char micro F1)")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=0, help="val micro F1 이 이만큼 epoch 연속 안 오르면 중단(0=끔)")
    ap.add_argument("--class-weights", default="none", choices=["none", "sqrt", "inv"], help="토큰 라벨 빈도 기반 가중치")
    ap.add_argument("--weight-clip", type=float, default=10.0)
    ap.add_argument("--policy", default="all", choices=["all", "first"])
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-root", default=str(nc.RUNS_ROOT))
    ap.add_argument("--data-dir", default=str(nc.DATA_OUT))
    ap.add_argument("--no-save-model", action="store_true")
    ap.add_argument("--save-every-epoch", action="store_true",
                    help="매 에폭 종료 시 가중치를 epoch_<k>/ 에 추가로 저장(기본 꺼짐, 학습 계산·best 저장은 그대로, --no-save-model 이면 무시)")
    return ap.parse_args(argv)


def file_digest(path):
    """데이터 버전 표시용 sha256 앞 12자리."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _data_build_info(data_dir):
    """데이터 생성 시드·틀당 문장 수(leak_report.json)와 stats.json 해시."""
    d = Path(data_dir)
    try:
        rep = json.loads((d / "leak_report.json").read_text(encoding="utf-8"))
        return {"seed": rep.get("seed"), "per_template": rep.get("per_template"),
                "stats_sha256_12": file_digest(d / "stats.json")}
    except Exception:  # noqa: BLE001
        return None


def git_head():
    """저장소 HEAD 짧은 해시(없으면 None)."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=nc.REPO_ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def class_weights(enc, mode, clip):
    """학습 토큰 라벨 빈도로 손실 가중치를 만든다(모드 none 이면 None).

    @param enc: encode_rows 결과(학습)
    @param mode: none|sqrt|inv
    @param clip: 최대 가중치
    @returns: (가중치 list 또는 None, 라벨별 토큰 수)
    """
    counts = [0] * len(nc.TAGS)
    for e in enc:
        for t in e["labels"]:
            if t != ta.IGNORE:
                counts[t] += 1
    if mode == "none":
        return None, counts
    total = sum(counts)
    w = []
    for c in counts:
        base = total / (len(counts) * max(c, 1))
        w.append(min(base ** (0.5 if mode == "sqrt" else 1.0), clip))
    return w, counts


def seqeval_scores(enc, pred_ids):
    """토큰 수준 BIO 를 seqeval 로 채점(entity-level P/R/F1 + 라벨별).

    @param enc: encode_rows 결과(정답 labels 포함)
    @param pred_ids: 행별 예측 태그 id
    @returns: seqeval classification_report(output_dict) + overall
    """
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    y_true, y_pred = [], []
    for e, p in zip(enc, pred_ids, strict=False):
        pairs = [(nc.ID2TAG[t], nc.ID2TAG[q]) for t, q in zip(e["labels"], p, strict=False) if t != ta.IGNORE]
        y_true.append([a for a, _ in pairs])
        y_pred.append([b for _, b in pairs])
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rep = {k: ({kk: float(vv) for kk, vv in v.items()}) for k, v in rep.items()}
    rep["overall"] = {"precision": float(precision_score(y_true, y_pred, zero_division=0)),
                      "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                      "f1": float(f1_score(y_true, y_pred, zero_division=0))}
    return rep


def evaluate_split(model, tok, rows, args, device):
    """한 검증 집합의 seqeval(토큰 BIO)·문자 span(exact/overlap) 지표.

    @param model: 모델
    @param tok: tokenizer
    @param rows: 검증 행
    @param args: 인자
    @param device: torch.device
    @returns: dict
    """
    enc = ta.encode_rows(rows, tok, args.max_len, args.policy)
    ids = ni.predict_tag_ids(model, enc, args.batch_size * 2, device, tok.pad_token_id or 0, not args.no_fp16)
    spans = [ta.decode_spans(p, e["offsets"], e["special"], args.policy, e["tokens"]) for p, e in zip(ids, enc, strict=False)]
    golds = [r["spans"] for r in rows]
    return {"seqeval": seqeval_scores(enc, ids), "char_exact": nc.span_metrics(golds, spans, "exact"),
            "char_overlap": nc.span_metrics(golds, spans, "overlap")}


def main(argv=None):
    import numpy as np
    import torch
    from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup

    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_fp16

    train_rows = [r for spec in args.train_files for r in ni.load_rows(spec, args.data_dir, args.seed)]
    val_sets = {spec: ni.load_rows(spec, args.data_dir, args.seed) for spec in args.val_files}
    n_base = len(train_rows)
    n_klue = 0
    if args.klue_ratio > 0:
        n_klue = round(n_base * args.klue_ratio / (1 - args.klue_ratio))
        train_rows += ni.load_rows(f"{args.klue_file}:{n_klue}", args.data_dir, args.seed)
    random.Random(args.seed).shuffle(train_rows)
    tok = ta.load_tokenizer(args.model)
    train_enc = ta.encode_rows(train_rows, tok, args.max_len, args.policy)
    w, tag_counts = class_weights(train_enc, args.class_weights, args.weight_clip)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(nc.TAGS), id2label=nc.ID2TAG, label2id=nc.TAG2ID).to(device)
    nd = ["bias", "LayerNorm.weight"]
    groups = [{"params": [p for n, p in model.named_parameters() if not any(x in n for x in nd)],
               "weight_decay": args.weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(x in n for x in nd)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = (len(train_enc) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, device=device) if w else None, ignore_index=ta.IGNORE)

    data_files = {}
    for spec in [*args.train_files, *args.val_files]:
        p = Path(args.data_dir) / f"{spec.partition(':')[0]}.jsonl"
        data_files[spec] = {"sha256_12": file_digest(p), "rows": sum(1 for _ in open(p, encoding="utf-8"))}
    log = {"run_id": run_id, "started": datetime.now().isoformat(timespec="seconds"), "git_head": git_head(),
           "config": {k: v for k, v in vars(args).items()}, "device": str(device),
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "data_files": data_files, "train_sentences": len(train_rows), "train_sentences_non_klue": n_base,
           "train_sentences_klue": n_klue, "klue_share_actual": n_klue / max(len(train_rows), 1),
           "data_build": _data_build_info(args.data_dir), "tag_token_counts": dict(zip(nc.TAGS, tag_counts, strict=False)),
           "class_weights": dict(zip(nc.TAGS, w, strict=False)) if w else None, "epochs": []}
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None

    best, bad, step_times = -1.0, 0, []
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_enc)))
        random.Random(args.seed + ep).shuffle(order)
        ep_loss, n_steps = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            batch = ni.collate([train_enc[j] for j in order[i : i + args.batch_size]], tok.pad_token_id or 0)
            batch = {k: v.to(device) for k, v in batch.items()}
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss = loss_fn(logits.float().view(-1, logits.size(-1)), batch["labels"].view(-1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.time() - t0)
            ep_loss += float(loss)
            n_steps += 1
        val = {spec: evaluate_split(model, tok, rows, args, device) for spec, rows in val_sets.items()}
        # best/early stopping 기준: --select-on 검증 집합의 char micro F1(--select-metric)
        key = f"char_{args.select_metric}"
        if args.select_on == "mean" or args.select_on not in val:
            score = float(np.mean([v[key]["micro"]["f1"] for v in val.values()])) if val else 0.0
        else:
            score = float(val[args.select_on][key]["micro"]["f1"])
        log["epochs"].append({"epoch": ep, "train_loss": ep_loss / max(n_steps, 1), "val_score": score,
                              "val": val, "elapsed_s": time.time() - t_start})
        print(f"epoch {ep}: loss {ep_loss / max(n_steps, 1):.4f} val({args.select_on}) char-{args.select_metric} F1 {score:.4f}", flush=True)
        if args.save_every_epoch and not args.no_save_model:
            # 읽기 전용 저장: 손실·옵티마이저·스케줄·난수 상태를 건드리지 않는다.
            ep_dir = run_dir / f"epoch_{ep}"
            model.save_pretrained(ep_dir)
            log["epochs"][-1]["checkpoint_sha256"] = hashlib.sha256((ep_dir / "model.safetensors").read_bytes()).hexdigest()
        if score > best:
            best, bad = score, 0
            if not args.no_save_model:
                model.save_pretrained(run_dir / "best")
                tok.save_pretrained(run_dir / "best")
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                print("early stopping")
                break
    log["best_val_score"] = best
    log["best_epoch"] = max(log["epochs"], key=lambda e: e["val_score"])["epoch"] if log["epochs"] else None
    log["stopped_epoch"] = log["epochs"][-1]["epoch"] if log["epochs"] else None
    best_dir = run_dir / "best"
    log["checkpoint_mb"] = (sum(f.stat().st_size for f in best_dir.rglob("*") if f.is_file()) / 2**20) if best_dir.exists() else None
    log["weights_mb"] = (sum(f.stat().st_size for f in best_dir.glob("*.safetensors")) / 2**20) if best_dir.exists() else None
    log["step_time_s"] = {"mean": float(np.mean(step_times)), "median": float(np.median(step_times)),
                          "p95": float(np.percentile(step_times, 95)), "steps": len(step_times)}
    log["max_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    log["total_train_time_s"] = time.time() - t_start
    log["finished"] = datetime.now().isoformat(timespec="seconds")
    (run_dir / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"로그: {run_dir / 'train_log.json'}")
    return log


if __name__ == "__main__":
    main()
