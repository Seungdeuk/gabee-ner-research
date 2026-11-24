"""학습·평가가 함께 쓰는 입출력 보조: 데이터 지정자 읽기, 배치 만들기, 모델 예측 -> 문자 span."""

import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ner_common as nc  # noqa: E402
import tokenize_align as ta  # noqa: E402


def load_rows(spec, data_dir=None, seed=42):
    """``이름`` 또는 ``이름:N``(무작위 N 문장) 지정자로 jsonl 을 읽는다.

    @param spec: 예) "synth_train", "klue_train:3000"
    @param data_dir: 데이터 폴더(기본 nc.DATA_OUT)
    @param seed: 부분 추출 시드
    @returns: 행 목록
    """
    import random

    name, _, n = spec.partition(":")
    rows = nc.read_jsonl(Path(data_dir or nc.DATA_OUT) / f"{name}.jsonl")
    if n:
        rows = random.Random(seed).sample(rows, min(int(n), len(rows)))
    return rows


def collate(batch, pad_id=0):
    """encode_rows 결과 dict 목록을 패딩해 텐서 dict 로 만든다.

    @param batch: encode_rows 의 항목 목록
    @param pad_id: 패딩 토큰 id
    @returns: {input_ids, attention_mask, labels}
    """
    import torch

    m = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), m), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), m), dtype=torch.long)
    lab = torch.full((len(batch), m), ta.IGNORE, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        att[i, :n] = torch.tensor(b["attention_mask"])
        lab[i, :n] = torch.tensor(b["labels"])
    return {"input_ids": ids, "attention_mask": att, "labels": lab}


def predict_tag_ids(model, enc, batch_size, device, pad_id=0, use_amp=True):
    """인코딩된 행들의 토큰별 예측 태그 id 를 낸다.

    @param model: 토큰 분류 모델
    @param enc: encode_rows 결과
    @param batch_size: 배치 크기
    @param device: torch.device
    @param pad_id: 패딩 id
    @param use_amp: CUDA 에서 fp16 autocast 사용
    @returns: 행별 태그 id 목록(패딩 제외, 길이 = 토큰 수)
    """
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(enc), batch_size):
            chunk = enc[i : i + batch_size]
            b = collate(chunk, pad_id)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
                logits = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device)).logits
            pred = logits.argmax(-1).cpu().tolist()
            for c, p in zip(chunk, pred, strict=False):
                out.append(p[: len(c["input_ids"])])
    return out


def predict_spans(model, tok, rows, max_len, batch_size, device, policy="all", use_amp=True):
    """문장들의 문자 span 예측(subword -> 문자 offset 복원).

    @param model: 토큰 분류 모델
    @param tok: fast tokenizer
    @param rows: [{text, spans}]
    @param max_len: 최대 토큰 길이
    @param batch_size: 배치 크기
    @param device: torch.device
    @param policy: 정렬 정책
    @param use_amp: fp16 autocast
    @returns: (문장별 예측 span 목록, encode_rows 결과)
    """
    enc = ta.encode_rows(rows, tok, max_len, policy)
    pads = predict_tag_ids(model, enc, batch_size, device, tok.pad_token_id or 0, use_amp)
    spans = [ta.decode_spans(p, e["offsets"], e["special"], policy, e["tokens"]) for p, e in zip(pads, enc, strict=False)]
    return spans, enc
