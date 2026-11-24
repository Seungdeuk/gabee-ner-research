"""문자 span <-> 토큰 BIO 라벨 정렬(HF fast tokenizer 의 offset_mapping 사용)과 정렬 오류율 검증.

정렬 정책(문서화)
- 토큰이 어떤 span 과 한 글자라도 겹치면 그 span 의 토큰이다(겹침이 가장 큰 span 선택). span 의 첫 토큰은 ``B-라벨``, 이어지는 토큰은
  ``I-라벨``, 겹치는 span 이 없으면 ``O``. [CLS]/[SEP]/패딩은 -100(손실 제외).
- ``policy="all"``(기본): span 안의 모든 서브워드 토큰에 B/I 를 붙인다(문자 span 복원이 가능하다).
- ``policy="first"``: 단어 중간 서브워드 토큰(wordpiece 의 ``##`` 로 시작, 또는 sentencepiece 의 접두 표시가 없는 연속 조각 중 이름 안쪽)은
  -100 으로 두고 span 의 첫 토큰과 단어 첫 토큰만 학습한다. 예측 복원은 span 끝을 같은 단어의 마지막 서브워드까지 확장한다.
- 복원(``decode_spans``): B 로 시작하거나 앞 토큰과 라벨이 다른 I 에서 새 span 이 시작하고, 첫 토큰 시작~마지막 토큰 끝이 문자 span 이다.
- 한계: 토큰 경계가 span 경계와 다르면(예: 이름+조사가 한 토큰 ``철수가``) 복원 span 이 정답과 달라진다. ``alignment_report`` 가
  그 비율(복원 span == 원래 span)을 센다.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

import ner_common as nc  # noqa: E402

IGNORE = -100


def load_tokenizer(name):
    """fast tokenizer 를 로드한다(offset_mapping 필수).

    @param name: HF 모델 이름 또는 경로
    @returns: PreTrainedTokenizerFast
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError(f"{name}: fast tokenizer 가 아니라 offset_mapping 을 쓸 수 없다")
    return tok


def _is_continuation(token):
    return token.startswith("##")


def align_labels(offsets, spans, special_mask, tokens=None, policy="all"):
    """토큰 offset 과 문자 span 으로 토큰별 태그 id 를 만든다.

    @param offsets: [(start, end)] 토큰 문자 offset
    @param spans: [{label, start, end}] 문자 span (겹치지 않는다고 가정)
    @param special_mask: 특수 토큰이면 1 인 목록
    @param tokens: 토큰 문자열 목록(policy="first" 에 필요)
    @param policy: "all" 또는 "first"
    @returns: (태그 id 목록(-100 포함), 한 토큰이 둘 이상 span 과 겹친 횟수)
    """
    spans = sorted(spans, key=lambda s: s["start"])
    labels, started, conflicts = [], set(), 0
    for i, ((ts, te), sp_flag) in enumerate(zip(offsets, special_mask, strict=False)):
        if sp_flag or te <= ts:
            labels.append(IGNORE)
            continue
        best, best_ov, hits = None, 0, 0
        for si, sp in enumerate(spans):
            ov = min(te, sp["end"]) - max(ts, sp["start"])
            if ov > 0:
                hits += 1
                if ov > best_ov:
                    best, best_ov = si, ov
        conflicts += hits > 1
        if best is None:
            tag = "O"
        else:
            kind = "I" if best in started else "B"
            started.add(best)
            tag = f"{kind}-{spans[best]['label']}"
        if policy == "first" and tokens is not None and _is_continuation(tokens[i]) and not tag.startswith("B-"):
            labels.append(IGNORE)
            continue
        labels.append(nc.TAG2ID[tag])
    return labels, conflicts


def decode_spans(tag_ids, offsets, special_mask=None, policy="all", tokens=None):
    """토큰 태그 id 열에서 문자 span 을 복원한다.

    @param tag_ids: 토큰별 태그 id(-100 은 직전 태그를 잇는 것으로 본다)
    @param offsets: 토큰 offset
    @param special_mask: 특수 토큰 1
    @param policy: "first" 이면 span 끝을 같은 단어의 마지막 서브워드(## 연속)까지 확장
    @param tokens: 토큰 문자열(policy="first" 에 필요)
    @returns: [{label, start, end}]
    """
    spans, cur = [], None
    n = len(tag_ids)
    for i in range(n):
        if special_mask is not None and special_mask[i]:
            if cur:
                spans.append(cur)
                cur = None
            continue
        ts, te = offsets[i]
        tid = tag_ids[i]
        if tid == IGNORE or te <= ts:
            continue
        tag = nc.ID2TAG[tid]
        if tag == "O":
            if cur:
                spans.append(cur)
                cur = None
            continue
        bi, lab = tag.split("-", 1)
        if cur and bi == "I" and cur["label"] == lab:
            cur["end"] = te
        else:
            if cur:
                spans.append(cur)
            cur = {"label": lab, "start": ts, "end": te}
    if cur:
        spans.append(cur)
    if policy == "first" and tokens is not None:
        for sp in spans:
            for i in range(n - 1):
                if offsets[i][1] == sp["end"] and _is_continuation(tokens[i + 1]):
                    j = i + 1
                    while j < n and _is_continuation(tokens[j]):
                        sp["end"] = offsets[j][1]
                        j += 1
                    break
    return spans


def encode_rows(rows, tok, max_len=128, policy="all"):
    """행 목록을 모델 입력(input_ids, attention_mask, labels)과 복원용 offset 으로 바꾼다.

    @param rows: [{text, spans}] (spans 는 label/start/end)
    @param tok: fast tokenizer
    @param max_len: 최대 토큰 길이(잘림)
    @param policy: 정렬 정책
    @returns: dict 목록 {input_ids, attention_mask, labels, offsets, special, tokens, truncated, lost_spans}
    """
    out = []
    for r in rows:
        enc = tok(r["text"], truncation=True, max_length=max_len, return_offsets_mapping=True,
                  return_special_tokens_mask=True)
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        toks = tok.convert_ids_to_tokens(enc["input_ids"])
        labels, conflicts = align_labels(offsets, r["spans"], enc["special_tokens_mask"], toks, policy)
        last = max((e for (s, e), sp in zip(offsets, enc["special_tokens_mask"], strict=False) if not sp), default=0)
        full_len = len(tok(r["text"], add_special_tokens=True)["input_ids"])
        out.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels,
                    "offsets": offsets, "special": enc["special_tokens_mask"], "tokens": toks,
                    "truncated": full_len > max_len, "lost_spans": sum(1 for s in r["spans"] if s["end"] > last),
                    "conflicts": conflicts})
    return out


def alignment_report(rows, tok, max_len=128, policy="all", max_examples=8):
    """gold span -> 토큰 BIO -> 복원 span 이 원래와 일치하는 비율과 잘림 비율을 잰다.

    @param rows: 행 목록
    @param tok: fast tokenizer
    @param max_len: 최대 토큰 길이
    @param policy: 정렬 정책
    @param max_examples: 불일치 예시 수
    @returns: dict(spans, exact_recovered, exact_rate, overlap_recovered, per_label, truncated_sentences, truncated_rate, lost_spans, examples)
    """
    enc = encode_rows(rows, tok, max_len, policy)
    total = ok = ov_ok = lost = 0
    per = {lab: [0, 0] for lab in nc.NER_LABELS}
    trunc = sum(e["truncated"] for e in enc)
    conflicts = sum(e["conflicts"] for e in enc)
    examples, reasons = [], collections.Counter()
    for r, e in zip(rows, enc, strict=False):
        pred = decode_spans(e["labels"], e["offsets"], e["special"], policy, e["tokens"])
        lost += e["lost_spans"]
        for g in r["spans"]:
            total += 1
            per[g["label"]][1] += 1
            if any(p["label"] == g["label"] and p["start"] == g["start"] and p["end"] == g["end"] for p in pred):
                ok += 1
                per[g["label"]][0] += 1
            else:
                near = [p for p in pred if p["label"] == g["label"] and min(p["end"], g["end"]) > max(p["start"], g["start"])]
                if near:
                    ov_ok += 1
                    p = near[0]
                    reasons["경계가 토큰 안쪽(조사/서브워드 결합)" if (p["end"] > g["end"] or p["start"] < g["start"]) else "경계 불일치(짧게 복원)"] += 1
                elif g["end"] > max((o[1] for o, s in zip(e["offsets"], e["special"], strict=False) if not s), default=0):
                    reasons["잘림으로 span 손실"] += 1
                else:
                    reasons["복원 실패(기타)"] += 1
                if len(examples) < max_examples:
                    examples.append({"text": r["text"][:60], "gold": r["text"][g["start"] : g["end"]],
                                     "pred": [r["text"][p["start"] : p["end"]] for p in pred]})
    return {"spans": total, "exact_recovered": ok, "exact_rate": ok / total if total else 1.0,
            "overlap_recovered_only": ov_ok, "per_label_exact": {k: (v[0] / v[1] if v[1] else None) for k, v in per.items()},
            "truncated_sentences": trunc, "truncated_rate": trunc / len(rows) if rows else 0.0, "lost_spans": lost,
            "token_multi_span_conflicts": conflicts, "mismatch_reasons": dict(reasons), "examples": examples,
            "max_len": max_len, "policy": policy, "sentences": len(rows)}


def main():
    ap = argparse.ArgumentParser(description="정렬 오류율·잘림 비율 검증")
    ap.add_argument("--model", default="monologg/koelectra-small-v3-discriminator")
    ap.add_argument("--files", nargs="+", default=["synth_train", "baseline_train", "klue_train"])
    ap.add_argument("--max-lens", nargs="+", type=int, default=[128, 256])
    ap.add_argument("--policy", default="all", choices=["all", "first"])
    ap.add_argument("--out", default=None, help="JSON 보고서 경로")
    a = ap.parse_args()
    tok = load_tokenizer(a.model)
    rep = {}
    for f in a.files:
        rows = nc.read_jsonl(nc.DATA_OUT / f"{f}.jsonl")
        for ml in a.max_lens:
            rep[f"{f}@{ml}"] = alignment_report(rows, tok, ml, a.policy)
            r = rep[f"{f}@{ml}"]
            print(f"{f} max_len={ml}: exact {r['exact_recovered']}/{r['spans']} = {r['exact_rate']:.4f}, "
                  f"잘림 문장 {r['truncated_sentences']} ({r['truncated_rate']:.4f}), 손실 span {r['lost_spans']}")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
