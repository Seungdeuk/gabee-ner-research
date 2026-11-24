"""BIO 왕복 불일치 원인 분해(건수만). 기존 학습 데이터(R9F·R10)와 같은 비율인지 비교한다. 문장 원문은 출력하지 않는다."""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "train"))
import tokenize_align as ta  # noqa: E402

PERSON = ("PERSON", "PERSON_ALIAS")
ALLOWED = PERSON + ("ORGANIZATION", "LOCATION", "ADDRESS")


def rd(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def diag(rows, tok):
    enc = ta.encode_rows(rows, tok, 128, "all")
    c = Counter()
    ids = []
    for r, e in zip(rows, enc, strict=False):
        gold = sorted((s["label"], s["start"], s["end"]) for s in r["spans"] if s["label"] in ALLOWED)
        dec = ta.decode_spans(e["labels"], e["offsets"], e["special"], "all", e["tokens"])
        got = sorted((s["label"], s["start"], s["end"]) for s in dec)
        if gold == got:
            continue
        offs = [o for o, sp in zip(e["offsets"], e["special"], strict=False) if not sp]
        inside = False
        for _, a, b in gold:
            for (x, y) in offs:
                if x < a < y or x < b < y:
                    inside = True
        c["boundary_inside_token" if inside else "other"] += 1
        c["len_diff_gold_vs_decoded"] += len(gold) != len(got)
        ids.append(r["id"])
    return len(rows), dict(c), ids


if __name__ == "__main__":
    tok = ta.load_tokenizer("monologg/koelectra-small-v3-discriminator")
    D11 = Path(r"C:\Users\<user>\venvs\ner-runs\data_r11")
    D10 = Path(r"C:\Users\<user>\venvs\ner-runs\data_r10")
    out = {}
    for name, p in (("r11_Dprime_train", D11 / "r11_Dprime_train.jsonl"), ("r11_val", D11 / "r11_val.jsonl"), ("r11_dev", D11 / "r11_dev.jsonl"),
                    ("ref:r10_full_train", D10 / "r10_full_train.jsonl"), ("ref:r9_Ffull_train", D10 / "r9_Ffull_train.jsonl"), ("ref:synth_train", D10 / "synth_train.jsonl"),
                    ("ref:r10_dev", D10 / "r10_dev.jsonl")):
        n, c, ids = diag(rd(p), tok)
        out[name] = {"rows": n, "mismatch": sum(v for k, v in c.items() if k in ("boundary_inside_token", "other")), "causes": c, "rate_pct": round(100 * sum(v for k, v in c.items() if k in ("boundary_inside_token", "other")) / n, 2)}
        if name.startswith("r11"):
            out[name]["ids_head"] = ids[:5]
    print(json.dumps(out, ensure_ascii=False, indent=1))
