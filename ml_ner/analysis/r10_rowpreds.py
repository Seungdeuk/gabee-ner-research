"""R10 오류 분석용 행 단위 예측 덤프(개발 데이터 전용, 학습 없음).

대상: R10 개발셋(r10_dev) + R9 회귀 5세트(개발용). R9 봉인셋·R10 독립 평가셋은 경로 자체가 없다.
모델 16개(S0, R9F x5, C10 x5, R10-full x5)를 한 번에 하나씩 로드해 추론한다. 서비스 코드(`ner.py`)는 수정하지 않고,
`NerEngine._decode` 를 프로세스 안에서만 감싸 토큰별 확률을 잡는다.

행마다 저장: 최종 span(서비스 파이프라인 결과 = 채점 대상), 모델 원시 span(규칙 병합 전), 정답 구간의 PERSON 확률(최대),
음성 문장은 pair_word 위치의 PERSON 확률(최대), 채점 판정(full/partial/masked/neg_flagged/neg_clean).

사용: python r10_rowpreds.py --out-dir <dir> [--only LABEL ...]   (apps/api 를 cwd 로, API venv 파이썬)
"""

import argparse
import json
import sys
from pathlib import Path

API = Path(r"C:\Users\<user>\Documents\gabee-ner\apps\api")
sys.path.insert(0, str(API))
sys.path.insert(0, str(API / "scripts"))

import evaluate_r10_dev as E  # noqa: E402  (1글자 이름 제외 패치 포함, 채점 코드 동일)

lib = E.lib
ner = lib.ner
D = E.D

BASE = Path(r"C:\Users\<user>\venvs\ner-runs")
SEEDS = [7, 13, 21, 42, 99]
cap = {}
_orig = ner.NerEngine._decode


def _hook(self, plain, starts_map, ends_map, offsets, logits):
    import numpy as np

    sh = logits - logits.max(axis=1, keepdims=True)
    pr = np.exp(sh)
    pr /= pr.sum(axis=1, keepdims=True)
    out = _orig(self, plain, starts_map, ends_map, offsets, logits)
    cap["last"] = (starts_map, ends_map, list(offsets), pr, dict(self._labels), out)
    return out


ner.NerEngine._decode = _hook


def models():
    base = json.loads((BASE / "r10" / "dev" / "baselines.json").read_text(encoding="utf-8"))
    out = [("S0", BASE / "service_v2b")]
    for s in SEEDS:
        out.append((f"R9F-s{s}", Path(base[f"R9F-s{s}"]["model_dir"])))
    for arm in ("C10", "R10-full"):
        for s in SEEDS:
            out.append((f"{arm}-s{s}", BASE / "r10" / "export" / f"r10-{arm}-s{s}"))
    return out


def tok_probs(text):
    """마지막 추론의 토큰별 (원문 시작, 원문 끝, P(PERSON 계열), P(non-O))."""
    if "last" not in cap:
        return []
    starts_map, ends_map, offsets, pr, labels, _ = cap["last"]
    person_ids = [i for i, t in labels.items() if t.endswith("PERSON") or t.endswith("PERSON_ALIAS")]
    o_ids = [i for i, t in labels.items() if t == "O"]
    res = []
    for k, (cs, ce) in enumerate(offsets):
        if ce <= cs:
            continue
        if starts_map is not None and ends_map is not None:
            a, b = starts_map[cs], ends_map[ce - 1]
        else:
            a, b = cs, ce
        pp = float(sum(pr[k, i] for i in person_ids))
        po = float(sum(pr[k, i] for i in o_ids)) if o_ids else 0.0
        res.append((a, b, pp, 1.0 - po))
    return res


def max_p(toks, a, b):
    v = [t[2] for t in toks if max(t[0], a) < min(t[1], b)]
    return max(v) if v else None


def run_rows(rows, set_name):
    out = []
    for r in rows:
        text = r["text"]
        ner._cache.clear()
        cap.pop("last", None)
        final = [(s.start, s.end) for s in ner.detect_with_ner(text, ()) if s.kind.name == "PERSON"]
        toks = tok_probs(text)
        raw = [(s.label, s.start, s.end, round(s.score, 3)) for s in cap["last"][5]] if "last" in cap else []
        golds = lib.gold_person(r)
        rec = lib.score_rows([r], keep_records=True, pred_fn=lambda _r, f=final: f)["records"][0]
        item = {"id": r.get("id"), "set": set_name, "stress": r.get("stress_type"), "gold": golds, "final": final, "raw": raw, "cls": rec["cls"]}
        if golds:
            item["p_gold"] = [max_p(toks, a, b) for a, b in golds]
        w = r.get("pair_word")
        if w and not golds:
            pos = []
            i = text.find(w)
            while i >= 0:
                pos.append(max_p(toks, i, i + len(w)))
                i = text.find(w, i + 1)
            item["p_word"] = max((p for p in pos if p is not None), default=None)
        out.append(item)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip-r9sets", action="store_true")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    dev = E.rows_of(E.R10_DEV)
    r9 = {} if a.skip_r9sets else {n: D.load(p) for n, p in D.MAIN5.items()}
    for label, d in models():
        if a.only and label not in a.only:
            continue
        outp = a.out_dir / f"rowpreds_{label}.jsonl"
        if outp.exists():
            print("SKIP(이미 있음)", label, flush=True)
            continue
        fp = lib.set_model(d, None)
        recs = run_rows(dev, "r10_dev")
        for n, rs in r9.items():
            recs += run_rows(rs, n)
        tmp = outp.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps({"label": label, "model_dir": str(d), "fingerprint12": fp}, ensure_ascii=False) + "\n")
            for x in recs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        tmp.replace(outp)
        print("DONE", label, fp, len(recs), flush=True)


if __name__ == "__main__":
    main()
