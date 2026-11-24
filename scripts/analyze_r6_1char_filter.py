"""R6의 "1자 길이필터 탈락" 22건 전수 조사(토큰 단위) — 읽기 전용. R7 전 진단용.

v12 "PERSON" 문장 전체에 R6 모델을 돌려, 완전노출 원인이 "1자 길이필터 탈락"인 사례를 전부 뽑고
각각 (1) 정답 PERSON 구간 (2) 토큰 분할·오프셋 (3) 토큰별 BIO 태그·확률 (4) 병합된 원시 span
(5) merge_ner 최종 결과를 나란히 기록한다. 토큰화는 `tokenizers.Tokenizer` + onnxruntime을
`ner.py`와 같은 모델 파일로 직접 돌려(운영 코드 재사용·미수정) 토큰 단위 세부를 그대로 보여준다.

ner.py·detectors.py·모델·DB·설정 미수정, v12 데이터 미수정, 외부 LLM 미호출, 재학습 없음.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from app.shared.config import settings  # noqa: E402
from app.shared.privacy import ner  # noqa: E402
from app.shared.privacy.types import Kind  # noqa: E402

V12 = ROOT / "app" / "shared" / "privacy" / "eval" / "style_v12_locked" / "v12_locked.jsonl"


def load_rows():
    rows = [json.loads(l) for l in V12.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("group") == "PERSON"]


def gold_span(row):
    for sp in row.get("spans", []):
        if sp.get("kind") in ("PERSON", "PERSON_ALIAS"):
            return sp["start"], sp["end"]
    return None


def covered(preds, g):
    cov = set()
    for s in preds:
        if s.kind is Kind.PERSON:
            cov.update(range(max(s.start, g[0]), min(s.end, g[1])))
    return cov


class TokenInspector:
    """tokenizers+onnxruntime 로 토큰별 태그·확률을 직접 뽑는다(ner.py 의 _decode 와 같은 로직,
    읽기 전용 사본 — ner.py 는 전혀 수정하지 않는다)."""

    def __init__(self, model_dir: str):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        self.tk = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tk.no_truncation()
        self.tk.no_padding()
        meta = json.loads((d / "labels.json").read_text(encoding="utf-8"))
        self.id2label = {int(k): v for k, v in meta["id2label"].items()}
        self.cls_id = int(meta["cls_token_id"])
        self.sep_id = int(meta["sep_token_id"])
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def tokens(self, text: str):
        import numpy as np

        enc = self.tk.encode(text)
        keep = [i for i, m in enumerate(enc.special_tokens_mask) if not m]
        ids = [enc.ids[i] for i in keep]
        offs = [enc.offsets[i] for i in keep]
        toks = [enc.tokens[i] for i in keep]
        seq = [self.cls_id, *ids, self.sep_id]
        input_ids = np.array([seq], dtype=np.int64)
        mask = np.ones_like(input_ids)
        feed = {"input_ids": input_ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        logits = self.sess.run(None, feed)[0][0][1:-1]
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = __import__("numpy").exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        pred = probs.argmax(axis=1)
        out = []
        for i in range(len(ids)):
            tag = self.id2label.get(int(pred[i]), "O")
            out.append({
                "token": toks[i], "char_start": offs[i][0], "char_end": offs[i][1],
                "tag": tag, "prob": round(float(probs[i, pred[i]]), 4),
            })
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--r6-candidate", required=True)
    ap.add_argument("--r5-candidate", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows = load_rows()

    def run_cause(model_dir, fp12_expected):
        ner.reset()
        settings.privacy_ner_mode = "required"
        settings.privacy_ner_model_dir = model_dir
        settings.privacy_ner_threads = 2
        status = ner.ner_status()
        assert status["fingerprint"][:12] == fp12_expected, status["fingerprint"][:12]
        ner._cache.clear()
        detail = {}
        for r in rows:
            text = r["text"]
            g = gold_span(r)
            raw = ner.spans_for(text) or []
            final = ner.detect_with_ner(text, ())
            final_person = [(s.start, s.end) for s in final if s.kind is Kind.PERSON]
            n = g[1] - g[0]
            leaked = n - len(covered(final, g))
            exact = g in final_person
            full = leaked >= n
            overlapping = [s for s in raw if max(s.start, g[0]) < min(s.end, g[1])]
            person_overlap = [s for s in overlapping if s.label in ("PERSON", "PERSON_ALIAS")]
            non_person = [s for s in overlapping if s.label not in ("PERSON", "PERSON_ALIAS")]
            if not full:
                cause = None
            elif not overlapping:
                cause = "후보 미생성"
            elif not person_overlap:
                cause = f"ORG/LOC 오분류({'/'.join(sorted({s.label for s in non_person}))})"
            else:
                short = [s for s in person_overlap if (s.end - s.start) < 2]
                long_ok = [s for s in person_overlap if (s.end - s.start) >= 2 and s.score >= ner.NER_MIN_SCORE]
                long_low = [s for s in person_overlap if (s.end - s.start) >= 2 and s.score < ner.NER_MIN_SCORE]
                if long_ok:
                    cause = "경계 불일치/병합 충돌(참고)"
                elif short and not long_low:
                    cause = "1자 길이필터 탈락"
                elif long_low:
                    cause = "신뢰도 부족"
                else:
                    cause = "기타"
            detail[r["id"]] = {
                "id": r["id"], "word": r.get("word"), "category": r.get("category"), "style": r.get("style"),
                "word_type": r.get("word_type"), "text": text, "gold": text[g[0]:g[1]], "gold_span": list(g),
                "exact": exact, "full_exposure": full, "cause": cause,
                "raw_spans": [(s.label, s.start, s.end, text[s.start:s.end], round(s.score, 4)) for s in raw],
                "final_person_spans": [(s.start, s.end, text[s.start:s.end], round(s.score, 4)) for s in final if s.kind is Kind.PERSON],
            }
        return detail

    r6_detail = run_cause(a.r6_candidate, "c59b174ba126")
    r5_detail = run_cause(a.r5_candidate, "a270ea570419")

    r6_1char_ids = [rid for rid, d in r6_detail.items() if d["cause"] == "1자 길이필터 탈락"]
    r5_1char_ids = [rid for rid, d in r5_detail.items() if d["cause"] == "1자 길이필터 탈락"]
    print(f"R6 1자 길이필터 탈락: {len(r6_1char_ids)}건, R5: {len(r5_1char_ids)}건")
    print(f"R6에만 있음(R5엔 없던 신규): {len(set(r6_1char_ids) - set(r5_1char_ids))}")
    print(f"R5에만 있음(R6에서 해결): {len(set(r5_1char_ids) - set(r6_1char_ids))}")
    print(f"둘 다 1자필터: {len(set(r6_1char_ids) & set(r5_1char_ids))}")

    insp = TokenInspector(a.r6_candidate)
    insp5 = TokenInspector(a.r5_candidate)

    cases = []
    for rid in r6_1char_ids:
        d = r6_detail[rid]
        text = d["text"]
        tok6 = insp.tokens(text)
        tok5 = insp5.tokens(text)
        g = d["gold_span"]
        tok6_overlap = [t for t in tok6 if max(t["char_start"], g[0]) < min(t["char_end"], g[1])]
        tok5_overlap = [t for t in tok5 if max(t["char_start"], g[0]) < min(t["char_end"], g[1])]
        cases.append({
            **d,
            "r6_tokens_overlapping_gold": tok6_overlap,
            "r5_tokens_overlapping_gold": tok5_overlap,
            "r6_all_tokens": tok6,
            "in_r5_too": rid in r5_1char_ids,
        })

    out = {
        "r6_1char_count": len(r6_1char_ids), "r5_1char_count": len(r5_1char_ids),
        "new_in_r6": sorted(set(r6_1char_ids) - set(r5_1char_ids)),
        "fixed_from_r5": sorted(set(r5_1char_ids) - set(r6_1char_ids)),
        "persisted": sorted(set(r6_1char_ids) & set(r5_1char_ids)),
        "cases": cases,
    }
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
