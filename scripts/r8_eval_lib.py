"""R8 평가 공용 라이브러리 - 서비스 모델과 후보를 같은 코드·같은 설정으로 채점한다(읽기 전용, 모델·DB·설정 미수정).

이전 채점(`evaluate_r6_candidate.eval_rows`)은 정답이 여러 개인 문장에서 **첫 정답 구간만** 정답으로 보고 나머지 이름을
오탐으로 셌다. 이 라이브러리는 문장의 모든 PERSON 계열 정답 구간을 채점한다(`legacy_first_only=True` 로 이전 방식도 재현한다).

판정 정의:
- 최종 예측 = ``ner.detect_with_ner(text, ())`` 의 Kind.PERSON 구간(가족 이름 사전 비움, 규칙+NER 병합 결과).
- Exact: (start, end) 가 정답과 정확히 같으면 TP. 못 맞춘 정답은 FN. 어떤 정답과도 정확히 같지 않은 예측은 FP.
- 노출: 정답 구간 글자 중 PERSON 예측으로 덮이지 않은 글자 수. 전부 안 덮이면 완전 노출, 일부면 부분 노출, 없으면 완전 가림.
  전체 노출률 = (완전 + 부분) / 정답 구간 수.
- 비PERSON 오탐률 = PERSON 정답이 없는 문장 중 PERSON 을 하나라도 낸 문장 비율.
- 경계 FP = 정답과 겹치지만 exact 가 아닌 예측, 순수 FP = 어떤 정답과도 겹치지 않는 예측(정답 있는 문장 안의 오탐).
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.shared.config import settings  # noqa: E402
from app.shared.privacy import ner  # noqa: E402
from app.shared.privacy.types import Kind  # noqa: E402

PERSON_KINDS = ("PERSON", "PERSON_ALIAS")


def set_model(model_dir, expect_fp12=None, threads=2):
    """서비스 설정을 바꾸지 않고(프로세스 안에서만) 모델 폴더를 지정하고 지문을 확인한다."""
    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = str(model_dir)
    settings.privacy_ner_threads = threads
    st = ner.ner_status()
    fp12 = st["fingerprint"][:12]
    if expect_fp12 and fp12 != expect_fp12:
        raise SystemExit(f"지문 불일치: got {fp12}, expected {expect_fp12}")
    ner._cache.clear()
    return fp12


def gold_person(row, key_candidates=("kind", "label", "mapped_kind")):
    """행의 PERSON 계열 정답 구간 목록(정렬)."""
    out = []
    for sp in row.get("spans", []):
        kinds = [sp.get(k) for k in key_candidates if sp.get(k)]
        if any(k in PERSON_KINDS for k in kinds):
            out.append((sp["start"], sp["end"]))
    return sorted(out)


def wilson(k, n, z=1.96):
    """이항 비율의 Wilson 95% 신뢰구간(비율 단위)."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None  # 2PR/(P+R) 와 같은 값, TP 0 이면 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def final_person_spans(text):
    return [(s.start, s.end) for s in ner.detect_with_ner(text, ()) if s.kind is Kind.PERSON]


def score_rows(rows, legacy_first_only=False, keep_records=False, pred_fn=None):
    """rows 전체를 채점한다. keep_records 이면 문장별 판정을 함께 돌려준다."""
    tp = fp = fn = 0
    fp_boundary = fp_pure = 0
    full = part = masked = 0
    neg_n = neg_flag = 0
    recs = []
    for r in rows:
        text = r["text"]
        golds = gold_person(r)
        pred = pred_fn(r) if pred_fn else final_person_spans(text)
        if legacy_first_only and golds:
            g_all = golds
            golds = golds[:1]
        if golds:
            gset = set(golds)
            tp += sum(1 for g in golds if g in pred)
            fn += sum(1 for g in golds if g not in pred)
            fp += sum(1 for p in pred if p not in gset)
            for p in pred:
                if p in gset:
                    continue
                if any(max(p[0], g[0]) < min(p[1], g[1]) for g in golds):
                    fp_boundary += 1
                else:
                    fp_pure += 1
            cls_list = []
            for g in golds:
                cov = set()
                for a, b in pred:
                    cov.update(range(max(a, g[0]), min(b, g[1])))
                leaked = (g[1] - g[0]) - len(cov)
                c = "full" if leaked >= g[1] - g[0] else ("partial" if leaked > 0 else "masked")
                cls_list.append((g, c, leaked))
                if c == "full":
                    full += 1
                elif c == "partial":
                    part += 1
                else:
                    masked += 1
            if keep_records:
                recs.append({"id": r.get("id"), "text": text, "gold": golds, "pred": pred, "cls": [c for _, c, _ in cls_list]})
        else:
            neg_n += 1
            neg_flag += int(bool(pred))
            if keep_records:
                recs.append({"id": r.get("id"), "text": text, "gold": [], "pred": pred, "cls": ["neg_flagged" if pred else "neg_clean"]})
    n_gold = full + part + masked
    out = {
        "n_sentences": len(rows), "n_gold_spans": n_gold, "n_neg_sentences": neg_n,
        "prf": prf(tp, fp, fn), "fp_boundary": fp_boundary, "fp_pure": fp_pure,
        "full": full, "partial": part, "masked": masked,
        "full_rate": full / n_gold if n_gold else None,
        "partial_rate": part / n_gold if n_gold else None,
        "total_exposure_rate": (full + part) / n_gold if n_gold else None,
        "total_exposure_ci": wilson(full + part, n_gold),
        "neg_flagged": neg_flag,
        "nonperson_fpr": neg_flag / neg_n if neg_n else None,
        "nonperson_fpr_ci": wilson(neg_flag, neg_n),
    }
    if keep_records:
        out["records"] = recs
    return out


def macro(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
