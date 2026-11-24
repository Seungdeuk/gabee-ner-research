"""R3(문형 충돌 보강) 후보와 기존 서비스 모델을 여러 평가셋에서 한 번에 비교한다 — 읽기 전용.

대상 평가셋(사용자 지시 2026-11-16, 조건 6·7):
- r3_heldout_eval(학습 미사용 holdout 4단어, 일반화 1차 확인)
- v9_locked(완전 신규 10단어, **최종 1회용** 독립 테스트 — 결과를 보고 재튜닝 안 함)
- ambiguity_probe(48절, 회귀 확인용 — 독립 근거 아님)
- step2_synth_draft(56문장, 회귀 확인용 — 독립 근거 아님, 사용자 지시 조건 1)
- v8_independent(기존 일반 평가셋, 753문장)

지표: entity-level exact P/R/F1, 비PERSON 오탐률(최종 경로), 문자 단위 완전노출/부분노출률
(position-based, 이름 글자 겹침 집합 기준). ner.py·모델·DB·설정 미수정, 외부 LLM 미호출.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from app.shared.config import settings  # noqa: E402
from app.shared.privacy import ner  # noqa: E402
from app.shared.privacy.types import Kind  # noqa: E402

EVAL_ROOT = ROOT / "app" / "shared" / "privacy" / "eval"
DATASETS = {
    "r3_heldout": Path(r"C:\Users\<user>\venvs\ner-runs\data_r3_phrase\r3_heldout_eval.jsonl"),
    "v9_locked": EVAL_ROOT / "style_v9_locked" / "v9_locked.jsonl",
    "ambiguity_probe_REGRESSION": EVAL_ROOT / "style_ambiguity_probe" / "ambiguity_probe.jsonl",
    "draft56_REGRESSION": EVAL_ROOT / "step2_synth_draft" / "synth_draft.jsonl",
    "v8_independent_GENERAL": EVAL_ROOT / "style_v8_independent" / "v8_independent.jsonl",
}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def gold_spans(row: dict) -> list[tuple[int, int]]:
    return [(sp["start"], sp["end"]) for sp in row.get("spans", []) if sp.get("kind") in ("PERSON", "PERSON_ALIAS")]


def covered(spans: list, g: tuple[int, int]) -> set[int]:
    cov: set[int] = set()
    for s in spans:
        if s.kind is Kind.PERSON:
            cov.update(range(max(s.start, g[0]), min(s.end, g[1])))
    return cov


def eval_dataset(rows: list[dict]) -> dict:
    tp = fp = fn = 0
    full_exposure = partial_exposure = fully_masked = 0
    pos_total = 0
    neg_total = neg_flagged = 0
    by_group: dict[str, Counter] = defaultdict(Counter)

    for r in rows:
        text = r["text"]
        golds = gold_spans(r)
        preds = ner.detect_with_ner(text, ())
        pred_person = [(s.start, s.end) for s in preds if s.kind is Kind.PERSON]
        group = r.get("group") or r.get("category") or "_all"

        if golds:
            pos_total += 1
            for g in golds:
                n = g[1] - g[0]
                exact = g in pred_person
                tp += int(exact)
                if not exact:
                    fn += 1
                leaked = n - len(covered(preds, g))
                if leaked >= n:
                    full_exposure += 1
                elif leaked > 0:
                    partial_exposure += 1
                else:
                    fully_masked += 1
                by_group[group]["total"] += 1
                by_group[group]["exact"] += int(exact)
                by_group[group]["full_exposure"] += int(leaked >= n)
            for p in pred_person:
                if p not in golds:
                    fp += 1
        else:
            neg_total += 1
            flagged = bool(pred_person)
            neg_flagged += int(flagged)
            by_group[group]["total"] += 1
            by_group[group]["flagged"] += int(flagged)

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) else None
    return {
        "pos_sentences": pos_total, "neg_sentences": neg_total,
        "exact": {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1},
        "char_exposure": {
            "full_exposure": full_exposure, "partial_exposure": partial_exposure, "fully_masked": fully_masked,
            "full_exposure_rate": round(full_exposure / (full_exposure + partial_exposure + fully_masked), 4) if (full_exposure + partial_exposure + fully_masked) else None,
        },
        "nonperson_fpr": round(neg_flagged / neg_total, 4) if neg_total else None,
        "by_group": {k: dict(v) for k, v in by_group.items()},
    }


def run_model(model_key: str, model_dir: str, expect_fp12: str | None) -> dict:
    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = model_dir
    settings.privacy_ner_threads = 2
    status = ner.ner_status()
    fp12 = status["fingerprint"][:12]
    if expect_fp12 and fp12 != expect_fp12:
        raise SystemExit(f"[{model_key}] 지문 불일치: got {fp12}, expected {expect_fp12}")
    ner._cache.clear()
    out = {"fingerprint12": fp12}
    for name, path in DATASETS.items():
        if not path.exists():
            out[name] = {"error": f"파일 없음: {path}"}
            continue
        rows = load_rows(path)
        out[name] = eval_dataset(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--service", required=True)
    ap.add_argument("--r3-candidate", required=True)
    ap.add_argument("--r1", default=None)
    ap.add_argument("--service-fp12", default="46210943c45f")
    ap.add_argument("--r1-fp12", default="c536eadeb5a1")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    out = {
        "service": run_model("service", a.service, a.service_fp12),
        "r3_candidate": run_model("r3_candidate", a.r3_candidate, None),
    }
    if a.r1:
        out["r1_s7"] = run_model("r1_s7", a.r1, a.r1_fp12)

    for key, r in out.items():
        print(f"\n== {key} (fp12={r['fingerprint12']}) ==")
        for name in DATASETS:
            d = r.get(name, {})
            if "error" in d:
                print(f"  {name}: {d['error']}")
                continue
            print(f"  {name}: exact-F1={d['exact']['f1']}, full_exposure_rate={d['char_exposure']['full_exposure_rate']}, nonperson_fpr={d['nonperson_fpr']}")

    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
