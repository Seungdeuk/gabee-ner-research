"""v8 독립 평가(2026-11-16): 처음 보는 이름·새 템플릿에서 조사 후처리가 여전히 유효한지 검증.

읽기 전용. v8_independent.jsonl(v7·학습 데이터와 전혀 안 겹침, build_v8_independent.py 로 생성)만 쓴다.
`trial_trim`(analyze_v7_followup.py)은 **이 평가 결과를 보고 바꾸지 않는다** — v7에서 이미 고정된 규칙을
그대로 가져와 쓴다. 운영 코드(`ner.py`)·모델 파일·DB·운영 설정 미수정. 외부 LLM 호출 없음(로컬 ONNX
추론만 함).

측정: entity-level exact-match P/R/F1, 완전미탐률, 부분노출률(문자 단위), 문자 잔존율, 문장 단위 노출률
(이름 1글자라도 남는 문장 비율), 위치(시작/중간/끝)·이름범주별 분해, 비인명(사물문맥·순수비인명) 오탐,
이름문맥(일반단어가 실제 이름으로 쓰인 경우) recall. trial_trim 적용 전/후 전부 비교.

사용: ``python scripts/analyze_v8_independent.py <서비스 모델 폴더> <R1 모델 폴더> [--out 결과.json]``
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from analyze_v7_followup import predict, trial_trim  # noqa: E402  (바꾸지 않고 그대로 재사용)
from app.shared.config import settings  # noqa: E402
from app.shared.privacy import ner  # noqa: E402
from app.shared.privacy.types import Kind  # noqa: E402

V8 = ROOT / "app" / "shared" / "privacy" / "eval" / "style_v8_independent" / "v8_independent.jsonl"
EXPECT_FP12 = {"service": "46210943c45f", "r1_s7": "c536eadeb5a1"}


def load_v8() -> list[dict]:
    return [json.loads(line) for line in V8.read_text(encoding="utf-8").splitlines() if line.strip()]


def mask(text: str, spans: list) -> str:
    out = list(text)
    for s in spans:
        if s.kind is Kind.PERSON:
            for i in range(max(0, s.start), min(len(out), s.end)):
                out[i] = "●"
    return "".join(out)


def covered(g: dict, preds: list) -> set[int]:
    cov: set[int] = set()
    for s in preds:
        if s.kind is Kind.PERSON:
            cov.update(range(max(s.start, g["start"]), min(s.end, g["end"])))
    return cov


def exact_bucket(g: dict, preds: list) -> str:
    person = [s for s in preds if s.kind is Kind.PERSON]
    if any(s.start == g["start"] and s.end == g["end"] for s in person):
        return "정확탐지"
    cov = covered(g, preds)
    return "완전미탐" if not cov else "경계불일치"


def run_model(model_key: str, model_dir: str, rows: list[dict]) -> dict:
    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = model_dir
    settings.privacy_ner_threads = 2
    status = ner.ner_status()
    fp12 = status["fingerprint"][:12]
    if EXPECT_FP12.get(model_key) and fp12 != EXPECT_FP12[model_key]:
        raise SystemExit(f"[{model_key}] 지문 불일치: got {fp12}")
    if not status["active"]:
        raise SystemExit(f"[{model_key}] NER 엔진 비활성")
    ner._cache.clear()

    raw = {r["id"]: predict(r["text"], tuple(r.get("known_names", ()))) for r in rows}
    text_by_id = {r["id"]: r["text"] for r in rows}
    trimmed = {rid: [trial_trim(text_by_id[rid], s) for s in spans] for rid, spans in raw.items()}

    def summarize(preds_by_id: dict[str, list]) -> dict:
        tp = fp = fn = 0
        total_gold_chars = leaked_chars = 0
        n_pos = n_full_exp = n_partial_exp = 0
        by_category: dict[str, Counter] = defaultdict(Counter)
        by_position: dict[str, Counter] = defaultdict(Counter)
        failure_examples: list[dict] = []
        neg_by_category: dict[str, dict] = defaultdict(lambda: {"total": 0, "flagged": 0})

        for r in rows:
            preds = preds_by_id[r["id"]]
            person = [s for s in preds if s.kind is Kind.PERSON]
            if not r["spans"]:
                fp += len(person)
                d = neg_by_category[r["category"].rsplit("_", 1)[0]]
                d["total"] += 1
                d["flagged"] += int(bool(person))
                continue
            g = r["spans"][0]
            n_pos += 1
            n = g["end"] - g["start"]
            total_gold_chars += n
            cov = covered(g, preds)
            leaked = n - len(cov)
            leaked_chars += leaked
            if leaked >= n:
                n_full_exp += 1
            elif leaked > 0:
                n_partial_exp += 1
            bucket = exact_bucket(g, preds)
            if bucket == "정확탐지":
                tp += 1
                fp += len(person) - 1
            else:
                fn += 1
                fp += len(person)
            base_cat = r["category"].rsplit("_", 1)[0]
            by_category[base_cat][bucket] += 1
            by_position[r["position"]][bucket] += 1
            if bucket != "정확탐지" and len(failure_examples) < 30:
                failure_examples.append(
                    {
                        "id": r["id"],
                        "category": r["category"],
                        "position": r["position"],
                        "name": r["name"],
                        "text": r["text"],
                        "gold": r["text"][g["start"] : g["end"]],
                        "predicted": [r["text"][s.start : s.end] for s in person],
                        "bucket": bucket,
                        "leaked_chars": leaked,
                    }
                )

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n_positive": n_pos,
            "char_leak_rate": round(leaked_chars / total_gold_chars, 4) if total_gold_chars else 0.0,
            "full_exposure_rate": round(n_full_exp / n_pos, 4) if n_pos else 0.0,
            "partial_exposure_rate": round(n_partial_exp / n_pos, 4) if n_pos else 0.0,
            "any_exposure_rate": round((n_full_exp + n_partial_exp) / n_pos, 4) if n_pos else 0.0,
            "by_category": {k: dict(v) for k, v in by_category.items()},
            "by_position": {k: dict(v) for k, v in by_position.items()},
            "neg_by_category": {k: {**v, "fpr": round(v["flagged"] / v["total"], 4) if v["total"] else 0.0} for k, v in neg_by_category.items()},
            "failure_examples": failure_examples,
        }

    before = summarize(raw)
    after = summarize(trimmed)

    # before/after 비교: 노출 개선/악화 사례
    improved, worsened = [], []
    for r in rows:
        if not r["spans"]:
            continue
        g = r["spans"][0]
        n = g["end"] - g["start"]
        lb = n - len(covered(g, raw[r["id"]]))
        la = n - len(covered(g, trimmed[r["id"]]))
        if la < lb or la > lb:
            ex = {
                "id": r["id"], "name": r["name"], "text": r["text"],
                "before_masked": mask(r["text"], raw[r["id"]]), "after_masked": mask(r["text"], trimmed[r["id"]]),
                "leaked_before": lb, "leaked_after": la,
            }
            (improved if la < lb else worsened).append(ex)

    return {
        "fingerprint12": fp12,
        "before_trim": before,
        "after_trim": after,
        "improved_count": len(improved),
        "worsened_count": len(worsened),
        "improved_examples": improved[:10],
        "worsened_examples": worsened,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("service_model_dir")
    ap.add_argument("r1_model_dir")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = load_v8()
    print(f"v8 {len(rows)}개(긍정 {sum(1 for r in rows if r['spans'])}, 비인명 {sum(1 for r in rows if not r['spans'])})")

    out = {
        "service": run_model("service", args.service_model_dir, rows),
        "r1_s7": run_model("r1_s7", args.r1_model_dir, rows),
    }
    for key, r in out.items():
        b, a = r["before_trim"], r["after_trim"]
        print(f"\n== {key} ==")
        print(f"  P/R/F1 전: {b['precision']}/{b['recall']}/{b['f1']}  후: {a['precision']}/{a['recall']}/{a['f1']}")
        print(f"  잔존율 전: {b['char_leak_rate']*100:.2f}% 후: {a['char_leak_rate']*100:.2f}%")
        print(f"  완전노출률 전: {b['full_exposure_rate']*100:.2f}% 후: {a['full_exposure_rate']*100:.2f}%")
        print(f"  노출1건이상 전: {b['any_exposure_rate']*100:.2f}% 후: {a['any_exposure_rate']*100:.2f}%")
        print(f"  개선 {r['improved_count']}건 / 악화 {r['worsened_count']}건")
        print(f"  비인명 FPR(전): {b['neg_by_category']}")

    out_path = args.out or (ROOT / "app" / "shared" / "privacy" / "eval" / "style_v8_independent" / "results_v8.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
