"""v8 독립 평가셋(753문장) 사후 분석: 운영 경로(규칙+NER, 시범 후처리 미적용) 기준 이름 노출 수준.

**중요(재사용 고지)**: v8_independent.jsonl 은 이미 45절에서 한 번 평가에 썼다. 이번 분석은 **그 결과를
다른 지표(글자 단위 노출 등급)로 다시 보는 사후 분석**이고, 이 결과를 보고 트리밍 규칙이나 v8 데이터를
고치지 않는다. "성능 개선 후 새 독립 검증"이 필요하면 v8 을 재사용하지 말고 새 셋(v9)을 만들어야 한다.

적용 경로: `ner.detect_with_ner`(서비스가 실제로 쓰는 경로) — 조사 제거 시범 후처리(`trial_trim`)는
**적용하지 않는다**. 비교용으로 규칙 단독(`detectors.detect`)·NER 단독도 같이 돌린다(known_names 는 v8
전체가 비어 있어, 가족 이름 사전의 기여는 이 평가에서 구조적으로 0이다 — 31절에서 이미 등록 시 효과를
확인했으므로 여기서 다시 보지 않는다).

지표(글자 단위, 이름 길이 n, 안 가려진 글자 수 leaked):
- 완전 가림(leaked=0) · 1글자 노출(leaked=1) · 2글자 이상 부분노출(2<=leaked<n) · 전체 노출(leaked=n)
- 1음절 이름(n=1)은 "노출(leaked=1)/가림(leaked=0)" 둘뿐이라 위 4분류에서 빼고 따로 집계한다.

운영 코드·모델 파일·DB·환경 설정 미수정. 외부 LLM 미호출(로컬 ONNX 추론만, "최종 전송 문자열"은 이
스크립트 안에서 문자열로만 만들어 확인한다 — 실제로 어디에도 전송하지 않는다).

사용: ``python scripts/analyze_v8_exposure_breakdown.py <서비스 모델 폴더> <R1 모델 폴더> [--out 결과.json]``
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
from app.shared.privacy.detectors import detect  # noqa: E402
from app.shared.privacy.types import Kind  # noqa: E402

V8 = ROOT / "app" / "shared" / "privacy" / "eval" / "style_v8_independent" / "v8_independent.jsonl"
EXPECT_FP12 = {"service": "46210943c45f", "r1_s7": "c536eadeb5a1"}


def load_v8() -> list[dict]:
    return [json.loads(line) for line in V8.read_text(encoding="utf-8").splitlines() if line.strip()]


def rule_only(text: str, known: tuple) -> list:
    return detect(text, known)


def ner_only(text: str, known: tuple) -> list:
    raw = ner.spans_for(text) or []
    return ner.merge_ner(text, [], raw)


def integrated(text: str, known: tuple) -> list:
    return ner.detect_with_ner(text, known)  # 시범 후처리(trial_trim) 적용 안 함 — 운영 경로 그대로


def covered(g: dict, preds: list) -> set[int]:
    cov: set[int] = set()
    for s in preds:
        if s.kind is Kind.PERSON:
            cov.update(range(max(s.start, g["start"]), min(s.end, g["end"])))
    return cov


def mask(text: str, preds: list) -> str:
    out = list(text)
    for s in preds:
        if s.kind is Kind.PERSON:
            for i in range(max(0, s.start), min(len(out), s.end)):
                out[i] = "●"
    return "".join(out)


def diagnose_full_exposure(text: str, g: dict, raw_spans: list) -> str:
    """전체노출(leaked=n) 원인: 후보 미생성 / 다른 라벨 오분류 / 낮은 신뢰도(임계값 탈락) / 후처리 제거."""
    overlapping = [s for s in raw_spans if max(s.start, g["start"]) < min(s.end, g["end"])]
    person_overlap = [s for s in overlapping if s.label in ("PERSON", "PERSON_ALIAS")]
    if not overlapping:
        return "후보 미생성"
    if not person_overlap:
        return f"다른 라벨로 오분류({'/'.join(sorted({s.label for s in overlapping}))})"
    if all(s.score < ner.NER_MIN_SCORE for s in person_overlap):
        return "낮은 신뢰도(점수 임계값 탈락)"
    return "후처리 제거(merge_ner 트리밍·길이·친족어 필터)"


def classify(n: int, leaked: int) -> str:
    if n == 1:
        return "1음절_노출" if leaked == 1 else "1음절_가림"
    if leaked == 0:
        return "완전가림"
    if leaked == n:
        return "전체노출"
    if leaked == 1:
        return "1글자노출"
    return "2글자이상_부분노출"


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

    pos_rows = [r for r in rows if r["spans"]]

    preds_by_method: dict[str, dict] = {}
    for method_key, fn in (("규칙단독", rule_only), ("NER단독", ner_only), ("통합(운영경로)", integrated)):
        preds_by_method[method_key] = {r["id"]: fn(r["text"], tuple(r.get("known_names", ()))) for r in rows}

    def summarize(preds_by_id: dict[str, list]) -> dict:
        overall = Counter()
        by_position: dict[str, Counter] = defaultdict(Counter)
        by_freqband: dict[str, Counter] = defaultdict(Counter)
        full_exposure_examples = []
        partial_examples = []
        cause_counts = Counter()

        for r in pos_rows:
            g = r["spans"][0]
            n = g["end"] - g["start"]
            preds = preds_by_id[r["id"]]
            leaked = n - len(covered(g, preds))
            cat = classify(n, leaked)
            overall[cat] += 1
            by_position[r["position"]][cat] += 1
            base_cat = r["category"].rsplit("_", 1)[0]
            if base_cat in ("신규고빈도", "신규저빈도"):
                by_freqband[base_cat][cat] += 1

            if cat in ("전체노출", "1음절_노출"):
                raw_candidates = ner.spans_for(r["text"]) or []
                cause = diagnose_full_exposure(r["text"], g, raw_candidates)
                cause_counts[cause] += 1
                if len(full_exposure_examples) < 12:
                    full_exposure_examples.append(
                        {
                            "id": r["id"], "name": r["name"], "position": r["position"],
                            "text": r["text"], "gold": r["text"][g["start"] : g["end"]],
                            "masked": mask(r["text"], preds), "cause": cause,
                        }
                    )
            elif cat in ("1글자노출", "2글자이상_부분노출") and len(partial_examples) < 12:
                partial_examples.append(
                    {
                        "id": r["id"], "name": r["name"], "position": r["position"],
                        "text": r["text"], "gold": r["text"][g["start"] : g["end"]],
                        "masked": mask(r["text"], preds), "leaked_chars": leaked, "category": cat,
                    }
                )

        n_multi = sum(1 for r in pos_rows if (r["spans"][0]["end"] - r["spans"][0]["start"]) > 1)
        n_single = len(pos_rows) - n_multi

        def rate(cat: str, denom: int) -> float:
            return round(overall[cat] / denom, 4) if denom else 0.0

        return {
            "n_positive_sentences": len(pos_rows),
            "n_multi_char_names": n_multi,
            "n_single_char_names": n_single,
            "counts": dict(overall),
            "rates_multi_char_base": {
                "완전가림률": rate("완전가림", n_multi),
                "1글자노출률": rate("1글자노출", n_multi),
                "2글자이상_부분노출률": rate("2글자이상_부분노출", n_multi),
                "전체노출률": rate("전체노출", n_multi),
            },
            "single_char_names": {
                "총": n_single,
                "노출": overall["1음절_노출"],
                "가림": overall["1음절_가림"],
                "노출률": rate("1음절_노출", n_single),
            },
            "sentence_level_full_exposure_rate": rate("전체노출", len(pos_rows)),
            "by_position": {k: dict(v) for k, v in by_position.items()},
            "by_freqband": {k: dict(v) for k, v in by_freqband.items()},
            "full_exposure_cause_counts": dict(cause_counts),
            "full_exposure_examples": full_exposure_examples,
            "partial_exposure_examples": partial_examples,
        }

    out = {"fingerprint12": fp12}
    for method_key, preds_by_id in preds_by_method.items():
        out[method_key] = summarize(preds_by_id)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("service_model_dir")
    ap.add_argument("r1_model_dir")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = load_v8()
    print(f"v8 {len(rows)}개(긍정 {sum(1 for r in rows if r['spans'])}) — 사후 분석, 재사용 데이터")

    out = {
        "service": run_model("service", args.service_model_dir, rows),
        "r1_s7": run_model("r1_s7", args.r1_model_dir, rows),
    }
    for key, r in out.items():
        print(f"\n== {key} (지문 {r['fingerprint12']}) ==")
        m = r["통합(운영경로)"]
        print(f"  운영경로 rates(2자+ 이름 기준): {m['rates_multi_char_base']}")
        print(f"  문장 단위 전체노출률: {m['sentence_level_full_exposure_rate']}")
        print(f"  1음절 이름: {m['single_char_names']}")
        print(f"  전체노출 원인: {m['full_exposure_cause_counts']}")
        print(f"  규칙단독 counts: {r['규칙단독']['counts']}")
        print(f"  NER단독 counts: {r['NER단독']['counts']}")

    out_path = args.out or (ROOT / "app" / "shared" / "privacy" / "eval" / "style_v8_independent" / "exposure_breakdown.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
