"""v7 사후 진단(2026-11-16): 시범 조사 후처리(trial_trim)가 실제 개인정보 노출을 줄이는지 검증.

**중요(조건 8)**: 이 스크립트는 `analyze_v7_followup.py`가 `trial_trim` 규칙을 만들 때 이미 들여다본
v7_namefreq.jsonl(2,400+78문장)을 그대로 또 쓴다. 즉 "후처리 개발에 쓴 데이터로 그 후처리를 검증"하는
**사후 진단(post-hoc)**이지, 독립 검증이 아니다 — 아래 결과만으로 실서비스 적용을 확정해서는 안 된다
(8절 참고). 운영 코드(`ner.py`)·모델 파일·DB·운영 설정은 전혀 건드리지 않는다. 외부 LLM 호출 없음(이
파이프라인은 로컬 ONNX 추론만 쓰고, 애초에 LLM을 호출하지 않는다).

F1/Precision/Recall은 "탐지"가 기준이라, 이미 정답을 전부 덮고 추가로 조사까지 가져간 과잉 예측(= 이미
안전함)과, 실제로 이름 글자를 못 덮은 예측(= 유출)을 둘 다 "틀림"으로 섞어 센다. 이 스크립트는 **글자
단위로 가려졌는지만** 본다 — F1이 올라도 유출이 그대로이거나 늘 수 있다는 걸 직접 보이기 위함이다.

사용: ``python scripts/analyze_v7_privacy_exposure.py <서비스 모델 폴더> <R1 모델 폴더> [--out 결과.json]``
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from analyze_v7_followup import load_v7, predict, trial_trim  # noqa: E402
from app.shared.config import settings  # noqa: E402
from app.shared.privacy import ner  # noqa: E402
from app.shared.privacy.types import Kind  # noqa: E402

EXPECT_FP12 = {"service": "46210943c45f", "r1_s7": "c536eadeb5a1"}


def mask(text: str, spans: list) -> str:
    """원문 위치(start,end) 기준으로만 가린다 — 문자열 검색이 아니라 인덱스 연산이라, 같은 글자가
    문장 다른 자리에 또 나와도 틀리지 않는다(조건 3)."""
    out = list(text)
    for s in spans:
        if s.kind is Kind.PERSON:
            for i in range(max(0, s.start), min(len(out), s.end)):
                out[i] = "●"
    return "".join(out)


def leak_status(n: int, leaked: int) -> str:
    if leaked == 0:
        return "노출없음"
    if leaked >= n:
        return "완전노출"
    return "부분노출"


def covered_indices(g: dict, preds: list) -> set[int]:
    cov: set[int] = set()
    for s in preds:
        if s.kind is Kind.PERSON:
            cov.update(range(max(s.start, g["start"]), min(s.end, g["end"])))
    return cov


def analyze_model(model_key: str, model_dir: str, rows: list[dict]) -> dict:
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

    raw_by_id = {r["id"]: predict(r["text"], tuple(r.get("known_names", ()))) for r in rows}
    trimmed_by_id = {rid: [trial_trim(next(r for r in rows if r["id"] == rid)["text"], s) for s in spans] for rid, spans in raw_by_id.items()}
    text_by_id = {r["id"]: r["text"] for r in rows}
    trimmed_by_id = {rid: [trial_trim(text_by_id[rid], s) for s in spans] for rid, spans in raw_by_id.items()}

    pos_rows = [r for r in rows if r["spans"]]
    neg_rows = [r for r in rows if not r["spans"]]

    total_gold_chars = 0
    total_leaked_before = total_leaked_after = 0
    status_counts_before: dict[str, int] = {"완전노출": 0, "부분노출": 0, "노출없음": 0}
    status_counts_after: dict[str, int] = {"완전노출": 0, "부분노출": 0, "노출없음": 0}
    improved, worsened, unchanged_exposed = [], [], 0

    for r in pos_rows:
        g = r["spans"][0]
        n = g["end"] - g["start"]
        total_gold_chars += n
        before_cov = covered_indices(g, raw_by_id[r["id"]])
        after_cov = covered_indices(g, trimmed_by_id[r["id"]])
        leaked_before = n - len(before_cov)
        leaked_after = n - len(after_cov)
        total_leaked_before += leaked_before
        total_leaked_after += leaked_after
        status_counts_before[leak_status(n, leaked_before)] += 1
        status_counts_after[leak_status(n, leaked_after)] += 1

        example = {
            "id": r["id"],
            "name": r["name"],
            "text": r["text"],
            "gold": r["text"][g["start"] : g["end"]],
            "before_masked": mask(r["text"], raw_by_id[r["id"]]),
            "after_masked": mask(r["text"], trimmed_by_id[r["id"]]),
            "leaked_before": leaked_before,
            "leaked_after": leaked_after,
        }
        if leaked_after < leaked_before:
            improved.append(example)
        elif leaked_after > leaked_before:
            worsened.append(example)
        elif leaked_before > 0:
            unchanged_exposed += 1

    ends_with_i = [e for e in worsened if e["name"].endswith("이")]

    # ---- 비인명(과마스킹) 출력 변화 ----
    neg_output_changes = []
    neg_flagged_before = neg_flagged_after = 0
    for r in neg_rows:
        before_text = mask(r["text"], raw_by_id[r["id"]])
        after_text = mask(r["text"], trimmed_by_id[r["id"]])
        if any(s.kind is Kind.PERSON for s in raw_by_id[r["id"]]):
            neg_flagged_before += 1
        if any(s.kind is Kind.PERSON for s in trimmed_by_id[r["id"]]):
            neg_flagged_after += 1
        if before_text != after_text:
            neg_output_changes.append({"word": r["name"], "text": r["text"], "before": before_text, "after": after_text})

    n_pos = len(pos_rows)
    return {
        "fingerprint12": fp12,
        "n_positive_sentences": n_pos,
        "total_gold_chars": total_gold_chars,
        "char_leak_rate_before": round(total_leaked_before / total_gold_chars, 4),
        "char_leak_rate_after": round(total_leaked_after / total_gold_chars, 4),
        "full_exposure_rate_before": round(status_counts_before["완전노출"] / n_pos, 4),
        "full_exposure_rate_after": round(status_counts_after["완전노출"] / n_pos, 4),
        "partial_exposure_rate_before": round(status_counts_before["부분노출"] / n_pos, 4),
        "partial_exposure_rate_after": round(status_counts_after["부분노출"] / n_pos, 4),
        "any_exposure_rate_before": round((status_counts_before["완전노출"] + status_counts_before["부분노출"]) / n_pos, 4),
        "any_exposure_rate_after": round((status_counts_after["완전노출"] + status_counts_after["부분노출"]) / n_pos, 4),
        "status_counts_before": status_counts_before,
        "status_counts_after": status_counts_after,
        "improved_count": len(improved),
        "worsened_count": len(worsened),
        "improved_examples": improved[:8],
        "worsened_examples": worsened,
        "worsened_ending_in_i": ends_with_i,
        "neg_flagged_before": neg_flagged_before,
        "neg_flagged_after": neg_flagged_after,
        "neg_total": len(neg_rows),
        "neg_output_text_changed_count": len(neg_output_changes),
        "neg_output_text_changed_examples": neg_output_changes[:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("service_model_dir")
    ap.add_argument("r1_model_dir")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = load_v7()
    print(f"v7 {len(rows)}개(긍정 {sum(1 for r in rows if r['spans'])}, 비인명 {sum(1 for r in rows if not r['spans'])}) — 후처리 개발에 쓴 셋 재사용(사후 진단)")

    out = {
        "service": analyze_model("service", args.service_model_dir, rows),
        "r1_s7": analyze_model("r1_s7", args.r1_model_dir, rows),
    }
    for key, r in out.items():
        print(f"\n== {key} ==")
        print(f"  잔존율(글자): {r['char_leak_rate_before']*100:.2f}% -> {r['char_leak_rate_after']*100:.2f}%")
        print(f"  완전노출률: {r['full_exposure_rate_before']*100:.2f}% -> {r['full_exposure_rate_after']*100:.2f}%")
        print(f"  부분노출률: {r['partial_exposure_rate_before']*100:.2f}% -> {r['partial_exposure_rate_after']*100:.2f}%")
        print(f"  노출 1건이상 비율: {r['any_exposure_rate_before']*100:.2f}% -> {r['any_exposure_rate_after']*100:.2f}%")
        print(f"  개선 {r['improved_count']}건 / 악화 {r['worsened_count']}건(그중 '이'로 끝나는 이름 {len(r['worsened_ending_in_i'])}건)")
        print(f"  비인명 과마스킹(플래그): {r['neg_flagged_before']}/{r['neg_total']} -> {r['neg_flagged_after']}/{r['neg_total']}, 출력 텍스트 달라진 문장 {r['neg_output_text_changed_count']}건")

    out_path = args.out or (ROOT / "app" / "shared" / "privacy" / "eval" / "style_v7_namefreq" / "privacy_exposure_analysis.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
