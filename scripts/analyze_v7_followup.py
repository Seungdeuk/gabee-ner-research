"""v7 후속 분석(2026-11-16): 조사 경계오류 상세·일반단어 이름사용 비교·시범 후처리·가명처리 잔존 검사.

읽기 전용. v7_namefreq.jsonl 과 새로 지은 "이름 문맥" 테스트 문장만 쓴다(blind_v6 안 씀). 운영 코드
(`app/shared/privacy/ner.py`)는 전혀 수정하지 않는다 — 시범 후처리는 이 스크립트 안에서 예측 Span 목록을
받은 "뒤"에만 적용하는, 별도의 순수 함수(`trial_trim`)다.

1) 경계불일치 사례 전부(이 스크립트 실행 시점의 개수, v7 결과의 "경계불일치" 버킷과 동일 정의)를 조사
   유무·붙은 조사 종류별로 분류하고, 정답과 최종 예측을 나란히 보여준다.
2) 일반명사이자 실제 등록 이름이기도 한 단어(바다·노을·하늘·이슬·새벽 등, 공개 이름 빈도 통계에 실제 등록
   건수 있음) 26개를 "사물 문맥"(기존 v7 비인명 문장)과 "이름 문맥"(성 붙임/호격, 새로 지은 문장)에서
   recall 을 비교한다.
3) 시범 후처리(`trial_trim`: PERSON 구간 끝의 "이/가"가 받침 규칙과 맞을 때만 1글자 제거, 2글자 미만으로는
   안 줄임)를 적용 전/후로 v7 전체 Recall·경계오류율·Precision과 비인명 FPR을 비교한다.
4) "가명처리 결과" 기준: 최종 예측 구간으로 정답 이름 글자를 전부 가렸다고 가정했을 때, 가려지지 않고 남는
   글자 비율(문자 단위 잔존율)을 전/후로 비교하고, 대표 사례의 원문→가명화 결과를 보여준다.

사용: ``python scripts/analyze_v7_followup.py <서비스 모델 폴더> <R1 모델 폴더> [--out 결과.json]``
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
from app.shared.privacy.types import Kind, Span  # noqa: E402

V7 = ROOT / "app" / "shared" / "privacy" / "eval" / "style_v7_namefreq" / "v7_namefreq.jsonl"
EXPECT_FP12 = {"service": "46210943c45f", "r1_s7": "c536eadeb5a1"}

REAL_NAME_WORDS = {"바다": 4, "노을": 5, "하늘": 98, "이슬": 4, "새벽": 10}
ALL_NEG_WORDS = [
    "바다", "노을", "하늘", "구름", "시냇물", "소나기", "무지개", "들판", "안개", "이슬",
    "파도", "폭포", "단풍", "눈송이", "별빛", "햇살", "바람", "새벽", "노래방", "운동장",
    "놀이터", "도서관", "미술관", "박물관", "수족관", "놀이공원",
]
SURNAME = "김"  # 이름-문맥 테스트용 고정 성(결과 해석을 단순하게 유지하기 위해 하나만 사용)


def has_batchim(ch: str) -> bool:
    return (ord(ch) - 0xAC00) % 28 != 0


def trial_trim(text: str, s: Span) -> Span:
    """시범 후처리(운영 코드 아님): PERSON 구간 끝의 이/가가 받침 규칙과 맞을 때만 1글자 줄인다."""
    if s.kind is not Kind.PERSON or s.end - s.start <= 2:
        return s
    last = text[s.end - 1]
    prev = text[s.end - 2]
    if last == "이" and has_batchim(prev):
        return Span(s.kind, s.start, s.end - 1, s.score, s.source, s.canonical)
    if last == "가" and not has_batchim(prev):
        return Span(s.kind, s.start, s.end - 1, s.score, s.source, s.canonical)
    return s


def name_context_rows() -> list[dict]:
    rows = []
    for w in ALL_NEG_WORDS:
        surname_name = SURNAME + w
        rows.append(
            {
                "id": f"ctx-surname-{w}",
                "word": w,
                "condition": "성붙임(이름문맥)",
                "text": f"{surname_name} 오늘 유치원 다녀왔대",
                "spans": [{"kind": "PERSON", "start": 0, "end": len(surname_name)}],
                "known_names": [],
            }
        )
        rows.append(
            {
                "id": f"ctx-vocative-{w}",
                "word": w,
                "condition": "호격(이름문맥)",
                "text": f"{w}야 빨리 밥 먹어",
                "spans": [{"kind": "PERSON", "start": 0, "end": len(w)}],
                "known_names": [],
            }
        )
    return rows


def load_v7() -> list[dict]:
    return [json.loads(line) for line in V7.read_text(encoding="utf-8").splitlines() if line.strip()]


def predict(text: str, known: tuple) -> list[Span]:
    return ner.detect_with_ner(text, known)


def classify_exact(g: dict, preds: list[Span]) -> tuple[str, Span | None]:
    person = [s for s in preds if s.kind is Kind.PERSON]
    for s in person:
        if s.start == g["start"] and s.end == g["end"]:
            return "정확탐지", s
    cov: set[int] = set()
    for s in person:
        cov.update(range(max(s.start, g["start"]), min(s.end, g["end"])))
    n = g["end"] - g["start"]
    if not cov:
        return "완전미탐", None
    best = max(
        (s for s in person if max(s.start, g["start"]) < min(s.end, g["end"])),
        key=lambda s: len(range(max(s.start, g["start"]), min(s.end, g["end"]))),
        default=None,
    )
    return "경계불일치", best


def describe_mismatch(text: str, g: dict, pred: Span | None) -> str:
    if pred is None:
        return "예측 없음(여러 조각으로 쪼개졌거나 검출 실패)"
    if pred.start == g["start"] and pred.end == g["end"] + 1 and text[g["end"]] in "이가":
        return f"조사 포함(끝에 '{text[g['end']]}' 더 가져감)"
    if pred.start > g["start"]:
        return f"앞부분 누락({text[g['start']:pred.start]!r} 못 덮음)"
    if pred.end < g["end"]:
        return f"뒷부분 누락({text[pred.end:g['end']]!r} 못 덮음)"
    if pred.start == g["start"] and pred.end > g["end"]:
        return f"끝에 더 가져감({text[g['end']:pred.end]!r})"
    return "기타"


def mask(text: str, spans: list[Span]) -> str:
    out = list(text)
    for s in spans:
        if s.kind is Kind.PERSON:
            for i in range(s.start, s.end):
                if 0 <= i < len(out):
                    out[i] = "O"
    return "".join(out)


def residual_leak(rows: list[dict], preds_by_id: dict[str, list[Span]]) -> tuple[float, list[dict]]:
    total_chars = leaked_chars = 0
    examples = []
    for r in rows:
        if not r["spans"]:
            continue
        g = r["spans"][0]
        preds = preds_by_id[r["id"]]
        covered = set()
        for s in preds:
            if s.kind is Kind.PERSON:
                covered.update(range(max(s.start, g["start"]), min(s.end, g["end"])))
        n = g["end"] - g["start"]
        leaked = n - len(covered)
        total_chars += n
        leaked_chars += leaked
        if leaked and len(examples) < 10:
            examples.append(
                {
                    "text": r["text"],
                    "gold_name": r["text"][g["start"] : g["end"]],
                    "masked": mask(r["text"], preds),
                    "leaked_chars": leaked,
                }
            )
    rate = leaked_chars / total_chars if total_chars else 0.0
    return rate, examples


def run_model(model_key: str, model_dir: str, v7_rows: list[dict], ctx_rows: list[dict]) -> dict:
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

    # ---- v7 전체: 원본 예측 ----
    raw_preds: dict[str, list[Span]] = {}
    for r in v7_rows:
        raw_preds[r["id"]] = predict(r["text"], tuple(r.get("known_names", ())))
    trimmed_preds: dict[str, list[Span]] = {
        rid: [trial_trim(next(r["text"] for r in v7_rows if r["id"] == rid), s) for s in spans]
        for rid, spans in raw_preds.items()
    }
    # 위 trimmed_preds 계산은 text 조회가 느리므로(선형 탐색) id->text 매핑을 미리 만든다
    text_by_id = {r["id"]: r["text"] for r in v7_rows}
    trimmed_preds = {rid: [trial_trim(text_by_id[rid], s) for s in spans] for rid, spans in raw_preds.items()}

    def summarize(preds_by_id: dict[str, list[Span]]) -> dict:
        tp = fp = fn = 0
        buckets = Counter()
        mismatch_detail = []
        by_particle_boundary = Counter()
        for r in v7_rows:
            preds = preds_by_id[r["id"]]
            person = [s for s in preds if s.kind is Kind.PERSON]
            if not r["spans"]:
                fp += len(person)
                continue
            g = r["spans"][0]
            bucket, best = classify_exact(g, preds)
            buckets[bucket] += 1
            exact_idx = next((i for i, s in enumerate(person) if s.start == g["start"] and s.end == g["end"]), None)
            if exact_idx is not None:
                tp += 1
                fp += len(person) - 1
            else:
                fn += 1
                fp += len(person)
            if bucket == "경계불일치":
                particle_key = "조사있음" if r["has_particle"] else "조사없음"
                by_particle_boundary[particle_key] += 1
                mismatch_detail.append(
                    {
                        "id": r["id"],
                        "name": r["name"],
                        "category": r["category"],
                        "has_surname": r["has_surname"],
                        "has_particle": r["has_particle"],
                        "text": r["text"],
                        "gold": r["text"][g["start"] : g["end"]],
                        "predicted": [r["text"][s.start : s.end] for s in person],
                        "reason": describe_mismatch(r["text"], g, best),
                    }
                )
        neg_flagged = sum(
            1 for r in v7_rows if not r["spans"] and any(s.kind is Kind.PERSON for s in preds_by_id[r["id"]])
        )
        neg_total = sum(1 for r in v7_rows if not r["spans"])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        leak_rate, leak_examples = residual_leak(v7_rows, preds_by_id)
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "buckets": dict(buckets),
            "boundary_mismatch_by_particle": dict(by_particle_boundary),
            "neg_fpr": round(neg_flagged / neg_total, 4) if neg_total else 0.0,
            "residual_char_leak_rate": round(leak_rate, 4),
            "residual_leak_examples": leak_examples,
            "mismatch_detail": mismatch_detail,
        }

    before = summarize(raw_preds)
    after = summarize(trimmed_preds)

    # ---- 이름 문맥 vs 사물 문맥 ----
    neg_rows = [r for r in v7_rows if not r["spans"]]
    obj_recall: dict[str, dict] = {}
    for r in neg_rows:
        preds = raw_preds[r["id"]]
        flagged = any(s.kind is Kind.PERSON for s in preds)
        d = obj_recall.setdefault(r["name"], {"total": 0, "flagged": 0})
        d["total"] += 1
        d["flagged"] += int(flagged)

    ctx_results = []
    ctx_recall_by_word: dict[str, dict] = defaultdict(lambda: {"성붙임(이름문맥)": None, "호격(이름문맥)": None})
    for r in ctx_rows:
        preds = predict(r["text"], ())
        g = r["spans"][0]
        bucket, best = classify_exact(g, preds)
        exact = bucket == "정확탐지"
        ctx_results.append(
            {
                "word": r["word"],
                "condition": r["condition"],
                "text": r["text"],
                "exact_detected": exact,
                "predicted": [r["text"][s.start : s.end] for s in preds if s.kind is Kind.PERSON],
            }
        )
        ctx_recall_by_word[r["word"]][r["condition"]] = exact

    name_context_table = []
    for w in ALL_NEG_WORDS:
        obj = obj_recall.get(w, {"total": 0, "flagged": 0})
        name_context_table.append(
            {
                "word": w,
                "real_registered_count": REAL_NAME_WORDS.get(w, 0),
                "object_context_false_positive_rate": round(obj["flagged"] / obj["total"], 4) if obj["total"] else None,
                "surname_context_detected": ctx_recall_by_word[w]["성붙임(이름문맥)"],
                "vocative_context_detected": ctx_recall_by_word[w]["호격(이름문맥)"],
            }
        )

    return {
        "fingerprint12": fp12,
        "before_trim": before,
        "after_trim": after,
        "name_context_table": name_context_table,
        "name_context_raw": ctx_results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("service_model_dir")
    ap.add_argument("r1_model_dir")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    v7_rows = load_v7()
    ctx_rows = name_context_rows()
    print(f"v7 {len(v7_rows)}개, 이름문맥 테스트 {len(ctx_rows)}개")

    out = {
        "service": run_model("service", args.service_model_dir, v7_rows, ctx_rows),
        "r1_s7": run_model("r1_s7", args.r1_model_dir, v7_rows, ctx_rows),
    }
    for key, res in out.items():
        b, a = res["before_trim"], res["after_trim"]
        print(f"\n== {key} ==")
        print(f"  trim 전: P={b['precision']} R={b['recall']} F1={b['f1']} buckets={b['buckets']} negFPR={b['neg_fpr']} 잔존율={b['residual_char_leak_rate']}")
        print(f"  trim 후: P={a['precision']} R={a['recall']} F1={a['f1']} buckets={a['buckets']} negFPR={a['neg_fpr']} 잔존율={a['residual_char_leak_rate']}")
        print(f"  경계불일치(trim 전) 조사별: {b['boundary_mismatch_by_particle']}")

    out_path = args.out or (ROOT / "app" / "shared" / "privacy" / "eval" / "style_v7_namefreq" / "followup_analysis.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
