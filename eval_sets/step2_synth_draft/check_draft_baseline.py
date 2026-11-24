"""초안 56문장에 대한 현재 모델의 베이스라인(실패 확인용) — 읽기 전용, 학습에 안 씀."""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))

from app.shared.config import settings  # noqa: E402
from app.shared.privacy import ner  # noqa: E402
from app.shared.privacy.types import Kind  # noqa: E402

EXPECT_FP12 = {"service": "46210943c45f", "r1_s7": "c536eadeb5a1"}


def run(model_key: str, model_dir: str, rows: list[dict]) -> dict:
    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = model_dir
    settings.privacy_ner_threads = 2
    status = ner.ner_status()
    fp12 = status["fingerprint"][:12]
    if EXPECT_FP12.get(model_key) and fp12 != EXPECT_FP12[model_key]:
        raise SystemExit(f"[{model_key}] 지문 불일치: {fp12}")
    ner._cache.clear()

    person_ok = 0
    person_total = 0
    nonperson_fp = 0
    nonperson_total = 0
    fails = []
    for r in rows:
        final = ner.detect_with_ner(r["text"], ())
        person_final = [s for s in final if s.kind is Kind.PERSON]
        if r["context_type"] == "PERSON":
            g = r["spans"][0]
            person_total += 1
            exact = any(s.start == g["start"] and s.end == g["end"] for s in person_final)
            person_ok += int(exact)
            if not exact:
                fails.append({"id": r["id"], "text": r["text"], "final_person_spans": [(s.start, s.end, round(s.score, 3)) for s in person_final]})
        else:
            nonperson_total += 1
            nonperson_fp += int(bool(person_final))
    return {
        "fingerprint12": fp12,
        "person_recall": f"{person_ok}/{person_total}",
        "nonperson_fpr": f"{nonperson_fp}/{nonperson_total}",
        "person_fails": fails,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("service_model_dir")
    ap.add_argument("r1_model_dir")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (HERE / "synth_draft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {"service": run("service", args.service_model_dir, rows), "r1_s7": run("r1_s7", args.r1_model_dir, rows)}
    for k, r in out.items():
        print(f"== {k} == PERSON recall {r['person_recall']}, 비PERSON FPR {r['nonperson_fpr']}")
        for f in r["person_fails"]:
            print("  FAIL:", f["id"], f["text"], f["final_person_spans"])
    (HERE / "draft_baseline_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
