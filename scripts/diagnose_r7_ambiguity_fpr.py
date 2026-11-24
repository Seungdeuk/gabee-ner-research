"""R7의 중의성(48절 ambiguity_probe) 오탐 역행(1.11%->3.33%) 원인 진단 — 읽기 전용.
R6/R7 각각에서 어떤 비PERSON 문장이 틀렸는지 전부 찾아 비교한다."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from app.shared.config import settings  # noqa: E402
from app.shared.privacy import ner  # noqa: E402
from app.shared.privacy.types import Kind  # noqa: E402

OUT = open(sys.argv[1] if len(sys.argv) > 1 else HERE / "r7_amb_fpr_out.txt", "w", encoding="utf-8")
def p(*a):
    print(*a, file=OUT)

AMB = ROOT / "app/shared/privacy/eval/style_ambiguity_probe/ambiguity_probe.jsonl"
rows = [json.loads(l) for l in AMB.read_text(encoding="utf-8").splitlines() if l.strip()]
neg_rows = [r for r in rows if not any(sp.get("kind") in ("PERSON", "PERSON_ALIAS") for sp in r.get("spans", []))]
p(f"neg rows: {len(neg_rows)}")

for key, model_dir, fp12 in [
    ("R6", r"C:\Users\<user>\venvs\ner-runs\export\r6-candidate-s42", "c59b174ba126"),
    ("R7", r"C:\Users\<user>\venvs\ner-runs\export\r7-candidate-s42", "098cb439c7e4"),
]:
    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = model_dir
    settings.privacy_ner_threads = 2
    status = ner.ner_status()
    assert status["fingerprint"][:12] == fp12, status["fingerprint"][:12]
    p(f"\n=== {key} ===")
    for r in neg_rows:
        text = r["text"]
        final = ner.detect_with_ner(text)
        final_person = [(s.start, s.end, text[s.start:s.end]) for s in final if s.kind is Kind.PERSON]
        if final_person:
            raw = ner.spans_for(text) or []
            raw_person = [(s.label, s.start, s.end, round(s.score, 3), text[s.start:s.end]) for s in raw if s.label in ("PERSON", "PERSON_ALIAS")]
            p(f"  {r['id']} word={r['word']} group={r['group']} text={text!r} flagged={final_person} raw={raw_person}")
OUT.close()
