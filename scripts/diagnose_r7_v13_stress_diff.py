"""R7(+B)가 v13의 어떤 완전노출 단어를 고쳤는지/못 고쳤는지 비교 — 읽기 전용."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from app.shared.config import settings  # noqa: E402
from app.shared.privacy import ner  # noqa: E402
from app.shared.privacy.types import Kind  # noqa: E402

V13 = ROOT / "app/shared/privacy/eval/style_v13_locked/v13_locked.jsonl"
rows = [json.loads(l) for l in V13.read_text(encoding="utf-8").splitlines() if l.strip()]


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


OUT = open(sys.argv[1], "w", encoding="utf-8")
def p(*a):
    print(*a, file=OUT)

results = {}
for key, model_dir, fp12 in [
    ("R6", r"C:\Users\<user>\venvs\ner-runs\export\r6-candidate-s42", "c59b174ba126"),
    ("R7", r"C:\Users\<user>\venvs\ner-runs\export\r7-candidate-s42", "098cb439c7e4"),
]:
    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = model_dir
    settings.privacy_ner_threads = 2
    status = ner.ner_status()
    assert status["fingerprint"][:12] == fp12
    full_ids = set()
    for r in rows:
        g = gold_span(r)
        if g is None:
            continue
        final = ner.detect_with_ner(r["text"])
        n = g[1] - g[0]
        leaked = n - len(covered(final, g))
        if leaked >= n:
            full_ids.add(r["id"])
    results[key] = full_ids
    p(f"{key}: 완전노출 {len(full_ids)}건")

only_r6 = results["R6"] - results["R7"]
only_r7 = results["R7"] - results["R6"]
both = results["R6"] & results["R7"]
p(f"\nR6만 완전노출(R7이 고침): {len(only_r6)}건")
by_id = {r["id"]: r for r in rows}
for rid in sorted(only_r6):
    r = by_id[rid]
    p(f"  {rid} word={r.get('word')} cat={r.get('category')} style={r.get('style')} text={r['text']!r}")
p(f"\nR7만 완전노출(R7이 새로 틀림): {len(only_r7)}건")
for rid in sorted(only_r7):
    r = by_id[rid]
    p(f"  {rid} word={r.get('word')} cat={r.get('category')} style={r.get('style')} text={r['text']!r}")
p(f"\n둘 다 완전노출(변화 없음): {len(both)}건")
for rid in sorted(both):
    r = by_id[rid]
    p(f"  {rid} word={r.get('word')} cat={r.get('category')}")
OUT.close()
