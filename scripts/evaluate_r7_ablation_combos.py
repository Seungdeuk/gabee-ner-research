"""R7 ablation 2단계 — 블록 조합(P+R, P+Amb, R+Amb) 상호작용 확인. evaluate_r7_ablation.py 재사용."""
import json
import sys
from pathlib import Path

import evaluate_r7_ablation as abl

HERE = Path(__file__).resolve().parent

COMBO_MODELS = [
    ("PR-s42", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-pr-s42", "2914b68ad315"),
    ("PR-s7", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-pr-s7", "83c94f3babda"),
    ("PAmb-s42", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-pamb-s42", "32b849f80307"),
    ("PAmb-s7", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-pamb-s7", "d38ba679c06d"),
    ("RAmb-s42", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-ramb-s42", "9cdf185ce03e"),
    ("RAmb-s7", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-ramb-s7", "d0e44331b7e1"),
]

bucket_rows = {b: abl.ev4.load_rows(paths) for b, paths in abl.ev4.BUCKETS.items()}
one_char_rows = abl.ev4.load_rows([abl.ev4.ONE_CHAR_BUCKET])
amb_rows = [json.loads(l) for l in abl.AMB.read_text(encoding="utf-8").splitlines() if l.strip()]

out = {}
for key, model_dir, fp12 in COMBO_MODELS:
    combo = abl.ev4.run_combo(key, model_dir, fp12, "normal", bucket_rows, one_char_rows)
    combo["ambiguity_probe_specific"] = abl.check_specific_ambiguity(amb_rows)
    combo["alias_alias_concat"] = abl.check_alias_alias(None)
    out[key] = combo
    c = combo["통합(매크로평균)"]
    amb_hits = [rid.split("-")[1] for rid, info in combo["ambiguity_probe_specific"].items() if info["flagged"]]
    over = combo["alias_alias_concat"]["over_merged"]
    print(f"{key}: 전체노출={c['exposure']['total_rate']} 오탐(macro)={c['nonperson_fpr_macro']} 48절={amb_hits} alias_alias={over}")

Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
