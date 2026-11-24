"""R7(+B)의 인접 엔티티 과잉병합 확인 — check_adjacent_entity_merge.py의 run()을 R7에 재사용."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_adjacent_entity_merge as cae  # noqa: E402

R6 = r"C:\Users\<user>\venvs\ner-runs\export\r6-candidate-s42"
R7 = r"C:\Users\<user>\venvs\ner-runs\export\r7-candidate-s42"

out = {
    "R6_normal": cae.run(R6, "c59b174ba126", False),
    "R6_B": cae.run(R6, "c59b174ba126", True),
    "R7_normal": cae.run(R7, "098cb439c7e4", False),
    "R7_B": cae.run(R7, "098cb439c7e4", True),
}
Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

with open(sys.argv[2], "w", encoding="utf-8") as f:
    for cfg, cases in out.items():
        print(f"== {cfg} ==", file=f)
        for c in cases:
            flag = " <<< 과잉병합 의심" if c["over_merged_guess"] else ""
            print(f"  {c['case']}: {c['final_spans']}{flag}", file=f)

# R6_normal 대비 변화(수정됨/새로 틀림)만 요약
with open(sys.argv[3], "w", encoding="utf-8") as f:
    base = {c["case"]: c["final_spans"] for c in out["R6_normal"]}
    for cfg in ["R6_B", "R7_normal", "R7_B"]:
        print(f"== {cfg} vs R6_normal ==", file=f)
        for c in out[cfg]:
            if c["final_spans"] != base[c["case"]]:
                print(f"  {c['case']}: R6_normal={base[c['case']]} -> {cfg}={c['final_spans']}", file=f)
