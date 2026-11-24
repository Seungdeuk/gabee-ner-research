"""R7 ablation — 어떤 보강 블록이 노출 개선·중의성 오탐 증가·무공백 인접 별명 과잉병합에 기여했는지
분리한다. 읽기 전용. R6/R7(각 대표 시드)을 기준으로 두고, P(무공백 인접 쌍만)/R(실사용 단일개체만)/
Amb(중의성 대비만)/Vol(크기만 맞춘 중립 데이터) 네 단일 블록을 각 3시드씩 평가한다. v14는 사용했던
개발셋을 "원인 분석·회귀 확인" 용도로만 재사용한다(최종 독립 평가 근거 아님). `ner.py`/`merge_ner`/
운영 모델/DB/설정 미수정, 재학습은 이미 끝난 것만 평가, 외부 LLM 미호출.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from app.shared.config import settings  # noqa: E402
from app.shared.privacy import ner  # noqa: E402
from app.shared.privacy.types import Kind  # noqa: E402

import evaluate_r7_4way as ev4  # noqa: E402
import check_adjacent_entity_merge as cae  # noqa: E402

V14 = ROOT / "app/shared/privacy/eval/style_v14_locked"
ev4.BUCKETS = {
    "실사용": [V14 / "v14_realistic.jsonl"],
    "중의성": [V14 / "v14_ambiguity.jsonl"],
    "스트레스": [V14 / "v14_stress.jsonl"],
}
ev4.ONE_CHAR_BUCKET = V14 / "v14_one_char.jsonl"

AMB = ROOT / "app/shared/privacy/eval/style_ambiguity_probe/ambiguity_probe.jsonl"
SPECIFIC_AMB_IDS = ["amb-보라-nonperson-3", "amb-미래-nonperson-2", "amb-지성-nonperson-3"]

MODELS = [
    ("R6(기준)", r"C:\Users\<user>\venvs\ner-runs\export\r6-candidate-s42", "c59b174ba126"),
    ("R7(전체, 기준)", r"C:\Users\<user>\venvs\ner-runs\export\r7-candidate-s42", "098cb439c7e4"),
    ("P-s42", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-p-s42", "8ac33d04eab6"),
    ("P-s7", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-p-s7", "505e86fd732c"),
    ("P-s21", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-p-s21", "fb722c081379"),
    ("R-s42", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-r-s42", "983f754c3ddf"),
    ("R-s7", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-r-s7", "c5388f8d02a0"),
    ("R-s21", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-r-s21", "b184729c3b2f"),
    ("Amb-s42", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-amb-s42", "8040eb3a0f39"),
    ("Amb-s7", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-amb-s7", "7fd2cc134888"),
    ("Amb-s21", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-amb-s21", "8d04ab32345b"),
    ("Vol-s42", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-vol-s42", "8c126979f3f4"),
    ("Vol-s7", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-vol-s7", "54938877e9ac"),
    ("Vol-s21", r"C:\Users\<user>\venvs\ner-runs\export\r7abl-vol-s21", "76f6a27ff363"),
]


def gold_span(row):
    for sp in row.get("spans", []):
        if sp.get("kind") in ("PERSON", "PERSON_ALIAS"):
            return sp["start"], sp["end"]
    return None


def check_specific_ambiguity(amb_rows):
    by_id = {r["id"]: r for r in amb_rows}
    out = {}
    for rid in SPECIFIC_AMB_IDS:
        r = by_id[rid]
        text = r["text"]
        final = ner.detect_with_ner(text)
        flagged = [(s.start, s.end, text[s.start:s.end]) for s in final if s.kind is Kind.PERSON]
        out[rid] = {"text": text, "flagged": bool(flagged), "spans": flagged}
    return out


def check_alias_alias(cae_rows):
    name, text, entities = next(c for c in cae.CASES if c[0] == "alias_alias_concat")
    final = ner.detect_with_ner(text)
    spans = [(s.kind.name, s.start, s.end, text[s.start:s.end]) for s in final]
    over_merged = any(
        s.kind is Kind.PERSON and sum(1 for e in entities if text[s.start:s.end].find(e) >= 0) >= 2
        for s in final
    )
    return {"text": text, "final_spans": spans, "over_merged": over_merged}


def main():
    bucket_rows = {b: ev4.load_rows(paths) for b, paths in ev4.BUCKETS.items()}
    one_char_rows = ev4.load_rows([ev4.ONE_CHAR_BUCKET])
    amb_rows = [json.loads(l) for l in AMB.read_text(encoding="utf-8").splitlines() if l.strip()]

    out = {}
    for key, model_dir, fp12 in MODELS:
        combo = ev4.run_combo(key, model_dir, fp12, "normal", bucket_rows, one_char_rows)
        combo["ambiguity_probe_specific"] = check_specific_ambiguity(amb_rows)
        combo["alias_alias_concat"] = check_alias_alias(None)
        out[key] = combo

        c = combo["통합(매크로평균)"]
        amb_hits = sum(1 for v in combo["ambiguity_probe_specific"].values() if v["flagged"])
        over = combo["alias_alias_concat"]["over_merged"]
        print(f"{key}: 통합전체노출={c['exposure']['total_rate']} 오탐(macro)={c['nonperson_fpr_macro']} "
              f"| 48절 3건중 오탐={amb_hits}/3 | alias_alias 과잉병합={over}")

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "r7_ablation_result.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
