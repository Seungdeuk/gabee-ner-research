"""R8 개발 평가 - 서비스 모델과 후보 시드를 같은 코드·설정으로 개발 평가셋에서 측정하고 사전 등록 규칙으로 판정한다.

사전 등록: docs/security/NER_R8_PREREGISTRATION.md (평가 전에 커밋). 읽기 전용: ner.py·모델·DB·설정 미수정(프로세스 안 설정만),
외부 호출 없음, 재학습 없음. 봉인 평가셋 new180 은 이 스크립트가 열지 않는다.

사용::

    python scripts/evaluate_r8_dev.py --model service <dir> 46210943c45f --model s7 <dir> ... --out result.json
    python scripts/evaluate_r8_dev.py --select result.json     # 사전 등록 게이트·선정 규칙 적용
"""

import argparse
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r8_eval_lib as lib  # noqa: E402
from r8_eval_lib import ner  # noqa: E402

ROOT = HERE.parent
EV = ROOT / "app" / "shared" / "privacy" / "eval"
NEMO = Path(r"C:\Users\<user>\Documents\gabee-ner-backup-20261118\scratch\nemotron")
MAIN5 = {
    "v7_namefreq": EV / "style_v7_namefreq" / "v7_namefreq.jsonl",
    "realistic_family": EV / "style_family_realistic" / "realistic_family.jsonl",
    "v14_realistic": EV / "style_v14_locked" / "v14_realistic.jsonl",
    "nemotron_full": NEMO / "nemotron_full_2500.jsonl",
    "nemotron_given": NEMO / "nemotron_given_2500.jsonl",
}
SAFETY_FP = {"ambiguity_probe": EV / "style_ambiguity_probe" / "ambiguity_probe.jsonl"}
REF = {
    "v12_locked": EV / "style_v12_locked" / "v12_locked.jsonl",
    "v13_locked": EV / "style_v13_locked" / "v13_locked.jsonl",
    "v8_independent": EV / "style_v8_independent" / "v8_independent.jsonl",
    "r7b_realistic": EV / "style_r7b_dev" / "r7b_realistic.jsonl",
}
ADJ = EV / "style_r7b_dev" / "r7b_adjacent.jsonl"
CAE = HERE / "check_adjacent_entity_merge.py"
FP_SETS = ("v7_namefreq", "realistic_family", "v14_realistic", "ambiguity_probe")


def load(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def cae_cases():
    """check_adjacent_entity_merge.py 의 CASES 를 import 없이 읽는다(실험 B 모듈 import 회피)."""
    tree = ast.parse(CAE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CASES":
            return ast.literal_eval(node.value)
    raise SystemExit("CASES 를 찾지 못함")


def overmerge_count():
    """한 PERSON 구간이 서로 다른 두 개체 이상을 삼킨 문장 수(이전 라운드와 같은 판정)."""
    items = []
    for r in load(ADJ):
        golds = lib.gold_person(r)
        if len(golds) >= 2:
            items.append((r["text"], [r["text"][a:b] for a, b in golds]))
    for _name, text, entities in cae_cases():
        items.append((text, entities))
    hits = []
    for text, ents in items:
        final = ner.detect_with_ner(text, ())
        over = any(s.kind.name == "PERSON" and sum(1 for e in ents if text[s.start:s.end].find(e) >= 0) >= 2 for s in final)
        if over:
            hits.append(text)
    return {"count": len(hits), "total_items": len(items), "texts": hits}


def run_model(label, model_dir, fp12, legacy_check):
    fp = lib.set_model(model_dir, fp12)
    out = {"label": label, "model_dir": str(model_dir), "fingerprint12": fp, "main5": {}, "safety": {}, "ref": {}}
    for name, p in MAIN5.items():
        rows = load(p)
        out["main5"][name] = lib.score_rows(rows)
        if legacy_check and name in ("v7_namefreq", "realistic_family", "v14_realistic", "nemotron_full", "nemotron_given"):
            out["main5"][name]["legacy_first_only"] = lib.score_rows(rows, legacy_first_only=True)
    for name, p in SAFETY_FP.items():
        out["safety"][name] = lib.score_rows(load(p))
    for name, p in REF.items():
        out["ref"][name] = lib.score_rows(load(p))
    out["overmerge"] = overmerge_count()
    # 요약(매크로)
    m5 = out["main5"]
    out["macro"] = {
        "total_exposure": lib.macro([m5[k]["total_exposure_rate"] for k in MAIN5]),
        "full_exposure": lib.macro([m5[k]["full_rate"] for k in MAIN5]),
        "partial_exposure": lib.macro([m5[k]["partial_rate"] for k in MAIN5]),
        "exact_f1": lib.macro([m5[k]["prf"]["f1"] for k in MAIN5]),
        "precision": lib.macro([m5[k]["prf"]["precision"] for k in MAIN5]),
        "recall": lib.macro([m5[k]["prf"]["recall"] for k in MAIN5]),
        "nonperson_fpr": lib.macro([({**m5, **out["safety"]}[k])["nonperson_fpr"] for k in FP_SETS]),
    }
    return out


def select(result):
    """사전 등록 §5: 게이트 G1~G4 → 개선 요건(-1.0%p) → 선정. 결과 JSON 의 service 와 시드들을 읽는다."""
    svc = result["service"]
    cands = {k: v for k, v in result.items() if k != "service" and isinstance(v, dict) and "main5" in v}
    verdict = {}
    for k, c in cands.items():
        g = {}
        fp_c = {s: ({**c["main5"], **c["safety"]}[s])["nonperson_fpr"] for s in FP_SETS}
        fp_s = {s: ({**svc["main5"], **svc["safety"]}[s])["nonperson_fpr"] for s in FP_SETS}
        g["G1_macro_fp_delta_pp"] = round(100 * (c["macro"]["nonperson_fpr"] - svc["macro"]["nonperson_fpr"]), 2)
        g["G1_worst_set_fp_delta_pp"] = round(100 * max(fp_c[s] - fp_s[s] for s in FP_SETS), 2)
        g["G1"] = g["G1_macro_fp_delta_pp"] <= 2.0 and g["G1_worst_set_fp_delta_pp"] <= 5.0
        g["G2_overmerge"] = [c["overmerge"]["count"], svc["overmerge"]["count"]]
        g["G2"] = c["overmerge"]["count"] <= svc["overmerge"]["count"]
        g["G3_macro_f1_delta_pp"] = round(100 * (c["macro"]["exact_f1"] - svc["macro"]["exact_f1"]), 2)
        g["G3"] = g["G3_macro_f1_delta_pp"] >= -1.0
        d4 = {s: round(100 * (c["main5"][s]["total_exposure_rate"] - svc["main5"][s]["total_exposure_rate"]), 2) for s in MAIN5}
        g["G4_set_exposure_delta_pp"] = d4
        g["G4"] = all(v <= 1.0 for v in d4.values())
        g["passes_gates"] = g["G1"] and g["G2"] and g["G3"] and g["G4"]
        g["macro_exposure_delta_pp"] = round(100 * (c["macro"]["total_exposure"] - svc["macro"]["total_exposure"]), 2)
        g["meets_improvement"] = g["macro_exposure_delta_pp"] <= -1.0
        g["eligible"] = g["passes_gates"] and g["meets_improvement"]
        verdict[k] = g
    elig = [k for k, g in verdict.items() if g["eligible"]]
    chosen = None
    if elig:
        def key(k):
            c = cands[k]["macro"]
            return (round(c["total_exposure"] * 1000) / 1000, c["nonperson_fpr"], -c["exact_f1"], int(k.lstrip("s")) if k.lstrip("s").isdigit() else 999)
        # 0.1%p 이내 동률 처리: 노출 매크로를 0.1%p 단위로 내림 비교
        best = min(cands[k]["macro"]["total_exposure"] for k in elig)
        tied = [k for k in elig if cands[k]["macro"]["total_exposure"] - best <= 0.001]
        chosen = sorted(tied, key=lambda k: (cands[k]["macro"]["nonperson_fpr"], -cands[k]["macro"]["exact_f1"], int(k.lstrip("s")) if k.lstrip("s").isdigit() else 999))[0]
    return {"verdict": verdict, "eligible": elig, "chosen": chosen}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", nargs="+", action="append", default=[], metavar=("LABEL", "DIR"), help="LABEL DIR [FP12]")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--legacy-check", action="store_true", help="이전 채점(첫 구간만) 수치도 함께 낸다(기준선 재현 확인)")
    ap.add_argument("--select", type=Path, help="결과 JSON 에 사전 등록 게이트·선정 규칙을 적용")
    a = ap.parse_args()
    if a.select:
        res = json.loads(a.select.read_text(encoding="utf-8"))
        print(json.dumps(select(res), ensure_ascii=False, indent=1))
        return
    res = {}
    for spec in a.model:
        label, d = spec[0], spec[1]
        fp = spec[2] if len(spec) > 2 else ""
        print(f"[{label}] {d} (expect fp {fp or '-'})", flush=True)
        res[label] = run_model(label, d, fp or None, a.legacy_check)
        m = res[label]["macro"]
        print(f"  macro 노출 {m['total_exposure']:.4f} F1 {m['exact_f1']:.4f} FP {m['nonperson_fpr']:.4f} 과잉병합 {res[label]['overmerge']['count']}", flush=True)
        if a.out:
            a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if "service" in res and len(res) > 1:
        print(json.dumps(select(res), ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
