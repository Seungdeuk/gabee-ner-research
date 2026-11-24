"""운영 교체 사전 등록 매니페스트(모델·코드·게이트·문서 해시) 작성·검증. 텍스트 파일은 줄바꿈을 LF 로 정규화해 해시한다(CRLF 체크아웃에서도 재현).

사용: python ops_prereg_check.py            # 검증(매니페스트와 현재 파일 대조, 모델 폴더 해시 포함)
      python ops_prereg_check.py --write    # 매니페스트 작성(사전 등록 확정 직전에만)
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OPS = REPO / "docs" / "security" / "r11_validation" / "ops_replacement"
MANIFEST = OPS / "ops_prereg_manifest.json"
FILES = [
    "docs/security/NER_R11_OPS_REPLACEMENT_PROTOCOL.md", "docs/security/NER_R11_OPS_DEPLOYMENT_CHECKLIST.md",
    "docs/security/NER_R11_EVALSET_SPEC.md", "docs/security/NER_R11_EVALSET_PRODUCER_INSTRUCTIONS.md", "docs/security/NER_R11_EVALSET_PRODUCER_ADDENDUM_OPS.md", "docs/security/NER_R11_EVALSET_SEAL_PROCEDURE.md",
    "docs/security/r11_validation/ops_replacement/ops_decision_criteria.json", "docs/security/r11_validation/ops_replacement/ops_model_lock.json",
    "docs/security/r11_validation/r11_reference_baselines.json", "docs/security/r11_validation/r11_recall_gate_fixed.json", "docs/security/r11_validation/r11_code_pin.json",
    "docs/security/r11_validation/r11_B_results/selection_final.json", "docs/security/r11_validation/r11_B_results/r11_B_summary.json",
    "apps/api/scripts/evaluate_ops_final.py", "apps/api/scripts/seal_r11_evalset.py", "apps/api/scripts/verify_r11_seal.py", "apps/api/scripts/validate_r11_evalset.py", "apps/api/scripts/tag_to_r11_jsonl.py",
    "apps/api/scripts/r11_eval_lib.py", "apps/api/scripts/r8_eval_lib.py", "apps/api/scripts/test_evaluate_ops_final.py", "apps/api/scripts/test_ops_criteria_frozen.py",
]


def sha_lf(p):
    return hashlib.sha256(Path(p).read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build():
    lock = json.loads((OPS / "ops_model_lock.json").read_text(encoding="utf-8"))
    models = {w: {f: {"sha256": h, "current": sha(Path(m["onnx_dir"]) / f)} for f, h in m["sha256"].items()} for w, m in lock["models"].items()}
    return {"version": "ops-1.0", "tag": "r11-ops-prereg-1.0", "files_sha256_lf": {f: sha_lf(REPO / f) for f in FILES}, "models": models}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    cur = build()
    bad_models = [f"{w}:{f}" for w, d in cur["models"].items() for f, v in d.items() if v["sha256"] != v["current"]]
    if a.write:
        if bad_models:
            raise SystemExit(f"모델 파일 해시가 모델 고정 파일과 다름: {bad_models}")
        for w in cur["models"].values():
            for v in w.values():
                v.pop("current")
        MANIFEST.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({"written": str(MANIFEST), "files": len(cur["files_sha256_lf"])}, ensure_ascii=False))
        return 0
    pin = json.loads(MANIFEST.read_text(encoding="utf-8"))
    changed = [f for f, h in pin["files_sha256_lf"].items() if cur["files_sha256_lf"].get(f) != h]
    missing = [f for f in FILES if f not in pin["files_sha256_lf"]]
    ok = not changed and not bad_models and not missing
    print(json.dumps({"ok": ok, "changed_files": changed, "unlisted_files": missing, "model_hash_mismatch": bad_models}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
