"""R11 독립 평가셋 봉인(사용자의 별도 봉인 승인 후 1회). 원문·정답은 출력하지 않는다. 지금은 실행하지 않는다(준비 단계).

순서: (1) 승인된 해시 파일(`--approved`)과 입력 파일 대조 → (2) strict 검증 재실행(오류 0 필요) → (3) 읽기 전용 복사 → (4) 봉인 매니페스트·MANIFEST.sha256 생성 → (5) 복사본 재확인.
봉인 폴더가 비어 있지 않으면 중단(덮어쓰기 금지). 봉인 폴더는 학습·선정 코드 경로 밖에 둔다.
사용: python seal_r11_evalset.py --incoming <폴더> --sealed <빈 폴더> --approved approved_hashes.json --prereg-commit <sha> --confirm
approved_hashes.json: {"r11_stress_independent.jsonl": "<sha256>", ...}  (사용자가 승인한 해시)
"""

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = "r11_stress_v1.jsonl"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def readonly(p):
    os.chmod(p, stat.S_IREAD)
    subprocess.run(["attrib", "+R", str(p)], capture_output=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--incoming", type=Path, required=True)
    ap.add_argument("--sealed", type=Path, required=True)
    ap.add_argument("--approved", type=Path, required=True)
    ap.add_argument("--prereg-commit", required=True)
    ap.add_argument("--reference-dir", action="append", default=[], type=Path)
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args(argv)
    if not a.confirm:
        raise SystemExit("--confirm 없이는 봉인하지 않는다")
    approved = json.loads(a.approved.read_text(encoding="utf-8"))
    if DATASET not in approved:
        raise SystemExit(f"승인 해시에 {DATASET} 가 없다")
    for n, h in approved.items():
        if not (a.incoming / n).exists() or sha(a.incoming / n) != h:
            raise SystemExit(f"승인된 해시와 불일치: {n}")
    a.sealed.mkdir(parents=True, exist_ok=True)
    if any(a.sealed.iterdir()):
        raise SystemExit("봉인 폴더가 비어 있지 않다(덮어쓰기 금지)")
    report = a.sealed.parent / "seal_validation_report.json"
    cmd = [sys.executable, "-X", "utf8", str(HERE / "validate_r11_evalset.py"), "--data", str(a.incoming / DATASET), "--set-kind", "stress", "--full-ids", "--strict", "--report-out", str(report)]
    for d in a.reference_dir:
        cmd += ["--reference-dir", str(d)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    rep = json.loads(report.read_text(encoding="utf-8"))
    if r.returncode != 0 or rep["errors"]:
        raise SystemExit(f"strict 검증 실패(봉인하지 않음): {list(rep['errors'])}")
    for n in approved:
        shutil.copyfile(a.incoming / n, a.sealed / n)
        assert sha(a.sealed / n) == approved[n], n
    manifest = {"seal_id": "R11-ops-independent-evalset", "status": "SEALED", "sealed_at_utc": datetime.now(timezone.utc).isoformat(), "spec": "NER_R11_EVALSET_SPEC.md r11-1.0",
                "preregistration": {"ops_criteria": "ops_decision_criteria.json", "git_commit": a.prereg_commit},
                "datasets": [{"name": DATASET, "rows": sum(1 for x in (a.sealed / DATASET).read_text(encoding="utf-8").splitlines() if x.strip()), "sha256": approved[DATASET], "strict_validation_errors": 0}],
                "human_review": "미완료 — 의미 품질은 자동 검증으로 대체할 수 없다(사양 §9 한계 기록 참조)",
                "usage_restrictions": ["학습·튜닝·후보 선정·개발 평가에 쓰지 않는다", "원본·라벨·id·순서를 수정하지 않는다", "S0 와 B-new seed 21 을 한 번씩만 평가(EVALUATED.lock)", "결과를 본 뒤 임계값·모델을 바꾸지 않는다"]}
    (a.sealed / "R11_OPS_SEAL_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    names = sorted(p.name for p in a.sealed.iterdir() if p.name != "MANIFEST.sha256")
    (a.sealed / "MANIFEST.sha256").write_text("".join(f"{sha(a.sealed / n)} *{n}\n" for n in names), encoding="utf-8")
    for p in a.sealed.iterdir():
        readonly(p)
    print(json.dumps({"sealed": True, "files": sorted(p.name for p in a.sealed.iterdir()), "dataset_sha256": approved[DATASET]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
