"""R9 v1.2 독립 평가셋 최종 봉인(사용자 승인 후 1회 실행). 원문·정답은 출력하지 않는다.

순서: (1) 승인된 해시와 입력 파일 대조 (2) strict 검증 재실행(오류 0 필요) (3) 읽기 전용 복사 (4) 최종 봉인 매니페스트와 MANIFEST.sha256 생성
(5) 복사본 해시·읽기 전용 확인. 봉인 폴더에 파일이 이미 있으면 중단한다(덮어쓰기 금지).
사용: python seal_r9_evalset.py --prereg-commit <sha> --confirm
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
REPO = HERE.parents[2]
ROOT = Path(r"C:\Users\<user>\Documents\gabee-ner-sealed-r9")
SRC, DST, REVIEW = ROOT / "incoming", ROOT / "sealed", ROOT / "review"
WORK = Path(r"C:\Users\<user>\venvs\ner-runs\r9\seal_run")
EXPECT = {
    "r9_representative_v1.2.jsonl": "110cceef97381a71125a96e1c08f4396878ce41d3676a57d938d464d2d313039",
    "r9_stress_v1.2.jsonl": "8f742bb7f321dd8c820f68ddcc49030c017c5af424a66967ae074b7e394ce66b",
    "r9_v1.2_NOTES.md": "c4910ab4588ef7755bbf771504e0ca59e74c16c4a0ee1c38ef7278172a29e3b3",
    "R9_v1.2_CLAUDE_HANDOFF.md": "5a3ee1c4a226efa53587ac7020961082afb3cde45d32f574eaf8d6c15bb62192",
}
REFERENCE_DIRS = [Path(r"C:\Users\<user>\venvs\ner-runs\data_r8"), REPO / "apps" / "api" / "app" / "shared" / "privacy" / "eval",
                  Path(r"C:\Users\<user>\Documents\gabee-ner-backup-20261118\scratch\nemotron"), Path(r"C:\Users\<user>\Documents\gabee-ner-sealed-new180\gabee_ner_new180_v1.jsonl")]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def readonly(p):
    os.chmod(p, stat.S_IREAD)
    subprocess.run(["attrib", "+R", str(p)], capture_output=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prereg-commit", required=True)
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    if not a.confirm:
        raise SystemExit("--confirm 없이는 봉인하지 않는다")
    # (1) 입력 해시
    for n, h in EXPECT.items():
        if sha(SRC / n) != h:
            raise SystemExit(f"승인된 해시와 불일치: {n}")
    if any(DST.iterdir()):
        raise SystemExit("봉인 폴더가 비어 있지 않다(덮어쓰기 금지)")
    WORK.mkdir(parents=True, exist_ok=True)
    # (2) strict 검증 재실행
    val = {}
    for kind in ("representative", "stress"):
        data = SRC / f"r9_{kind}_v1.2.jsonl"
        cmd = [sys.executable, "-X", "utf8", str(HERE / "validate_r9_evalset.py"), "--data", str(data), "--set-kind", kind, "--full-ids", "--strict",
               "--report-out", str(WORK / f"report_{kind}.json"), "--annotations-out", str(WORK / f"annot_{kind}.json")]
        for d in REFERENCE_DIRS:
            cmd += ["--reference-dir", str(d)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        rep = json.loads((WORK / f"report_{kind}.json").read_text(encoding="utf-8"))
        if r.returncode != 0 or rep["errors"]:
            raise SystemExit(f"strict 검증 실패: {kind} {list(rep['errors'])}")
        val[kind] = {"exit_code": r.returncode, "errors": 0, "warnings": {c: {"count": v["count"], "unique_ids": v["unique_ids"]} for c, v in rep["warnings"].items()},
                     "composition": {k: rep["info"][k] for k in ("sentences", "person_sentences", "other_only_sentences", "negative_sentences", "person_spans", "person_alias_spans", "person_len_dist",
                                                                  "multi_person_sentences", "distinct_person_values", "kin_adjacent_person_sentences", "negatives_with_kin_term", "person_2char_share_pct", "alias_share_pct") if k in rep["info"]},
                     "within_set": rep["info"]["within_set"], "reference_check": rep["info"]["reference"], "rows_excluded": rep["info"]["rows_excluded"]}
        if kind == "stress":
            val[kind]["composition"]["stress_primary_counts"] = rep["info"]["stress_primary_counts"]
    # (3) 읽기 전용 복사
    copied = []
    for n in EXPECT:
        d = DST / n
        shutil.copyfile(SRC / n, d)
        copied.append(n)
    led_src = REVIEW / "r9_v1.2_review_706_ID_ONLY.csv"
    shutil.copyfile(led_src, DST / "r9_v1.2_review_706_ID_ONLY.csv")
    shutil.copyfile(SRC / "R9_v1.1_CLAUDE_POLICY_CHANGE_REQUEST.md", DST / "R9_v1.1_CLAUDE_POLICY_CHANGE_REQUEST.md")
    # 복사본 해시
    for n, h in EXPECT.items():
        assert sha(DST / n) == h, n
    readiness = json.loads((Path(r"C:\Users\<user>\venvs\ner-runs\r9\validation_v1.2\seal_readiness.json")).read_text(encoding="utf-8"))
    led = readiness["ledger"]
    now = datetime.now(timezone.utc)
    prereg = REPO / "docs" / "security" / "NER_R9_PREREGISTRATION.md"
    manifest = {
        "seal_id": "R9-v1.2-final-independent-evalset", "status": "SEALED", "sealed_at_utc": now.isoformat(), "sealed_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "approved_by": "user", "approval_date": "2026-11-19", "sealed_by_tool": "seal_r9_evalset.py (Claude Code)",
        "policy_version": "r9-1.1", "spec_version": "r9-1.1", "validator_version": "r9-1.1", "dataset_version": "v1.2", "author_of_evalset": "ChatGPT (synthetic, source=synthetic_gpt_r9)",
        "datasets": [{"name": f"r9_{k}_v1.2.jsonl", "set_kind": k, "path": str(DST / f"r9_{k}_v1.2.jsonl"), "rows": val[k]["composition"]["sentences"], "bytes": (DST / f"r9_{k}_v1.2.jsonl").stat().st_size,
                      "sha256": sha(DST / f"r9_{k}_v1.2.jsonl"), "strict_validation": val[k]} for k in ("representative", "stress")],
        "related_documents": [{"name": n, "path": str(DST / n), "sha256": sha(DST / n)} for n in ("r9_v1.2_NOTES.md", "R9_v1.2_CLAUDE_HANDOFF.md", "R9_v1.1_CLAUDE_POLICY_CHANGE_REQUEST.md", "r9_v1.2_review_706_ID_ONLY.csv")]
        + [{"name": rel, "path": str(REPO / rel), "sha256": sha(REPO / rel)} for rel in ("docs/security/NER_R9_COMMON_SPEC.md", "docs/security/NER_R9_GPT_EVALSET_INSTRUCTIONS.md")],
        "preregistration": {"path": str(prereg), "sha256": sha(prereg), "git_commit": a.prereg_commit, "branch": "feature/seungdeuk-ner-improve2",
                            "note": "봉인 전에 커밋된 사전 등록(후보 선정 규칙, 교체 권고 조건, 지표 정의, D-1 승인, 검수 승인, C1 재사용 승인)"},
        "human_review": {"unique_sentences_reviewed": led["total_reviewed"], "verdicts": led["verdicts"], "by_set": {"representative": 477, "stress": 229}, "reviewer": "user", "review_date": "2026-11-19",
                         "unreviewed_sentences": led["unreviewed_rows"], "required_min_met": True, "high_risk_requirements_all_met": led["high_risk"]["all_met"],
                         "limitations": ["표본 검수이며 전수 검수가 아니다(미검수 문장은 정상으로 간주하지 않음)", "검수 표본의 무작위성은 확인되지 않았다(제안 표본과 대표 34%, 스트레스 25%만 겹침, 층별 커버리지로 판단)",
                                         "Wilson 95% 신뢰구간은 무작위 표본을 전제로 한 통계적 참고값이며 라벨 오류율의 보증이 아니다", "자동 검증은 의미적 라벨 정확성을 보장하지 않는다"],
                         "carry_over_check": led["v1_1_carry_over"], "changed_vs_v1_1_all_reviewed": not led["changed_vs_v1.1"]["changed_not_in_ledger"]},
        "deviations": [{"id": "D-1", "description": "대표 PERSON 계열 3글자 이름 비율 55.3%(권장 상한 55%), 경미한 구성 편차", "approved": True, "approval_date": "2026-11-19",
                        "effects": "게이트·교체 권고 조건·정답 라벨·지표 정의 변경 없음. 최종 평가에서 이름 글자 수(2, 3, 4글자 이상)별 층화 지표를 추가"}],
        "c1_reuse": {"approved": True, "approval_date": "2026-11-19", "basis": "5개 시드 정적 동등성 검증(설정 차이 0, 데이터 해시 일치, KLUE 표본·순서 동일, 학습 코드 변경 없음)",
                     "caveats": ["재학습하지 않아 같은 설정 재학습과 GPU 비결정성에 따른 소수 차이가 있을 수 있다", "조기 종료 에폭이 시드마다 다르다(종료 5~8, 최적 3~6): 실제 종료 에폭·총 스텝을 평가 보고서에 명시"]},
        "usage_restrictions": [
            "원본·라벨·id·순서를 수정하지 않는다(수정이 필요하면 새 버전과 변경 이력을 만들고 이 봉인 세트는 개발용으로 분류한다)",
            "이 평가셋은 학습, 학습 데이터 생성 참고, 하이퍼파라미터 튜닝, 후처리 조정, 개발 평가에 쓰지 않는다",
            "최종 후보 모델 확정(사전 등록 §6) 전에는 이 평가셋에 모델 예측을 돌리거나 점수·정답을 확인하지 않는다",
            "열람이 필요한 작업은 검증기·평가 스크립트가 집계만 출력하는 방식으로 한다. 최종 평가는 서비스 모델과 확정 후보 1개에 대해 1회만 한다",
            "결과를 본 뒤 후보 교체·재학습·다른 시드 선택·후처리 조정을 하지 않는다"],
        "leak_blocking": {"procedure": "신규 R9 학습 데이터 생성기는 봉인 평가셋을 프로그램 안에서만 읽어 문장·정규화 문장·문형·이름·구간 값이 겹치는 후보를 거르고 건수만 보고한다(내용 출력·참고 생성 금지). 예시 문장·이름은 차단 목록에 넣는다",
                          "reference_corpora_at_seal": val["representative"]["reference_check"]},
        "forbidden_actions_not_performed_in_seal_step": ["신규 학습 데이터 생성", "모델 학습", "개발 평가", "독립 최종 평가", "운영 모델 교체·배포", "푸시·PR·병합"],
    }
    mp = DST / "R9_v1.2_SEAL_MANIFEST.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    # MANIFEST.sha256 (매니페스트 자신 포함)
    names = sorted(p.name for p in DST.iterdir() if p.name != "MANIFEST.sha256")
    (DST / "MANIFEST.sha256").write_text("".join(f"{sha(DST / n)} *{n}\n" for n in names), encoding="utf-8")
    (DST / "SEAL_README.md").write_text(
        "# R9 v1.2 독립 평가셋 봉인 폴더\n\n상태: SEALED (2026-11-19, 사용자 승인). 이 폴더의 파일은 읽기 전용이며 수정·삭제·덮어쓰기 금지.\n\n"
        "- 데이터: `r9_representative_v1.2.jsonl`(1,600문장), `r9_stress_v1.2.jsonl`(800문장)\n- 해시·검증 결과·승인 기록: `R9_v1.2_SEAL_MANIFEST.json`, `MANIFEST.sha256`\n"
        "- 확인: `python apps/api/scripts/verify_r9_seal.py`\n\n사용 제한은 매니페스트의 `usage_restrictions` 와 `docs/security/NER_R9_PREREGISTRATION.md` §6, §7, §10 을 따른다. 최종 후보 확정 전에는 열람·예측 실행·튜닝에 쓰지 않는다.\n",
        encoding="utf-8")
    names = sorted(p.name for p in DST.iterdir() if p.name != "MANIFEST.sha256")
    (DST / "MANIFEST.sha256").write_text("".join(f"{sha(DST / n)} *{n}\n" for n in names), encoding="utf-8")
    for p in DST.iterdir():
        readonly(p)
    print(json.dumps({"sealed": True, "files": sorted(p.name for p in DST.iterdir()), "datasets": {d["name"]: d["sha256"] for d in manifest["datasets"]}, "manifest": str(mp), "manifest_sha256": sha(mp)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
