"""R11 B군 학습 직전 최종 점검(학습·추론 없음). 사전 등록 NER_R11_PREREGISTRATION.md r11-1.4 §4-4·§10, r11-2.0(`NER_R11_PREREGISTRATION_v2_0.md`).

모드
 - `--mode research`(기본, r11-2.0): 사람 의미 검수 없이 **연구용** 학습이 가능한지 확인한다. 자동 품질 검증 재실행(확정 구조 오류 0, 확정 표면 오류 = 알려진 Development 3행 이하)과 상태 기록을 확인한다.
   **이 모드의 통과는 사람 검수 통과가 아니며 정식 검증·운영 교체 자격을 뜻하지 않는다**(출력의 quality_status 에 항상 명시).
 - `--mode strict`: r11-1.4 의 사람 검수 요건(검수 집계 verdict==pass, 미판정 0)을 그대로 요구한다. 현재는 충족되지 않았다.
공통 점검: B군 9,915행·해시, A군(C10) 해시, 평가 코드·사전 등록 문서·기준선 해시 고정(`r11_code_pin.json`)과 태그, GPU·RAM·디스크, 실행기 순차·중단 설정.
첫 GPU 실행의 `best/`==`epoch_best/` 일치는 실행기 `cands` 단계가 자동 확인한다(불일치 시 중단).
사용: python r11_pretrain_check.py [--mode research|strict] [--write-pin]   (--write-pin 은 모든 점검 통과 후 최종 고정 때만)
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = Path(r"C:\Users\<user>\venvs\ner-runs\data_r11")
V = REPO / "docs/security/r11_validation"
PIN = V / "r11_code_pin.json"
PY = r"C:\Users\<user>\venvs\gabee-ner\Scripts\python.exe"
V2_DOC = "docs/security/NER_R11_PREREGISTRATION_v2_0.md"
PREREG_TAGS = {"research": "r11-prereg-2.0", "strict": "r11-prereg-1.4"}
PIN_FILES = ["apps/api/scripts/r11_eval_lib.py", "apps/api/scripts/evaluate_r11_dev.py", "apps/api/scripts/evaluate_r11_val.py", "apps/api/scripts/aggregate_r11_baselines.py",
             "apps/api/scripts/select_r11.py", "apps/api/scripts/g12_r11.py", "apps/api/scripts/g12_latency.py", "apps/api/scripts/r11_flagged_sensitivity.py", "ml/ner/train/g12_parity.py",
             "ml/ner/train/run_r11_B.py", "ml/ner/train/r11_derive.py", "ml/ner/train/train_ner.py", "ml/ner/train/build_r11_B.py", "ml/ner/train/r11_pretrain_check.py",
             "ml/ner/data/r11_auto_qa.py", "ml/ner/data/review_r11_data.py", "ml/ner/data/r11_review_ingest.py",
             "docs/security/NER_R11_PREREGISTRATION.md", V2_DOC, "docs/security/NER_R11_AUTOQA_TRANSITION_PROPOSAL.md",
             "docs/security/r11_validation/r11_reference_baselines.json", "docs/security/r11_validation/r11_recall_gate_fixed.json", "docs/security/r11_validation/r11_auto_qa_report.json",
             "docs/security/r11_validation/r11_suspects22.json", "docs/security/r11_validation/R11_DATA_MANIFEST.sha256", "docs/security/r11_validation/r11_baseline_raw_results.sha256"]
DATA_PIN = ["r11_Dprime_train.jsonl", "r11_val.jsonl", "r11_dev.jsonl", "r11_B_extra.jsonl", "baseline_train.jsonl", "baseline_val.jsonl", "synth_train.jsonl", "synth_val.jsonl", "klue_train.jsonl",
            "klue_val.jsonl", "r9_Ffull_train.jsonl", "r11_B_assembly_report.json"]
REVIEW = DATA / "r11_review_summary.json"
KNOWN_F_DEV = 3  # 알려진 확정 표면 오류(Development 조사 받침 3행): 이보다 늘어나면 실패


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def check_review_strict():
    if not REVIEW.exists():
        return {"ok": False, "reason": "사람 검수 집계 결과 없음(r11_review_summary.json): 검수 미완료는 통과가 아니다(r11-1.4 요건 미충족)"}
    d = json.loads(REVIEW.read_text(encoding="utf-8"))
    sets = d.get("sets", {})
    undec = sum(v.get("undecided", 0) for v in sets.values())
    return {"ok": d.get("verdict") == "pass" and undec == 0, "verdict": d.get("verdict"), "undecided_total": undec}


def check_quality_research():
    """연구용 모드의 품질 점검: 자동 검증 재실행. 사람 검수 결과를 만들거나 가정하지 않는다."""
    out_r, out_s = DATA / "r11_auto_qa_report_recheck.json", DATA / "r11_suspects22_recheck.json"
    subprocess.run([PY, str(REPO / "ml/ner/data/r11_auto_qa.py"), "--out-report", str(out_r), "--out-suspects", str(out_s)], check=True, capture_output=True,
                   env={**os.environ, "HF_HOME": r"C:\Users\<user>\venvs\hf-cache", "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"})
    rep = json.loads(out_r.read_text(encoding="utf-8"))
    pinned = json.loads((V / "r11_auto_qa_report.json").read_text(encoding="utf-8"))
    S = {k: sum(rep[k]["S_definitive_structural"].values()) for k in ("train", "val", "dev")}
    F = {k: sum(rep[k]["F_definitive_surface_label_neutral"].values()) for k in ("train", "val", "dev")}
    same_as_recorded = all(rep[k][g] == pinned[k][g] for k in ("train", "val", "dev") for g in ("S_definitive_structural", "F_definitive_surface_label_neutral", "M_semantic_review_warnings"))
    v2 = (REPO / V2_DOC).read_text(encoding="utf-8") if (REPO / V2_DOC).exists() else ""
    statements = {"human_review_519_incomplete": "사람 의미 검수: 519건 미완료" in v2, "no_formal_validation_or_replacement": "정식 검증 및 운영 교체 자격 없음" in v2, "r11_1_4_not_retroactively_met": "소급" in v2}
    status = {
        "자동 검증": "완료" if all(v == 0 for v in S.values()) else "확정 구조 오류 있음",
        "확정 구조 오류": S, "확정 표면 오류": F,
        "사람 의미 검수": "519건 미완료(미판정은 통과가 아님)",
        "해결되지 않은 의미 품질 위험": "외래어 의심 대조쌍 7, 별명 충돌 2행, 단서 약한 쌍 45, 기계 반복 12행, 전체 자연스러움·합성 분포 균질성(r11-2.0 문서 §3)",
        "연구용 학습": "가능(조건부)", "정식 검증 및 운영 교체 자격": "없음",
    }
    ok = all(v == 0 for v in S.values()) and F["train"] == 0 and F["val"] == 0 and F["dev"] <= KNOWN_F_DEV and same_as_recorded and all(statements.values())
    return {"ok": ok, "mode": "research(자동 검증 기반, 사람 검수 대체 아님)", "S_by_set": S, "F_by_set": F, "same_as_recorded_report": same_as_recorded, "statements_in_v2_doc": statements,
            "human_review_summary_present": REVIEW.exists(), "quality_status": status}


def check_data():
    rep = json.loads((DATA / "r11_B_assembly_report.json").read_text(encoding="utf-8"))
    man = {}
    for line in (V / "R11_DATA_MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, n = line.split("  ", 1)
            man[n] = h
    skip = lambda n: n.startswith("review/") or n.startswith("g12_test") or "recheck" in n or n in ("r11_review_auto_report.json", "r11_auto_qa_report.json", "r11_suspects22.json", "r11_flagged_sensitivity.json", "r11_auto_qa_ids.json", "r11_review_summary.json", "r11_pretrain_check_report.json", "r11_quality_status.json")  # noqa: E731
    mism = [n for n, h in man.items() if (DATA / n).exists() and not skip(n) and sha(DATA / n) != h]
    return {"ok": rep["rows"]["total"] == 9915 and rep["steps_per_epoch"] == 310 and sha(DATA / "r11_B_extra.jsonl") == rep["r11_B_extra_sha256"] and not mism
            and all(v["copy_identical"] for v in rep["r9f_files_copied"].values()) and rep["extra_sentences_equal_to_r11_val_dev"] == 0,
            "rows": rep["rows"], "steps_per_epoch": rep["steps_per_epoch"], "manifest_files_checked": len(man), "manifest_mismatch": mism}


def check_a_group():
    out = DATA / "a_group_integrity_recheck.json"
    subprocess.run([sys.executable, str(REPO / "ml/ner/analysis/r11_check_a_group.py"), "--manifest", str(REPO / "docs/security/r10_validation/r10_runs_manifest.json"), "--out", str(out)], check=True, capture_output=True)
    d = json.loads(out.read_text(encoding="utf-8"))
    return {"ok": bool(d["all_match"]), "n_runs": d["n_runs"], "note": "C10 재학습 금지(재추론만)"}


def check_pin(write, mode):
    tag_name = PREREG_TAGS[mode]
    cur = {f: sha(REPO / f) for f in PIN_FILES}
    cur_data = {f: sha(DATA / f) for f in DATA_PIN}
    raw = {p.name: sha(p) for p in sorted(DATA.glob("baselines_*.json"))}
    tag = git("rev-parse", "--verify", "-q", f"refs/tags/{tag_name}^{{commit}}")
    dirty = [f for f in PIN_FILES if git("status", "--porcelain", "--", f)]
    res = {"git_head": git("rev-parse", "--short", "HEAD"), f"prereg_tag_{tag_name}": tag[:8] or None, "uncommitted_pinned_files": dirty}
    if write:
        PIN.write_text(json.dumps({"mode": mode, "git_head": res["git_head"], "prereg_tag": tag_name, "prereg_tag_commit": tag[:8], "files": cur, "data_files": cur_data, "baseline_raw_results": raw,
                                   "quality_status": "자동 검증 기반 연구용(사람 의미 검수 519건 미완료, 정식 검증·운영 교체 자격 없음)" if mode == "research" else "strict"}, ensure_ascii=False, indent=1), encoding="utf-8")
        res["pin_written"] = str(PIN)
        res["ok"] = bool(tag) and not dirty
        return res
    if not PIN.exists():
        return {**res, "ok": False, "reason": "코드·데이터 고정 파일(r11_code_pin.json) 없음: 전체 점검 통과 후 --write-pin 으로 고정"}
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    diff = [f for f in PIN_FILES if pin["files"].get(f) != cur[f]]
    ddiff = [f for f in DATA_PIN if pin["data_files"].get(f) != cur_data[f]]
    rdiff = [n for n in set(raw) | set(pin["baseline_raw_results"]) if pin["baseline_raw_results"].get(n) != raw.get(n)]
    return {**res, "ok": bool(tag) and not diff and not ddiff and not rdiff and not dirty, "changed_files_since_pin": diff, "changed_data_since_pin": ddiff, "changed_baseline_raw_since_pin": rdiff}


def check_runner():
    out = subprocess.run([PY, str(HERE / "run_r11_B.py")], capture_output=True, text=True, encoding="utf-8", errors="replace")
    txt = out.stdout
    j = json.loads(txt[: txt.index("RUN ")]) if "RUN " in txt else {}
    seeds = [ln.split()[-1] for ln in txt.splitlines() if ln.startswith("RUN ")]
    src = (HERE / "run_r11_B.py").read_text(encoding="utf-8")
    return {"ok": bool(j.get("ok")) and seeds == [f"r11-B-s{s}" for s in (7, 13, 21, 42, 99)] and "드라이런" in txt and "C10" not in src.split("def train_cmd")[1].split("def gpu_info")[0],
            "failed_checks": j.get("failed_checks"), "seeds_in_order": seeds, "avail_ram_mb": j.get("avail_ram_mb"), "free_disk_gb": j.get("free_disk_gb"), "gpu": j.get("gpu"),
            "stop_on_failure_no_auto_retry": "sys.exit(p.returncode)" in src, "run_level_done_marker": "pipeline_done.json" in src}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["research", "strict"], default="research")
    ap.add_argument("--write-pin", action="store_true")
    a = ap.parse_args()
    q = check_review_strict() if a.mode == "strict" else check_quality_research()
    rep = {"mode": a.mode, "1_quality": q, "2_B_data_hashes": check_data(), "3_A_group_C10": check_a_group(), "5_6_runner_resources_sequence": check_runner()}
    pre_ok = all(v.get("ok") for v in rep.values() if isinstance(v, dict))
    if a.write_pin and not pre_ok:
        rep["4_code_and_prereg_pin"] = {"ok": False, "reason": "다른 점검이 통과하지 않아 --write-pin 을 실행하지 않았다(임의 수정·진행 금지, 보고 필요)"}
    else:
        rep["4_code_and_prereg_pin"] = check_pin(a.write_pin, a.mode)
    rep["ALL_OK"] = all(v.get("ok") for v in rep.values() if isinstance(v, dict))
    rep["interpretation"] = ("연구용 학습 준비 점검 통과. 이는 사람 검수 통과가 아니며 정식 검증·운영 교체 자격과 무관하다." if a.mode == "research" and rep["ALL_OK"] else
                             "점검 미통과 또는 strict 모드" if not rep["ALL_OK"] else "strict 점검 통과(사람 검수 통과)")
    (DATA / "r11_pretrain_check_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0 if rep["ALL_OK"] else 1


if __name__ == "__main__":
    sys.exit(main())
