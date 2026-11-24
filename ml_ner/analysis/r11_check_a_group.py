"""R11 A군(= R10 C10, 재학습 금지) 정합성 확인: 모델·ONNX·학습 로그·개발 결과·학습 데이터 SHA256 이 R10 매니페스트와 같은지 확인한다.

학습·추론 없음, 파일 해시만 계산한다. 결과 JSON 에는 해시 일치 여부·스텝·종료 에폭만 담는다(문장·이름 없음).
사용: python r11_check_a_group.py --manifest <r10_runs_manifest.json> --runs-root <r10 root> --out <json>
"""

import argparse
import hashlib
import json
from pathlib import Path

DATA_DIRS = [Path(r"C:\Users\<user>\venvs\ner-runs\data_r10"), Path(r"C:\Users\<user>\venvs\ner-runs\data_r9")]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, default=Path(r"C:\Users\<user>\venvs\ner-runs\r10"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    runs = [r for r in json.loads(a.manifest.read_text(encoding="utf-8"))["runs"] if r["arm"] == "C10"]
    out, ok_all = [], True
    for r in runs:
        rid = r["run_id"]
        paths = {"best/model.safetensors": a.runs_root / "runs" / rid / "best" / "model.safetensors",
                 "export/model.onnx": a.runs_root / "export" / rid / "model.onnx",
                 "train_log.json": a.runs_root / "runs" / rid / "train_log.json",
                 "dev_result.json": a.runs_root / "dev" / f"{rid}.json"}
        files = {}
        for k, h in r["sha256"].items():
            p = paths.get(k)
            files[k] = "missing" if not p or not p.exists() else ("match" if sha(p) == h else "MISMATCH")
        data = {}
        for k, h in r["train_data_sha256"].items():
            found = next((d / f"{k}.jsonl" for d in DATA_DIRS if (d / f"{k}.jsonl").exists()), None)
            data[k] = "missing" if not found else ("match" if sha(found) == h else "MISMATCH")
        ok = all(v == "match" for v in [*files.values(), *data.values()])
        ok_all &= ok
        out.append({"run_id": rid, "seed": r["seed"], "steps_total": r["steps_total"], "stopped_epoch": r["stopped_epoch"],
                    "best_epoch": r["best_epoch"], "model_files": files, "train_data": data, "all_match": ok})
    res = {"arm": "C10", "n_runs": len(out), "all_match": ok_all, "runs": out}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("all_match", ok_all, "->", a.out)


if __name__ == "__main__":
    main()
