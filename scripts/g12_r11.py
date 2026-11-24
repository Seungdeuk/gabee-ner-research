"""R11-G12 판정 오케스트레이터(후보 ONNX 의 PyTorch 일치와 CPU 지연·크기·RSS 를 S0 와 같은 조건에서 비교). 사전 등록 NER_R11_PREREGISTRATION.md §7-2 G12.

G12 조건(사전 등록): ONNX 토큰 일치 ≥ 99.9%, CPU 문장당 지연·모델 파일 크기·RSS 가 S0 대비 +30% 이하.
판정 통계(r11-1.4 로 명확화): 지연은 **문장당 p50(반복 사이 중앙값)의 후보/S0 비**, p95·평균 비는 보고만 한다. 한 항목이라도 어기면 G12 실패이며 후보 적격에서 제외된다(select_r11 이 G12 가 true 일 때만 eligible).
측정 순서는 드리프트를 줄이려고 S0, 후보, 후보, S0, S0, 후보(각 3회)로 교차한다(프로세스마다 모델 하나). 서비스 공용 NER 코드·운영 ONNX 는 수정하지 않는다(읽기 전용).
사용: python scripts/g12_r11.py --label B-new-s7 --onnx-dir DIR --torch-dir DIR --s0-dir DIR --out-dir DIR --summary g12_summary.json
"""

import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PY = r"C:\Users\<user>\venvs\gabee-ner\Scripts\python.exe"  # torch·onnxruntime(일치 비교)
API_PY = sys.executable  # 서비스 NER 경로(지연 측정)
LIMITS = {"token_agreement_pct_min": 99.9, "latency_ratio_max": 1.30, "size_ratio_max": 1.30, "rss_ratio_max": 1.30}
ORDER = ["s0", "cand", "cand", "s0", "s0", "cand"]


def judge(parity_pct, cand, s0):
    """cand, s0: {p50_ms, p95_ms, mean_ms, model_size_mb, rss_mb_after}(반복 측정의 중앙값). 반환: 항목별 값·통과와 G12 통과 여부."""
    r = {
        "token_agreement_pct": parity_pct,
        "latency_p50_ratio": cand["p50_ms"] / s0["p50_ms"],
        "latency_p95_ratio(report)": cand["p95_ms"] / s0["p95_ms"],
        "latency_mean_ratio(report)": cand["mean_ms"] / s0["mean_ms"],
        "size_ratio": cand["model_size_mb"] / s0["model_size_mb"],
        "rss_ratio": cand["rss_mb_after"] / s0["rss_mb_after"],
    }
    ok = {
        "parity": parity_pct >= LIMITS["token_agreement_pct_min"],
        "latency_p50": r["latency_p50_ratio"] <= LIMITS["latency_ratio_max"],
        "size": r["size_ratio"] <= LIMITS["size_ratio_max"],
        "rss": r["rss_ratio"] <= LIMITS["rss_ratio_max"],
    }
    return {"values": r, "pass_by_item": ok, "G12": all(ok.values())}


def med(runs, key):
    return st.median(x[key] for x in runs)


def aggregate_runs(runs):
    return {k: med(runs, k) for k in ("p50_ms", "p95_ms", "mean_ms", "model_size_mb", "rss_mb_after")}


def run_latency(onnx_dir, out):
    subprocess.run([API_PY, str(HERE / "g12_latency.py"), "--onnx-dir", str(onnx_dir), "--out", str(out)], check=True, cwd=str(REPO / "apps/api"),
                   env={**__import__("os").environ, "PYTHONPATH": ".", "PYTHONIOENCODING": "utf-8"}, capture_output=True)
    return json.loads(Path(out).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--torch-dir", type=Path, required=True)
    ap.add_argument("--s0-dir", type=Path, default=Path(r"C:\Users\<user>\venvs\ner-runs\service_v2b"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--skip-parity", action="store_true", help="시험용(실제 판정에서는 쓰지 않는다)")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    par = a.out_dir / f"{a.label}.parity.json"
    if not a.skip_parity:
        subprocess.run([PY, "-X", "utf8", str(REPO / "ml/ner/train/g12_parity.py"), "--model-dir", str(a.torch_dir), "--onnx-dir", str(a.onnx_dir), "--out", str(par)], check=True, capture_output=True)
        parity = json.loads(par.read_text(encoding="utf-8"))["token_agreement_pct"]
    else:
        parity = 100.0
    runs = {"s0": [], "cand": []}
    for i, who in enumerate(ORDER):
        d = a.s0_dir if who == "s0" else a.onnx_dir
        runs[who].append(run_latency(d, a.out_dir / f"{a.label}.lat{i}.{who}.json"))
    res = judge(parity, aggregate_runs(runs["cand"]), aggregate_runs(runs["s0"]))
    res.update({"label": a.label, "limits": LIMITS, "order": ORDER, "s0": aggregate_runs(runs["s0"]), "cand": aggregate_runs(runs["cand"]),
                "cpu_percent_before_each_run": [r["conditions"]["cpu_percent_before"] for w in runs.values() for r in w]})
    (a.out_dir / f"{a.label}.g12.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    if a.summary:
        cur = json.loads(a.summary.read_text(encoding="utf-8")) if a.summary.exists() else {}
        cur[a.label] = res["G12"]
        a.summary.write_text(json.dumps(cur, indent=1), encoding="utf-8")
    print(json.dumps({"label": a.label, "G12": res["G12"], "values": res["values"], "pass_by_item": res["pass_by_item"]}, indent=1))


if __name__ == "__main__":
    main()
