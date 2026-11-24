"""개인정보 탐지기 평가 스크립트(외부 호출 없음). 합성 데이터셋 app/shared/privacy/eval/pii_eval.jsonl 로 정밀도·재현율을 잰다.

실행(apps/api 에서): ``PYTHONPATH=. python scripts/eval_pii.py [--json | --md] [--write docs/security/PII_EVALUATION.md]``

- 일치 기준: 종류가 같고 겹침 비율(IoU = 겹친 길이 / 합친 길이) >= 0.5. 엄격 일치(시작·끝 같음)도 함께 보고한다.
- 음성 오탐률: 정답 구간이 없는 문장 중 탐지기가 무언가를 찾은 문장의 비율.
- --write: 대상 문서의 ``<!-- PII_EVAL_START -->`` ~ ``<!-- PII_EVAL_END -->`` 사이(탐지 성능 절)를 갱신한다. 없으면 파일 끝에 만든다.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.shared.privacy.detectors import detect  # noqa: E402

EVAL_DIR = ROOT / "app" / "shared" / "privacy" / "eval"
DATASET = EVAL_DIR / "pii_eval.jsonl"
HOLDOUT = EVAL_DIR / "pii_eval_holdout.jsonl"
SETS = {"main": ("학습형 세트", DATASET), "holdout": ("독립 보조 세트", HOLDOUT)}
START, END = "<!-- PII_EVAL_START -->", "<!-- PII_EVAL_END -->"
KINDS = ["PHONE", "EMAIL", "RRN", "CARD", "ACCOUNT", "SECRET", "ADDRESS", "PERSON"]


def load(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def match(gold: list[dict], pred: list, strict: bool) -> tuple[int, int, int, list, list]:
    """(TP, FP, FN, 놓친 정답, 잘못 찾은 예측). 한 예측은 한 정답과만 짝지어진다."""
    used: set[int] = set()
    tp = 0
    missed = []
    for g in gold:
        best, best_score = None, 0.0
        for i, p in enumerate(pred):
            if i in used or p.kind.value != g["kind"]:
                continue
            if strict:
                score = 1.0 if (p.start, p.end) == (g["start"], g["end"]) else 0.0
            else:
                score = iou((p.start, p.end), (g["start"], g["end"]))
            if score >= (1.0 if strict else 0.5) and score > best_score:
                best, best_score = i, score
        if best is None:
            missed.append(g)
        else:
            used.add(best)
            tp += 1
    extra = [p for i, p in enumerate(pred) if i not in used]
    return tp, len(extra), len(missed), missed, extra


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


STRESS_INPUTS = {
    "숫자 반복": lambda n: "1" * n,
    "숫자-하이픈 반복": lambda n: "1-" * (n // 2),
    "영문.점 반복": lambda n: "a." * (n // 2),
    "공백": lambda n: " " * n,
    "한글 반복": lambda n: "가" * n,
    "비번 반복": lambda n: "비번 " * (n // 3),
    "주소 반복": lambda n: "테헤란로 152 " * (n // 9),
    "전화 반복": lambda n: "010-1234-5678 " * (n // 14),
    "이름 반복": lambda n: "김철수씨 " * (n // 5),
}


def stress(chars: int = 100_000) -> dict:
    """반복 입력(최악에 가까운 조건)에서의 ms / 1,000자. 선형이면 평균과 비슷하다."""
    out = {}
    for label, gen in STRESS_INPUTS.items():
        text = gen(chars)
        t0 = time.perf_counter()
        detect(text, ("김민수", "이영희"))
        out[label] = round((time.perf_counter() - t0) * 1000 / (len(text) / 1000), 3)
    return out


def evaluate(rows: list[dict]) -> dict:
    counts = {m: {k: defaultdict(int) for k in KINDS} for m in ("relaxed", "strict")}
    examples: dict[str, list] = {"missed": [], "false_positive": [], "negative_fp": []}
    neg_total = neg_hit = 0
    chars = 0
    for row in rows:
        text, gold = row["text"], row["spans"]
        pred = detect(text, tuple(row.get("known_names", [])))
        chars += len(text)
        if not gold:
            neg_total += 1
            if pred:
                neg_hit += 1
                examples["negative_fp"].append(
                    {"text": text, "found": [(p.kind.value, text[p.start : p.end]) for p in pred]}
                )
        for mode in ("relaxed", "strict"):
            for kind in KINDS:
                g = [x for x in gold if x["kind"] == kind]
                p = [x for x in pred if x.kind.value == kind]
                tp, fp, fn, missed, extra = match(g, p, mode == "strict")
                c = counts[mode][kind]
                c["tp"] += tp
                c["fp"] += fp
                c["fn"] += fn
                c["gold"] += len(g)
                if mode == "relaxed":
                    for m in missed:
                        examples["missed"].append(
                            {"kind": kind, "text": text, "span": text[m["start"] : m["end"]]}
                        )
                    if gold:  # 음성 문장의 오탐은 negative_fp 로 따로 모은다
                        for e in extra:
                            examples["false_positive"].append(
                                {"kind": kind, "text": text, "found": text[e.start : e.end]}
                            )
    # 속도: 데이터셋 전체를 여러 번 돌려 ms / 1000자
    texts = [(r["text"], tuple(r.get("known_names", []))) for r in rows]
    rounds = 5
    t0 = time.perf_counter()
    for _ in range(rounds):
        for t, names in texts:
            detect(t, names)
    ms_per_1k = (time.perf_counter() - t0) * 1000 / (chars * rounds / 1000)
    result: dict = {
        "sentences": len(rows),
        "positive_sentences": len(rows) - neg_total,
        "negative_sentences": neg_total,
        "negative_false_positive_sentences": neg_hit,
        "negative_false_positive_rate": neg_hit / neg_total if neg_total else 0.0,
        "ms_per_1k_chars": round(ms_per_1k, 3),
        "per_kind": {},
        "overall": {},
        "examples": {k: v[:12] for k, v in examples.items()},
    }
    for mode in ("relaxed", "strict"):
        tot = defaultdict(int)
        for kind in KINDS:
            c = counts[mode][kind]
            p, r, f = prf(c["tp"], c["fp"], c["fn"])
            result["per_kind"].setdefault(kind, {})[mode] = {
                "gold": c["gold"], "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
                "miss_rate": round(c["fn"] / c["gold"], 3) if c["gold"] else 0.0,
            }  # fmt: skip
            for key in ("tp", "fp", "fn", "gold"):
                tot[key] += c[key]
        p, r, f = prf(tot["tp"], tot["fp"], tot["fn"])
        result["overall"][mode] = {
            "gold": tot["gold"], "tp": tot["tp"], "fp": tot["fp"], "fn": tot["fn"],
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
        }  # fmt: skip
    return result


def to_markdown(res: dict, title: str = "") -> str:
    lines = [
        f"**{title}**: {res['sentences']}문장 (양성 {res['positive_sentences']}, 음성 {res['negative_sentences']}), 합성 데이터.",
        "",
        "| 종류 | 정답 수 | P (겹침) | R (겹침) | F1 (겹침) | 누락률 | P (엄격) | R (엄격) | F1 (엄격) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind in KINDS:
        a, s = res["per_kind"][kind]["relaxed"], res["per_kind"][kind]["strict"]
        lines.append(
            f"| {kind} | {a['gold']} | {a['precision']:.3f} | {a['recall']:.3f} | {a['f1']:.3f} | {a['miss_rate']:.1%} | "
            f"{s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} |"
        )
    a, s = res["overall"]["relaxed"], res["overall"]["strict"]
    miss = a["fn"] / a["gold"] if a["gold"] else 0.0
    lines.append(
        f"| **전체(micro)** | {a['gold']} | {a['precision']:.3f} | {a['recall']:.3f} | {a['f1']:.3f} | {miss:.1%} | "
        f"{s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} |"
    )
    lines += [
        "",
        f"- 음성 문장 오탐률: {res['negative_false_positive_sentences']}/{res['negative_sentences']} = "
        f"{res['negative_false_positive_rate']:.1%} (개인정보가 없는 문장에서 무언가를 찾은 비율)",
        f"- 처리 속도: {res['ms_per_1k_chars']} ms / 1,000자 (이 컴퓨터, 단일 스레드, 데이터셋 5회 평균)",
        "",
        "**놓친 예(누락, 최대 12개)**",
        "",
    ]
    for e in res["examples"]["missed"]:
        lines.append(f'- {e["kind"]}: `{e["span"]}` <- "{e["text"]}"')
    lines += ["", "**양성 문장 안의 잘못 찾은 예(최대 12개)**", ""]
    for e in res["examples"]["false_positive"]:
        lines.append(f'- {e["kind"]}: `{e["found"]}` <- "{e["text"]}"')
    lines += ["", "**음성 문장 오탐 예(최대 12개)**", ""]
    for e in res["examples"]["negative_fp"]:
        found = ", ".join(f"{k}:`{v}`" for k, v in e["found"])
        lines.append(f'- {found} <- "{e["text"]}"')
    return "\n".join(lines)


def write_section(path: Path, body: str) -> None:
    block = f"{START}\n\n## 탐지 성능\n\n{body}\n\n{END}"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if START in existing and END in existing:
        head, rest = existing.split(START, 1)
        _, tail = rest.split(END, 1)
        out = head + block + tail
    else:
        out = (existing.rstrip() + "\n\n" if existing else "") + block + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out, encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--set",
        choices=["main", "holdout", "all"],
        default="all",
        help="main=학습형 세트(규칙 작성자가 만듦, 과적합 가능), holdout=독립 보조 세트, all=둘 다(기본)",
    )
    ap.add_argument("--dataset", type=Path, help="직접 지정한 jsonl (지정하면 --set 무시)")
    ap.add_argument("--json", action="store_true", help="JSON 만 출력")
    ap.add_argument("--md", action="store_true", help="마크다운 표만 출력")
    ap.add_argument(
        "--write",
        type=Path,
        help="이 문서의 '탐지 성능' 절을 갱신(예: docs/security/PII_EVALUATION.md)",
    )
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.dataset:
        chosen = {"custom": (args.dataset.name, args.dataset)}
    elif args.set == "all":
        chosen = SETS
    else:
        chosen = {args.set: SETS[args.set]}
    results = {key: evaluate(load(path)) for key, (_, path) in chosen.items()}
    worst = stress()
    parts = [to_markdown(res, chosen[key][0]) for key, res in results.items()]
    parts.append(
        "**최악 입력(반복 입력 10만 자) 처리 속도, ms / 1,000자**: "
        + ", ".join(f"{k} {v}" for k, v in worst.items())
    )
    md = "\n\n---\n\n".join(parts)
    if args.write:
        target = args.write if args.write.is_absolute() else ROOT.parents[1] / args.write
        write_section(target, md)
    if not args.md:
        print(
            json.dumps({"sets": results, "stress_ms_per_1k": worst}, ensure_ascii=False, indent=2)
        )
    if not args.json:
        print(md)


if __name__ == "__main__":
    main()
