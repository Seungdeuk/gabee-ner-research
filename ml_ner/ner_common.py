"""NER 학습 파이프라인 공통 코드(라벨, 경로, jsonl 입출력, span 지표). 서비스 코드(apps/)를 import 하지 않는다."""

import json
import os
import re
from pathlib import Path

NER_LABELS = ("PERSON", "PERSON_ALIAS", "ADDRESS", "LOCATION", "ORGANIZATION")
TAGS = ["O"] + [f"{p}-{lab}" for lab in NER_LABELS for p in ("B", "I")]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
ID2TAG = dict(enumerate(TAGS))

ML_DIR = Path(__file__).resolve().parent
REPO_ROOT = ML_DIR.parents[1]
BASELINE_DIR = REPO_ROOT / "apps" / "api" / "app" / "shared" / "privacy" / "eval" / "baseline_v2"
RUNS_ROOT = Path(os.environ.get("NER_RUNS_DIR", str(Path.home() / "venvs" / "ner-runs")))
DATA_OUT = RUNS_ROOT / "data"


def read_jsonl(path):
    """jsonl 파일을 읽어 dict 목록으로 돌려준다.

    @param path: 파일 경로
    @returns: list[dict]
    """
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    """dict 목록을 UTF-8 jsonl 로 쓴다(상위 폴더 자동 생성).

    @param path: 파일 경로
    @param rows: 쓸 dict 목록
    @returns: 쓴 행 수
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def normalize_text(text):
    """누출 비교용 정규화(공백·기호 제거, 소문자). Baseline 평가 스크립트와 같은 규칙.

    @param text: 문장
    @returns: 정규화 문자열
    """
    return re.sub(r"[\W_]+", "", text.lower())


def skeleton(text, spans):
    """정답 구간을 [라벨]로 바꾸고 숫자를 #로 바꾼 뒤 정규화한 문장 틀(누출 비교용).

    @param text: 문장
    @param spans: [{label|kind, start, end}]
    @returns: 정규화된 문장 틀
    """
    out, pos = "", 0
    for sp in sorted(spans, key=lambda s: s["start"]):
        lab = sp.get("label") or sp.get("kind")
        out += text[pos : sp["start"]] + f"[{lab}]"
        pos = sp["end"]
    out += text[pos:]
    return _skel_norm(out)


def _skel_norm(s):
    s = re.sub(r"\d+", "#", s)
    parts = re.split(r"(\[[A-Z_]+\])", s)
    return "".join(p if p.startswith("[") and p.endswith("]") else re.sub(r"[\W_]+", "", p.lower()) for p in parts)


def to_ner_spans(row):
    """Baseline 행(spans 의 kind)을 NER 라벨만 남긴 [{label,start,end}] 로 바꾼다.

    @param row: Baseline 또는 내부 형식 행
    @returns: NER 라벨 span 목록(start 순)
    """
    out = []
    for sp in row["spans"]:
        lab = sp.get("label") or sp.get("kind")
        if lab in NER_LABELS:
            out.append({"label": lab, "start": sp["start"], "end": sp["end"]})
    return sorted(out, key=lambda s: s["start"])


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f, "fnr": 1 - r if tp + fn else 0.0}


def _iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def span_metrics(golds, preds, mode):
    """문자 span 평가(Baseline eval_pii_v2 와 같은 규칙: 같은 라벨끼리 일대일 매칭).

    exact: 라벨·start·end 모두 같음. overlap: 라벨이 같고 한 글자라도 겹침(맞닿기만 하면 아님).

    @param golds: 문장별 정답 span 목록의 리스트 [[{label,start,end}]]
    @param preds: 문장별 예측 span 목록의 리스트(같은 길이)
    @param mode: "exact" 또는 "overlap"
    @returns: {"micro": prf, "per_label": {label: prf}}
    """
    cnt = {lab: [0, 0, 0] for lab in NER_LABELS}
    for g_list, p_list in zip(golds, preds, strict=False):
        for lab in NER_LABELS:
            gold = [s for s in g_list if s["label"] == lab]
            pred = [s for s in p_list if s["label"] == lab]
            used, tp = set(), 0
            for g in gold:
                best, best_score = None, 0.0
                for pi, p in enumerate(pred):
                    if pi in used:
                        continue
                    if mode == "exact":
                        if (p["start"], p["end"]) != (g["start"], g["end"]):
                            continue
                        score = 1.0
                    else:
                        if min(p["end"], g["end"]) <= max(p["start"], g["start"]):
                            continue
                        score = max(_iou((p["start"], p["end"]), (g["start"], g["end"])), 1e-9)
                    if score > best_score:
                        best, best_score = pi, score
                if best is not None:
                    used.add(best)
                    tp += 1
            cnt[lab][0] += tp
            cnt[lab][1] += len(pred) - len(used)
            cnt[lab][2] += len(gold) - tp
    per = {lab: _prf(*cnt[lab]) for lab in NER_LABELS}
    tp, fp, fn = (sum(cnt[lab][i] for lab in NER_LABELS) for i in range(3))
    return {"micro": _prf(tp, fp, fn), "per_label": per}
