"""개인정보 탐지 Baseline v2 평가 스크립트(외부 호출 없음, 합성 데이터만).

대상: ``app.shared.privacy.detectors.detect`` (규칙 + 가족 사전). gateway·정책 임계값을 거치지 않고 모든 Span 을 쓴다.
데이터: ``app/shared/privacy/eval/baseline_v2/{train,val,test}.jsonl`` (안전 픽스처, LABELING_GUIDE.md 기준).

실행(apps/api 에서)::

    PYTHONPATH=. python scripts/eval_pii_v2.py [--rounds 5] [--out-dir DIR]
    PYTHONPATH=. python scripts/eval_pii_v2.py --data-dir data/private_eval/baseline_v2   # 로컬 전용 원본 세트
    PYTHONPATH=. python scripts/eval_pii_v2.py --compare-data-dir data/private_eval/baseline_v2

세 가지 관점으로 낸다(모두 exact / overlap 두 모드).
- fine: 세분 라벨 단위(PHONE_PERSONAL, PHONE_PUBLIC, LOCATION, PERSON_ALIAS, ORGANIZATION …). 탐지기 예측은 평가용 대응
  (``assign_fine``)로 세분 라벨을 붙인다: PHONE 은 공개 대표번호 접두(1588·1577·1599·1644·080)면 PHONE_PUBLIC, 아니면
  PHONE_PERSONAL. 탐지기가 못 내는 라벨(PERSON_ALIAS, LOCATION, ORGANIZATION)은 '기존 탐지기에 없음'이라 FN 만 생긴다.
- mapped: 정답 라벨을 기존 탐지기 Kind 로 접어서(``LEGACY``) 비교. PHONE_*->PHONE, PERSON_ALIAS->PERSON.
  LOCATION·ORGANIZATION 은 대응하는 Kind 가 없어 별도 라벨로 두고(예측 불가) 따로 집계한다.
- policy: mask_default=true 인 정답을 놓친 정도(보호 재현율)와, 가릴 필요가 없는 것을 가린 예측 수(과마스킹).

exact: 라벨·start·end 모두 일치. overlap: 라벨이 같고 구간이 한 글자라도 겹침. 한 예측은 한 정답과만 짝지어진다.
FPR: 정답 구간이 없는 문장 중 탐지기가 무언가 찾은 문장의 비율. 처리시간: 워밍업 1회 후 ``--rounds`` 회 반복, 문장별 ms.
분할 누출 검사가 실패하면 측정하지 않고 종료한다. 기존 eval_pii.py 의 load/prf/iou 를 그대로 쓴다.
"""

import argparse
import csv
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_pii import KINDS, iou, load, prf  # noqa: E402

from app.shared.privacy.detectors import detect  # noqa: E402

V2_DIR = ROOT / "app" / "shared" / "privacy" / "eval" / "baseline_v2"
RESULTS_DIR = ROOT / "app" / "shared" / "privacy" / "eval" / "results"
SPLITS = ("train", "val", "test")
MODES = ("exact", "overlap")
FINE = [
    "PHONE_PERSONAL",
    "PHONE_PUBLIC",
    "EMAIL",
    "RRN",
    "CARD",
    "ACCOUNT",
    "SECRET",
    "ADDRESS",
    "LOCATION",
    "PERSON",
    "PERSON_ALIAS",
    "ORGANIZATION",
]
# 세분 라벨 -> 기존 탐지기 Kind (None = 대응 Kind 없음, 평가 전용)
LEGACY: dict[str, str | None] = {
    "PHONE_PERSONAL": "PHONE",
    "PHONE_PUBLIC": "PHONE",
    "EMAIL": "EMAIL",
    "RRN": "RRN",
    "CARD": "CARD",
    "ACCOUNT": "ACCOUNT",
    "SECRET": "SECRET",
    "ADDRESS": "ADDRESS",
    "LOCATION": None,
    "PERSON": "PERSON",
    "PERSON_ALIAS": "PERSON",
    "ORGANIZATION": None,
}
NO_DETECTOR = ("LOCATION", "ORGANIZATION")  # mapped 관점에서 따로 두는 평가 전용 라벨
MAPPED_KINDS = [*KINDS, *NO_DETECTOR]
UNSUPPORTED_FINE = (
    "PERSON_ALIAS",
    "LOCATION",
    "ORGANIZATION",
)  # 기존 탐지기가 세분 라벨로 낼 수 없는 것
VIEW_KINDS = {"fine": FINE, "mapped": MAPPED_KINDS}
PUBLIC_PHONE_PREFIX = re.compile(r"^\D*(1588|1577|1599|1644|080)")


# ---------------------------------------------------------------------------
# 데이터 검증·누출 검사
# ---------------------------------------------------------------------------
def load_splits(directory: Path = V2_DIR) -> dict[str, list[dict]]:
    return {s: load(directory / f"{s}.jsonl") for s in SPLITS}


def normalize_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def skeleton(row: dict) -> str:
    """정답 구간을 [종류]로 바꾸고 나머지 숫자를 #로 바꾼 문장 틀(같은 틀이 분할에 걸쳐 있는지 보는 용도)."""
    text, out, pos = row["text"], "", 0
    for sp in sorted(row["spans"], key=lambda s: s["start"]):
        out += text[pos : sp["start"]] + f"[{sp['kind']}]"
        pos = sp["end"]
    return re.sub(r"\d+", "#", out + text[pos:])


def validate_dataset(rows: list[dict]) -> list[str]:
    """형식 오류 목록(비어 있으면 정상): 필드, 종류, 정책 필드 타입, 인덱스 범위, 겹침, 빈 구간."""
    problems = []
    need = {"id", "text", "spans", "known_names", "tags", "template_id", "split"}
    ids = set()
    for r in rows:
        rid = r.get("id", "?")
        if need - set(r):
            problems.append(f"{rid}: 필드 누락 {sorted(need - set(r))}")
            continue
        if rid in ids:
            problems.append(f"{rid}: id 중복")
        ids.add(rid)
        prev_end = -1
        for sp in sorted(r["spans"], key=lambda s: s["start"]):
            if sp["kind"] not in FINE:
                problems.append(f"{rid}: 알 수 없는 종류 {sp['kind']}")
            for field in ("mask_default", "context_sensitive"):
                if not isinstance(sp.get(field), bool):
                    problems.append(f"{rid}: {field} 필드가 bool 이 아님")
            if not (0 <= sp["start"] < sp["end"] <= len(r["text"])):
                problems.append(f"{rid}: 인덱스 범위 오류 {sp}")
                continue
            if sp["start"] < prev_end:
                problems.append(f"{rid}: 구간이 겹침 {sp}")
            prev_end = sp["end"]
            frag = r["text"][sp["start"] : sp["end"]]
            if frag != frag.strip():
                problems.append(f"{rid}: 구간 앞뒤에 공백 {frag!r}")
    return problems


_KO_DIGITS = {c: str(i) for i, c in enumerate("공일이삼사오육칠팔구")}
_OCR_DIGITS = {"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8"}
_SAFE_EMAIL = re.compile(r"@example\.(com|org|net)$")
_SAFE_PHONE = re.compile(r"^(?:0100000\d{4}|82100000\d{4}|0(?:2|31|32|51)000\d{4})$")


def span_digits(value: str) -> str:
    """한글 숫자·OCR 문자를 숫자로 되돌리고 숫자만 남긴다(안전성 검사용)."""
    return re.sub(r"\D", "", "".join(_KO_DIGITS.get(c, _OCR_DIGITS.get(c, c)) for c in value))


def _luhn_valid(d: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(d)):
        x = int(ch) * (2 if i % 2 else 1)
        total += x - 9 if x > 9 else x
    return total % 10 == 0


def check_fixture_safety(rows: list[dict], public_pans: set[str]) -> list[str]:
    """공개 저장소용 픽스처에 실존할 수 있는 식별자가 없는지 검사한다(비어 있으면 통과).

    - RRN: 13자리이면 성별 자리 3/4 이고 출생연도 2030 이후(아직 태어나지 않은 날짜)여야 한다.
    - CARD: 공개 시험 PAN 이거나 Luhn 이 실패하는 값(Luhn 유효한 15자리 34/37 시작도 거부)이어야 한다.
    - PHONE_PERSONAL: 010-0000-NNNN(+82 포함) 또는 지역번호-000-NNNN 꼴이어야 한다.
    - EMAIL: 예약 도메인 example.com/org/net 만.
    ACCOUNT 는 임의 숫자라 검사할 수 없다(잔여 위험)."""
    problems = []
    for r in rows:
        for sp in r["spans"]:
            v = r["text"][sp["start"] : sp["end"]]
            k, d = sp["kind"], span_digits(v)
            if k == "RRN" and len(d) == 13 and not (d[6] in "34" and int(d[:2]) >= 30):
                problems.append(f"{r['id']}: RRN 이 미래 출생 형식이 아님")
            elif k == "RRN" and len(d) != 13:
                problems.append(f"{r['id']}: RRN 이 13자리가 아님")
            elif k == "CARD":
                if d in public_pans:
                    continue
                if _luhn_valid(d) and (len(d) == 16 or d[:2] in ("34", "37")):
                    problems.append(f"{r['id']}: CARD 가 공개 시험 PAN 도 아니고 Luhn 유효")
            elif k == "PHONE_PERSONAL" and not _SAFE_PHONE.match(d):
                problems.append(f"{r['id']}: PHONE_PERSONAL 이 안전 대역이 아님")
            elif k == "EMAIL" and not _SAFE_EMAIL.search(v):
                problems.append(f"{r['id']}: EMAIL 이 예약 도메인이 아님")
    return problems


def load_public_pans() -> set[str]:
    """빌더(build_baseline_v2.py)에 적힌 공개 시험 PAN 목록."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_build_v2", V2_DIR / "build_baseline_v2.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_build_v2"] = mod
    spec.loader.exec_module(mod)
    return set(mod.PUBLIC_TEST_PANS)


def check_leakage(by_split: dict[str, list[dict]]) -> list[str]:
    """분할 누출 목록(비어 있으면 통과): 같은 template_id 가 두 분할, 정규화한 본문 중복, 문장 틀 공유."""
    problems = []
    tmpl: dict[str, set[str]] = defaultdict(set)
    text_seen: dict[str, tuple[str, str]] = {}
    skel: dict[str, set[str]] = defaultdict(set)
    for split, rows in by_split.items():
        for r in rows:
            tmpl[r["template_id"]].add(split)
            if r["split"] != split:
                problems.append(f"{r['id']}: split 필드 {r['split']} 가 파일 {split} 과 다름")
            key = normalize_text(r["text"])
            if key in text_seen:
                problems.append(f"본문 중복: {r['id']} == {text_seen[key][0]}")
            text_seen[key] = (r["id"], split)
            skel[skeleton(r)].add(split)
    problems += [f"template_id {t} 가 {sorted(s)} 에 걸침" for t, s in tmpl.items() if len(s) > 1]
    problems += [f"문장 틀이 {sorted(s)} 에 걸침: {k[:40]}" for k, s in skel.items() if len(s) > 1]
    return problems


# ---------------------------------------------------------------------------
# 매칭·지표
# ---------------------------------------------------------------------------
def match_kind(
    gold: list[dict], pred: list[dict], mode: str
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """같은 종류의 정답·예측을 일대일로 짝짓는다. (짝 목록 (g,p), 못 찾은 정답 번호, 남은 예측 번호).

    exact 는 시작·끝이 같아야 하고 overlap 은 한 글자라도 겹치면 된다(맞닿기만 하면 겹침 아님)."""
    used: set[int] = set()
    pairs, missed = [], []
    for gi, g in enumerate(gold):
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
                score = max(iou((p["start"], p["end"]), (g["start"], g["end"])), 1e-9)
            if score > best_score:
                best, best_score = pi, score
        if best is None:
            missed.append(gi)
        else:
            used.add(best)
            pairs.append((gi, best))
    return pairs, missed, [i for i in range(len(pred)) if i not in used]


def new_counts() -> dict[str, int]:
    return {"gold": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}


def assign_fine(kind: str, value: str) -> str:
    """탐지기 예측(기존 Kind)에 평가용 세분 라벨을 붙인다. 탐지기는 PHONE 한 종류뿐이라 접두로 공개번호를 구분한다."""
    if kind == "PHONE":
        return "PHONE_PUBLIC" if PUBLIC_PHONE_PREFIX.match(value) else "PHONE_PERSONAL"
    return kind


def _views(row: dict, pred_spans: list) -> dict[str, tuple[list[dict], list[dict]]]:
    text = row["text"]
    pred_legacy = [{"kind": p.kind.value, "start": p.start, "end": p.end} for p in pred_spans]
    pred_fine = [
        {**p, "kind": assign_fine(p["kind"], text[p["start"] : p["end"]])} for p in pred_legacy
    ]
    gold_fine = row["spans"]
    gold_mapped = [{**g, "kind": LEGACY[g["kind"]] or g["kind"]} for g in gold_fine]
    return {"fine": (gold_fine, pred_fine), "mapped": (gold_mapped, pred_legacy)}


def _err(row, etype, kind, gold, pred, note, view):
    text = row["text"]
    return {
        "id": row["id"],
        "split": row["split"],
        "view": view,
        "error_type": etype,
        "kind": kind,
        "gold_text": text[gold["start"] : gold["end"]] if gold else "",
        "pred_kind": pred["kind"] if pred else "",
        "pred_text": text[pred["start"] : pred["end"]] if pred else "",
        "tags": "|".join(row["tags"]),
        "template_id": row["template_id"],
        "note": note,
    }


def _same_kind_overlap(g: dict, p: dict) -> bool:
    return g["kind"] == p["kind"] and min(g["end"], p["end"]) > max(g["start"], p["start"])


def score_row(row: dict, pred_spans: list) -> dict:
    """한 문장의 지표 재료. counts[view][mode][kind], 정책 지표, 오류 목록(mapped 관점, overlap 기준)."""
    views = _views(row, pred_spans)
    counts = {v: {m: {k: new_counts() for k in VIEW_KINDS[v]} for m in MODES} for v in VIEW_KINDS}
    errors = []
    for view, (gold_all, pred_all) in views.items():
        for mode in MODES:
            for kind in VIEW_KINDS[view]:
                g = [x for x in gold_all if x["kind"] == kind]
                p = [x for x in pred_all if x["kind"] == kind]
                pairs, missed, extra = match_kind(g, p, mode)
                c = counts[view][mode][kind]
                c["gold"] += len(g)
                c["pred"] += len(p)
                c["tp"] += len(pairs)
                c["fn"] += len(missed)
                c["fp"] += len(extra)
                if view != "mapped" or mode != "overlap":
                    continue
                for gi in missed:
                    other = [
                        x
                        for x in pred_all
                        if x["kind"] != kind
                        and min(x["end"], g[gi]["end"]) > max(x["start"], g[gi]["start"])
                    ]
                    note = "다른 종류로 탐지" if other else ""
                    errors.append(
                        _err(row, "FN", kind, g[gi], other[0] if other else None, note, view)
                    )
                for pi in extra:
                    errors.append(_err(row, "FP", kind, None, p[pi], "", view))
                for gi, pi in pairs:
                    if (g[gi]["start"], g[gi]["end"]) != (p[pi]["start"], p[pi]["end"]):
                        errors.append(
                            _err(row, "BOUNDARY", kind, g[gi], p[pi], "겹치지만 경계가 다름", view)
                        )
    # 정책 관점: mask_default=true 정답 중 (같은 기존 Kind 의 예측과 겹친) 것, 가릴 필요 없는 곳을 가린 예측
    gold_m, pred_m = views["mapped"]
    protect = [g for g in gold_m if g["mask_default"]]
    hit = sum(1 for g in protect if any(_same_kind_overlap(g, p) for p in pred_m))
    over = sum(1 for p in pred_m if not any(_same_kind_overlap(g, p) for g in protect))
    policy = {
        "protect_gold": len(protect),
        "protect_hit": hit,
        "pred": len(pred_m),
        "overmask": over,
    }
    return {
        "counts": counts,
        "policy": policy,
        "errors": errors,
        "flagged": bool(pred_m),
        "n_gold": len(row["spans"]),
    }


def aggregate(records: list[dict], view: str, mode: str, kinds: list[str] | None = None) -> dict:
    """records 의 합계에서 P/R/F1/FNR. kinds 를 주지 않으면 그 관점의 모든 라벨."""
    tot = new_counts()
    for rec in records:
        for kind in kinds or VIEW_KINDS[view]:
            for key, val in rec["counts"][view][mode][kind].items():
                tot[key] += val
    p, r, f = prf(tot["tp"], tot["fp"], tot["fn"])
    fnr = tot["fn"] / tot["gold"] if tot["gold"] else 0.0
    return {**tot, "precision": p, "recall": r, "f1": f, "fnr": fnr}


def aggregate_policy(records: list[dict]) -> dict:
    t: dict[str, int] = defaultdict(int)
    for r in records:
        for k, v in r["policy"].items():
            t[k] += v
    return {
        **t,
        "protect_recall": t["protect_hit"] / t["protect_gold"] if t["protect_gold"] else 0.0,
        "overmask_rate": t["overmask"] / t["pred"] if t["pred"] else 0.0,
    }


def fpr_of(records: list[dict]) -> dict:
    neg = [r for r in records if r["n_gold"] == 0]
    hit = sum(1 for r in neg if r["flagged"])
    return {"neg_sentences": len(neg), "neg_flagged": hit, "fpr": hit / len(neg) if neg else 0.0}


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    k = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[k]


def timing(rows: list[dict], rounds: int) -> dict:
    """워밍업 1회 후 rounds 번 반복해 문장마다 ms 를 잰다."""
    items = [(r["text"], tuple(r["known_names"])) for r in rows]
    for t, n in items:
        detect(t, n)
    samples: list[float] = []
    for _ in range(rounds):
        for t, n in items:
            t0 = time.perf_counter_ns()
            detect(t, n)
            samples.append((time.perf_counter_ns() - t0) / 1e6)
    return {
        "samples": len(samples),
        "mean_ms": statistics.fmean(samples),
        "p95_ms": percentile(samples, 0.95),
        "max_ms": max(samples),
    }


# ---------------------------------------------------------------------------
# 실행·출력
# ---------------------------------------------------------------------------
def evaluate(by_split: dict[str, list[dict]]) -> list[dict]:
    records = []
    for split, rows in by_split.items():
        for row in rows:
            res = score_row(row, detect(row["text"], tuple(row["known_names"])))
            res.update(id=row["id"], split=split, tags=row["tags"])
            records.append(res)
    return records


CSV_COLS = [
    "view", "scope", "group", "mode", "sentences", "gold", "pred", "tp", "fp", "fn",
    "precision", "recall", "f1", "fnr", "neg_sentences", "neg_flagged", "fpr",
    "protect_recall", "overmask_rate",
]  # fmt: skip


def metric_rows(records: list[dict]) -> list[dict]:
    """CSV 한 줄씩. view: fine/mapped/policy. scope: overall / split / kind / split_kind / tag / fpr."""
    out = []
    blank = {k: "" for k in CSV_COLS}

    def add(view, scope, group, recs, kinds=None):
        for mode in MODES:
            m = aggregate(recs, view, mode, kinds)
            out.append(
                {**blank, "view": view, "scope": scope, "group": group, "mode": mode,
                 "sentences": len(recs), **m}
            )  # fmt: skip

    tags = sorted({t for r in records for t in r["tags"]})
    by_split = {s: [r for r in records if r["split"] == s] for s in SPLITS}
    for view in VIEW_KINDS:
        supported = [k for k in FINE if k not in UNSUPPORTED_FINE] if view == "fine" else KINDS
        add(view, "overall", "all", records)
        add(view, "overall", "detector_supported_only", records, supported)
        for s, sub in by_split.items():
            add(view, "split", s, sub)
            add(view, "split", f"{s}_detector_supported_only", sub, supported)
            for k in VIEW_KINDS[view]:
                add(view, "split_kind", f"{s}/{k}", sub, [k])
        for k in VIEW_KINDS[view]:
            add(view, "kind", k, records, [k])
        for t in tags:
            add(view, "tag", t, [r for r in records if t in r["tags"]])
    groups = [("all", records), *by_split.items()]
    groups += [(t, [r for r in records if t in r["tags"]]) for t in tags if t.startswith("neg_")]
    for g, recs in groups:
        f = fpr_of(recs)
        if f["neg_sentences"]:
            out.append(
                {**blank, "view": "any", "scope": "fpr", "group": g, "mode": "any",
                 "sentences": len(recs), **f}
            )  # fmt: skip
        pol = aggregate_policy(recs)
        out.append(
            {**blank, "view": "policy", "scope": "policy", "group": g, "mode": "overlap",
             "sentences": len(recs), "gold": pol["protect_gold"], "pred": pol["pred"],
             "tp": pol["protect_hit"], "fp": pol["overmask"],
             "fn": pol["protect_gold"] - pol["protect_hit"],
             "protect_recall": pol["protect_recall"], "overmask_rate": pol["overmask_rate"]}
        )  # fmt: skip
    return out


def fmt(v):
    return round(v, 4) if isinstance(v, float) else v


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, "")) for k in cols})


def bar_svg(per_kind: dict[str, dict]) -> str:
    """세분 라벨별 P/R/F1(overlap) 과 FNR(exact·overlap) 막대그래프. 외부 라이브러리 없이 SVG 를 직접 쓴다.

    라벨 이름 뒤 * 는 기존 탐지기가 세분 라벨로 낼 수 없는 평가 전용 라벨(항상 0)."""
    kinds = [k for k in FINE if per_kind[k]["overlap"]["gold"]]
    w, h, left, top = 1100, 700, 60, 60
    panel_h = 230
    colors = {
        "P": "#4c78a8",
        "R": "#f58518",
        "F1": "#54a24b",
        "exact": "#e45756",
        "overlap": "#72b7b2",
    }
    title = "Baseline v2 - per-label metrics (rule detector, all splits; * = no detector label)"
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="sans-serif" font-size="11">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
        f'<text x="60" y="28" font-size="16" font-weight="bold">{title}</text>',
    ]

    def panel(y0, name, series):
        out.append(f'<text x="{left}" y="{y0 - 14}" font-weight="bold">{name}</text>')
        for i in range(6):
            v = i / 5
            y = y0 + panel_h - v * panel_h
            out.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{w - 20}" y2="{y:.1f}" stroke="#dddddd"/>'
            )
            out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{v:.1f}</text>')
        gw = (w - 20 - left) / len(kinds)
        bw = gw * 0.8 / len(series)
        for gi, k in enumerate(kinds):
            for si, (_, color, fn) in enumerate(series):
                val = fn(per_kind[k])
                x = left + gi * gw + gw * 0.1 + si * bw
                bh = val * panel_h
                out.append(
                    f'<rect x="{x:.1f}" y="{y0 + panel_h - bh:.1f}" width="{bw - 2:.1f}" height="{bh:.1f}" fill="{color}"/>'
                )
                out.append(
                    f'<text x="{x + (bw - 2) / 2:.1f}" y="{y0 + panel_h - bh - 3:.1f}" text-anchor="middle" font-size="8">{val:.2f}</text>'
                )
            star = "*" if k in UNSUPPORTED_FINE else ""
            cx = left + gi * gw + gw / 2
            out.append(
                f'<text x="{cx:.1f}" y="{y0 + panel_h + 16}" text-anchor="middle" font-size="9">{k}{star}</text>'
            )
            out.append(
                f'<text x="{cx:.1f}" y="{y0 + panel_h + 28}" text-anchor="middle" font-size="9">n={per_kind[k]["overlap"]["gold"]}</text>'
            )
        for si, (label, color, _) in enumerate(series):
            lx = w - 20 - 100 * (len(series) - si)
            out.append(
                f'<rect x="{lx}" y="{y0 - 24}" width="10" height="10" fill="{color}"/><text x="{lx + 14}" y="{y0 - 14}">{label}</text>'
            )

    panel(top + 20, "Precision / Recall / F1 (overlap match)", [
        ("Precision", colors["P"], lambda m: m["overlap"]["precision"]),
        ("Recall", colors["R"], lambda m: m["overlap"]["recall"]),
        ("F1", colors["F1"], lambda m: m["overlap"]["f1"]),
    ])  # fmt: skip
    panel(top + 20 + panel_h + 90, "False negative rate (lower is better)", [
        ("FNR exact", colors["exact"], lambda m: m["exact"]["fnr"]),
        ("FNR overlap", colors["overlap"], lambda m: m["overlap"]["fnr"]),
    ])  # fmt: skip
    out.append("</svg>")
    return "\n".join(out)


def _cmp_row(group: str, ma: dict, mb: dict) -> dict:
    return {
        "group": group,
        "gold": ma["gold"],
        "safe_recall": ma["recall"],
        "private_recall": mb["recall"],
        "recall_diff": ma["recall"] - mb["recall"],
        "safe_precision": ma["precision"],
        "private_precision": mb["precision"],
        "precision_diff": ma["precision"] - mb["precision"],
    }


def compare_rows(a: list[dict], b: list[dict]) -> list[dict]:
    """안전 픽스처(a)와 원본 세트(b)의 mapped·overlap 재현율/정밀도 차이(값은 담지 않는 집계만)."""
    out = [_cmp_row("all", aggregate(a, "mapped", "overlap"), aggregate(b, "mapped", "overlap"))]
    for k in MAPPED_KINDS:
        ma, mb = aggregate(a, "mapped", "overlap", [k]), aggregate(b, "mapped", "overlap", [k])
        out.append(_cmp_row(f"kind:{k}", ma, mb))
    for t in sorted({t for r in a for t in r["tags"]}):
        ma = aggregate([r for r in a if t in r["tags"]], "mapped", "overlap")
        mb = aggregate([r for r in b if t in r["tags"]], "mapped", "overlap")
        out.append(_cmp_row(f"tag:{t}", ma, mb))
    return out


def load_checked(directory: Path) -> dict[str, list[dict]]:
    by_split = load_splits(directory)
    problems = [p for rows in by_split.values() for p in validate_dataset(rows)]
    leaks = check_leakage(by_split)
    print(
        f"[{directory.name}] 형식 검증: {'통과' if not problems else f'실패 {len(problems)}건'} / "
        f"분할 누출 검사: {'통과' if not leaks else f'실패 {len(leaks)}건'}"
    )
    for p in (problems + leaks)[:20]:
        print("  -", p)
    if problems or leaks:
        raise SystemExit(1)
    return by_split


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=5, help="처리시간 측정 반복 횟수")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="기본: 저장소 eval/results (--data-dir 를 바꾸면 그 폴더의 results)",
    )
    ap.add_argument("--data-dir", type=Path, default=V2_DIR)
    ap.add_argument(
        "--compare-data-dir",
        type=Path,
        default=None,
        help="비교할 두 번째 세트(예: 로컬 원본 세트)",
    )
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data_dir = args.data_dir.resolve()
    out = args.out_dir or (RESULTS_DIR if data_dir == V2_DIR.resolve() else data_dir / "results")

    import app

    app_path = Path(app.__file__).resolve()
    print(f"app 경로: {app_path}")
    by_split = load_checked(data_dir)
    if data_dir == V2_DIR.resolve():  # 공개 저장소 세트는 안전 검사를 통과해야 한다
        unsafe = check_fixture_safety(
            [r for rows in by_split.values() for r in rows], load_public_pans()
        )
        print(f"안전 픽스처 검사: {'통과' if not unsafe else f'실패 {len(unsafe)}건'}")
        if unsafe:
            for u in unsafe[:10]:
                print(f"  - {u}")
            return 1

    records = evaluate(by_split)
    mrows = metric_rows(records)
    errors = [e for r in records for e in r["errors"]]
    all_rows = [r for rows in by_split.values() for r in rows]
    timings = {"all": timing(all_rows, args.rounds)}
    for s in SPLITS:
        timings[s] = timing(by_split[s], args.rounds)
    long_rows = [r for r in all_rows if "long_text" in r["tags"]]
    timings["long_text(500+자)"] = timing(long_rows, args.rounds)
    timings["short(<500자)"] = timing(
        [r for r in all_rows if "long_text" not in r["tags"]], args.rounds
    )

    per_fine = {k: {m: aggregate(records, "fine", m, [k]) for m in MODES} for k in FINE}
    per_mapped = {k: {m: aggregate(records, "mapped", m, [k]) for m in MODES} for k in MAPPED_KINDS}
    fine_supported = [k for k in FINE if k not in UNSUPPORTED_FINE]
    summary = {
        "app_path": str(app_path),
        "data_dir": data_dir.name,
        "sentences": {s: len(rows) for s, rows in by_split.items()},
        "format_validation": "pass",
        "leakage_check": "pass",
        "overall": {v: {m: aggregate(records, v, m) for m in MODES} for v in VIEW_KINDS},
        "overall_detector_supported_only": {
            "fine": {m: aggregate(records, "fine", m, fine_supported) for m in MODES},
            "mapped": {m: aggregate(records, "mapped", m, KINDS) for m in MODES},
        },
        "per_kind_fine": per_fine,
        "per_kind_mapped": per_mapped,
        "per_split_mapped": {
            s: {m: aggregate([r for r in records if r["split"] == s], "mapped", m) for m in MODES}
            for s in SPLITS
        },
        "policy": {
            "all": aggregate_policy(records),
            **{s: aggregate_policy([r for r in records if r["split"] == s]) for s in SPLITS},
        },
        "fpr": {
            "all": fpr_of(records),
            **{s: fpr_of([r for r in records if r["split"] == s]) for s in SPLITS},
        },
        "timing_ms_per_sentence": timings,
        "error_counts_mapped_overlap": {
            t: sum(1 for e in errors if e["error_type"] == t) for t in ("FN", "FP", "BOUNDARY")
        },
    }
    write_csv(out / "baseline_v2_metrics.csv", mrows, CSV_COLS)
    write_csv(
        out / "baseline_v2_errors.csv",
        errors,
        ["id", "split", "view", "error_type", "kind", "gold_text", "pred_kind", "pred_text",
         "tags", "template_id", "note"],
    )  # fmt: skip
    (out / "baseline_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=fmt),
        encoding="utf-8",
        newline="\n",
    )
    (out / "baseline_v2_metrics.svg").write_text(bar_svg(per_fine), encoding="utf-8", newline="\n")

    if args.compare_data_dir:
        other = load_checked(args.compare_data_dir.resolve())
        crow = compare_rows(records, evaluate(other))
        write_csv(
            out / "baseline_v2_safe_vs_private.csv",
            crow,
            ["group", "gold", "safe_recall", "private_recall", "recall_diff",
             "safe_precision", "private_precision", "precision_diff"],
        )  # fmt: skip
        a = crow[0]
        print(
            f"안전 vs 원본(mapped, overlap) 재현율 {a['safe_recall']:.3f} vs {a['private_recall']:.3f}, "
            f"정밀도 {a['safe_precision']:.3f} vs {a['private_precision']:.3f}"
        )
        for r in sorted(crow, key=lambda r: -abs(r["recall_diff"]))[:8]:
            print(f"  {r['group']}: 재현율 차이 {r['recall_diff']:+.3f} (n={r['gold']})")

    for view in ("fine", "mapped"):
        ov = summary["overall"][view]
        o, e = ov["overlap"], ov["exact"]
        print(
            f"[{view}] 전체 overlap P/R/F1 = {o['precision']:.3f}/{o['recall']:.3f}/{o['f1']:.3f}, "
            f"exact = {e['precision']:.3f}/{e['recall']:.3f}/{e['f1']:.3f}"
        )
    sup = summary["overall_detector_supported_only"]["mapped"]["overlap"]
    print(
        f"[mapped, 탐지기 지원 라벨만] overlap P/R/F1 = {sup['precision']:.3f}/{sup['recall']:.3f}/{sup['f1']:.3f}"
    )
    print(
        f"FPR(음성 문장) = {summary['fpr']['all']['neg_flagged']}/{summary['fpr']['all']['neg_sentences']}"
    )
    pol = summary["policy"]["all"]
    print(
        f"정책: 보호 재현율 {pol['protect_recall']:.3f}, 과마스킹 예측 {pol['overmask']}/{pol['pred']}"
    )
    print(
        f"처리시간 평균 {timings['all']['mean_ms']:.3f} ms, P95 {timings['all']['p95_ms']:.3f} ms"
    )
    print(f"결과 저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
