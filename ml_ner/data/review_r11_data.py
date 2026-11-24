"""R11 생성 데이터 품질 검수(자동 항목 + 사람 검수용 로컬 표본). 사전 등록 NER_R11_PREREGISTRATION.md §4-3.

자동 항목(건수만 출력): 구간 무결성, 토큰 정렬 BIO 왕복, 조사 받침 일치, B1·C2·LONGNAME·별명 규칙, 대조쌍 문맥 차이, 접두·접미 구절 반복(스크립트 결합), 외래어 의심 단어 수, 형식 이상.
사람 검수용 표본: 세트·유형별 무작위 추출(시드 고정) → 로컬 HTML(오프라인, 외부 전송 없음)·CSV. 문장 원문은 저장소 밖(`data_r11/review/`)에만 둔다.
사용: python review_r11_data.py --out-report report.json [--packet-dir DIR]
"""

import argparse
import difflib
import html
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "train"))
import build_r11_data as b11  # noqa: E402

DATA = b11.OUT_DEFAULT
PERSON = ("PERSON", "PERSON_ALIAS")
PUNCT = " \t.,!?~·()[]\"'’”“…:;-/ㅋㅎㅠㅜ"
PARTICLE_START = "이가은는을를과와랑에의도만로"
KIN = ("엄마", "아빠", "할머니", "할아버지", "이모", "삼촌", "고모", "외삼촌", "누나", "언니", "오빠", "형", "동생", "아들", "딸")
LOAN_END = ("터", "더", "스", "크", "트", "즈", "드", "커", "나")  # 외래어 일반명사 의심 어말(휴리스틱, 사람이 확인)
SAMPLE = {"train": {"pairs": 60}, "val": {"pairs": 40},
          "dev": {"pairs": 60, "B1": 25, "C2": 25, "NEWNAME": 25, "LONGNAME": 25, "EXISTING": 20, "NEG_ORGLOC": 15, "NEG_FAMILY": 15, "NEG_D2": 15, "NEG_GENERAL": 15}}
SEED = 20261126


def rd(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def batchim(ch):
    return (ord(ch) - 0xAC00) % 28 != 0 if "가" <= ch <= "힣" else None


def masked_text(r):
    """이름·대조쌍 단어를 가린 문장(틀 비교용)."""
    t = r["text"]
    for s in sorted(r["spans"], key=lambda x: -x["start"]):
        t = t[: s["start"]] + "<N>" + t[s["end"]:]
    w = r.get("pair_word")
    if w:
        t = t.replace(w, "<W>")
    return t


def auto_checks(name, rows, tok):
    import tokenize_align as ta

    rep = {"rows": len(rows)}
    must = set()  # 사람이 반드시 볼 행(자동 검사에 걸린 행)
    bad = Counter()
    # 1) 구간 무결성
    for r in rows:
        t, prev = r["text"], -1
        for s in sorted(r["spans"], key=lambda x: x["start"]):
            v = t[s["start"] : s["end"]]
            if not (0 <= s["start"] < s["end"] <= len(t)):
                bad["span_out_of_range"] += 1
            elif v != v.strip() or any(c in v for c in "()[]\"'.,!?"):
                bad["span_edge_or_punct"] += 1
            if s["start"] < prev:
                bad["span_overlap"] += 1
            prev = s["end"]
            if s["label"] in PERSON and v in KIN:
                bad["person_is_kinship_term"] += 1
            if s["label"] in PERSON and (re.search(r"\d", v) or re.search(r"[A-Za-z]", v)):
                bad["person_has_digit_or_latin"] += 1
            if s["label"] in PERSON and len(v) < 2:
                bad["person_one_char"] += 1
    rep["span_integrity_violations"] = dict(bad)
    # 2) 토큰 정렬 BIO 왕복(훈련 파이프라인과 같은 정렬 함수, 학습 때 쓰는 max_len 128)
    enc = ta.encode_rows(rows, tok, 128, "all")
    rt_bad = trunc = 0
    for r, e in zip(rows, enc, strict=False):
        dec = ta.decode_spans(e["labels"], e["offsets"], e["special"], "all", e["tokens"])
        gold = [(s["label"], s["start"], s["end"]) for s in r["spans"] if s["label"] in PERSON + ("ORGANIZATION", "LOCATION", "ADDRESS")]
        got = [(s["label"], s["start"], s["end"]) for s in dec]
        if sorted(gold) != sorted(got):
            rt_bad += 1
        trunc += len(e["tokens"]) >= 128
    rep["bio_roundtrip_mismatch_rows"] = rt_bad
    rep["truncated_rows(>=128 tokens)"] = trunc
    # 3) 조사 받침 일치(이름 뒤 이/가·은/는·을/를·와/과)
    pairs_ok = {"이": True, "가": False, "은": True, "는": False, "을": True, "를": False, "과": True, "와": False}
    checked = viol = 0
    for r in rows:
        t = r["text"]
        for s in r["spans"]:
            if s["label"] not in PERSON or s["end"] >= len(t):
                continue
            c, last = t[s["end"]], t[s["end"] - 1]
            nxt = t[s["end"] + 1] if s["end"] + 1 < len(t) else " "
            if c in pairs_ok and (nxt in PUNCT or not ("가" <= nxt <= "힣")) and batchim(last) is not None:
                checked += 1
                if batchim(last) != pairs_ok[c]:
                    viol += 1
                    must.add(r["id"])
    rep["josa_checked_vs_batchim_violation"] = [checked, viol]
    # 4) 유형별 규칙(dev)
    if name == "dev":
        b1_bad = c2_bad = long_bad = 0
        for r in rows:
            sp = [s for s in r["spans"] if s["label"] in PERSON]
            if r["type"] == "B1":
                for s in sp:
                    nxt = r["text"][s["end"]] if s["end"] < len(r["text"]) else " "
                    kinfollow = any(r["text"][s["end"] :].lstrip().startswith(k) for k in KIN) or r["text"][s["end"] :].startswith(("씨", "님"))
                    if nxt in PARTICLE_START and not kinfollow:
                        b1_bad += 1
            if r["type"] == "C2":
                ok = any(b["start"] - a["end"] <= 3 for a, b in zip(sp, sp[1:], strict=False))
                c2_bad += not ok
                if not ok:
                    must.add(r["id"])
            if r["type"] == "LONGNAME" and not any(s["end"] - s["start"] >= 4 for s in sp if s["label"] == "PERSON"):
                long_bad += 1
        rep["B1_name_followed_by_particle(rows)"] = b1_bad
        rep["C2_rows_without_adjacent_pair"] = c2_bad
        rep["LONGNAME_rows_without_4plus_name"] = long_bad
    # 5) 별명
    al = [r["text"][s["start"] : s["end"]] for r in rows for s in r["spans"] if s["label"] == "PERSON_ALIAS"]
    names = {r["text"][s["start"] : s["end"]] for r in rows for s in r["spans"] if s["label"] == "PERSON"}
    for r in rows:
        for s_ in r["spans"]:
            if s_["label"] == "PERSON_ALIAS":
                v_ = r["text"][s_["start"] : s_["end"]]
                if v_ in KIN or v_ in names:
                    must.add(r["id"])
    rep["alias"] = {"spans": len(al), "distinct": len(set(al)), "kinship_or_name_collision": sum(1 for x in al if x in KIN or x in names), "len_outside_2_4": sum(1 for x in al if not 2 <= len(x) <= 4)}
    # 6) 대조쌍 문맥 차이
    pairs = defaultdict(dict)
    for r in rows:
        if r.get("pair_id"):
            pairs[r["pair_id"]][r["pair_role"]] = r
    sims = []
    for p in pairs.values():
        if {"person", "noun"} <= set(p):
            sims.append(difflib.SequenceMatcher(None, masked_text(p["person"]), masked_text(p["noun"])).ratio())
    if sims:
        rep["pair_context_similarity"] = {"pairs": len(sims), "ge_0.95": sum(s >= 0.95 for s in sims), "ge_0.90": sum(s >= 0.90 for s in sims), "ge_0.80": sum(s >= 0.80 for s in sims),
                                          "mean": round(sum(sims) / len(sims), 3)}
        words = [p["person"]["pair_word"] for p in pairs.values() if {"person", "noun"} <= set(p)]
        for p in pairs.values():
            if {"person", "noun"} <= set(p) and len(p["person"]["pair_word"]) == 3 and p["person"]["pair_word"].endswith(LOAN_END):
                must.add(p["person"]["id"])
                must.add(p["noun"]["id"])
        rep["pair_words_loanword_suspect_by_ending(len3)"] = sum(1 for w in set(words) if len(w) == 3 and w.endswith(LOAN_END))
    # 7) 접두·접미 구절 반복(스크립트 결합 의심)
    def words3(t, first):
        w = t.split()
        return " ".join(w[:3] if first else w[-3:])

    pre = Counter(words3(masked_text(r), True) for r in rows)
    suf = Counter(words3(masked_text(r), False) for r in rows)
    flagged = {r["id"] for r in rows if pre[words3(masked_text(r), True)] >= 5 or suf[words3(masked_text(r), False)] >= 5}
    rep["repeated_phrase"] = {"prefix_phrases_ge5": sum(1 for c in pre.values() if c >= 5), "suffix_phrases_ge5": sum(1 for c in suf.values() if c >= 5),
                              "rows_in_repeated_prefix_or_suffix": len(flagged), "share": round(len(flagged) / len(rows), 3), "max_prefix_repeat": max(pre.values()), "max_suffix_repeat": max(suf.values())}
    byfile = defaultdict(lambda: [0, 0])
    for r in rows:
        byfile[r["frame_file"]][0] += 1
        byfile[r["frame_file"]][1] += r["id"] in flagged
    rep["repeated_phrase_by_author_file"] = {k: v for k, v in byfile.items() if v[1]}
    # 8) 형식 이상
    fmt = Counter()
    for r in rows:
        t = r["text"]
        fmt["double_space"] += "  " in t
        fmt["edge_space"] += t != t.strip()
        ub = (t.count("(") != t.count(")")) or (t.count('"') % 2 == 1)
        fmt["unbalanced_bracket_or_quote"] += ub
        if ub:
            must.add(r["id"])
        toks = t.split()
        fmt["same_word_3x"] += any(c >= 3 for c in Counter(toks).values())
    rep["format_anomalies(rows)"] = dict(fmt)
    rep["must_review_rows"] = len(must)
    return rep, flagged | set(), must


def sample_rows(name, rows, flagged, rng, must=frozenset()):
    """세트·유형별 무작위 표본. 대조쌍은 쌍 단위로 뽑는다. 접두·접미 반복 행은 별도 추가."""
    spec = SAMPLE[name]
    out = []
    pairs = defaultdict(dict)
    for r in rows:
        if r.get("pair_id"):
            pairs[r["pair_id"]][r["pair_role"]] = r
    pk = sorted(k for k, v in pairs.items() if {"person", "noun"} <= set(v))
    for k in rng.sample(pk, spec["pairs"]):
        out += [pairs[k]["person"], pairs[k]["noun"]]
    if name == "dev":
        for t, n in spec.items():
            if t == "pairs":
                continue
            c = [r for r in rows if r["type"] == t]
            out += rng.sample(c, n)
    chosen = {r["id"] for r in out}
    extra = [r for r in rows if r["id"] in flagged and r["id"] not in chosen and not r.get("pair_id")]
    out += rng.sample(extra, min(10, len(extra)))
    chosen = {r["id"] for r in out}
    out += [r for r in rows if r["id"] in must and r["id"] not in chosen]  # 자동 검사에 걸린 행은 전수 포함
    return out


def render(r, name):
    t = r["text"]
    parts, pos = [], 0
    for s in sorted(r["spans"], key=lambda x: x["start"]):
        parts.append(html.escape(t[pos : s["start"]]))
        parts.append(f"<mark class='{s['label']}'>{html.escape(t[s['start']:s['end']])}<sub>{s['label']}</sub></mark>")
        pos = s["end"]
    parts.append(html.escape(t[pos:]))
    return "".join(parts)


PAGE = """<!doctype html><meta charset="utf-8"><title>R11 데이터 표본 검수(로컬)</title>
<style>body{font:14px/1.5 system-ui,'Malgun Gothic',sans-serif;margin:16px;max-width:1100px}table{border-collapse:collapse;width:100%}
td,th{border:1px solid #ccc;padding:4px 6px;vertical-align:top}mark{background:#ffe08a}mark.PERSON_ALIAS{background:#ffc9a8}sub{font-size:10px;color:#555}
.meta{color:#666;font-size:12px}textarea{width:180px;height:44px}.pair{background:#f3f7ff}</style>
<h2>R11 데이터 표본 검수 (로컬 전용: 외부 전송 없음, 이 파일을 외부 LLM·서비스에 올리지 마세요)</h2>
<p>각 문장에서 노란색(이름)·주황색(별명) 구간만 PERSON/PERSON_ALIAS 로 라벨된 것입니다. 항목마다 판단해 저장 후 JSON 으로 내려받으세요.</p>
<ul><li><b>자연스러움</b>: 1=자연, 2=어색하지만 사용 가능, 3=부자연(접두·접미 구절이 억지로 붙은 느낌 포함)</li>
<li><b>라벨 정확</b>: 이름/별명이 아닌 것에 라벨이 붙었거나 이름에 라벨이 빠진 곳이 없는가(일반명사로 쓰인 단어는 라벨 없음이 정답)</li>
<li><b>경계 정확</b>: 구간이 이름 부분만 정확히 포함하는가(조사·호칭·따옴표 제외)</li>
<li><b>쌍 구분</b>(파란 행, 같은 단어 쌍): 사람 문장은 문맥상 사람, 명사 문장은 문맥상 일반명사로 명확히 읽히는가</li>
<li><b>외래어/별명/조사</b>: 단어가 외래어 일반명사이거나 별명·조사 결합이 이상하면 메모</li></ul>
<table id="t"><tr><th>#</th><th>세트·유형</th><th>문장</th><th>자연스러움</th><th>라벨</th><th>경계</th><th>쌍 구분</th><th>메모</th></tr>
%ROWS%</table><p><button onclick="save()">JSON 저장(내려받기)</button> <span id="msg"></span></p>
<script>
const ids=%IDS%;
function save(){const out=ids.map((id,i)=>{const g=n=>{const e=document.querySelector(`[name='${n}${i}']:checked`);return e?e.value:null};
return {id,natural:g('nat'),label_ok:g('lab'),boundary_ok:g('bnd'),pair_ok:g('pair'),note:document.querySelector(`[name='note${i}']`).value}});
const b=new Blob([JSON.stringify(out,null,1)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='r11_review_result.json';a.click();document.getElementById('msg').textContent=out.length+'개 저장';}
</script>"""


def packet(sample, outdir, must_ids=frozenset(), only_must=False, filename="review_packet.html"):
    """표본 519개는 그대로 두고 화면 순서만 바꾼다: 자동 검사에 걸린 행(must_ids)을 맨 앞에 '우선' 으로 표시. only_must 이면 그 행만 담은 별도 파일."""
    outdir.mkdir(parents=True, exist_ok=True)
    rows_html, ids = [], []
    sample = sorted(sample, key=lambda x: (x[1]["id"] not in must_ids)) if must_ids else sample
    if only_must:
        sample = [x for x in sample if x[1]["id"] in must_ids]
    for i, (name, r) in enumerate(sample):
        ids.append(r["id"])
        radio = lambda n, vals: "".join(f"<label><input type='radio' name='{n}{i}' value='{v}'>{v}</label> " for v in vals)  # noqa: E731
        is_pair = bool(r.get("pair_id"))
        rows_html.append(
            f"<tr class='{'pair' if is_pair else ''}'><td>{i + 1}{' <b style="color:#c00">우선</b>' if r['id'] in must_ids else ''}</td><td>{name}/{r.get('type', 'train-pair')}<div class='meta'>{html.escape(r['id'])}"
            f"{' · ' + r['pair_role'] if is_pair else ''}</div></td><td>{render(r, name)}</td><td>{radio('nat', ['1', '2', '3'])}</td>"
            f"<td>{radio('lab', ['Y', 'N'])}</td><td>{radio('bnd', ['Y', 'N'])}</td><td>{radio('pair', ['Y', 'N']) if is_pair else '-'}</td><td><textarea name='note{i}'></textarea></td></tr>")
    page = PAGE.replace("%ROWS%", "\n".join(rows_html)).replace("%IDS%", json.dumps(ids))
    (outdir / filename).write_text(page, encoding="utf-8")
    if not only_must:
        (outdir / "review_sample_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        (outdir / "review_priority_ids.json").write_text(json.dumps(sorted(must_ids)), encoding="utf-8")
    import csv

    if only_must:
        return len(ids)
    with open(outdir / "review_packet.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "set", "type", "pair_role", "text_with_spans", "natural(1-3)", "label_ok(Y/N)", "boundary_ok(Y/N)", "pair_ok(Y/N)", "note"])
        for name, r in sample:
            txt = r["text"]
            for s in sorted(r["spans"], key=lambda x: -x["start"]):
                txt = txt[: s["start"]] + "[[" + txt[s["start"] : s["end"]] + "|" + s["label"] + "]]" + txt[s["end"] :]
            w.writerow([r["id"], name, r.get("type", ""), r.get("pair_role", ""), txt, "", "", "", "", ""])
    return len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-report", type=Path, required=True)
    ap.add_argument("--packet-dir", type=Path)
    a = ap.parse_args()
    import tokenize_align as ta

    tok = ta.load_tokenizer("monologg/koelectra-small-v3-discriminator")
    sets = {"train": rd(DATA / "r11_Dprime_train.jsonl"), "val": rd(DATA / "r11_val.jsonl"), "dev": rd(DATA / "r11_dev.jsonl")}
    report, sample, must_all = {"seed": SEED}, [], set()
    rng = random.Random(SEED)
    for name, rows in sets.items():
        rep, flagged, must = auto_checks(name, rows, tok)
        must_all |= must
        report[name] = rep
        sample += [(name, r) for r in sample_rows(name, rows, flagged, rng, must)]
    report["human_review_sample"] = {"total_items": len(sample), "by_set": dict(Counter(n for n, _ in sample))}
    if a.packet_dir:
        report["human_review_sample"]["packet_items_written"] = packet(sample, a.packet_dir, must_all)
        report["human_review_sample"]["priority_items"] = packet(sample, a.packet_dir, must_all, only_must=True, filename="review_priority_packet.html")
    a.out_report.parent.mkdir(parents=True, exist_ok=True)
    a.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
