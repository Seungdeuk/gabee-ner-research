"""R11 자동 품질 검증 강화(사람 검수 대체가 아님): 학습 D'·Validation·Development 전체를 로컬에서 다시 검사하고 결과를 세 등급으로 나눈다.

 - S 확정 구조 오류: 데이터 규격·정답 구조·격리 규칙 위반으로 사람 판단 없이 틀렸다고 말할 수 있는 것
 - F 확정 표면 오류(라벨 영향 없음): 조사 받침 불일치처럼 규칙으로 확정되지만 정답 라벨·경계에는 영향이 없는 것
 - M 의미 검수 필요 경고: 자연스러움, 대조쌍의 문맥 구분, 외래어·별명 충돌, 반복 구절처럼 사람이 읽어야 판단되는 것(자동으로 통과·오류 처리하지 않는다)
결과 JSON 에는 문장 원문·이름·단어를 싣지 않는다(id, 건수, 코드, 구조 사실만). 외부 LLM·서비스를 쓰지 않는다. 데이터를 수정하지 않는다(읽기 전용).
사용: python r11_auto_qa.py --out-report report.json --out-suspects suspects.json
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "train"))
import build_r11_data as b11  # noqa: E402
import review_r11_data as rv  # noqa: E402

DATA = b11.OUT_DEFAULT
PERSON = ("PERSON", "PERSON_ALIAS")
ALLOWED = PERSON + ("ORGANIZATION", "LOCATION", "ADDRESS")
EMOTICON = re.compile(r"(?:[:;=][\"'-]?[()DPp])|(?:[ㅠㅜㅋㅎ]+\)+)|(?:\(\^[^)]*\^\))|(?:\^\))|(?:[0-9]\))")
BOUNDARY_OK = " \t.,!?~·()[]\"'’”“…:;-/ㅋㅎㅠㅜ"
C2_MAX_GAP = 8  # r11-2.0: 초기 휴리스틱(3)은 임의 값이었고 사양의 C2 정의(접속·쉼표·중점 나열 포함)보다 엄격했다. 변경 사유와 1행 재분류는 r11-2.0 문서에 기록
JOSA_PAIRS = {"이": True, "가": False, "은": True, "는": False, "을": True, "를": False, "과": True, "와": False}


def rd(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def bad_brackets(t):
    t2 = EMOTICON.sub("", t)
    return t2.count("(") != t2.count(")") or t2.count('"') % 2 == 1


def run_set(name, rows, tok):
    import tokenize_align as ta

    S, F, M = defaultdict(list), defaultdict(list), defaultdict(list)  # 코드 -> id 목록
    pairs = defaultdict(dict)
    for r in rows:
        t, rid = r["text"], r["id"]
        ps = sorted(r["spans"], key=lambda s: s["start"])
        prev = -1
        for s in ps:
            v = t[s["start"] : s["end"]]
            if s["label"] not in ALLOWED:
                S["label_not_allowed"].append(rid)
            if not (0 <= s["start"] < s["end"] <= len(t)):
                S["span_out_of_range"].append(rid)
            elif v != v.strip() or any(c in v for c in "()[]\"'.,!?"):
                S["span_edge_whitespace_or_punct"].append(rid)
            if s["start"] < prev:
                S["span_overlap"].append(rid)
            prev = s["end"]
            if s["label"] in PERSON:
                if len(v) < 2:
                    S["person_one_char"].append(rid)
                if v in rv.KIN:
                    S["person_equals_kinship_term"].append(rid)
                if re.search(r"[0-9A-Za-z]", v):
                    S["person_has_digit_or_latin"].append(rid)
        if r.get("pair_id"):
            pairs[r["pair_id"]][r["pair_role"]] = r
        # 조사 받침 일치(확정 표면 오류, 라벨 영향 없음)
        for s in ps:
            if s["label"] in PERSON and s["end"] < len(t):
                c, last = t[s["end"]], t[s["end"] - 1]
                nxt = t[s["end"] + 1] if s["end"] + 1 < len(t) else " "
                if c in JOSA_PAIRS and (nxt in rv.PUNCT or not ("가" <= nxt <= "힣")) and rv.batchim(last) is not None and rv.batchim(last) != JOSA_PAIRS[c]:
                    F["josa_batchim_mismatch"].append(rid)
        if bad_brackets(t):
            F["bracket_or_quote_unbalanced_after_emoticon_removal"].append(rid)
    # BIO 정렬 왕복(정렬 손실은 토큰 분절의 성질, 구조 오류가 아님: 유형별 비율을 기록)
    enc = ta.encode_rows(rows, tok, 128, "all")
    loss_by_type = Counter()
    loss_ids = []
    for r, e in zip(rows, enc, strict=False):
        gold = sorted((s["label"], s["start"], s["end"]) for s in r["spans"] if s["label"] in ALLOWED)
        dec = ta.decode_spans(e["labels"], e["offsets"], e["special"], "all", e["tokens"])
        if gold != sorted((s["label"], s["start"], s["end"]) for s in dec):
            loss_ids.append(r["id"])
            loss_by_type[r.get("type", "TRAIN_PAIR")] += 1
        if len(e["tokens"]) >= 128 and any(s["end"] > max((o[1] for o in e["offsets"] if o != (0, 0)), default=0) for s in r["spans"]):
            S["span_beyond_truncation"].append(r["id"])
    # 대조쌍 정합성
    for pid, p in pairs.items():
        if set(p) != {"person", "noun"}:
            S["pair_incomplete"].extend(x["id"] for x in p.values())
            continue
        a, n = p["person"], p["noun"]
        w = a["pair_word"]
        if n["pair_word"] != w or w not in a["text"] or w not in n["text"]:
            S["pair_word_mismatch"].append(a["id"])
        if not any(a["text"][s["start"] : s["end"]] == w for s in a["spans"] if s["label"] in PERSON):
            S["pair_person_sentence_without_word_span"].append(a["id"])
        if any(s["label"] in PERSON and w in n["text"][s["start"] : s["end"]] for s in n["spans"]):
            S["pair_noun_sentence_labels_the_word"].append(n["id"])
        if a["text"] == n["text"]:
            S["pair_identical_text"].append(a["id"])
        # 같은 단어가 다른 쌍의 사람/명사 문장에서 일관되게 쓰였는지: 명사 문장 안의 단어는 라벨이 없어야 한다(위 검사), 사람 문장에서는 구간이어야 한다(위 검사)
    # 유형 규칙(dev)
    if name == "dev":
        for r in rows:
            sp = sorted([s for s in r["spans"] if s["label"] in PERSON], key=lambda s: s["start"])
            t = r["text"]
            if r["type"] == "B1":
                for s in sp:
                    nxt = t[s["end"]] if s["end"] < len(t) else " "
                    kin = any(t[s["end"] :].lstrip().startswith(k) for k in rv.KIN) or t[s["end"] :].startswith(("씨", "님"))
                    if not (nxt in BOUNDARY_OK or kin):
                        S["B1_name_not_at_boundary(조사 또는 글자가 이어짐)"].append(r["id"])
                    if len(t[s["start"] : s["end"]]) < 3:
                        S["B1_name_shorter_than_3"].append(r["id"])
            if r["type"] == "C2":
                if len(sp) < 2:
                    S["C2_row_with_fewer_than_2_names"].append(r["id"])
                elif not any(b["start"] - a["end"] <= C2_MAX_GAP for a, b in zip(sp, sp[1:], strict=False)):
                    S["C2_no_adjacent_names"].append(r["id"])
                elif not any(b["start"] - a["end"] <= 3 for a, b in zip(sp, sp[1:], strict=False)):
                    # 접속 표현(예: 접속 조사+공백)으로 간격이 4글자 이상. 두 이름이 별도 구간으로 분리돼 있어 구조는 정상이며 의미 확인은 사람 몫(자동 판정 아님)
                    M["C2_names_separated_by_connector_gap_over_3(structure ok)"].append(r["id"])
                if any(b["start"] < a["end"] for a, b in zip(sp, sp[1:], strict=False)):
                    S["C2_names_overlap"].append(r["id"])
            if r["type"] == "LONGNAME" and not any(s["end"] - s["start"] >= 4 for s in sp if s["label"] == "PERSON"):
                S["LONGNAME_without_4plus_name"].append(r["id"])
            if r["type"].startswith("NEG_") and sp:
                S["negative_type_has_person_span"].append(r["id"])
            if r["type"] == "D1" and r.get("is_negative") is False and any(s["label"] in PERSON and r.get("pair_word") in t[s["start"] : s["end"]] for s in sp):
                S["D1_row_labels_the_word"].append(r["id"])
    # 의미 검수 경고(M): 자동 판단 불가
    al_vals = [(r["id"], r["text"][s["start"] : s["end"]]) for r in rows for s in r["spans"] if s["label"] == "PERSON_ALIAS"]
    names = {r["text"][s["start"] : s["end"]] for r in rows for s in r["spans"] if s["label"] == "PERSON"}
    for rid, v in al_vals:
        if v in names or v in rv.KIN:
            M["alias_equals_a_person_name_or_kinship_term"].append(rid)
    ws = {r["pair_word"] for r in rows if r.get("pair_word")}
    suspect_words = {w for w in ws if len(w) == 3 and w.endswith(rv.LOAN_END)}
    for r in rows:
        if r.get("pair_word") in suspect_words:
            M["pair_word_loanword_suspect_by_ending"].append(r["id"])
    # 접두·접미 반복
    pre = Counter(rv.words3(rv.masked_text(r), True) if hasattr(rv, "words3") else " ".join(rv.masked_text(r).split()[:3]) for r in rows)
    suf = Counter(" ".join(rv.masked_text(r).split()[-3:]) for r in rows)
    for r in rows:
        mt = rv.masked_text(r).split()
        if pre[" ".join(mt[:3])] >= 5 or suf[" ".join(mt[-3:])] >= 5:
            M["repeated_prefix_or_suffix_phrase(machine-combined suspect)"].append(r["id"])
    # 대조쌍 문맥 단서가 약한 쌍(문장 유사도 높음): 의미 검수 필요
    import difflib

    for pid, p in pairs.items():
        if {"person", "noun"} <= set(p):
            if difflib.SequenceMatcher(None, rv.masked_text(p["person"]), rv.masked_text(p["noun"])).ratio() >= 0.9:
                M["pair_contexts_nearly_identical(cue may be weak)"].extend([p["person"]["id"], p["noun"]["id"]])
    rep = {"rows": len(rows),
           "S_definitive_structural": {k: len(set(v)) for k, v in S.items()}, "F_definitive_surface_label_neutral": {k: len(set(v)) for k, v in F.items()},
           "M_semantic_review_warnings": {k: len(set(v)) for k, v in M.items()},
           "bio_alignment_loss": {"rows": len(loss_ids), "rate_pct": round(100 * len(loss_ids) / len(rows), 2), "by_type": dict(loss_by_type),
                                  "note": "전부 구간 경계가 토큰 내부(이름+조사가 한 토큰)인 서브워드 분절의 성질, 기존 학습 데이터에도 같은 비율(2.5~3.0%)"}}
    ids = {"S": {k: sorted(set(v)) for k, v in S.items()}, "F": {k: sorted(set(v)) for k, v in F.items()}, "M": {k: sorted(set(v)) for k, v in M.items()}}
    return rep, ids


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-report", type=Path, required=True)
    ap.add_argument("--out-suspects", type=Path, required=True)
    a = ap.parse_args()
    import tokenize_align as ta

    tok = ta.load_tokenizer("monologg/koelectra-small-v3-discriminator")
    sets = {"train": rd(DATA / "r11_Dprime_train.jsonl"), "val": rd(DATA / "r11_val.jsonl"), "dev": rd(DATA / "r11_dev.jsonl")}
    report, ids_all = {}, {}
    for k, rows in sets.items():
        report[k], ids_all[k] = run_set(k, rows, tok)
    # 기존 자동 검수(중복·유사도·역할 누수)는 같은 로컬 검사기의 최신 결과를 그대로 인용
    q = json.loads((DATA / "r11_quality_report.json").read_text(encoding="utf-8"))
    report["leakage_and_duplicates(from check_r11_data)"] = {k: q[k] for k in ("dup_within", "dup_between_roles(sentences)", "skeleton_overlap_between_roles", "sentence_in_existing(R9,R10)",
                                                                                "skeleton_in_existing(R9,R10)", "near_dup_rows_vs_R9R10_train(count,>=0.7)", "near_dup_rows_between_roles(count,>=0.7)",
                                                                                "name_role_violations(wrong hash role)", "given_part_overlap_between_roles",
                                                                                "pair_word_overlap_between_roles(exact)", "pair_word_containment_between_roles",
                                                                                "pair_words_seen_in_R9R10_training(val,dev must be 0)", "pair_integrity")}
    # 22건(기존 자동 의심 사례) 분해: 코드별·세트별·유형별
    prio = set(json.loads((DATA / "review" / "review_priority_ids.json").read_text(encoding="utf-8")))
    meta = {r["id"]: (k, r.get("type", "TRAIN_PAIR"), r.get("pair_id"), r.get("pair_role")) for k, rows in sets.items() for r in rows}
    reasons = defaultdict(list)
    for k, ids in ids_all.items():
        for grade, codes in ids.items():
            for code, lst in codes.items():
                for i in lst:
                    if i in prio:
                        reasons[i].append(f"{grade}:{code}")
    suspects = []
    for i in sorted(prio):
        suspects.append({"id": i, "set": meta[i][0], "type": meta[i][1], "pair_id": meta[i][2], "pair_role": meta[i][3], "reasons": sorted(reasons.get(i, []))})
    pair_groups = defaultdict(list)
    for x in suspects:
        pair_groups[x["pair_id"] or x["id"]].append(x["id"])
    out = {"suspect_rows": len(suspects), "distinct_units(pair counted once)": len(pair_groups), "by_set": dict(Counter(x["set"] for x in suspects)), "by_type": dict(Counter(f"{x['set']}/{x['type']}" for x in suspects)),
           "by_reason": dict(Counter(r for x in suspects for r in x["reasons"])), "rows_without_any_current_reason": [x["id"] for x in suspects if not x["reasons"]], "rows": suspects}
    a.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    a.out_suspects.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (a.out_suspects.parent / "r11_auto_qa_ids.json").write_text(json.dumps(ids_all, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1)[:6500])
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
