"""R9 평가셋 규격 준수·교차 중복·라벨 형식 이상 후보 점검. 원문·이름·정답 구간은 출력하지 않고 건수와 문장 id 만 낸다(id 목록은 파일로 저장).

사용: python check_r9_compliance.py --rep R.jsonl --stress S.jsonl --out ids.json
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validate_r9_evalset as V  # noqa: E402

KIN = {"엄마", "아빠", "할머니", "할아버지", "삼촌", "이모", "고모", "언니", "오빠", "누나", "형", "동생", "큰아들", "작은아들", "큰딸", "막내", "아들", "딸", "이모부", "고모부", "외할머니", "외할아버지", "시어머니", "아버지", "어머니"}
HONORIFIC_END = ("씨", "님", "선생님", "교수님", "대표님", "사장님")
JOSA_TAIL = ("에", "에서", "으로", "로", "에게", "한테", "이랑", "랑", "은", "는", "이", "가", "을", "를", "의")
ONE_CHAR_GLYPHS = "봄솔별율결담온겸찬휘"
ONE_CHAR_PAT = re.compile(r"(?<![가-힣])[" + ONE_CHAR_GLYPHS + r"](?:이|아|야|이가|이는|이랑|이한테|한테|이를|이도)(?![가-힣])")


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def per(rows, spec):
    return {k: v for k, v in spec.items()}


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True, type=Path)
    ap.add_argument("--stress", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--prev-rep", type=Path, help="이전 버전 대표 파일(변경 범위 집계용)")
    ap.add_argument("--prev-stress", type=Path)
    a = ap.parse_args()
    rep, st = load(a.rep), load(a.stress)
    ids = defaultdict(list)
    out = {}

    # ---- 교차 중복(대표 <-> 스트레스) ----
    def keyed(rows):
        return {r["id"]: r for r in rows}
    R, S = keyed(rep), keyed(st)
    n_rep, n_st = {V.norm(r["text"]): r["id"] for r in rep}, {V.norm(r["text"]): r["id"] for r in st}
    exact_cross = [(n_rep[k], n_st[k]) for k in n_rep if k in n_st]
    sk_rep = defaultdict(list)
    for r in rep:
        sk_rep[V.skeleton(r["text"], r["spans"])].append(r["id"])
    sk_cross = [(sk_rep[V.skeleton(r["text"], r["spans"])][0], r["id"]) for r in st if V.skeleton(r["text"], r["spans"]) in sk_rep]
    grams_r = {r["id"]: V.grams(V.skeleton(r["text"], r["spans"])) for r in rep}
    idx = defaultdict(list)
    for rid, g in grams_r.items():
        for x in g:
            idx[x].append(rid)
    sim_cross = []
    for r in st:
        g = V.grams(V.skeleton(r["text"], r["spans"]))
        if not g:
            continue
        share = Counter()
        for x in g:
            for o in idx.get(x, ()):
                share[o] += 1
        for o, c in share.items():
            if c >= 0.6 * len(g) and len(g & grams_r[o]) / len(g | grams_r[o]) >= 0.6:
                sim_cross.append((o, r["id"]))
                break
    pv = lambda rows: {s["value"] for r in rows for s in r["spans"] if s["label"] in ("PERSON", "PERSON_ALIAS")}
    name_cross = pv(rep) & pv(st)
    out["cross_set"] = {"exact_or_normalized_sentence_pairs": len(exact_cross), "skeleton_equal_pairs": len(sk_cross), "skeleton_similar_4gram_ge_0.6_rows": len(sim_cross),
                        "person_values_in_both_sets": len(name_cross), "rep_person_values": len(pv(rep)), "stress_person_values": len(pv(st))}
    ids["CROSS_SENTENCE_PAIRS"] = [f"{x}|{y}" for x, y in exact_cross]
    ids["CROSS_SKELETON_EQUAL_PAIRS"] = [f"{x}|{y}" for x, y in sk_cross]
    ids["CROSS_SKELETON_SIMILAR_PAIRS"] = [f"{x}|{y}" for x, y in sim_cross]
    ids["CROSS_SHARED_PERSON_VALUE_ROWS"] = [r["id"] for r in rep + st if any(s["value"] in name_cross for s in r["spans"] if s["label"] in ("PERSON", "PERSON_ALIAS"))]

    # ---- 세트 안 근접 중복 쌍(id 쌍) ----
    for tag, rows in (("REP", rep), ("STRESS", st)):
        gs = {r["id"]: V.grams(V.skeleton(r["text"], r["spans"])) for r in rows}
        ix = defaultdict(list)
        for rid, g in gs.items():
            for x in g:
                ix[x].append(rid)
        pairs = set()
        for rid, g in gs.items():
            share = Counter()
            for x in g:
                for o in ix[x]:
                    if o != rid:
                        share[o] += 1
            for o, c in share.items():
                if c >= 0.7 * len(g) and len(g & gs[o]) / len(g | gs[o]) >= 0.7:
                    pairs.add(tuple(sorted((rid, o))))
        ids[f"{tag}_NEAR_DUPLICATE_PAIRS"] = [f"{x}|{y}" for x, y in sorted(pairs)]

    # ---- 라벨 형식 이상 후보 ----
    for tag, rows in (("REP", rep), ("STRESS", st)):
        for r in rows:
            t = r["text"]
            for s in r["spans"]:
                v, lab = s["value"], s["label"]
                if lab == "ACCOUNT" and not re.fullmatch(r"[\d\- ]+", v):
                    ids[f"{tag}_ACCOUNT_VALUE_NOT_DIGITS"].append(r["id"])
                if lab == "PHONE" and not re.fullmatch(r"01[016789][-. ]?\d{3,4}[-. ]?\d{4}", v):
                    ids[f"{tag}_PHONE_VALUE_NOT_MOBILE_PATTERN"].append(r["id"])
                if lab == "EMAIL" and not re.fullmatch(r"[\w.+-]+@[\w-]+\.[\w.-]+", v):
                    ids[f"{tag}_EMAIL_VALUE_BAD"].append(r["id"])
                if lab == "ADDRESS" and not re.search(r"\d", v):
                    ids[f"{tag}_ADDRESS_NO_DIGIT_MAYBE_LOCATION"].append(r["id"])
                if lab == "ADDRESS" and v.endswith(JOSA_TAIL):
                    ids[f"{tag}_ADDRESS_ENDS_WITH_JOSA"].append(r["id"])
                if lab in ("PERSON", "PERSON_ALIAS"):
                    if v in KIN:
                        ids[f"{tag}_PERSON_VALUE_IS_KIN_TERM"].append(r["id"])
                    if v.endswith(HONORIFIC_END) and lab == "PERSON":
                        ids[f"{tag}_PERSON_VALUE_ENDS_WITH_HONORIFIC"].append(r["id"])
                    if lab == "PERSON" and " " in v:
                        ids[f"{tag}_PERSON_VALUE_HAS_SPACE"].append(r["id"])
                    if lab == "PERSON_ALIAS" and not (" " in v or v.endswith(tuple(KIN)) or len(v) >= 2):
                        ids[f"{tag}_ALIAS_SHAPE"].append(r["id"])
                    if lab == "PERSON" and v.endswith(JOSA_TAIL) and len(v) >= 3 and s["end"] < len(t) and False:
                        pass
            if ONE_CHAR_PAT.search(t):
                ids[f"{tag}_POSSIBLE_ONE_CHAR_NAME_PATTERN"].append(r["id"])
            if r["is_negative"] and any(k in t for k in KIN) and False:
                pass

    # ---- 미라벨 이름 의심의 성격(설계 짝 D1/D1P 여부) ----
    pv_st_by_type = defaultdict(set)
    for r in st:
        for s in r["spans"]:
            if s["label"] == "PERSON":
                pv_st_by_type[r.get("primary_stress_type")].add(s["value"])
    d1p_vals = pv_st_by_type.get("D1P", set())
    unl = defaultdict(Counter)
    for tag, rows, allvals in (("REP", rep, pv(rep)), ("STRESS", st, pv(st))):
        for r in rows:
            covered = [False] * len(r["text"])
            for s in r["spans"]:
                for i in range(s["start"], s["end"]):
                    covered[i] = True
            hits = []
            for v in allvals:
                if len(v) < 2:
                    continue
                i = r["text"].find(v)
                while i >= 0:
                    if not any(covered[i:i + len(v)]):
                        hits.append(v)
                        break
                    i = r["text"].find(v, i + 1)
            if hits:
                kind = "D1_pair_by_design" if (tag == "STRESS" and r.get("primary_stress_type") == "D1" and all(h in d1p_vals for h in hits)) else "other"
                unl[tag][(r.get("primary_stress_type") or r["category"] if tag == "STRESS" else "all", kind, "len3+" if any(len(h) >= 3 for h in hits) else "len2")] += 1
                ids[f"{tag}_UNLABELED_NAME_SUSPECT_{kind.upper()}"].append(r["id"])
                if any(len(h) >= 3 for h in hits):
                    ids[f"{tag}_UNLABELED_NAME_SUSPECT_LEN3PLUS"].append(r["id"])
    out["unlabeled_name_suspects"] = {t: {"|".join(k): v for k, v in c.items()} for t, c in unl.items()}
    # D1 <-> D1P 짝 현황: D1P 값이 D1 문장에 나타나는 비율(집계만)
    d1_rows = [r for r in st if r.get("primary_stress_type") == "D1"]
    d1p_in_d1 = {v for v in d1p_vals if any(v in r["text"] for r in d1_rows)}
    d1_with_pair = sum(1 for r in d1_rows if any(v in r["text"] for v in d1p_vals))
    out["d1_d1p_pairing"] = {"d1p_values": len(d1p_vals), "d1p_values_found_in_some_d1_sentence": len(d1p_in_d1), "d1_sentences_with_a_d1p_value": d1_with_pair, "d1_sentences": len(d1_rows)}

    # ---- 규격 준수 항목(집계) ----
    L = lambda rows: sum(1 for r in rows if len(r["text"]) >= 60)
    pl = lambda rows: [len(s["value"]) for r in rows for s in r["spans"] if s["label"] in ("PERSON", "PERSON_ALIAS")]
    lr = pl(rep)
    out["spec_checks"] = {
        "rep_len_ge_60_share_pct": pct(L(rep), len(rep)), "stress_len_ge_60_share_pct": pct(L(st), len(st)),
        "rep_person_2char_pct": pct(sum(1 for x in lr if x == 2), len(lr)), "rep_person_3char_pct": pct(sum(1 for x in lr if x == 3), len(lr)),
        "rep_person_4plus_pct": pct(sum(1 for x in lr if x >= 4), len(lr)),
        "rep_alias_pct": pct(sum(1 for r in rep for s in r["spans"] if s["label"] == "PERSON_ALIAS"), len(lr)),
        "rep_multi_person_pct_of_person_sentences": pct(sum(1 for r in rep if sum(1 for s in r["spans"] if s["label"] in ("PERSON", "PERSON_ALIAS")) >= 2), 950),
        "rep_id_prefix_ok": all(re.fullmatch(r"R9R-\d{4}", r["id"]) for r in rep), "stress_id_prefix_ok": all(re.fullmatch(r"R9S-\d{4}", r["id"]) for r in st),
        "rep_split_set_ok": all(r.get("split_set") == "representative" for r in rep), "stress_split_set_ok": all(r.get("split_set") == "stress" for r in st),
        "stress_has_primary_and_list": all(r.get("primary_stress_type") and isinstance(r.get("stress_types"), list) and r["primary_stress_type"] in r["stress_types"] for r in st),
        "sources": dict(Counter(r["source"] for r in rep + st)), "batches_rep": len({r.get("batch") for r in rep}), "batches_stress": len({r.get("batch") for r in st}),
        "template_id_present_rep": sum(1 for r in rep if r.get("template_id")), "template_id_present_stress": sum(1 for r in st if r.get("template_id")),
        "context_entities_rows": {"rep": sum(1 for r in rep if r.get("context_entities")), "stress": sum(1 for r in st if r.get("context_entities"))},
        "negative_with_context_entities_stress_A3N": sum(1 for r in st if r.get("primary_stress_type") == "A3N" and r.get("context_entities")),
        "stress_person_len_by_type": {t: dict(sorted(Counter(len(s["value"]) for r in st if r.get("primary_stress_type") == t for s in r["spans"] if s["label"] == "PERSON").items())) for t in ("A1", "A3", "B1", "B2", "C1", "C2", "D3", "D1P")},
    }
    # 문형 편중: 문장 끝 3글자 상위 1위 점유율, 첫 단어(공백 기준) 상위 1위 점유율(문자열은 출력하지 않음)
    for tag, rows in (("rep", rep), ("stress", st)):
        end = Counter(V.norm(r["text"])[-3:] for r in rows)
        first = Counter(r["text"].split()[0] if r["text"].split() else "" for r in rows)
        out["spec_checks"][f"{tag}_top1_sentence_ending3_share_pct"] = pct(end.most_common(1)[0][1], len(rows))
        out["spec_checks"][f"{tag}_top1_first_token_share_pct"] = pct(first.most_common(1)[0][1], len(rows))
        out["spec_checks"][f"{tag}_distinct_sentence_endings3"] = len(end)

    # ---- 기존 평가 코드 호환(정답=예측 만점): 모델 없이 공통 형식 + 정책 처리 확인 ----
    import evaluate_r8_new180 as E
    comp = {}
    for tag, rows in (("rep", rep), ("stress", st)):
        rows_ = [r for r in rows if not r.get("exclude")]
        oracle = lambda r: [(s["mapped_kind"], s["start"], s["end"], 0.95) for s in r["spans"]]
        s = E.summarize(rows_, E.analyze(oracle, rows_))
        comp[tag] = {"oracle_person_f1": s["person_family"]["exact"]["f1"], "oracle_exposure": s["person_family"]["total_exposure_rate"],
                     "oracle_protected_all": s["protected_all_kinds"]["rate"], "account_policy": s["account_block_policy"]}
    out["eval_code_compat"] = comp

    # ---- r9-1.1 PERSON_ALIAS 정책 점검(별명 자체만, 가족 호칭은 구간 밖) ----
    KIN_ALL = KIN | {"외삼촌", "외숙모", "큰이모", "작은이모", "큰엄마", "작은엄마"}
    KIN_RX = "|".join(sorted(KIN_ALL, key=len, reverse=True))
    pol = {}
    for tag, rows in (("rep", rep), ("stress", st)):
        al = [(r, s_) for r in rows for s_ in r["spans"] if s_["label"] == "PERSON_ALIAS"]
        per = [(r, s_) for r in rows for s_ in r["spans"] if s_["label"] == "PERSON"]

        def kin_inside(v):
            toks = v.split()
            return v in KIN_ALL or any(t in KIN_ALL for t in toks) or any(v.endswith(k) and len(v) > len(k) for k in KIN_ALL)

        for r, s_ in al:
            if kin_inside(s_["value"]):
                ids[f"{tag.upper()}_ALIAS_CONTAINS_KIN_TERM"].append(r["id"])
        for r, s_ in per:
            if kin_inside(s_["value"]):
                ids[f"{tag.upper()}_PERSON_VALUE_CONTAINS_KIN_TERM"].append(r["id"])
        adj = []
        for r, s_ in per:
            if V.kin_follows(r["text"], s_["end"]):
                adj.append(r["id"])
        for rid in adj:
            ids[f"{tag.upper()}_PERSON_FOLLOWED_BY_KIN_TERM"].append(rid)
        pol[tag] = {"alias_spans": len(al), "alias_share_of_person_family_pct": pct(len(al), len(al) + len(per)),
                    "alias_contains_kin": len(set(ids[f"{tag.upper()}_ALIAS_CONTAINS_KIN_TERM"])),
                    "person_contains_kin": len(set(ids[f"{tag.upper()}_PERSON_VALUE_CONTAINS_KIN_TERM"])),
                    "person_spans_followed_by_kin_term": len(adj),
                    "alias_len_dist": dict(sorted(Counter(len(s_["value"]) for _, s_ in al).items())),
                    "alias_sentences": len({r["id"] for r, _ in al}), "alias_distinct_values": len({s_["value"] for _, s_ in al}),
                    "alias_value_equals_a_person_value": len({s_["value"] for _, s_ in al} & {s_["value"] for _, s_ in per})}
    out["policy_r9_1_1"] = pol
    # ---- 이전 버전 대비 변경 범위(id 기준, 원문 미출력) ----
    if a.prev_rep and a.prev_stress:
        diff = {}
        for tag, cur, prev in (("rep", rep, load(a.prev_rep)), ("stress", st, load(a.prev_stress))):
            pc = {r["id"]: r for r in prev}
            cc = {r["id"]: r for r in cur}
            same_text = [i for i in cc if i in pc and cc[i]["text"] == pc[i]["text"]]
            same_all = [i for i in same_text if cc[i]["spans"] == pc[i]["spans"] and cc[i].get("primary_stress_type") == pc[i].get("primary_stress_type")]
            sa = set(same_all)
            changed_spans_only = [i for i in same_text if i not in sa]
            changed_text = [i for i in cc if i in pc and cc[i]["text"] != pc[i]["text"]]
            diff[tag] = {"rows_v1": len(pc), "rows_v1_1": len(cc), "ids_added": len(set(cc) - set(pc)), "ids_removed": len(set(pc) - set(cc)),
                         "unchanged": len(same_all), "text_unchanged_labels_changed": len(changed_spans_only), "text_changed": len(changed_text)}
            ids[f"DIFF_{tag.upper()}_TEXT_UNCHANGED_LABELS_CHANGED"] = changed_spans_only
            ids[f"DIFF_{tag.upper()}_TEXT_CHANGED"] = changed_text
            kinds = Counter()
            for i in changed_spans_only:
                la = sorted(x["label"] for x in pc[i]["spans"])
                lb = sorted(x["label"] for x in cc[i]["spans"])
                kinds["+".join(la) + " -> " + "+".join(lb)] += 1
            diff[tag]["label_change_patterns"] = dict(kinds.most_common(8))
            diff[tag]["changed_text_alias_before"] = sum(1 for i in changed_text if any(x["label"] == "PERSON_ALIAS" for x in pc[i]["spans"]))
            diff[tag]["changed_text_alias_after"] = sum(1 for i in changed_text if any(x["label"] == "PERSON_ALIAS" for x in cc[i]["spans"]))
        out["diff_vs_prev"] = diff
    out["id_list_counts"] = {k: len(dict.fromkeys(v)) for k, v in sorted(ids.items())}
    a.out.write_text(json.dumps({"summary": out, "ids": {k: list(dict.fromkeys(v)) for k, v in ids.items()}}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
