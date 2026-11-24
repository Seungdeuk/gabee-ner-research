"""R11 데이터 생성기: 학습 D'(대조쌍 800행), Validation(600행), Development(1,920행). 사전 등록 NER_R11_PREREGISTRATION.md §4·§5.

- 틀은 `r11_frames/{train,val,dev}/`(역할마다 **다른 작성 에이전트**, id 접두어 분리). 단어 목록은 `r11_frames/words/{train,val,dev}.json`(역할별 작성자).
- 이름은 이름 부분(성 제외 마지막 2~3글자) 해시 `sha256("r11:"+키) mod 10`: 0 → dev, 1 → val, 2 → 독립 테스트 후보 풀(쓰지 않음), 3~9 → 학습.
- 명사형 이름 단어는 역할별로 서로 겹치지 않고(정확·부분 문자열 포함 0, 우선순위 dev > val > train), dev·val 단어는 R9·R10 학습에서 쓰이지 않은 것만 쓴다.
- R9 봉인 평가셋은 읽지 않는다(Guards10 상속). 문장·문장 틀(skeleton)은 R9·R10 학습·개발 데이터와도 역할 사이에서도 겹치지 않아야 한다(한 틀은 전체에서 1회만).
산출: --out-dir 의 r11_Dprime_train.jsonl, r11_val.jsonl, r11_dev.jsonl, r11_build_report.json(건수만, 원문 없음)
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_r8_data as r8
import build_r9_data as b9
import build_r10_data as b10

FR_SRC = Path(r"C:\Users\<user>\venvs\ner-runs\data_r11\src_frames")  # 원본 틀·단어 목록은 저장소 밖(비공개), 저장소에는 명세와 해시만 둔다
FR = FR_SRC if FR_SRC.exists() else HERE / "r11_frames"
OUT_DEFAULT = Path(r"C:\Users\<user>\venvs\ner-runs\data_r11")
R10_DIR = b10.OUT_DEFAULT
HANGUL_ONLY = re.compile(r"^[가-힣]+$")

# 역할별 틀 파일(stem)
FRAME_STEMS = {
    "train": ["pairs_a", "pairs_b", "pairs_c"],
    "val": ["pairs_a", "pairs_b"],
    "dev": ["pairs_a", "pairs_b", "pairs_c", "b1", "c2", "msg_new", "longname", "existing", "neg_orgloc", "neg_family", "neg_d2", "neg_general"],
}
# 목표 행 수
TARGET_TRAIN_PAIRS = 400
TARGET_VAL_PAIRS = 300
TARGET_DEV = {"pairs": 400, "b1": 150, "c2": 120, "msg_new": 150, "longname": 120, "existing": 100,
              "neg_orgloc": 120, "neg_family": 120, "neg_d2": 120, "neg_general": 120}
DEV_TYPE = {"b1": "B1", "c2": "C2", "msg_new": "NEWNAME", "longname": "LONGNAME", "existing": "EXISTING",
            "neg_orgloc": "NEG_ORGLOC", "neg_family": "NEG_FAMILY", "neg_d2": "NEG_D2", "neg_general": "NEG_GENERAL"}
DEV_NOISE_BUCKET = {"b1": "B", "c2": "C", "msg_new": "N", "longname": "N", "existing": "N",
                    "neg_orgloc": "G", "neg_family": "G", "neg_d2": "G", "neg_general": "G"}


# 기존 이름 음절 집합이 작아(R9·R10 이 대부분 소진) 역할별 미사용 이름이 모자라므로 흔한 이름 음절을 더한다(합성 이름, 실존 인물 아님)
EXTRA_A = "강건경광규기나다대도동래루리마명모미민바범보빈사상새서석선설성세소솔수숙순승시신아안애양어언여연영예오완용우운원위유윤은이인일자장재정제조종주준중지진찬창채천철초충태하한해헌현형호홍화환훈휘희"
EXTRA_B = "경근나노누다담도동라란랑래량려령로록룡리림마만매명모목문미민바반별보부비빈사산상서석선설성소솔송수숙순슬승시식신아안애야양여연열영예오완요용우운원월위유율은응이익인자장재정제종주준중지진찬창채철초추춘충태하한해행헌혁현형혜호홍화환훈휘희"
b9.A_SYL = b9.A_SYL + "".join(c for c in dict.fromkeys(EXTRA_A) if c not in b9.A_SYL)
b9.B_SYL = b9.B_SYL + "".join(c for c in dict.fromkeys(EXTRA_B) if c not in b9.B_SYL)
EXTRA_NICK = ["콩", "솜", "몽", "뭉", "꿍", "뚜", "깜", "쭈", "찌", "뽀", "토", "두", "루", "누", "푸", "쿠", "동글", "말랑", "보들", "포동", "방실", "토실", "몽글", "초롱", "새콤", "달콤", "고소", "오동", "도톰", "아롱"]
b9.NICK_STEMS = list(b9.NICK_STEMS) + [x for x in EXTRA_NICK if x not in b9.NICK_STEMS]


class NearIndex:
    """문장 틀 4-gram Jaccard 근접 중복 색인(기존 학습 데이터 + 이미 채택한 R11 행)."""

    def __init__(self):
        self.idx, self.grams = defaultdict(list), []

    def add(self, sk):
        g = r8.ngrams(sk)
        if g:
            self.grams.append(g)
            for x in g:
                self.idx[x].append(len(self.grams) - 1)

    def hit(self, sk, thr=0.7):
        g = r8.ngrams(sk)
        if not g:
            return False
        share = Counter()
        for x in g:
            for i in self.idx.get(x, ()):
                share[i] += 1
        return any(len(g & self.grams[i]) / len(g | self.grams[i]) >= thr for i, _ in share.most_common(30))


NEAR = NearIndex()


def key_of(name):
    return name if len(name) <= 2 else name[-2:]


def role_of(name):
    h = int(hashlib.sha256(("r11:" + key_of(name)).encode()).hexdigest(), 16) % 10
    return "dev" if h == 0 else "val" if h == 1 else "test" if h == 2 else "train"


def overlap(a, b):
    return a == b or (len(a) >= 2 and len(b) >= 2 and (a in b or b in a))


class Guards11(b10.Guards10):
    """R10 학습·개발 데이터의 이름·문장·틀도 기존 데이터로 취급한다."""

    def __init__(self):
        super().__init__()
        self.seen_words = set()
        for f in ("r10_full_train", "r10_val_extra", "r10_dev"):
            p = R10_DIR / f"{f}.jsonl"
            for r in r8.read_jsonl(p):
                self.existing_sents.add(r8.norm(r["text"]))
                sp = [{"start": s["start"], "end": s["end"]} for s in r["spans"]]
                self.existing_skels.add(r8.skeleton_of(r["text"], sp))
                if r.get("pair_word"):
                    self.seen_words.add(r["pair_word"])
                for s in r["spans"]:
                    if s.get("label") in ("PERSON", "PERSON_ALIAS"):
                        v = r["text"][s["start"] : s["end"]]
                        self.r9_names.add(v)
                        self.r9_given |= b10.given_parts(v)
        self.seen_words |= self.r9_names | self.r9_given | set(b9.NAMEWORDS_NOUN) | set(b10.NAMEWORDS_R10)


class R11Names:
    """역할(train/val/dev)별 이름 풀. b9.Names 와 같은 인터페이스(pick, split, nature, objects)."""

    CAP = 4

    def __init__(self, g, rng):
        self.used, self.idx = Counter(), Counter()
        self.fallbacks = Counter()
        self.split = {}
        orig = b9.hbucket
        built = {}
        for mode in ("dev", "val"):
            # b9.Names 는 hbucket(x) < 16 을 val 분할로 쓴다. 이름 키가 mode 역할이면 val, 아니면 train 으로 보내 같은 코드를 재사용
            b9.hbucket = lambda key, mod=100, _m=mode: ((0 if role_of(key) == _m else 99) if mod == 100 else orig(key, mod))
            try:
                n = b9.Names(g, random.Random(rng.random()))
                n._build()
            finally:
                b9.hbucket = orig
            built[mode] = n
        kinds = [k for k in built["dev"].split if k != "nameword"]
        for k in kinds:
            self.split[k] = {
                "dev": built["dev"].split[k]["val"],
                "val": built["val"].split[k]["val"],
                "train": [x for x in built["dev"].split[k]["train"] if role_of(x) == "train"],
            }
        self.split["nameword"] = {"train": [], "val": [], "dev": []}
        locs = [x for x in b9.synth.CITIES + b9.synth.STATIONS + b9.synth.LANDMARKS if x not in g.dev_exact]
        orgs = [p + s for P, S in ((b9.synth.HOSP_PREFIX, b9.synth.HOSP_SUFFIX), (b9.synth.SCHOOL_PREFIX, b9.synth.SCHOOL_SUFFIX),
                                   (b9.synth.COMPANY_PREFIX, b9.synth.COMPANY_SUFFIX)) for p in P for s in S]
        orgs = [x for x in orgs if x not in g.dev_exact]
        self.cfg = {"locs": {r: locs for r in ("train", "val", "dev")}, "orgs": {r: orgs for r in ("train", "val", "dev")}}
        nat = [w for w in dict.fromkeys(b9.NATURE) if w not in g.dev_words and w not in g.dev_exact]
        obj = [w for w in dict.fromkeys(b9.OBJECTS) if w not in g.dev_words and w not in g.dev_exact]
        self.nature = {r: nat for r in ("train", "val", "dev")}
        self.objects = {r: obj for r in ("train", "val", "dev")}

    def pick(self, kind, role):
        pool = self.split[kind][role]
        for fb in ("full3", "given2"):
            if (len(pool) < 20 and kind != "nickname" or not pool) and kind != fb:  # 20개 미만 풀은 한두 이름이 반복되므로 쓰지 않는다
                self.fallbacks[(kind, role)] += 1  # 역할 분할 후 빈 풀은 일반 이름 풀로 대체(보고서에 횟수 기록)
                kind = fb
                pool = self.split[kind][role]
        if not pool:
            raise RuntimeError(f"이름 풀 비어 있음: {kind}/{role}")
        for _ in range(len(pool)):
            n = pool[self.idx[(kind, role)] % len(pool)]
            self.idx[(kind, role)] += 1
            if self.used[n] < self.CAP:
                self.used[n] += 1
                return n
        n = pool[self.idx[(kind, role)] % len(pool)]
        self.idx[(kind, role)] += 1
        self.used[n] += 1
        return n


def load_words(g):
    """역할별 단어 목록: 형식 검사, 기존 데이터 단어 제외(dev·val), 역할 간 포함 관계 0(우선순위 dev > val > train)."""
    raw = {}
    for r in ("train", "val", "dev"):
        lst = json.loads((FR / "words" / f"{r}.json").read_text(encoding="utf-8"))
        ex = FR / "words" / f"{r}_extra.json"
        if ex.exists():
            lst += json.loads(ex.read_text(encoding="utf-8"))
        raw[r] = lst
    stats = {}

    def clean(lst):
        out = []
        for w in dict.fromkeys(x.strip() for x in lst):
            if 2 <= len(w) <= 3 and HANGUL_ONLY.match(w) and g.name_ok(w, allow_common=True):
                out.append(w)
        return out

    c = {r: clean(raw[r]) for r in raw}
    stats["raw"] = {r: len(raw[r]) for r in raw}
    stats["format_ok"] = {r: len(c[r]) for r in c}
    for r in ("dev", "val"):  # 기준선 모델이 학습에서 본 적 없는 단어만
        c[r] = [w for w in c[r] if w not in g.seen_words and w not in g.r9_given]
    stats["after_seen_filter"] = {r: len(c[r]) for r in c}
    dev = c["dev"]
    val = [w for w in c["val"] if not any(overlap(w, d) for d in dev)]
    train = [w for w in c["train"] if not any(overlap(w, d) for d in dev + val)]
    # 이름 풀과 겹치지 않게
    stats["final"] = {"dev": len(dev), "val": len(val), "train": len(train)}
    return {"dev": dev, "val": val, "train": train}, stats


def load_role_frames(role):
    b9.FRAMES = FR / role
    b9.FRAME_FILES = set(FRAME_STEMS[role])
    frames, prob = b9.load_frames()
    raw = {}
    for stem in FRAME_STEMS[role]:
        p = FR / role / f"{stem}.json"
        if p.exists():
            for it in json.loads(p.read_text(encoding="utf-8")):
                raw[it["id"]] = it
    for fr in frames:
        it = raw.get(fr["id"])
        fr["pair_id"], fr["role"] = (it.get("pair_id"), it.get("role")) if it else (None, None)
        fr["author"] = f"{role}/{fr['file']}"
    return frames, prob


def finish(fr, segs, rng, bucket, g, seen_norm, seen_skel, extra):
    segs = b9.apply_noise(segs, rng, *b10.NOISE[bucket])
    text, spans, rule = b9.finalize(segs)
    key = r8.norm(text)
    sk = r8.skeleton_of(text, spans)
    if key in seen_norm:
        g.rejected["dup_sentence"] += 1
        return None
    if sk in seen_skel:
        g.rejected["dup_skeleton_cross"] += 1
        return None
    if NEAR.hit(sk):
        g.rejected["near_dup_4gram"] += 1
        return None
    if not g.text_ok(text, spans):
        return None
    row = {"id": "", "text": text, "spans": spans, "source": "synth", "template_id": fr["id"],
           "category": f"r11-{'-'.join(fr['tags'][:2]) or fr['file']}", "augment": True, "r11": True,
           "frame_file": fr["author"], "frame_tag": ",".join(fr["tags"]), "_key": key, "_sk": sk, **extra}
    if rule:
        row["rule_spans"] = rule
    return row


def build_pairs(frames, quota, role, words, names, cfg, g, rng, seen_norm, seen_skel, n2_kind):
    pairs = defaultdict(dict)
    for f in frames:
        if f.get("pair_id"):
            pairs[f["pair_id"]][f["role"]] = f
    ids = sorted(k for k, v in pairs.items() if {"person", "noun"} <= set(v))
    rng.shuffle(ids)
    wlist = list(words)
    rng.shuffle(wlist)
    rows, made, wi, tried = [], 0, 0, 0
    word_use = Counter()
    for pid in ids:
        if made >= quota:
            break
        tried += 1
        word = None
        for _ in range(len(wlist)):  # 한 단어 최대 2쌍
            w = wlist[wi % len(wlist)]
            wi += 1
            if word_use[w] < 2:
                word = w
                break
        if word is None:
            break
        pr = pairs[pid]
        pair_rows = []
        for r_ in ("person", "noun"):
            fr = pr[r_]

            def roles(k, num, _w=word, _id=fr["id"]):
                if k in ("NW", "WN"):
                    return _w
                if k == "N" and n2_kind:
                    return names.pick(n2_kind, role)
                return b9.name_roles("N", "msg_person", names, role, rng, _id)(k, num)

            segs, _ = b9.fill(fr, cfg, names, role, rng, g, roles)
            row = finish(fr, segs, rng, "D", g, seen_norm, seen_skel,
                         {"pair_id": f"{role}-{pid}", "pair_role": r_, "pair_word": word})
            if row is None:
                pair_rows = None
                break
            pair_rows.append(row)
        if not pair_rows or len(pair_rows) != 2:
            continue
        p_row, n_row = pair_rows
        person_ok = any(p_row["text"][x["start"] : x["end"]] == word for x in p_row["spans"])
        noun_ok = word in n_row["text"] and not any(n_row["text"][x["start"] : x["end"]] == word for x in n_row["spans"])
        if p_row["_key"] == n_row["_key"]:
            g.rejected["pair_identical_text"] += 1  # 사람·명사 문장이 같으면 같은 문장에 다른 정답이 붙는 모순
            continue
        if p_row["_sk"] == n_row["_sk"] or not (person_ok and noun_ok):
            g.rejected["pair_invalid"] += 1
            continue
        for r_ in pair_rows:
            seen_norm.add(r_["_key"])
            seen_skel.add(r_["_sk"])
            NEAR.add(r_["_sk"])
        word_use[word] += 1
        rows += pair_rows
        made += 1
    return rows, {"pairs_made": made, "pairs_target": quota, "frames_pairs_available": len(ids), "distinct_words": len(word_use),
                  "max_word_pairs": max(word_use.values()) if word_use else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    b9.EXISTING_DIR = b10.R9_DIR
    g = Guards11()
    rng = random.Random(a.seed)
    names = R11Names(g, rng)
    words, wstats = load_words(g)
    cfg = names.cfg
    seen_norm, seen_skel = set(), set()
    for f in ("baseline_train", "synth_train", "r9_Ffull_train"):
        for r in r8.read_jsonl(b10.R9_DIR / f"{f}.jsonl"):
            NEAR.add(r8.skeleton_of(r["text"], [{"start": x["start"], "end": x["end"]} for x in r["spans"]]))
    for f in ("r10_full_train", "r10_val_extra"):
        for r in r8.read_jsonl(R10_DIR / f"{f}.jsonl"):
            NEAR.add(r8.skeleton_of(r["text"], [{"start": x["start"], "end": x["end"]} for x in r["spans"]]))
    report = {"seed": a.seed, "policy": "r9-1.1", "sealed_access": "없음(R9 봉인 평가셋을 읽지 않음)", "words": wstats, "name_pool_sizes": {
        k: {r: len(v[r]) for r in ("train", "val", "dev")} for k, v in names.split.items()}}

    # --- 학습 D' ---
    tf, tprob = load_role_frames("train")
    train_rows, tinfo = build_pairs(tf, TARGET_TRAIN_PAIRS, "train", words["train"], names, cfg, g, rng, seen_norm, seen_skel, "given2")
    report["train"] = {"frames": len(tf), "frame_problems": tprob, **tinfo}
    # --- Validation ---
    vf, vprob = load_role_frames("val")
    val_rows, vinfo = build_pairs(vf, TARGET_VAL_PAIRS, "val", words["val"], names, cfg, g, rng, seen_norm, seen_skel, None)
    report["val"] = {"frames": len(vf), "frame_problems": vprob, **vinfo}
    # --- Development ---
    df, dprob = load_role_frames("dev")
    dev_rows, dinfo = build_pairs([f for f in df if f["file"].startswith("pairs_")], TARGET_DEV["pairs"], "dev", words["dev"], names, cfg, g, rng,
                                  seen_norm, seen_skel, None)
    for r in dev_rows:
        r["type"] = "D1P" if r["pair_role"] == "person" else "D1"
    seen_names = []
    for f in ("baseline_train", "synth_train", "r9_Ffull_train"):
        for r in r8.read_jsonl(b10.R9_DIR / f"{f}.jsonl"):
            for s in r["spans"]:
                if s.get("label") == "PERSON":
                    v = r["text"][s["start"] : s["end"]]
                    if 2 <= len(v) <= 4:
                        seen_names.append(v)
    seen_names = sorted(set(seen_names))
    byfile = defaultdict(list)
    for f in df:
        if not f["file"].startswith("pairs_"):
            byfile[f["file"]].append(f)
    single_info = {}
    for stem, quota in TARGET_DEV.items():
        if stem == "pairs":
            continue
        fs = byfile[stem]
        rng.shuffle(fs)
        made = 0
        for fr in fs:
            if made >= quota:
                break
            tag = fr["tags"][0] if fr["tags"] else None
            if stem == "longname":
                def roles(k, num, _id=fr["id"]):
                    return names.pick("full4", "dev") if k == "N" else b9.name_roles("N", "msg_person", names, "dev", rng, _id)(k, num)
            elif stem == "existing":
                def roles(k, num, _id=fr["id"]):
                    return rng.choice(seen_names) if k == "N" else b9.name_roles("N", "msg_person", names, "dev", rng, _id)(k, num)
            elif stem == "b1":
                roles = b9.name_roles("B1", "err_B", names, "dev", rng, fr["id"])
            elif stem == "c2":
                roles = b9.name_roles("C2", "err_B", names, "dev", rng, fr["id"])
            else:
                roles = b9.name_roles(tag, "msg_person" if stem == "msg_new" else stem, names, "dev", rng, fr["id"])
            segs, _ = b9.fill(fr, cfg, names, "dev", rng, g, roles)
            row = finish(fr, segs, rng, DEV_NOISE_BUCKET[stem], g, seen_norm, seen_skel, {})
            if row is None:
                continue
            if stem in ("neg_orgloc", "neg_family", "neg_d2", "neg_general") and any(s["label"] in ("PERSON", "PERSON_ALIAS") for s in row["spans"]):
                g.rejected["neg_has_person"] += 1
                continue
            if stem == "longname" and not any(s["end"] - s["start"] >= 4 for s in row["spans"] if s["label"] == "PERSON"):
                g.rejected["longname_too_short"] += 1
                continue
            seen_norm.add(row["_key"])
            seen_skel.add(row["_sk"])
            NEAR.add(row["_sk"])
            row["type"] = DEV_TYPE[stem]
            dev_rows.append(row)
            made += 1
        single_info[stem] = {"frames": len(fs), "made": made, "target": quota}
    report["dev"] = {"frames": len(df), "frame_problems": dprob, **dinfo, "singles": single_info}

    # --- 마무리: id, 메타, 검증 ---
    ref_given = set(g.r9_given)
    for r in train_rows:
        for s in r["spans"]:
            if s["label"] in ("PERSON", "PERSON_ALIAS"):
                ref_given |= b10.given_parts(r["text"][s["start"] : s["end"]])
    out = {}
    for tag, rows, role in (("train", train_rows, "r11_Dprime_train"), ("val", val_rows, "r11_val"), ("dev", dev_rows, "r11_dev")):
        for i, r in enumerate(rows, 1):
            r["id"] = f"{role}-{i:05d}"
            r["split"] = role
            r["train_bucket"] = "r11_Dprime" if tag == "train" else role
            r["is_negative"] = not r["spans"]
            r.pop("_key", None), r.pop("_sk", None)
            if tag == "dev":
                r["name_seen_in_reference"] = [bool(b10.given_parts(r["text"][s["start"] : s["end"]]) & ref_given)
                                               for s in r["spans"] if s["label"] in ("PERSON", "PERSON_ALIAS")]
        out[role] = rows
    problems = {k: b9.validate(v, g) for k, v in out.items()}
    report["rows"] = {k: len(v) for k, v in out.items()}
    report["problems"] = {k: len(v) for k, v in problems.items()}
    report["problem_examples"] = {k: v[:5] for k, v in problems.items() if v}
    report["dev_by_type"] = dict(Counter(r["type"] for r in out["r11_dev"]))
    report["rejected(counts)"] = dict(g.rejected)
    report["name_pool_fallbacks"] = {f"{k[0]}/{k[1]}": v for k, v in names.fallbacks.items()}
    if not a.dry_run:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        for k, v in out.items():
            (a.out_dir / f"{k}.jsonl").write_text(
                "".join(json.dumps({kk: (sorted(vv) if isinstance(vv, set) else vv) for kk, vv in r.items()}, ensure_ascii=False) + "\n" for r in v),
                encoding="utf-8")
        (a.out_dir / "r11_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1)[:3500])


if __name__ == "__main__":
    main()
