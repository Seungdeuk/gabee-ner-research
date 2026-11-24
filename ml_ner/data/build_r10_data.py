"""R10 신규 학습 데이터 생성기(정책 r9-1.1 유지). 사전 등록 NER_R10_PREREGISTRATION.md §3.

R9 생성기(`build_r9_data.py`)의 슬롯 채우기·조사·잡음·검증 함수를 재사용하되 다음이 다르다.
- **R9 봉인 평가셋과 new180 은 읽지 않는다**(Guards10 이 봉인 경로를 건드리지 않는다). 개발 평가셋·기존 학습 데이터·R9 신규 데이터와의 겹침만 거른다.
- **신규 이름**: R9 학습·검증 데이터에 쓰인 이름 부분(given)과 R10 개발셋 예약 구간(이름 해시 mod 5 == 0)은 R10 학습에서 쓰지 않는다.
- D1P 는 같은 단어를 사람 문장(`NW1`)과 명사 문장(`WN1`)에 **짝으로** 채운다(1:1).
산출(`--out-dir`): r10_D/B/C/N/G_train, r10_full_train(전부), r10_val_extra(모니터링 전용), r10_build_report.json.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_r8_data as r8
import build_r9_data as b9

FRAMES = HERE / "r10_frames"
b9.FRAMES = FRAMES
PAIR_FILES = ["pairs_a", "pairs_b", "pairs_c"]
SINGLE_FILES = ["b1", "c2", "msg_new", "neg_guard"]
b9.FRAME_FILES = set(PAIR_FILES + SINGLE_FILES)
R9_DIR = Path(r"C:\Users\<user>\venvs\ner-runs\data_r9")
OUT_DEFAULT = Path(r"C:\Users\<user>\venvs\ner-runs\data_r10")
R9_NEW_FILES = ["r9_Ffull_train", "r9_val_extra"]
BASE_FILES = ["baseline_train", "baseline_val", "synth_train", "synth_val"]

# 버킷별 훈련 행 목표(합 2,100). 검증(모니터링)은 VAL_FRACTION 만큼 따로 만든다.
TARGET = {"D": 800, "B": 420, "C": 320, "N": 400, "G": 160}
NOISE = {  # (앞 잡음, 뒤 잡음, 띄어쓰기, 오타) 확률
    "D": (0.15, 0.20, 0.08, 0.05),
    "B": (0.25, 0.30, 0.10, 0.08),
    "C": (0.25, 0.30, 0.10, 0.08),
    "N": (0.30, 0.35, 0.12, 0.10),
    "G": (0.25, 0.30, 0.10, 0.08),
}
VAL_FRACTION = 0.18



# R10 확장 이름 단어(일반 명사이면서 사람 이름으로도 쓰이는 단어). R9 목록과 겹치면 걸러진다.
NAMEWORDS_R10 = """하늘빛 푸름 파도 모래 조약돌 자갈 풍경 이랑 두레 사랑 우정 미소 행운 산들 달무리 해님 달님 별님 구름이 은빛 금빛 솔바람 늘봄 한울
가온 나래 다솔 라온 시온 이든 하람 새론 아침 저녁 한낮 밤하늘 새날 오월 유월 시월 동백 수국 백합 해당화 제비꽃 할미꽃 능소화 산수유 앵두
살구 자두 모과 대추 밤톨 감자 고구마 옥수수 호박 수박 참외 딸기 포도 복숭아 석류 도라지 쑥갓 상추 보름 한가위 설날 단오 나무 숲 언덕 들판
호수 시내 강물 개울 폭포 오솔길 갯벌 노랑이 파랑새 참새 종달 제비 비둘기 기러기 두루미 따오기 반딧불 무지개 소나기 봄비 가랑비 이슬비 안개
서리 눈꽃 얼음 고드름 눈송이 햇볕 달맞이 별똥 은하수 샛바람 마파람 높새 소슬 단풍 은행잎 솔방울 도토리묵 감꽃 밤꽃 찔레 으름 보금자리 징검다리
두꺼비 다람쥐 고래 돌고래 아기곰 꽃잎 꽃씨 씨앗 새싹 풀잎 나뭇잎 이파리 가지 뿌리 줄기 열쇠 방패 거울 구슬비 종이배 연필 지우개 도화지 붓꽃
물감 바이올린 피리 북소리 종소리 메아리 노래 춤사위 이야기 그림 동화 보물 선물 편지 엽서 우표 소포 풍선 연날리 팽이 구슬치기 실타래 바늘 단추
비누 수건 이불 베개 방석 바구니 항아리 뚝배기 주전자 숟가락 젓가락 접시 도마 소쿠리 가마 화로 호롱 등불 촛불 횃불 반딧 초승 보름달 그믐"""
NAMEWORDS_R10 = [w for w in NAMEWORDS_R10.split() if 2 <= len(w) <= 4]


def reserved(name):
    """R10 개발셋 예약 이름·단어(학습에서 영구 제외). 이름 부분 해시 기준."""
    return b9.hbucket("r10dev:" + name, 5) == 0


def given_parts(v):
    out = {v}
    if len(v) >= 3:
        out.add(v[1:])
    if len(v) >= 4:
        out.add(v[2:])
    return out


class Guards10(b9.Guards):
    """봉인 평가셋을 읽지 않는 가드. R9 학습·검증 이름은 신규 이름 조건으로 막는다."""

    def __init__(self):
        self.sealed_sents, self.sealed_skels = set(), set()
        self.sealed_grams, self.sealed_vals = [], set()
        self.sealed_names, self.sealed_given = set(), set()
        self.gram_index = defaultdict(list)
        self.dev_exact, self.dev_given, self.dev_words, self.dev_sents = (
            r8.collect_dev_blocks()
        )
        self.dev_skels = set()
        for f in list(r8.EVAL_DIR.rglob("*.jsonl")) + list(r8.NEMO_DIR.glob("*.jsonl")):
            for r in r8.read_jsonl(f):
                t = r.get("text")
                if isinstance(t, str):
                    sp = [
                        {"start": s["start"], "end": s["end"]}
                        for s in r.get("spans", [])
                        if "start" in s
                    ]
                    self.dev_skels.add(r8.skeleton_of(t, sp))
        self.common = self._common_words()
        self.existing_sents, self.existing_skels = set(), set()
        self.r9_names, self.r9_given = set(), set()
        for f in BASE_FILES + R9_NEW_FILES:
            for r in r8.read_jsonl(R9_DIR / f"{f}.jsonl"):
                self.existing_sents.add(r8.norm(r["text"]))
                sp = [{"start": s["start"], "end": s["end"]} for s in r["spans"]]
                self.existing_skels.add(r8.skeleton_of(r["text"], sp))
                for s in r["spans"]:
                    if s.get("label") in ("PERSON", "PERSON_ALIAS"):
                        v = r["text"][s["start"] : s["end"]]
                        self.r9_names.add(v)
                        self.r9_given |= given_parts(v)
        self.rejected = Counter()

    def name_ok(self, full, given=None, allow_common=False):
        given = given or full
        if allow_common:  # 일반 명사형 이름 단어(D1P): R9 와 겹쳐도 되지만 예약 구간은 제외
            return not reserved(full) and super().name_ok(full, given, True)
        # 보수적 판정: 이름 부분 경계를 행에 기록하지 않으므로 어떤 분해(전체·뒤 2~3글자)든 예약 구간이면 학습에서 뺀다
        cands = given_parts(full) | given_parts(given)
        if full in self.r9_names or given in self.r9_given or any(reserved(c) for c in cands):
            return False
        return super().name_ok(full, given, False)

    def text_ok(self, text, spans):
        if not super().text_ok(text, spans):
            return False
        if r8.skeleton_of(text, spans) in self.existing_skels:
            self.rejected["r9_train_skeleton"] += 1
            return False
        return True


def load_frames():
    frames, prob = b9.load_frames()
    raw = {}
    for f in PAIR_FILES:
        p = FRAMES / f"{f}.json"
        if p.exists():
            for it in json.loads(p.read_text(encoding="utf-8")):
                raw[it["id"]] = it
    for fr in frames:
        it = raw.get(fr["id"])
        if it:
            fr["pair_id"], fr["role"] = it.get("pair_id"), it.get("role")
    return frames, prob


def finish(fr, segs, rng, bucket, g, seen, extra):
    segs = b9.apply_noise(segs, rng, *NOISE[bucket])
    text, spans, rule = b9.finalize(segs)
    key = r8.norm(text)
    if key in seen or not g.text_ok(text, spans):
        return None
    row = {
        "id": "",
        "text": text,
        "spans": spans,
        "source": "synth",
        "template_id": fr["id"],
        "category": f"r10-{bucket}-{'-'.join(fr['tags'][:2]) or fr['file']}",
        "augment": True,
        "r10": True,
        "train_bucket": f"r10_{bucket}",
        "frame_file": fr["file"],
        "frame_tag": ",".join(fr["tags"]),
        **extra,
    }
    if rule:
        row["rule_spans"] = rule
    seen.add(key)
    return row


def build(frames, names, g, seed, cfg):
    rng = random.Random(seed)
    seen = set()
    out = {sp: defaultdict(list) for sp in ("train", "val")}
    byfile = defaultdict(list)
    for f in frames:
        byfile[f["file"]].append(f)

    def split_of(key):
        return "val" if b9.hbucket(key) < 15 else "train"

    # D: 쌍
    pairs = defaultdict(dict)
    for f in frames:
        if f.get("pair_id"):
            pairs[f["pair_id"]][f["role"]] = f
    pairs = {k: v for k, v in pairs.items() if {"person", "noun"} <= set(v)}
    for sp in ("train", "val"):
        ps = sorted(k for k in pairs if split_of(k) == sp)
        quota = TARGET["D"] if sp == "train" else round(TARGET["D"] * VAL_FRACTION)
        made, ring, tries = 0, 0, 0
        while made < quota and tries < quota * 30:
            tries += 1
            pid = ps[ring % len(ps)]
            ring += 1
            word = names.pick("nameword", sp)
            pr = pairs[pid]
            rows = []
            for role, kind in (("person", "NW"), ("noun", "WN")):  # 정합성: 아래에서 짝 검증
                fr = pr[role]

                def roles(k, num, _w=word, _fr=fr):
                    if k in ("NW", "WN"):
                        return _w
                    return b9.name_roles("N", "msg_person", names, sp, rng, _fr["id"])(
                        k, num
                    )

                segs, _ = b9.fill(fr, cfg, names, sp, rng, g, roles)
                row = finish(
                    fr, segs, rng, "D", g, seen, {"pair_id": pid, "pair_role": role, "pair_word": word}
                )
                if row is None:
                    rows = None
                    break
                rows.append(row)
            if rows and len(rows) == 2:
                p_row, n_row = rows
                person_ok = any(p_row["text"][x["start"] : x["end"]] == word for x in p_row["spans"])
                noun_ok = word in n_row["text"] and not any(
                    n_row["text"][x["start"] : x["end"]] == word for x in n_row["spans"]
                )
                if not (person_ok and noun_ok):
                    seen.discard(r8.norm(p_row["text"]))
                    seen.discard(r8.norm(n_row["text"]))
                    continue
                out[sp]["D"].extend(rows)
                made += 2
    # 단일 틀 버킷
    plan = {"B": ("b1", None), "C": ("c2", None), "N": ("msg_new", None), "G": ("neg_guard", None)}
    for bucket, (fname, _) in plan.items():
        for sp in ("train", "val"):
            fs = [f for f in byfile[fname] if split_of(f["id"]) == sp] or byfile[fname][:5]
            quota = TARGET[bucket] if sp == "train" else round(TARGET[bucket] * VAL_FRACTION)
            made, ring, tries = 0, 0, 0
            while made < quota and tries < quota * 40:
                tries += 1
                fr = fs[ring % len(fs)]
                ring += 1
                tag = fr["tags"][0] if fr["tags"] else None
                roles = b9.name_roles(
                    "B1" if bucket == "B" else tag, "err_B" if bucket in ("B", "C") else fname, names, sp, rng, fr["id"]
                )
                segs, _ = b9.fill(fr, cfg, names, sp, rng, g, roles)
                row = finish(fr, segs, rng, bucket, g, seen, {})
                if row:
                    out[sp][bucket].append(row)
                    made += 1
            if made < quota:
                print(f"경고 행 부족 {bucket}/{sp}: {made}/{quota}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    b9.EXISTING_DIR = R9_DIR
    frames, fprob = load_frames()
    g = Guards10()
    rng = random.Random(a.seed)
    names = b9.Names(g, rng)
    names._build()
    for sp in ("train", "val"):
        have = set(names.split["nameword"][sp])
        add = [
            w
            for w in dict.fromkeys(NAMEWORDS_R10)
            if w not in have
            and (b9.hbucket(w) >= 16) == (sp == "train")
            and g.name_ok(w, allow_common=True)
        ]
        names.split["nameword"][sp] = names.split["nameword"][sp] + add
    # 보수적: 예약 단어(개발셋 전용)를 부분 문자열로 포함하거나 그 안에 든 학습 단어는 뺀다(예: 개발 '소리' vs 학습 '종소리')
    res_words = [
        w for w in dict.fromkeys(list(b9.NAMEWORDS_NOUN) + NAMEWORDS_R10) if reserved(w) and len(w) >= 2
    ]
    for sp in ("train", "val"):
        names.split["nameword"][sp] = [
            w for w in names.split["nameword"][sp] if not any(r in w or w in r for r in res_words)
        ]
    # 일반 단어(자연물·사물·이름 단어)는 R9 main() 과 같은 규칙으로 train/val 분할
    def sp_words(lst):
        ok = [w for w in dict.fromkeys(lst) if w not in g.dev_words and w not in g.dev_exact and not reserved(w)]
        return {"train": [w for w in ok if b9.hbucket(w) >= 16], "val": [w for w in ok if b9.hbucket(w) < 16]}

    names.nature, names.objects = sp_words(b9.NATURE), sp_words(b9.OBJECTS)
    locs = [x for x in b9.synth.CITIES + b9.synth.STATIONS + b9.synth.LANDMARKS if x not in g.dev_exact]
    orgs = [
        p + s
        for P, S in (
            (b9.synth.HOSP_PREFIX, b9.synth.HOSP_SUFFIX),
            (b9.synth.SCHOOL_PREFIX, b9.synth.SCHOOL_SUFFIX),
            (b9.synth.COMPANY_PREFIX, b9.synth.COMPANY_SUFFIX),
        )
        for p in P
        for s in S
    ]
    orgs = [x for x in orgs if x not in g.dev_exact]
    cfg = {
        "locs": {"train": [x for x in locs if b9.hbucket(x) >= 16], "val": [x for x in locs if b9.hbucket(x) < 16]},
        "orgs": {"train": [x for x in orgs if b9.hbucket(x) >= 16], "val": [x for x in orgs if b9.hbucket(x) < 16]},
    }
    out = build(frames, names, g, a.seed, cfg)
    n = 0
    files, val = {}, []
    for b in TARGET:
        files[f"r10_{b}_train"] = out["train"][b]
        val += out["val"][b]
    for sp, rows in (("train", [r for b in TARGET for r in out["train"][b]]), ("val", val)):
        for r in rows:
            n += 1
            r["id"] = f"r10-{sp}-{n:05d}"
            r["split"] = sp
    files["r10_full_train"] = [r for b in TARGET for r in out["train"][b]]
    files["r10_val_extra"] = val
    problems = b9.validate(files["r10_full_train"] + val, g)
    report = {
        "seed": a.seed,
        "policy": "r9-1.1",
        "sealed_access": "없음(R9 봉인 평가셋·new180 을 읽지 않음)",
        "frames": {"count": len(frames), "problems": fprob, "by_file": dict(Counter(f["file"] for f in frames))},
        "rows": {k: len(v) for k, v in files.items()},
        "problems": len(problems),
        "problem_examples": problems[:10],
        "rejected_by_reason(counts only)": dict(g.rejected),
        "reserved_rule": "이름 해시 mod 5 == 0 은 R10 개발셋 전용(학습 제외)",
    }
    if not a.dry_run:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        for k, v in files.items():
            (a.out_dir / f"{k}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in v), encoding="utf-8"
            )
        (a.out_dir / "r10_build_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
