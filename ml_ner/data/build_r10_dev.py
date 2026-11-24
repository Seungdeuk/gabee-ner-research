"""R10 개발 평가셋 생성기(후보 선정용, 학습에 쓰지 않는다). 사전 등록 NER_R10_PREREGISTRATION.md §6.

- 틀은 `r10_dev_frames/`(학습 틀과 **다른 작성 에이전트·다른 틀**, 틀 id 겹침 없음). 모든 틀을 1회만 쓴다(틀 재사용 없음).
- 이름·명사형 이름 단어는 **R10 예약 구간**(이름 부분 해시 mod 5 == 0)에서만 고르고, R9·R10 학습에서 쓴 이름 부분은 피한다 → 신규 이름 층.
- R9 봉인 평가셋은 읽지 않는다(Guards10 상속). 산출: r10_dev.jsonl(+보고서). 각 행에 stress_type·name_seen_in_reference 메타.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_r8_data as r8
import build_r9_data as b9
import build_r10_data as b10

DEV_FRAMES = HERE / "r10_dev_frames"
b9.FRAMES = DEV_FRAMES
FILES = {"dev_pairs", "dev_b1c2", "dev_msg_neg"}
b9.FRAME_FILES = FILES
OUT = Path(r"C:\Users\<user>\venvs\ner-runs\data_r10")


class GuardsDev(b10.Guards10):
    """예약 구간의 이름만 허용(새 이름), R9 학습·검증에서 쓴 이름 부분은 제외."""

    def name_ok(self, full, given=None, allow_common=False):
        given = given or full
        if not b10.reserved(given if not allow_common else full):
            return False
        if not allow_common and (full in self.r9_names or given in self.r9_given):
            return False
        return b9.Guards.name_ok(self, full, given, allow_common)


def stress_of(fr):
    t = fr["tags"]
    if fr["file"] == "dev_pairs":
        return "D1P" if fr["role"] == "person" else "D1"
    if "B1" in t:
        return "B1"
    if "C2" in t:
        return "C2"
    if "person" in t:
        return "NEWNAME"
    return {"D2": "D2", "kin": "KIN_ONLY", "locorg": "A3N", "noun": "D1_STANDALONE"}.get(t[0], "GENERIC")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--train-dir", type=Path, default=OUT)
    a = ap.parse_args()
    b9.EXISTING_DIR = b10.R9_DIR
    frames, fprob = b9.load_frames()
    raw = {}
    for it in json.loads((DEV_FRAMES / "dev_pairs.json").read_text(encoding="utf-8")):
        raw[it["id"]] = it
    for fr in frames:
        it = raw.get(fr["id"])
        if it:
            fr["pair_id"], fr["role"] = it.get("pair_id"), it.get("role")
    g = GuardsDev()
    rng = random.Random(a.seed)
    names = b9.Names(g, rng)
    names._build()
    for sp in ("train", "val"):
        have = set(names.split["nameword"][sp])
        names.split["nameword"][sp] += [
            w
            for w in dict.fromkeys(b10.NAMEWORDS_R10)
            if w not in have and (b9.hbucket(w) >= 16) == (sp == "train") and g.name_ok(w, allow_common=True)
        ]
    # 개발셋은 한 분할("train" 풀)만 쓰되 val 풀도 합쳐 크기를 확보한다
    for k in list(names.split):
        names.split[k]["train"] = names.split[k]["train"] + names.split[k]["val"]
    ok_words = lambda lst: [w for w in dict.fromkeys(lst) if w not in g.dev_words and w not in g.dev_exact]  # noqa: E731
    names.nature = {"train": ok_words(b9.NATURE), "val": []}
    names.objects = {"train": ok_words(b9.OBJECTS), "val": []}
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
    cfg = {"locs": {"train": locs}, "orgs": {"train": [x for x in orgs if x not in g.dev_exact]}}
    # 학습 데이터에서 쓴 이름(신규 이름 판정용)
    train_given = set()
    for f in ("r10_full_train", "r10_val_extra"):
        for r in r8.read_jsonl(a.train_dir / f"{f}.jsonl"):
            for s in r["spans"]:
                if s.get("label") in ("PERSON", "PERSON_ALIAS"):
                    train_given |= b10.given_parts(r["text"][s["start"] : s["end"]])
    ref_given = train_given | g.r9_given
    seen = set()
    rows = []
    byid = {f["id"]: f for f in frames}
    pairs = {}
    for f in frames:
        if f.get("pair_id"):
            pairs.setdefault(f["pair_id"], {})[f["role"]] = f
    done_pair = set()
    for f in frames:
        fr = f
        if fr.get("pair_id"):
            if fr["pair_id"] in done_pair or not {"person", "noun"} <= set(pairs[fr["pair_id"]]):
                continue
            done_pair.add(fr["pair_id"])
            word = names.pick("nameword", "train")
            for role in ("person", "noun"):
                pf = pairs[fr["pair_id"]][role]

                def roles(k, num, _w=word, _id=pf["id"]):
                    if k in ("NW", "WN"):
                        return _w
                    return b9.name_roles("N", "msg_person", names, "train", rng, _id)(k, num)

                segs, _ = b9.fill(pf, cfg, names, "train", rng, g, roles)
                row = b10.finish(pf, segs, rng, "D", g, seen, {"pair_id": fr["pair_id"], "pair_role": role, "pair_word": word})
                if row:
                    row["stress_type"] = stress_of(pf)
                    rows.append(row)
            continue
        tag = fr["tags"][0] if fr["tags"] else None
        bucket = "B" if "B1" in fr["tags"] else "C" if "C2" in fr["tags"] else "N" if "person" in fr["tags"] else "G"
        roles = b9.name_roles("B1" if bucket == "B" else tag, "err_B" if bucket in ("B", "C") else fr["file"], names, "train", rng, fr["id"])
        segs, _ = b9.fill(fr, cfg, names, "train", rng, g, roles)
        row = b10.finish(fr, segs, rng, bucket, g, seen, {})
        if row:
            row["stress_type"] = stress_of(fr)
            rows.append(row)
    for i, r in enumerate(rows, 1):
        r["id"] = f"r10dev-{i:05d}"
        r["split"] = "r10_dev"
        r["train_bucket"] = "r10_dev"
        r["is_negative"] = not r["spans"]
        r["name_seen_in_reference"] = [
            b10.given_parts(r["text"][s["start"] : s["end"]]) & ref_given != set()
            for s in r["spans"]
            if s["label"] in ("PERSON", "PERSON_ALIAS")
        ]
    problems = b9.validate(rows, g)
    vals = [r["text"][s["start"] : s["end"]] for r in rows for s in r["spans"] if s["label"] in ("PERSON", "PERSON_ALIAS")]
    flat_seen = [x for r in rows for x in r["name_seen_in_reference"]]
    rep = {
        "rows": len(rows),
        "by_stress_type": dict(Counter(r["stress_type"] for r in rows)),
        "negatives": sum(r["is_negative"] for r in rows),
        "person_spans": len(vals),
        "distinct_person_values": len(set(vals)),
        "max_person_reuse": max(Counter(vals).values()) if vals else 0,
        "person_spans_name_unseen": sum(1 for x in flat_seen if not x),
        "person_spans_name_seen": sum(1 for x in flat_seen if x),
        "reserved_violations": sum(1 for v in set(vals) if len(v) == 2 and not b10.reserved(v)),
        "problems": len(problems),
        "frames": {"count": len(frames), "problems": fprob, "by_file": dict(Counter(f["file"] for f in frames))},
        "rejected(counts)": dict(g.rejected),
        "sealed_access": "없음",
        "note": "개발 평가용. 학습·체크포인트 선택에 쓰지 않는다. 후보 선정(게이트)에만 쓴다.",
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "r10_dev.jsonl").write_text(
        "".join(json.dumps({k: (sorted(v) if isinstance(v, set) else v) for k, v in r.items()}, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    (a.out_dir / "r10_dev_build_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
