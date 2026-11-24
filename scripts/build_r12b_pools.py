"""R12b 데이터 작성용 풀 배정(사전 등록 NER_R12B_PREREGISTRATION.md §3).

- 단어: r12b_words.CANDIDATES 중 R11·R12 의 모든 pair_word(학습·검증·개발)와 겹치지 않는 '새 단어'를 val/dev/train 으로 나눈다.
  학습 풀은 새 단어 + R11·R12 학습에만 쓰였던 단어(r11_dev/val·r12_dev/val 단어는 절대 아님)로 채운다.
- 이름: build_r12_pools 와 같은 후보(Nemotron-Personas-Korea 이름, CC BY 4.0)에서 R12 풀·R11 정답 이름·r12_dev/val 정답 이름과 겹치지 않는 것만 쓴다.
- 세트 간 겹침 0 을 assert. 결과는 data_r12b/pools.json (Git 비추적).
"""

import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_r12_pools as B  # noqa: E402
import r12b_words  # noqa: E402

ROOT = Path(r"C:\Users\<user>\venvs\ner-runs")
DATA12 = ROOT / "data_r12"
OUT = ROOT / "data_r12b"
SEED = 20261124
EXCLUDE = set("수줍 흐뭇 얄리 동실 소복 송알 아련 어슴 졸졸 포슬 호젓 훈풍 사뿐 파릇 몽글 도란 보송 앙증 옹글 여린 푸릇 아롱 은결 지니 노닥 시나브로 소담이 쾌청 곱단 그리메 가온 꿈결 오롱이 하얀 해맑이 오월 몽글몽글 풍경 무늬 세잎 새알 단비".split())
NW = {"train": 300, "val": 60, "dev": 60}
NF = {"train": 560, "val": 100, "dev": 100}
NB = {"train": 450, "val": 100, "dev": 100}


def pair_words12(files):
    out = set()
    for f in files:
        for r in B.rows_of(DATA12 / f):
            if r.get("pair_word"):
                out.add(r["pair_word"])
    return out


def person_values(files, base):
    out = set()
    for f in files:
        for r in B.rows_of(base / f):
            for s in r.get("spans", []):
                if (s.get("label") or s.get("mapped_kind")) in ("PERSON", "PERSON_ALIAS"):
                    out.add(r["text"][s["start"]:s["end"]])
    return out


def main():
    rng = random.Random(SEED)
    cand = []
    for w in r12b_words.CANDIDATES.split():
        if 2 <= len(w) <= 3 and w not in cand and w not in EXCLUDE:
            cand.append(w)
    old = json.loads((DATA12 / "pools.json").read_text(encoding="utf-8"))
    r11_tr = B.pair_words_in(["r11_B_extra.jsonl", "r11_Dprime_train.jsonl"])
    r11_dv = B.pair_words_in(["r11_dev.jsonl", "r11_val.jsonl"])
    r12_tr, r12_v, r12_d = set(old["words"]["train"]), set(old["words"]["val"]), set(old["words"]["dev"])
    used_all = r11_tr | r11_dv | r12_tr | r12_v | r12_d
    banned_for_train = r11_dv | r12_v | r12_d  # R12b 학습에 절대 쓰지 않는 단어(기존 개발·검증 단어)
    fresh = [w for w in cand if w not in used_all]
    rng.shuffle(fresh)
    reusable = sorted((r11_tr | r12_tr) - banned_for_train)
    rng.shuffle(reusable)
    if len(fresh) < NW["val"] + NW["dev"] + 50:
        raise SystemExit(f"새 단어 부족: {len(fresh)}")
    val_w, dev_w = fresh[:NW["val"]], fresh[NW["val"]:NW["val"] + NW["dev"]]
    rest = fresh[NW["val"] + NW["dev"]:]
    train_w = (rest + reusable)[:NW["train"]]
    wp = {"train": train_w, "val": val_w, "dev": dev_w}
    for a in wp:
        for b in wp:
            if a < b:
                assert not (set(wp[a]) & set(wp[b]))
    assert not (set(train_w) & banned_for_train)
    assert not ((set(val_w) | set(dev_w)) & used_all)
    # 이름
    vocab = {t for t in json.loads(B.TOK.read_text(encoding="utf-8"))["model"]["vocab"] if not t.startswith(("##", "[")) and all("가" <= c <= "힣" for c in t)}
    nv_full = json.loads((B.SCRATCH / "nv_names_full.json").read_text(encoding="utf-8"))["full"]
    seen = B.gold_names(["r11_dev.jsonl", "r11_val.jsonl", "baseline_val.jsonl", "synth_val.jsonl"]) | person_values(["r12_dev.jsonl", "r12_val.jsonl"], DATA12)
    seen_given = {n[1:] for n in seen if len(n) == 3} | {n for n in seen if len(n) == 2}
    allwords = set(B.WORDS.split()) | set(cand) | used_all
    old_full = {n for v in old["full_names"].values() for n in v}
    old_bare = {n for v in old["bare_names"].values() for n in v}
    fulls = sorted(n for n, c in nv_full.items() if c >= 2 and n not in vocab and n not in seen and len(n) == 3 and n[1:] not in vocab and n[1:] not in allwords and n not in old_full)
    given = collections.Counter()
    for n, c in nv_full.items():
        if len(n) == 3:
            given[n[1:]] += c
    bares = sorted(g for g, c in given.items() if c >= 3 and g not in vocab and g not in seen_given and g not in allwords and len(g) == 2 and g not in old_bare)
    rng.shuffle(fulls)
    rng.shuffle(bares)

    def split(seq, n):
        out, i = {}, 0
        for k in ("train", "val", "dev"):
            out[k] = seq[i:i + n[k]]
            i += n[k]
        return out

    fp, bp = split(fulls, NF), split(bares, NB)
    # BND(끝글자가 조사·어미처럼 보이는 이름)용: 위 배정에 쓰이지 않은 후보 중 끝 글자가 아·야·이·가·도·랑·은·는 인 전체 이름
    endings = set("아야이가도랑은는")
    left = [n for n in fulls[sum(NF.values()):] if n[-1] in endings]
    NBND = {"train": 40, "val": 12, "dev": 12}
    bnd = split(left, NBND)
    OUT.mkdir(exist_ok=True)
    (OUT / "pools.json").write_text(json.dumps({"words": wp, "full_names": fp, "bare_names": bp, "bnd_names": bnd}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"단어 후보 {len(cand)} → 새 단어 {len(fresh)} (재사용 가능 학습 단어 {len(reusable)}) | train {len(train_w)} val {len(val_w)} dev {len(dev_w)}")
    print(f"전체 이름 후보 {len(fulls)} → 배정 {sum(len(v) for v in fp.values())}, 성 없는 이름 후보 {len(bares)} → 배정 {sum(len(v) for v in bp.values())}")


if __name__ == "__main__":
    main()
