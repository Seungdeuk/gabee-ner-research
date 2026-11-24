"""R12b 데이터 품질 게이트 실패 시의 데이터 수정(사전 등록 §3: 실패하면 학습하지 않고 데이터를 고친다).
r12b_extra 의 대조쌍에서 한쪽 라벨에 몰린 단서(순도 ≥ 0.80, 6회 이상)를 가진 쌍을 **쌍 단위(사람+명사 같이)** 로 제거해 쌍 균형을 지킨다.
단서를 가진 문장이 다수 라벨 쪽인 쌍부터 단서 점수가 큰 순서로 제거. 개발·검증셋과 모델 결과는 보지 않고 학습 쌍 안의 문맥 단서 분류기 정확도(단어 단위 5-겹 CV)만 본다.
원본은 r12b_extra_unpruned.jsonl 로 보존. 쌍 제거 상한은 전체 쌍의 25%.
"""

import collections
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r12b_data_diagnosis as D  # noqa: E402

D12B = Path(r"C:\Users\<user>\venvs\ner-runs\data_r12b")
SRC = D12B / "r12b_extra.jsonl"
BAK = D12B / "r12b_extra_unpruned.jsonl"
MAX_FRAC = 0.25


def cv(pairs):
    words = sorted({p["word"] for p in pairs})
    random.Random(1).shuffle(words)
    acc = []
    for i in range(5):
        f = set(words[i::5])
        acc.append(D.acc(D.nb_train([p for p in pairs if p["word"] not in f]), [p for p in pairs if p["word"] in f]))
    return sum(acc) / 5


def main():
    if not BAK.exists():
        shutil.copyfile(SRC, BAK)
    rows = [json.loads(x) for x in BAK.read_text(encoding="utf-8").splitlines() if x.strip()]
    pairs = D.pair_rows(rows)
    pid_of = {r["id"]: r.get("pair_id") for r in rows}
    n0 = len({pid_of[p["id"]] for p in pairs})
    removed = set()
    log = []
    for it in range(60):
        cur = [p for p in pairs if pid_of[p["id"]] not in removed]
        a = cv(cur)
        cov = D.asym(cur, D.ctx_feats, min_n=6)
        log.append({"iter": it, "pairs_left": len({pid_of[p["id"]] for p in cur}), "cv5": round(a, 4), "pure_cues": cov["pure_cues"], "covered%": cov["covered_rows%"]})
        if a <= 0.70 or len(removed) >= MAX_FRAC * n0:
            break
        cnt = collections.defaultdict(lambda: [0, 0])
        for p in cur:
            for k in set(D.ctx_feats(p)):
                cnt[k][p["person"]] += 1
        pure = {k: (v, max(v) / sum(v), v[1] >= v[0]) for k, v in cnt.items() if sum(v) >= 6 and max(v) / sum(v) >= 0.80}
        score = collections.Counter()
        for p in cur:
            for k in set(D.ctx_feats(p)):
                if k in pure and pure[k][2] == p["person"]:  # 이 문장이 그 단서의 다수 라벨 쪽
                    score[pid_of[p["id"]]] += pure[k][1] * sum(pure[k][0])
        if not score:
            break
        for pid, _ in score.most_common(max(4, n0 // 60)):
            removed.add(pid)
    keep = [r for r in rows if r.get("pair_id") not in removed]
    SRC.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    final = D.pair_rows(keep)
    rep = {"pairs_before": n0, "pairs_removed": len(removed), "rows_before": len(rows), "rows_after": len(keep), "final_cv5": round(cv(final), 4), "log": log}
    (Path(r"C:\Users\<user>\venvs\ner-runs\r12b\analysis") / "prune_cues.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "log"}, ensure_ascii=False), log[-1])


if __name__ == "__main__":
    main()
