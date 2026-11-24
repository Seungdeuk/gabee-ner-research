"""R12b 학습 전 데이터 품질 게이트(사전 등록 §3). 개발 결과(모델 성능)를 보지 않는다. 문맥 단서 분류기(낱말 가림, 진단 스크립트와 같은 코드)로
r12b_extra 쌍 문장을 5-겹 교차검증하고 r12b_dev·r12_dev·r11_dev 에 적용한다. 기준: CV ≤ 70%, 각 개발셋 ≤ 72%, 90% 이상 쏠린 단서가 쌍 문장을 덮는 비율 ≤ 25%.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r12b_data_diagnosis as D  # noqa: E402

D12B = Path(r"C:\Users\<user>\venvs\ner-runs\data_r12b")
D12 = Path(r"C:\Users\<user>\venvs\ner-runs\data_r12")


def load(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    train = D.pair_rows(load(D12B / "r12b_extra.jsonl"))
    sets = {"r12b_dev": D.pair_rows(load(D12B / "r12b_dev.jsonl")), "r12_dev": D.pair_rows(load(D12 / "r12_dev.jsonl")), "r11_dev": D.pair_rows(load(D12 / "r11_dev.jsonl"))}
    words = sorted({p["word"] for p in train})
    random.Random(1).shuffle(words)
    folds = [set(words[i::5]) for i in range(5)]  # 단어 단위로 나눠 같은 단어가 학습·검증에 걸치지 않게 한다
    accs = []
    for f in folds:
        tr = [p for p in train if p["word"] not in f]
        te = [p for p in train if p["word"] in f]
        accs.append(D.acc(D.nb_train(tr), te))
    m = D.nb_train(train)
    res = {"pairs_train": len(train), "cv5_word_split": [round(a, 4) for a in accs], "cv5_mean": round(sum(accs) / 5, 4),
           "dev_acc": {n: D.acc(m, rows) for n, rows in sets.items()},
           "pure_any_cue": D.asym(train, D.ctx_feats), "next_char": D.particle_table(train)}
    res["gate"] = {"cv<=0.70": res["cv5_mean"] <= 0.70, "dev<=0.72": all(v <= 0.72 for v in res["dev_acc"].values()), "pure_cover<=25%": res["pure_any_cue"]["covered_rows%"] <= 25.0}
    res["passed"] = all(res["gate"].values())
    out = Path(r"C:\Users\<user>\venvs\ner-runs\r12b\analysis")
    out.mkdir(parents=True, exist_ok=True)
    (out / "quality_gate.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("pairs_train", "cv5_mean", "dev_acc", "gate", "passed")}, ensure_ascii=False, indent=1))
    print("pure cues:", res["pure_any_cue"]["pure_cues"], res["pure_any_cue"]["covered_rows%"], res["pure_any_cue"]["top"][:6])
    print("next_char:", res["next_char"])


if __name__ == "__main__":
    main()
