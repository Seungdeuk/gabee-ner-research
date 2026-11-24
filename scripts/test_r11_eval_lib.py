"""R11 평가 라이브러리 시험: 정답 구간 4분류, Recall/가림률/경계만 틀림 항등식, Precision(공식·전체), 쌍 문맥 정확률."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import r11_eval_lib as r11  # noqa: E402


def row(i, text, gold, **kw):
    return {"id": i, "text": text, "spans": [{"label": "PERSON", "start": a, "end": b} for a, b in gold], **kw}


def score(rows, preds):
    return r11.score_rows_r11(rows, pred_fn=lambda r: preds[r["id"]])


def test_four_way_classification_and_identity():
    # 정답 (0,3). 예측 (0,3)=exact, (0,4)=경계만 틀림(완전 가림), (0,2)=부분 노출, ()=완전 노출
    rows = [row(i, "가나다라마", [(0, 3)]) for i in range(4)]
    s = score(rows, {0: [(0, 3)], 1: [(0, 4)], 2: [(0, 2)], 3: []})
    assert (s["exact"], s["boundary_only"], s["partial"], s["full"]) == (1, 1, 1, 1)
    assert s["exact_recall"] == 0.25 and s["mask_rate"] == 0.5 and s["total_exposure_rate"] == 0.5
    assert abs(s["mask_rate"] - s["exact_recall"] - s["boundary_only_rate"]) < 1e-12  # 가림률 - Recall == 경계만 틀림
    assert s["identity_ok"] is True


def test_precision_official_excludes_fp_in_negative_rows_but_all_includes():
    rows = [row(0, "가나다라마", [(0, 3)]), {"id": 1, "text": "바사아자", "spans": []}]
    s = score(rows, {0: [(0, 3)], 1: [(1, 3)]})
    assert s["precision_official"] == 1.0  # 정답 없는 문장의 예측은 공식 Precision 에서 제외
    assert s["precision_all"] == 0.5  # 전체 Precision 은 포함
    assert s["nonperson_fpr"] == 1.0 and s["fp_in_neg_rows"] == 1


def test_pair_context_accuracy_requires_person_masked_and_noun_clean():
    rows = [
        row("p1", "가나다 왔다", [(0, 3)], pair_id="x1", pair_role="person", type="D1P"),
        {"id": "n1", "text": "가나다 좋다", "spans": [], "pair_id": "x1", "pair_role": "noun", "type": "D1"},
        row("p2", "라마바 왔다", [(0, 3)], pair_id="x2", pair_role="person", type="D1P"),
        {"id": "n2", "text": "라마바 좋다", "spans": [], "pair_id": "x2", "pair_role": "noun", "type": "D1"},
    ]
    preds = {"p1": [(0, 3)], "n1": [], "p2": [(0, 3)], "n2": [(0, 3)]}
    res = r11.score_dataset(rows, pred_fn=lambda r: preds[r["id"]])
    assert res["pairs"]["n_pairs"] == 2 and res["pairs"]["context_correct"] == 1
    assert res["pairs"]["pair_context_accuracy"] == 0.5
    assert res["by_type"]["D1P"]["exact_recall"] == 1.0 and res["D1_fpr"] == 0.5
    assert res["identity_ok_all"] is True
