"""offset 정렬·BIO 변환·KLUE 변환 단위 시험."""

import ner_common as nc
import pytest
import tokenize_align as ta
from build_ner_data import klue_to_row


def _tags(names):
    return [nc.TAG2ID[n] for n in names]


def test_align_and_decode_roundtrip_fake_offsets():
    # "철수한테 엄마 전화" : 토큰 [CLS] 철수 ##한테 엄마 전화 [SEP]
    text = "철수한테 엄마 전화"
    offsets = [(0, 0), (0, 2), (2, 4), (5, 7), (8, 10), (0, 0)]
    special = [1, 0, 0, 0, 0, 1]
    spans = [{"label": "PERSON", "start": 0, "end": 2}]
    labels, conflicts = ta.align_labels(offsets, spans, special)
    assert labels == [-100, nc.TAG2ID["B-PERSON"], nc.TAG2ID["O"], nc.TAG2ID["O"], nc.TAG2ID["O"], -100]
    assert conflicts == 0
    assert ta.decode_spans(labels, offsets, special) == [{"label": "PERSON", "start": 0, "end": 2}]
    assert text[0:2] == "철수"


def test_multi_token_span_gets_b_then_i():
    offsets = [(0, 0), (0, 2), (2, 4), (4, 6), (0, 0)]
    special = [1, 0, 0, 0, 1]
    spans = [{"label": "ADDRESS", "start": 0, "end": 6}]
    labels, _ = ta.align_labels(offsets, spans, special)
    assert labels[1:4] == _tags(["B-ADDRESS", "I-ADDRESS", "I-ADDRESS"])
    assert ta.decode_spans(labels, offsets, special) == spans


def test_adjacent_spans_with_different_labels_split():
    offsets = [(0, 2), (2, 4)]
    spans = [{"label": "PERSON", "start": 0, "end": 2}, {"label": "ORGANIZATION", "start": 2, "end": 4}]
    labels, _ = ta.align_labels(offsets, spans, [0, 0])
    assert ta.decode_spans(labels, offsets) == spans


def test_i_without_b_starts_new_span_and_label_change_splits():
    offsets = [(0, 2), (3, 5), (6, 8)]
    ids = _tags(["I-PERSON", "I-PERSON", "I-LOCATION"])
    assert ta.decode_spans(ids, offsets) == [{"label": "PERSON", "start": 0, "end": 5},
                                              {"label": "LOCATION", "start": 6, "end": 8}]


def test_first_policy_ignores_continuation_subwords_but_keeps_span_start():
    offsets = [(0, 1), (1, 2), (2, 3)]
    tokens = ["철", "##수", "##가"]
    spans = [{"label": "PERSON", "start": 0, "end": 2}]
    labels, _ = ta.align_labels(offsets, spans, [0, 0, 0], tokens, policy="first")
    assert labels[0] == nc.TAG2ID["B-PERSON"]
    assert labels[1] == -100  # 단어 중간 서브워드는 손실 제외


def test_klue_conversion_maps_labels_and_drops_dt_qt_ti():
    tokens = list("김철수는 서울에서 3월 10일 삼성에 갔다")
    tags = [12] * len(tokens)
    text = "".join(tokens)

    def mark(sub, b, i):
        s = text.index(sub)
        tags[s] = b
        for k in range(s + 1, s + len(sub)):
            tags[k] = i

    mark("김철수", 6, 7)  # B-PS I-PS
    mark("서울", 2, 3)  # LC
    mark("3월 10일", 0, 1)  # DT -> O
    mark("삼성", 4, 5)  # OG
    row = klue_to_row(tokens, tags, 0, "train")
    got = [(row["text"][s["start"] : s["end"]], s["label"]) for s in row["spans"]]
    assert got == [("김철수", "PERSON"), ("서울", "LOCATION"), ("삼성", "ORGANIZATION")]


def test_klue_stray_i_and_consecutive_b_split():
    row = klue_to_row(list("가나다라"), [7, 7, 6, 7], 0, "val")  # I I B I (stray I 로 시작)
    assert [(s["start"], s["end"]) for s in row["spans"]] == [(0, 2), (2, 4)]
    row2 = klue_to_row(list("가나"), [6, 6], 0, "val")  # B B
    assert len(row2["spans"]) == 2


def test_span_metrics_exact_vs_overlap():
    gold = [[{"label": "PERSON", "start": 0, "end": 3}]]
    pred = [[{"label": "PERSON", "start": 0, "end": 4}]]
    assert nc.span_metrics(gold, pred, "exact")["micro"]["tp"] == 0
    assert nc.span_metrics(gold, pred, "overlap")["micro"]["tp"] == 1
    # 맞닿기만 하면 겹침이 아니다
    pred2 = [[{"label": "PERSON", "start": 3, "end": 5}]]
    assert nc.span_metrics(gold, pred2, "overlap")["micro"]["tp"] == 0


@pytest.fixture(scope="module")
def tok():
    try:
        return ta.load_tokenizer("monologg/koelectra-small-v3-discriminator")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer 를 로드할 수 없음: {e}")


def test_real_tokenizer_alignment_recovers_spaced_name(tok):
    text = "어제 김철수 선생님이랑 강남역에서 만났어"
    spans = [{"label": "PERSON", "start": 3, "end": 6}, {"label": "LOCATION", "start": 13, "end": 16}]
    assert text[3:6] == "김철수" and text[13:16] == "강남역"
    rep = ta.alignment_report([{"text": text, "spans": spans}], tok, 128)
    assert rep["exact_recovered"] == 2 and rep["truncated_sentences"] == 0


def test_real_tokenizer_truncation_counts_lost_spans(tok):
    text = "가 " * 200 + "김철수"
    spans = [{"label": "PERSON", "start": len(text) - 3, "end": len(text)}]
    rep = ta.alignment_report([{"text": text, "spans": spans}], tok, 32)
    assert rep["truncated_sentences"] == 1 and rep["lost_spans"] == 1
