"""서버 런타임용 NER 추론 참조 구현. onnxruntime + tokenizers + numpy 만 쓴다(torch·transformers 불필요).

흐름: tokenizer.json 로드 -> 인코딩(offset 포함, 자르지 않음) -> 내용 토큰(특수 토큰 제외)을 슬라이딩 윈도우로 나눔
-> 각 윈도우 앞뒤에 [CLS]/[SEP] 를 붙여 ONNX 추론 -> 토큰별 로짓을 겹침 구간에서 윈도우 중앙에 가까운 쪽으로 합침
-> BIO 디코딩 -> 문자 단위 span ``{label, start, end}``(end 는 배타적, 파이썬 슬라이스 기준).

토큰 수가 ``max_len - 2``(=126) 이하인 문장은 윈도우가 하나이고 학습 때의 입력과 같다.
BIO 디코딩 규칙은 학습 코드(``ml/ner/train/tokenize_align.py`` 의 ``decode_spans``, policy="all")와 같다.
"""

import json
from pathlib import Path

MAX_LEN = 128
DEFAULT_STRIDE_OVERLAP = 32  # 이웃 윈도우가 겹치는 내용 토큰 수


def plan_windows(n, size, overlap):
    """내용 토큰 n 개를 겹치는 윈도우 (start, end) 목록으로 나눈다.

    @param n: 내용 토큰 수
    @param size: 윈도우 내용 토큰 수(max_len - 2)
    @param overlap: 이웃 윈도우 겹침 토큰 수
    @returns: [(start, end)]; n <= size 이면 [(0, n)], n == 0 이면 []
    """
    if n <= 0:
        return []
    if n <= size:
        return [(0, n)]
    step = size - overlap
    out, s = [], 0
    while True:
        e = min(s + size, n)
        out.append((s, e))
        if e >= n:
            return out
        s += step


def softmax_free_argmax(logits):
    """로짓의 argmax 태그 id 목록(소프트맥스 불필요).

    @param logits: numpy (n, num_labels)
    @returns: list[int]
    """
    return [int(i) for i in logits.argmax(-1)]


def decode_bio(tag_ids, offsets, id2label):
    """토큰 태그 id 열에서 문자 span 을 복원한다(학습 코드 decode_spans 의 policy=all 과 동일 규칙).

    @param tag_ids: 내용 토큰별 태그 id
    @param offsets: 내용 토큰별 (start, end) 문자 offset
    @param id2label: {id: "B-PERSON" ...}
    @returns: [{label, start, end}]
    """
    spans, cur = [], None
    for tid, (ts, te) in zip(tag_ids, offsets, strict=False):
        if te <= ts:
            continue
        tag = id2label[tid]
        if tag == "O":
            if cur:
                spans.append(cur)
                cur = None
            continue
        bi, lab = tag.split("-", 1)
        if cur and bi == "I" and cur["label"] == lab:
            cur["end"] = te
        else:
            if cur:
                spans.append(cur)
            cur = {"label": lab, "start": ts, "end": te}
    if cur:
        spans.append(cur)
    return spans


class OnnxNer:
    """ONNX 토큰 분류 모델 + tokenizers 로 문자 span 을 예측한다.

    @param model_dir: model.onnx(또는 model_file), tokenizer.json, labels.json 이 있는 폴더
    @param model_file: 모델 파일 이름(기본 model.onnx, 양자화본은 model.int8.onnx)
    @param threads: intra_op 스레드 수(None 이면 onnxruntime 기본)
    @param overlap: 윈도우 겹침 토큰 수
    """

    def __init__(self, model_dir, model_file="model.onnx", threads=None, overlap=DEFAULT_STRIDE_OVERLAP, session=True):
        from tokenizers import Tokenizer

        d = Path(model_dir)
        self.tk = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tk.no_truncation()
        self.tk.no_padding()
        meta = json.loads((d / "labels.json").read_text(encoding="utf-8"))
        self.id2label = {int(k): v for k, v in meta["id2label"].items()}
        self.max_len = int(meta.get("max_len", MAX_LEN))
        self.pad_id = int(meta.get("pad_token_id", 0))
        cls_fb = self.tk.token_to_id("[CLS]") if self.tk.token_to_id("[CLS]") is not None else self.tk.token_to_id("<s>")
        sep_fb = self.tk.token_to_id("[SEP]") if self.tk.token_to_id("[SEP]") is not None else self.tk.token_to_id("</s>")
        self.cls_id = int(meta["cls_token_id"] if meta.get("cls_token_id") is not None else cls_fb)
        self.sep_id = int(meta["sep_token_id"] if meta.get("sep_token_id") is not None else sep_fb)
        self.size = self.max_len - 2
        self.overlap = overlap
        self.sess = None
        if session:
            import onnxruntime as ort

            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = int(threads)
                so.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(str(d / model_file), so, providers=["CPUExecutionProvider"])
            self.input_names = {i.name for i in self.sess.get_inputs()}

    # --- 토큰화 ---
    def encode(self, text):
        """텍스트를 내용 토큰 id·offset 으로 인코딩한다(특수 토큰 제외, 자르지 않음).

        @param text: 문장
        @returns: (ids, offsets) — 특수 토큰을 뺀 목록
        """
        enc = self.tk.encode(text)
        keep = [i for i, m in enumerate(enc.special_tokens_mask) if not m]
        return [enc.ids[i] for i in keep], [enc.offsets[i] for i in keep]

    # --- 모델 호출(서브클래스가 바꿀 수 있다) ---
    def run_batch(self, seqs):
        """[CLS] ... [SEP] 가 붙은 id 시퀀스들(길이 다를 수 있음)을 패딩해 로짓을 돌려준다.

        @param seqs: list[list[int]]
        @returns: list[numpy (len_i, num_labels)] — 패딩 제외
        """
        import numpy as np

        m = max(len(s) for s in seqs)
        ids = np.full((len(seqs), m), self.pad_id, dtype=np.int64)
        att = np.zeros((len(seqs), m), dtype=np.int64)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = s
            att[i, : len(s)] = 1
        feeds = {"input_ids": ids, "attention_mask": att}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        out = self.sess.run(["logits"], feeds)[0]
        return [out[i, : len(s)] for i, s in enumerate(seqs)]

    # --- 윈도우 합치기 ---
    def logits_for(self, ids, batch_windows=True):
        """내용 토큰 id 전체의 토큰별 로짓(윈도우 겹침은 중앙에 가까운 윈도우 값을 사용).

        @param ids: 내용 토큰 id 목록
        @param batch_windows: 한 문장의 윈도우들을 한 배치로 묶을지
        @returns: numpy (len(ids), num_labels), 토큰이 없으면 None
        """
        import numpy as np

        wins = plan_windows(len(ids), self.size, self.overlap)
        if not wins:
            return None
        seqs = [[self.cls_id, *ids[s:e], self.sep_id] for s, e in wins]
        outs = self.run_batch(seqs) if batch_windows else [self.run_batch([q])[0] for q in seqs]
        nl = outs[0].shape[-1]
        res = np.zeros((len(ids), nl), dtype=np.float32)
        best = np.full(len(ids), -1.0)
        for (s, e), lg in zip(wins, outs, strict=False):
            lg = lg[1:-1]
            n = e - s
            for j in range(n):
                centrality = min(j, n - 1 - j)  # 윈도우 가장자리에서의 거리: 클수록 문맥이 충분
                if centrality > best[s + j]:
                    best[s + j] = centrality
                    res[s + j] = lg[j]
        return res

    def predict(self, text):
        """한 문장(또는 긴 글)의 문자 span 목록.

        @param text: 입력 텍스트
        @returns: [{label, start, end}]
        """
        ids, offs = self.encode(text)
        lg = self.logits_for(ids)
        if lg is None:
            return []
        return decode_bio(softmax_free_argmax(lg), offs, self.id2label)

    def predict_many(self, texts, batch_size=32):
        """여러 문장을 길이순으로 묶어 배치 추론한다(결과는 입력 순서).

        @param texts: 문장 목록
        @param batch_size: 배치 크기
        @returns: list[list[span]]
        """
        enc = [self.encode(t) for t in texts]
        jobs = []  # (문장 번호, 윈도우 번호, seq)
        plans = []
        for si, (ids, _) in enumerate(enc):
            wins = plan_windows(len(ids), self.size, self.overlap)
            plans.append(wins)
            for wi, (s, e) in enumerate(wins):
                jobs.append((si, wi, [self.cls_id, *ids[s:e], self.sep_id]))
        jobs.sort(key=lambda j: len(j[2]))
        got = {}
        for b in range(0, len(jobs), batch_size):
            chunk = jobs[b : b + batch_size]
            for (si, wi, _), lg in zip(chunk, self.run_batch([j[2] for j in chunk]), strict=False):
                got[(si, wi)] = lg
        out = []
        import numpy as np

        for si, (ids, offs) in enumerate(enc):
            wins = plans[si]
            if not wins:
                out.append([])
                continue
            nl = got[(si, 0)].shape[-1]
            res = np.zeros((len(ids), nl), dtype=np.float32)
            best = np.full(len(ids), -1.0)
            for wi, (s, e) in enumerate(wins):
                lg = got[(si, wi)][1:-1]
                n = e - s
                for j in range(n):
                    c = min(j, n - 1 - j)
                    if c > best[s + j]:
                        best[s + j] = c
                        res[s + j] = lg[j]
            out.append(decode_bio(softmax_free_argmax(res), offs, self.id2label))
        return out
