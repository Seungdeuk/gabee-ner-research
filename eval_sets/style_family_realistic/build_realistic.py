"""실제 가족 메신저 대화에 더 가까운 문체로만 된 평가셋을 만든다(전부 합성 — 실제 대화는 절대 쓰지 않는다).

기사체·공문·명단 같은 문체는 가족 채팅에서 거의 안 쓰이므로 빼고, 한 메시지에 가족 호칭·여러 이름·줄임말·
이모티콘·안부·일정 이야기가 섞인 "진짜 메신저스러운" 문장만 담는다. 학습 풀·baseline_v2·옛 v3·v4·v5(지금까지
쓴 모든 세트) 어디와도 이름·문장 틀이 겹치면 생성이 실패한다(style_v4.build_evalset 의 검증된 함수 재사용).

실행: python build_realistic.py --synth <ml/ner/data/synth.py> --baseline <baseline_v2 폴더 ...> \
      --extra <v3.jsonl> <v4.jsonl> <v5.jsonl> --out realistic_family.jsonl
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "style_v4"))

from build_evalset import (  # noqa: E402
    GIVEN_CANDIDATES,
    SURNAMES,
    _jsonl_skeletons,
    _name_excluded,
    _synth_skeletons,
    assert_no_overlap,
    render,
)
from exclusions import collect_excluded_names  # noqa: E402

# 실제 가족 채팅방에서 흔한 패턴: 호칭+이름 섞어 부르기, 안부·일정·소소한 사건을 한 메시지에 담기,
# 이모티콘·줄임말·물음표 반복 같은 메신저 말투. 문장 하나에 이름이 2개 이상 섞이는 경우도 넣는다.
TEMPLATES: dict[str, list[str]] = {
    "메신저": [
        "낼 3시에 {N} 데리러 갈 사람 있어?",
        "{N|이/가} 오늘 학원 안 간다고 쌤이 문자왔어",
        "엄마가 {N} 간식 사놨대 ㅎㅎ",
        "{N} 열나서 조퇴시켰어요 ㅠㅠ 데리러 가주실 분",
        "할머니 생신에 {N}이랑 {N2} 같이 가기로 했어",
        "{N} 성적표 나왔는데 이번엔 잘봤더라",
        "오늘 저녁엔 {N} 좋아하는 거 해줄까",
        "{N2}가 {N} 괴롭혀서 또 싸웠대 ㅋㅋ",
        "주말에 {N} 축구 경기 있어서 다같이 응원가자",
        "{N} 담임선생님이 상담 신청하라셔",
        "{N|이랑/랑} {N2} 둘다 감기 걸렸나봐 콜록콜록",
        "아빠가 {N} 용돈 줬다는데 들었어?",
        "{N} 전학 간다는 거 진짜야??",
        "삼촌이 {N} 보고싶다고 영상통화 하재",
        "{N3}이 {N}랑 {N2} 다 데리고 놀이터 갔어",
        "{N} 오늘 반찬 뭐 좋아하는지 알아?",
        "고모가 {N} 선물 사왔다고 전해달래",
        "{N|한테/한테} 숙제 다 했는지 물어봐줘",
        "{N}이랑 {N2} 둘이 또 투닥거리네 ㅋㅋㅋ",
        "{N} 어린이집 발표회 몇시더라",
    ],
}

# 비인명: 가족 채팅방에 흔한 잡다한 안부·집안일 메시지. 이름처럼 보이는 일반어도 섞는다.
NEGATIVES = [
    "낼 비 온대 우산 챙겨",
    "저녁 뭐 먹을지 투표좀",
    "오늘 하늘 완전 예쁘더라 ㅋㅋ",
    "바다 보러 갈까 주말에",
    "장보고 올게 뭐 필요한거 있어?",
    "에어컨 또 고장났나봐",
    "강아지 산책 누가 시킬거야",
    "학원비 입금했어 확인해줘",
    "택배 왔는데 누구꺼야",
    "주차 좀 당겨줘",
    "쓰레기 분리수거 해야돼",
    "전기세 많이 나왔다 ㅠㅠ",
    "오늘 날씨 왜케 덥냐",
    "청소기 또 고장났어",
    "휴지 다 썼나봐 사와야해",
    "기름 넣어야 할듯",
    "보일러 좀 봐줘 안켜져",
    "인터넷 끊겼다 기사님 불러야하나",
    "빨래 좀 널어줘",
    "화분에 물 좀 줘",
    "오늘 기쁨 두배다 월급날이야",
    "초원이 나오는 그 노래 또 나왔어",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--extra", nargs="+", required=True, help="v3·v4·v5 jsonl 등 추가 제외 자료")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=71177)
    ap.add_argument("--per-template", type=int, default=10)
    a = ap.parse_args()

    synth_path = Path(a.synth)
    baseline_dirs = [Path(p) for p in a.baseline]
    extra_paths = [Path(p) for p in a.extra]

    excluded = collect_excluded_names(synth_path, baseline_dirs, extra_paths)
    givens = [g for g in dict.fromkeys(GIVEN_CANDIDATES) if not _name_excluded(g, excluded)]
    if len(givens) < 60:
        raise SystemExit(f"처음 보는 given 이름이 너무 적다({len(givens)}개, 최소 60개 필요)")

    template_texts = _synth_skeletons(synth_path)
    for d in baseline_dirs:
        for f in sorted(d.glob("*.jsonl")):
            template_texts |= _jsonl_skeletons(f)
    for p in extra_paths:
        template_texts |= _jsonl_skeletons(p)

    import random

    rng = random.Random(a.seed)
    rows: list[dict] = []

    def add(cat, text, spans, tpl_id):
        rows.append(
            {
                "id": f"fam-{len(rows) + 1:04d}",
                "category": cat,
                "text": text,
                "spans": spans,
                "known_names": [],
                "template_id": tpl_id,
            }
        )

    kinds = {"N": "PERSON", "N2": "PERSON", "N3": "PERSON"}
    for cat, tpls in TEMPLATES.items():
        for ti, tpl in enumerate(tpls):
            for _ in range(a.per_template):
                slots = {
                    "N": rng.choice(SURNAMES) + rng.choice(givens),
                    "N2": rng.choice(SURNAMES) + rng.choice(givens),
                    "N3": rng.choice(SURNAMES) + rng.choice(givens),
                }
                text, spans = render(tpl, slots, kinds)
                add(cat, text, spans, f"{cat}-{ti + 1:02d}")
    for i, s in enumerate(NEGATIVES):
        add("비인명", s, [], f"비인명-{i + 1:02d}")

    assert_no_overlap(rows, excluded, template_texts)

    Path(a.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    cats: dict[str, list[int]] = {}
    for r in rows:
        c = cats.setdefault(r["category"], [0, 0])
        c[0] += 1
        c[1] += len(r["spans"])
    sha = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()[:12]
    print(
        json.dumps(
            {
                "sentences": len(rows),
                "by_category(sentences,spans)": cats,
                "unseen_given_names": len(givens),
                "excluded_names": len(excluded),
                "sha256_12": sha,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
