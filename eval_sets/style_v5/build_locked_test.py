"""로드맵 5단계: 최종 독립 테스트셋(전부 합성)을 만든다. **한 번만 평가하고 다시 열지 않는다.**

학습 풀·baseline_v2·옛 v3(저장소 밖)·v4(저장소 안, devset으로 반복 참고함) 어디와도 이름·문장 틀이 겹치면
생성이 실패한다(겹침 검사는 style_v4.build_evalset 의 검증된 함수를 그대로 재사용한다). 이름 후보 풀은 v4 와
같은 방식(v3 실제 이름 음절 교차 조합)을 쓰지만, 제외 집합에 v4 도 들어가므로 v4 가 실제로 쓴 이름은 자동으로
빠진다. 문체·문장 틀은 v3·v4 와 전부 다른 새 문장으로 썼다.

실행: python build_locked_test.py --synth <ml/ner/data/synth.py> --baseline <baseline_v2 폴더 ...> \
      --extra <evalset_v3.jsonl> <evalset_v4.jsonl> --out locked_test_v5.jsonl
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

TEMPLATES: dict[str, list[str]] = {
    "가족대화": [
        "{N}이 오늘 일찍 온대",
        "{N} 밥 먹었는지 봐줘",
        "할아버지가 {N} 데리러 가신대",
        "{N|이랑/랑} 영화 보러 갈 거야",
        "{N} 오늘 늦잠 잤대",
        "{N|한테/한테} 숙제 도와달라고 했어",
        "{N|이/가} 전화 안 받아",
        "엄마가 {N} 챙겨주라고 했어",
    ],
    "배송수령": [
        "받는분 {N}",
        "고객명: {N}",
        "{N}님께 배송 중입니다",
        "주문자 {N}님 결제 확인됐어요",
        "교환 신청자 {N}",
        "{N}씨 경비실에 맡겨두었습니다",
        "환불 요청인 {N}",
        "입금 확인자명 {N}으로 처리할게요",
    ],
    "명단목록": [
        "출장자 명단: {N}, {N2}, {N3}",
        "① {N}  ② {N2}  ③ {N3}",
        "동호회 회원 - {N} / {N2} / {N3}",
        "오늘 담당자: {N}, {N2}",
        "대기자 명단에 {N}, {N2}님 추가했어",
        "팀 배정 1팀 {N} {N2} 2팀 {N3}",
        "참가자 목록 {N} · {N2}",
        "응시자 {N}, {N2}, {N3}",
    ],
    "공문안내": [
        "{N} 지도 교사께 전달 바랍니다",
        "{N} 학생 가정통신문을 안내드립니다",
        "면접 대상: {N} (서류 심사 통과)",
        "자녀 {N}의 예방접종 증명서를 제출해 주세요",
        "{N}님의 상담 일정이 확정되었습니다",
        "{N} 회원님 등급이 변경되었습니다",
        "대리인 {N} 날인",
        "제출인 {N} 서명",
    ],
    "기사체": [
        "{N} 부사장이 투자 계획을 발표했다",
        "{N} 구청장이 공약을 내놓았다",
        '피해자 {N}(29)씨는 "두렵다"고 밝혔다',
        "{N} 작가가 신작을 출간했다",
        "경찰은 {N|을/를} 유력한 용의자로 지목했다",
        "{N} 평론가가 영화를 소개했다",
        "{N} 대표팀 주장이 각오를 전했다",
        "{N} 이사장은 기부 소식을 알렸다",
    ],
    "호칭조사": [
        "{N}님 환영합니다",
        "{N} 씨 메일 주소 알려주세요",
        "{N}교수님 감사드립니다",
        "{N|이/가} 문자 보냈어",
        "{N|이랑/랑} 외출했어",
        "{N|한테/한테} 서류 전달해줘",
        "{N|의/의} 노트북이야",
        "{N|은/는} 출근했어?",
    ],
    "이름만": [
        "{G|이/} 아직 안 왔어",
        "{G|아/야} 조심해서 가",
        "{G|한테/한테} 메시지 보내봐",
        "{G|이랑/랑} 카페 갔어",
        "{G|은/는} 몸살이래",
        "{G|이/가} 합격했어",
        "방금 {G} 도착했대",
        "{G|네/네} 뒷집이야",
    ],
}

# 비인명 음성 문장: v4 와 다른 새 문장. 이름처럼 보이는 일반어는 계속 포함한다(과마스킹 회귀 확인용).
NEGATIVES = [
    "하늘색 원피스를 새로 샀어",
    "바다 냄새가 여기까지 난다",
    "기쁨도 잠깐이고 다시 바빠졌어",
    "초원 사진 보니까 힐링된다",
    "노을 질 때쯔음 나와봐",
    "구름 한 점 없이 맑다",
    "겨울방학 계획을 세웠어",
    "여름휴가 신청서를 냈어",
    "새벽 배송이 벌써 왔어",
    "달빛 산책 다녀왔어",
    "접수증을 다시 출력해줘",
    "지원서를 다시 써야겠어",
    "커튼 색깔을 바꿨어",
    "보람 있는 하루였어",
    "마당에 자갈을 깔았어",
    "저녁 메뉴는 김치찌개야",
    "미래를 대비해서 보험을 들었어",
    "소음 때문에 민원이 들어왔어",
    "온도가 갑자기 떨어졌대",
    "초록 불로 바뀌면 건너가자",
    "서가에서 책을 찾아왔어",
    "접수 마감일을 확인해줘",
    "등록 절차를 안내해줘",
    "진료 예약을 변경했대",
    "요금 고지서가 잘못 왔어",
    "결제 승인이 안 떨어졌어",
    "잔액 조회를 해봐",
    "배송 추적이 안 돼",
    "냉장고 정리를 좀 했어",
    "주차 공간이 부족해",
    "오늘 유난히 후텁지근하다",
    "탄산수 한 병 사다줘",
    "지하철을 놓쳐서 걸어왔어",
    "우산을 또 잃어버렸어",
    "회식 자리가 겹쳤어",
    "와이파이가 불안정해",
    "건조기가 멈춰버렸어",
    "환기 좀 시키자",
    "정류장 위치가 바뀌었대",
    "엘리베이터 점검 중이래",
]

ADDRESSES = [
    "서울특별시 송파구 올림픽로 435",
    "경기도 고양시 일산동구 중앙로 1275",
    "인천 부평구 부평대로 177 센트럴타워 801호",
    "부산광역시 수영구 광안해변로 219",
    "대전 중구 대종로 480 2층",
    "광주광역시 남구 효덕로 100",
    "서울 중구 세종대로 110 시청역 지하상가",
    "경남 창원시 의창구 중앙대로 300",
    "충북 청주시 서원구 산남로 62 1104호",
    "제주특별자치도 제주시 노형로 83",
    "강원도 강릉시 경강로 2007 3층",
    "전남 여수시 좌수영로 15",
    "사당역 4번 출구 뒤편 상가 502호",
    "울산 동구 방어진순환도로 1000 하버뷰 7층",
]
ADDR_T = [
    "거주지 주소는 {A} 입니다",
    "택배 주소: {A}",
    "{A} 근처로 와주세요",
    "근무지는 {A} 입니다",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--extra", nargs="+", required=True, help="옛 v3·v4 jsonl 등 추가 제외 자료")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=99117)
    ap.add_argument("--per-template", type=int, default=7)
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
                "id": f"v5-{len(rows) + 1:04d}",
                "category": cat,
                "text": text,
                "spans": spans,
                "known_names": [],
                "template_id": tpl_id,
            }
        )

    kinds = {"N": "PERSON", "N2": "PERSON", "N3": "PERSON", "G": "PERSON", "A": "ADDRESS"}
    for cat, tpls in TEMPLATES.items():
        for ti, tpl in enumerate(tpls):
            for _ in range(a.per_template):
                slots = {
                    "N": rng.choice(SURNAMES) + rng.choice(givens),
                    "N2": rng.choice(SURNAMES) + rng.choice(givens),
                    "N3": rng.choice(SURNAMES) + rng.choice(givens),
                    "G": rng.choice(givens),
                }
                text, spans = render(tpl, slots, kinds)
                add(cat, text, spans, f"{cat}-{ti + 1:02d}")
    for i, s in enumerate(NEGATIVES):
        add("비인명", s, [], f"비인명-{i + 1:02d}")
    for ti, tpl in enumerate(ADDR_T):
        for addr in ADDRESSES:
            text, spans = render(tpl, {"A": addr}, kinds)
            add("주소", text, spans, f"주소-{ti + 1:02d}")

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
