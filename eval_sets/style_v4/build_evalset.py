"""문체별 개인정보 평가셋 v4(전부 합성)을 만든다. 저장소 밖 옛 v3(488문장, 2026-11-03에 이미 1회 소비됨)를
재사용하지 않고, 학습 풀·baseline_v2·v3 모두와 **이름·문장 틀이 겹치면 생성이 실패**하게 다시 만든 devset이다.

- 이름: 성은 학습과 같은 풀을 쓰되, 이름(given)은 학습 풀(``ml/ner/data/synth.py`` GIVEN2/GIVEN1) + ``baseline_v2``
  전체 분할 + 옛 v3(``evalset_v3.jsonl``)의 PERSON span 문자열을 모두 뺀 것만 쓴다(``exclusions.collect_excluded_names``).
- 문장 틀: 세 출처(synth.py의 TEMPLATES류, baseline_v2, v3) 어디에도 없는 새 문장만 쓴다. 슬롯(``{...}``)·이름 span을
  지운 "뼈대"가 하나라도 겹치면 ``assert_no_overlap`` 이 ``SystemExit`` 을 올린다.
- 문체: 가족 대화 / 배송·수령 / 명단·목록 / 공문·안내 / 기사체 / 호칭·조사 변형 / 이름만 부름 / 주소 / 비인명.

실행: python build_evalset.py --synth <ml/ner/data/synth.py> --baseline <baseline_v2 폴더 ...> --v3 <evalset_v3.jsonl> --out evalset_v4.jsonl
"""

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from exclusions import collect_excluded_names  # noqa: E402

SURNAMES = list(
    "김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민나진지"
)

# 처음 보는 이름 후보. v3 는 고정된 ~140개 목록을 썼는데, 제외 집합에 v3 자체(483개 이름, 거의 이 목록 전부를 이미 썼다)까지
# 더하면 그 목록 거의 전부가 막힌다(실측: 140개 중 3개만 생존). 그래서 "실제 사람 이름에 쓰인 음절"끼리 성·이름 자리를
# 서로 바꿔 다시 묶는다(v3가 쓴 140개 이름을 첫 음절/둘째 음절로 쪼개 교차 조합) — 아무 한글 음절이나 섞지 않으므로
# 일반 명사·불변어가 나올 확률이 (완전 임의 조합보다) 낮다. 그래도 우연히 실제 단어와 같아지는 조합은 남을 수 있어
# 검토 중 발견된 것만 BLOCKLIST 로 뺀다(비속어 1건 포함) — 완전한 사전 검증은 아니다.
_V3_NAMES = (
    "민규 지환 현성 동우 승환 도율 건후 시완 은호 재이 하온 태율 주혁 해성 윤성 정민 상우 재현 우혁 성준 대윤 지석 원준 찬영 영훈 준영 기현 병헌 "
    "철민 동진 성호 한성 민혁 재호 진호 상민 종민 현석 도영 시형 경민 종훈 우성 태훈 용민 근호 호진 덕수 명철 서율 가윤 채윤 하연 다윤 서희 예나 "
    "지현 수연 민지 은지 혜원 보라 유경 세린 주하 시연 나영 은하 소정 효진 미래 현아 지수 아영 연주 소라 승아 다경 초원 은빈 채연 보경 윤지 도연 "
    "가현 시아 서영 예원 지은 혜인 소은 진아 미주 수현 영은 송이 지연 승희 준호 지성 동건 태영 민호 서원 유건 정원 하늘 도훈 승찬 재우 성우 현철 "
    "다온 채아 서우 하음 유하 나린 소윤 아인 이레 지안 시하 은유 도아 주은 보윤 세아 연아 가인 효주 미소 수민 은솔 하빈 채민 연수 지율 수호 이현"
).split()
_GIVEN_BLOCKLIST = {
    "연결",
    "시민",
    "철원",
    "한식",
    "세율",
    "대지",
    "한우",
    "진지",
    "송별",
    "해서",
    "해주",
    "양보",
    "주주",
    "보지",
    "해결",
    "보호",
    "지원",
    "정지",
    "소주",
}
_syl_firsts = sorted({t[0] for t in _V3_NAMES if len(t) == 2})
_syl_seconds = sorted(
    {t[1] for t in _V3_NAMES if len(t) == 2} - {"늘"}
)  # "늘"(하늘)은 이름 음절이 아니라 뺀다
GIVEN_CANDIDATES = sorted({a + b for a in _syl_firsts for b in _syl_seconds} - _GIVEN_BLOCKLIST)

TEMPLATES: dict[str, list[str]] = {
    "가족대화": [
        "{N} 지금 올라오고 있대",
        "{N} 지금 숙제하고 있대",
        "{N|한테/한테} 저녁 먹었는지 물어봐",
        "{N} 오늘 늦게 온다고 했어",
        "할머니가 {N} 보고 싶다셔",
        "{N|이랑/랑} 놀이공원 가기로 했어",
        "{N|에게/에게} 생일선물 보냈어",
        "{N|이/가} 방금 도착했대",
    ],
    "배송수령": [
        "받으실 분 {N}",
        "수취인 성명: {N}",
        "{N}님 앞으로 배송 출발했습니다",
        "주문하신 {N}고객님 상품이 발송됐어요",
        "반송 요청자 {N}",
        "{N}씨 택배 보관함에 넣어뒀습니다",
        "현금 수령인 {N}",
        "예약자 성명란에 {N} 적어주세요",
    ],
    "명단목록": [
        "출석자: {N}, {N2}, {N3}",
        "가) {N}  나) {N2}  다) {N3}",
        "동아리 명단 - {N} / {N2} / {N3}",
        "오늘 발표자: {N}, {N2}",
        "참가 확정자에 {N}, {N2}님 올렸어",
        "모둠 구성 1모둠 {N} {N2} 2모둠 {N3}",
        "가입자 목록 {N} · {N2}",
        "접수자 {N}, {N2}, {N3}",
    ],
    "공문안내": [
        "{N} 담당 선생님께 확인 바랍니다",
        "{N} 학생 보호자님께 안내드립니다",
        "면담 대상: {N} (2학년 1반)",
        "자녀 {N}의 체험학습 신청서를 제출해 주세요",
        "{N}님의 접종 예약이 완료되었습니다",
        "{N} 회원님 가입이 승인되었습니다",
        "법정대리인 {N} 서명",
        "신청인 {N} 직인",
    ],
    "기사체": [
        "{N} 대표가 사업 계획을 공개했다",
        "{N} 의원이 개정안을 제출했다",
        '목격자 {N}(37)씨는 "놀랐다"고 전했다',
        "{N} 감독이 인터뷰에서 소감을 밝혔다",
        "검찰은 {N|를/를} 참고인으로 소환했다",
        "{N} 연구원이 논문을 공개했다",
        "{N} 선수가 역전 홈런을 쳤다",
        "{N} 군수는 피해 지역을 점검했다",
    ],
    "호칭조사": [
        "{N}님 반갑습니다",
        "{N} 씨 번호 좀 알려줘",
        "{N}선생님 고맙습니다",
        "{N|이/가} 전화했었어",
        "{N|이랑/랑} 밥 먹었어",
        "{N|한테/한테} 메모 남겨줘",
        "{N|의/의} 우산이야",
        "{N|은/는} 집에 갔어?",
    ],
    "이름만": [
        "{G|이/} 벌써 끝났어",
        "{G|아/야} 늦었어 서둘러",
        "{G|한테/한테} 문자해봐",
        "{G|이랑/랑} 공원 갔어",
        "{G|은/는} 열 난대",
        "{G|이/가} 발표했어",
        "아까 {G} 학교 갔어",
        "{G|네/네} 옆집이야",
    ],
}

# 비인명 음성 문장: 이름처럼 보이는 일반어(하늘·바다·기쁨·초원 등, v3 과마스킹 이력)를 포함
NEGATIVES = [
    "하늘이 흐려서 우산을 챙겼어",
    "바다로 드라이브 다녀왔어",
    "기쁨이 넘치는 하루였어",
    "초원에서 사진을 많이 찍었어",
    "노을이 예뻐서 한참 봤어",
    "구름이 많아서 비 올 것 같아",
    "겨울 코트를 꺼내 입었어",
    "여름 휴가 계획을 세웠어",
    "새벽에 택배가 도착했어",
    "달빛이 밝아서 산책했어",
    "서류 접수증을 다시 뽑아줘",
    "지원 동기를 다시 써야 해",
    "분홍색 커튼을 새로 달았어",
    "보람을 느끼는 일이었어",
    "마당 잔디를 깎았어",
    "저녁은 비빔밥 먹자",
    "미래를 위해 저축을 늘렸어",
    "소음이 심해서 잠을 못 잤어",
    "지구 온도가 계속 오르고 있대",
    "초록색 가방을 새로 샀어",
    "도서관에서 자료를 찾아왔어",
    "접수 번호를 다시 확인해줘",
    "주차 등록 방법을 알려줘",
    "진료 시간이 변경됐대",
    "전화 요금이 많이 나왔어",
    "결제가 두 번 됐나봐",
    "통장 잔고를 확인해봐",
    "배송비가 추가됐다고 해",
    "냉동실에 재료가 다 떨어졌어",
    "주차장 출입구가 막혔어",
    "오늘따라 날씨가 쌀쌀하다",
    "아이스커피 한 잔 하자",
    "버스를 놓쳐서 택시 탔어",
    "지갑을 어디 두고 왔는지 모르겠어",
    "회의실 예약이 겹쳤어",
    "인터넷이 자꾸 끊겨",
    "세탁기가 또 고장났어",
    "창문을 열어서 환기했어",
    "정류장에서 한참 기다렸어",
    "엘리베이터가 공사 중이래",
]

ADDRESSES = [
    "서울특별시 강남구 테헤란로 152",
    "경기도 수원시 영통구 광교로 145",
    "인천 남동구 구월동 e스마트타워 702호",
    "부산광역시 동래구 충렬대로 167",
    "대전 서구 둔산로 100 7층",
    "광주광역시 서구 상무중앙로 56",
    "서울 영등포구 여의대로 108 IFC몰 3층",
    "경북 경주시 보문로 73",
    "충남 아산시 배방읍 희망로 20 305호",
    "제주특별자치도 서귀포시 중앙로 45",
    "강원도 원주시 북원로 2 4층",
    "전북 익산시 함라로 88",
    "신촌역 2번 출구 옆 오피스텔 1501호",
    "울산 중구 성남로 200 다온빌딩 6층",
]
ADDR_T = [
    "자택 주소는 {A} 입니다",
    "배송 주소: {A}",
    "{A}쪽으로 와주세요",
    "사업장 위치는 {A} 입니다",
]

JOSA = re.compile(r"\{(N2|N3|N|G|A)(?:\|([가-힣]*)/([가-힣]*))?\}")
_SLOT = re.compile(r"\{[^}]*\}")


def has_batchim(word: str) -> bool:
    c = word[-1]
    return "가" <= c <= "힣" and (ord(c) - 0xAC00) % 28 != 0


def render(template: str, slots: dict[str, str], kind_of: dict[str, str]):
    """틀을 채우고 (문장, 정답 구간 목록)을 돌려준다. 조사는 정답 구간에 넣지 않는다."""
    out, spans, pos = [], [], 0
    for m in JOSA.finditer(template):
        out.append(template[pos : m.start()])
        slot, with_b, without_b = m.group(1), m.group(2), m.group(3)
        word = slots[slot]
        start = sum(len(x) for x in out)
        out.append(word)
        spans.append({"kind": kind_of[slot], "start": start, "end": start + len(word)})
        if with_b is not None:
            out.append(with_b if has_batchim(word) else without_b)
        pos = m.end()
    out.append(template[pos:])
    return "".join(out), spans


def skeleton_from_spans(text: str, spans: list[dict]) -> str:
    """PERSON/ADDRESS span 을 지우고 남는 글자만 이어붙인 뼈대(이름·주소와 무관하게 "문장 틀"만 비교)."""
    out, pos = [], 0
    for sp in sorted(spans, key=lambda s: s["start"]):
        out.append(text[pos : sp["start"]])
        pos = sp["end"]
    out.append(text[pos:])
    return "".join(out)


_SYNTH_JOSA = re.compile(r"@([가-힣]+)/([가-힣]+)")


def skeleton_from_template(template: str) -> set[str]:
    """``{...}`` 슬롯을 지운 뼈대 집합을 돌려준다.

    내 템플릿은 조사를 슬롯 안({N|이/가})에 적어 한 번에 지워지지만, ``ml/ner/data/synth.py`` 는 조사를 슬롯
    **밖**(``{G}@이/가``)에 적는다. 이걸 통째로 지우면 "@이/가" 글자가 그대로 남아 어떤 생성 문장과도 다시는
    같아지지 않는다(뼈대 비교가 조용히 무력화된다) — 그래서 지우지 않고 가능한 조사 둘로 각각 펼쳐서 돌려준다
    (``{G}@이/가 X`` -> ``{"이 X", "가 X"}``), 실제로 조사가 어느 쪽으로 풀리든 뼈대 하나와는 반드시 맞아떨어지게.
    """
    base = _SLOT.sub("", template)
    skeletons = {base}
    while True:
        expanded: set[str] = set()
        changed = False
        for s in skeletons:
            m = _SYNTH_JOSA.search(s)
            if not m:
                expanded.add(s)
                continue
            changed = True
            for g in (m.group(1), m.group(2)):
                expanded.add(s[: m.start()] + g + s[m.end() :])
        skeletons = expanded
        if not changed:
            return skeletons


def _name_excluded(value: str, excluded_names: set[str]) -> bool:
    """``value`` 가 제외 집합의 어떤 이름과 같거나, 제외 집합의 이름이 ``value`` 로 끝나면(= 성 없이 given 만 써도
    그 given 이 이미 다른 성과 묶여 나온 적 있으면) True."""
    return any(e == value or e.endswith(value) for e in excluded_names)


def assert_no_overlap(rows: list[dict], excluded_names: set[str], template_texts: set[str]) -> None:
    """생성된 각 행이 제외 집합의 이름을 쓰거나 기존 문장 틀과 뼈대가 같으면 SystemExit(1)."""
    for row in rows:
        for sp in row.get("spans", []):
            if sp.get("kind") in ("PERSON", "PERSON_ALIAS"):
                value = row["text"][sp["start"] : sp["end"]]
                if _name_excluded(value, excluded_names):
                    raise SystemExit(f"이름 겹침: {value!r} (문장={row['text']!r})")
        skel = skeleton_from_spans(row["text"], row.get("spans", []))
        if skel in template_texts:
            raise SystemExit(f"문장 틀 겹침: {skel!r} (문장={row['text']!r})")


def _synth_skeletons(synth_path: Path) -> set[str]:
    spec = importlib.util.spec_from_file_location("_v4_synth_ref", synth_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    texts: list[str] = []
    for name in ("TEMPLATES", "AUG_TEMPLATES", "AUG_TEMPLATES_2", "AUG_TEMPLATES_3"):
        texts.extend(t for _, t in getattr(mod, name, []))
    out: set[str] = set()
    for t in texts:
        out |= skeleton_from_template(t)
    return out


def _jsonl_skeletons(path: Path) -> set[str]:
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.add(skeleton_from_spans(row["text"], row.get("spans", [])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--v3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--per-template", type=int, default=7)
    a = ap.parse_args()

    synth_path, v3_path = Path(a.synth), Path(a.v3)
    baseline_dirs = [Path(p) for p in a.baseline]

    excluded = collect_excluded_names(synth_path, baseline_dirs, [v3_path])
    # 주어진 이름(given)이 제외 집합의 어떤 전체 이름(성+이름)의 끝부분으로도 나타나면 뺀다(assert_no_overlap 과
    # 같은 _name_excluded 규칙). 성은 공유 풀이라 재사용하지만, given이 어떤 성과 결합해서라도 이미 쓰였으면
    # "새 이름"이 아니다.
    givens = [g for g in dict.fromkeys(GIVEN_CANDIDATES) if not _name_excluded(g, excluded)]
    if len(givens) < 60:
        raise SystemExit(
            f"처음 보는 given 이름이 너무 적다({len(givens)}개, 최소 60개 필요) — "
            "GIVEN_CANDIDATES 풀을 늘려야 한다"
        )

    template_texts = _synth_skeletons(synth_path)
    for d in baseline_dirs:
        for f in sorted(d.glob("*.jsonl")):
            template_texts |= _jsonl_skeletons(f)
    template_texts |= _jsonl_skeletons(v3_path)

    import random

    rng = random.Random(a.seed)
    rows: list[dict] = []

    def add(cat, text, spans, tpl_id):
        rows.append(
            {
                "id": f"v4-{len(rows) + 1:04d}",
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
