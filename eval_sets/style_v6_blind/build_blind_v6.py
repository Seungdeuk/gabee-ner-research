"""블라인드 최종 평가셋 v6 생성(27-3절 설계안 실행). 생성만 한다 — 모델 평가는 별도 승인 후.

지금까지 이번 세션에서 쓴 **학습·개발·진단·탐색** 이름·문장 틀 전부와 겹치지 않는 새 합성 문장을
만든다. R0~R1b-A의 학습/held-out 이름은 커밋된 생성 스크립트에서 그대로 불러오고, 이번 세션의
진단용 탐색 스크립트(스크래치패드, 저장소 밖)에서 쓴 이름은 ``PROBE_ONLY_NAMES`` 에 직접 적어
같이 제외한다 — "학습에는 안 썼지만 내가 들여다본 적 있는 이름"까지 깨끗하게 뺀다.

7개 범주: 일반 메신저(흔한 이름) · 낯선 이름(조합형) · 애칭·줄임말 · ORG/LOC 표면형(실제 단어+기관
접미) · ORG/LOC 문맥형(이름+기관명사) · 조사·스팬 경계(건강검진, 전부 정상 동작을 기대) · 비인명
(과마스킹 확인).

실제 가족 대화는 전혀 쓰지 않는다(합성만). 생성 후 ``check_blind_v6.py`` 로 중복·품질을 점검한다.
"""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[6]  # apps/api/app/shared/privacy/eval/style_v6_blind -> repo root
sys.path.insert(0, str(ROOT / "apps/api/app/shared/privacy/eval/style_v4"))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect_prior_names() -> set[str]:
    """R0~R1b-A 전부(커밋된 생성 스크립트에서 직접 불러옴) + 이번 세션 진단 전용 탐색 이름."""
    r1 = _load(ROOT / "ml/ner/data/build_r1_collision_data.py", "r1mod")
    r2 = _load(ROOT / "ml/ner/data/build_r2_nickname_data.py", "r2mod")
    r1b = _load(ROOT / "ml/ner/data/build_r1b_location_suffix_data.py", "r1bmod")

    names: set[str] = set()
    for n, _ in r1.TRAIN_NAMES + r1.HOLDOUT_NAMES:
        names.add(n)
    for full, nick in r2.TRAIN_PAIRS + r2.HOLDOUT_PAIRS:
        names.add(full)
        names.add(nick)
    for n, _ in r1b.TRAIN_PAIRS + r1b.HOLDOUT_PAIRS:
        names.add(n)

    # R0: 완전미탐 36건 교차재현(22절)에서 다룬 14개 고유 이름
    names |= {
        "서명소",
        "고한라",
        "성경안",
        "전송유",
        "주소온",
        "문건환",
        "고가소",
        "고가철",
        "우송소",
        "강한소",
        "배송우",
        "민소솔",
        "나호유",
        "유다성",
    }

    # 진단 전용(학습에는 안 썼지만 스크래치패드 탐색 스크립트에서 직접 테스트한 이름) - 22·25·26절
    names |= {
        # 애칭 탐색(check_nickname_r0*.py)
        "준이",
        "민준",
        "안이",
        "빈이",
        "원이",
        "진이",
        "영이",
        "린이",
        "우이",
        "지안",
        "도빈",
        "예원",
        "유진",
        "선율",
        "가영",
        "태민",
        "하린",
        "은우",
        "결이",
        "헌이",
        "택이",
        "웅이",
        "엽이",
        # 고가소 계열 표면형 탐색(probe_org_collision*.py, probe_so_suffix.py)
        "민준호",
        "김민준",
        "박서아",
        "저가민",
        "전세호",
        "월세준",
        "신축아",
        "고층영",
        "민준소",
        "배송소",
        "주차소",
        "정류소",
        "고가민",
        "고가호",
        "고가영",
        "고가준",
        # 장소·기관 접미 넓게 검증(validate_suffix_hypothesis.py) - 실제단어 접두 8 x 접미 5
        *(
            f"{p}{s}"
            for p in ["예매", "환승", "대기", "검역", "출입", "접수", "계산", "휴게"]
            for s in ["소", "원", "관", "실", "장"]
        ),
        # 같은 스크립트의 이름형(무의미) 접두 8 x 접미 5
        *(
            f"{p}{s}"
            for p in ["가온", "나린", "다훈", "라연", "마준", "사윤", "아결", "자빈"]
            for s in ["소", "원", "관", "실", "장"]
        ),
    }
    return names


PROBE_ONLY_NAMES = collect_prior_names()


def particles(has_batchim: bool) -> dict[str, str]:
    return {
        "i_ga": "이" if has_batchim else "가",
        "eun_neun": "은" if has_batchim else "는",
        "a_ya": "아" if has_batchim else "야",
        "eul_reul": "을" if has_batchim else "를",
        "irang_rang": "이랑" if has_batchim else "랑",
    }


def has_batchim(syllable: str) -> bool:
    code = ord(syllable[-1]) - 0xAC00
    return code % 28 != 0


# ---------------------------------------------------------------------------
# 1) 일반 메신저(흔한 이름) - 새 조합, 전부 PROBE_ONLY_NAMES 와 안 겹침
# ---------------------------------------------------------------------------
GENERAL_NAMES = [
    "강태오",
    "노은찬",
    "문서윤",
    "배하림",
    "신가은",
    "오지후",
    "임도윤",
    "조은서",
    "최하람",
    "한유준",
    "황나윤",
    "구시윤",
    "류채은",
    "백서진",
    "심민재",
    "양하은",
]

GENERAL_TEMPLATES = [
    "{N}{i_ga} 오늘 학원 끝나고 바로 온대",
    "{N} 치과 예약 몇시로 했어?",
    "엄마가 {N} 간식 냉장고에 넣어놨대",
    "{N} 오늘따라 기분 좋아보이던데",
    "{N}{i_ga} 숙제 다 했다고 자랑하더라",
    "{N} 사진 가족방에 좀 올려줘",
    "할머니가 {N} 이번 주말에 온다고 좋아하셔",
    "{N}{i_ga} 축구화 새로 사달래",
    "{N} 열 좀 내렸는지 확인해볼래",
    "{N}{irang_rang} 같이 도서관 가기로 했어",
]  # 마지막 2개는 val 격 없음(평가셋은 split 없음, 전부 평가용)


# ---------------------------------------------------------------------------
# 2) 낯선 이름(조합형) - R0 스타일의 "낯설게 들리는" 조합, 전부 신규
# ---------------------------------------------------------------------------
UNUSUAL_NAMES = [
    "노현소",
    "단우리",
    "매소안",
    "번가율",
    "삼우진",
    "안전호",
    "오래민",
    "자유현",
    "창해나",
    "파도윤",
    "하모건",
    "극지온",
]
UNUSUAL_TEMPLATES = [
    "{N} 오늘 저녁엔 뭐 먹고 싶대?",
    "{N}{i_ga} 어제 늦게까지 안 자더라",
    "할머니가 {N} 용돈 주셨대",
    "{N} 머리 자른 거 봤어?",
    "{N}{i_ga} 줄넘기 대회 나간대",
]


# ---------------------------------------------------------------------------
# 3) 애칭·줄임말 - 새 받침음절 애칭(마지막 음절+이), 전부 신규
# ---------------------------------------------------------------------------
NICKNAME_PAIRS = [
    ("서강", "강이"),
    ("도연", "연이"),
    ("하담", "담이"),
    ("고신", "신이"),
    ("주찬", "찬이"),
    ("나곤", "곤이"),
]  # (전체이름, 애칭) - 애칭은 전부 "이"로 끝남(받침 없음)
NICKNAME_TEMPLATES = [
    "{N} 오늘 유치원에서 노래 배웠대",
    "{N}{i_ga} 밥 잘 먹었는지 물어봐줘",
    "삼촌이 {N} 사진 보내달래",
    "{N} 요즘 부쩍 말이 늘었어",
    "{N}{i_ga} 어제 넘어져서 무릎 다쳤대",
    "할머니가 {N} 보고싶다고 전화하셨어",
    "{N} 데리러 몇시에 갈까",
    "{N}{i_ga} 숙제 다 했다고 그러더라",
    "{N} 사진 가족방에 올라왔던데 봤어?",
    "{N}{i_ga} 오늘 발표 잘했대",
]


# ---------------------------------------------------------------------------
# 4) ORG/LOC 표면형 - 실제 단어 접두 + 기관 접미(소/원/관/실/장), 전부 신규 조합
# ---------------------------------------------------------------------------
LOCSUF_SURFACE = [
    ("할인소", False),
    ("예약관", True),
    ("문의실", True),
    ("안내장", True),
    ("상환관", True),
    ("대출원", True),
    ("등록관", True),
    ("면접소", False),
]
LOCSUF_SURFACE_TEMPLATES = [
    "{N} 오늘 늦게 끝난대",
    "{N}{i_ga} 숙제 검사받았대",
    "{N} 사진 좀 보내줘",
    "삼촌이 {N} 데리러 간대",
    "{N}{i_ga} 오늘 상 받았대",
]


# ---------------------------------------------------------------------------
# 5) ORG/LOC 문맥형 - 이름+기관명사 바로 결합(25·26절 "유다성 어린이집" 패턴), 전부 신규
# ---------------------------------------------------------------------------
LOCCTX_NAMES = ["하도연", "서나율", "고은결", "배시후"]
LOCCTX_INSTITUTIONS = ["어린이집", "유치원", "학원", "학교"]
LOCCTX_TEMPLATES = [
    "{N} {INST} 행사가 몇시래?",
    "{N} {INST} 선생님이 전화하셨어",
    "{N} {INST} 다녀왔대",
]


# ---------------------------------------------------------------------------
# 6) 조사·스팬 경계 건강검진 - 다양한 조사가 전부 정상적으로 안 붙는지(=정답은 이름만) 확인
#    여기는 "실패를 찾는" 범주가 아니라 "이미 잘 되던 게 안 깨졌는지" 확인하는 범주다.
# ---------------------------------------------------------------------------
BOUNDARY_NAMES = [
    "강이준",
    "노을하",
    "민서온",
    "백하늘",
    "오지연",
    "장우빈",
    "조하늘",
    "최은우",
]
# 각 이름에 받침여부와 무관하게 여러 조사를 직접 붙여 문장을 만든다(조사 자체가 정답 밖으로 빠져야 함)
BOUNDARY_TEMPLATES = [
    "{N}{eun_neun} 오늘 늦게 온대",
    "{N}{i_ga} 숙제 다 했대",
    "{N}{eul_reul} 데리러 가야 돼",
    "{N}{a_ya} 밥 먹어",
    "{N}{irang_rang} 같이 놀이터 갔어",
    "{N}한테 급하게 전화 좀 넣어줘",
    "{N}네 옆집으로 이사왔대",
    "{N}{irang_rang} {N2} 둘이 싸웠대",
]


# ---------------------------------------------------------------------------
# 7) 비인명(과마스킹 확인) - 전부 새 문장, 일부는 알려진 위험 단어(하늘·바다 등) 재사용해 지속 확인
# ---------------------------------------------------------------------------
NEG_SENTENCES = [
    "오늘 하늘 색깔이 완전 예쁘더라",
    "바다 보러 주말에 갈까 생각 중이야",
    "노을 지는 거 사진 찍었어",
    "달빛이 너무 밝아서 산책 나갔어",
    "보람을 느끼는 하루였어",
    "초원 풍경이 그림 같더라",
    "소망을 적어서 벽에 붙여놨어",
    "인연이라는 게 참 신기해",
    "주유소에 들러서 기름 넣었어",
    "도서관에서 책 두 권 빌렸어",
    "체육관 바닥 공사 중이래",
    "교실에 에어컨 새로 달았대",
    "사무실 이전한다고 하더라",
    "시장에서 떡볶이 사왔어",
    "운동장 조명이 꺼졌더라",
    "방이 또 엉망이야 치워야겠어",
    "밥이 너무 빨리 식었어",
    "눈이 펑펑 오고 있어",
    "꿈이 너무 생생해서 깼어",
    "길이 막혀서 늦을 것 같아",
    "잠이 안 와서 뒤척였어",
    "짐이 생각보다 많더라",
    "힘이 다 빠졌어 좀 쉴래",
    "대기실에 사람이 너무 많더라",
    "세차장 다녀왔더니 차가 반짝여",
    "매표소 앞이 줄이 길더라",
    "박물관 전시가 꽤 괜찮았어",
    "수영장 물이 좀 차갑더라",
    "할인 행사 끝났대 아쉽다",
    "예약 시간 바뀌었다고 문자왔어",
]


# ---------------------------------------------------------------------------
# 행 생성
# ---------------------------------------------------------------------------
def make_row(rid: str, category: str, text: str, spans: list[tuple[int, int]]) -> dict:
    return {
        "id": rid,
        "category": category,
        "text": text,
        "spans": [{"kind": "PERSON", "start": s, "end": e} for s, e in spans],
        "known_names": [],
        "template_id": category,
    }


def gen_single_name(
    names: list[str], templates: list[str], category: str, start_rid: int
) -> tuple[list[dict], int]:
    rows, rid = [], start_rid
    for name in names:
        p = particles(has_batchim(name[-1]))
        for tpl in templates:
            rid += 1
            text = tpl.format(N=name, **p)
            s = text.find(name)
            rows.append(make_row(f"v6-{rid:04d}", category, text, [(s, s + len(name))]))
    return rows, rid


def gen_nicknames(
    pairs: list[tuple[str, str]], templates: list[str], category: str, start_rid: int
) -> tuple[list[dict], int]:
    rows, rid = [], start_rid
    for _full, nick in pairs:
        for tpl in templates:
            rid += 1
            text = tpl.format(N=nick, i_ga="가")
            s = text.find(nick)
            rows.append(make_row(f"v6-{rid:04d}", category, text, [(s, s + len(nick))]))
    return rows, rid


def gen_locsuffix(
    pairs: list[tuple[str, bool]], templates: list[str], category: str, start_rid: int
) -> tuple[list[dict], int]:
    rows, rid = [], start_rid
    for name, batchim in pairs:
        p = particles(batchim)
        for tpl in templates:
            rid += 1
            text = tpl.format(N=name, **p)
            s = text.find(name)
            rows.append(make_row(f"v6-{rid:04d}", category, text, [(s, s + len(name))]))
    return rows, rid


def gen_locctx(
    names: list[str], institutions: list[str], templates: list[str], category: str, start_rid: int
) -> tuple[list[dict], int]:
    rows, rid = [], start_rid
    for name in names:
        for inst in institutions:
            for tpl in templates:
                rid += 1
                text = tpl.format(N=name, INST=inst)
                s = text.find(name)
                rows.append(make_row(f"v6-{rid:04d}", category, text, [(s, s + len(name))]))
    return rows, rid


def gen_boundary(
    names: list[str], templates: list[str], category: str, start_rid: int
) -> tuple[list[dict], int]:
    rows, rid = [], start_rid
    for i, name in enumerate(names):
        p = particles(has_batchim(name[-1]))
        for tpl in templates:
            rid += 1
            if "{N2}" in tpl:
                n2 = names[(i + 1) % len(names)]
                text = tpl.format(N=name, N2=n2, **p)
                s1 = text.find(name)
                s2 = text.find(n2, s1 + len(name))
                rows.append(
                    make_row(
                        f"v6-{rid:04d}", category, text, [(s1, s1 + len(name)), (s2, s2 + len(n2))]
                    )
                )
            else:
                text = tpl.format(N=name, **p)
                s = text.find(name)
                rows.append(make_row(f"v6-{rid:04d}", category, text, [(s, s + len(name))]))
    return rows, rid


def gen_negatives(sentences: list[str], category: str, start_rid: int) -> tuple[list[dict], int]:
    rows, rid = [], start_rid
    for text in sentences:
        rid += 1
        rows.append(make_row(f"v6-{rid:04d}", category, text, []))
    return rows, rid


def skeleton_from_spans(text: str, spans: list[dict]) -> str:
    out, pos = [], 0
    for sp in sorted(spans, key=lambda s: s["start"]):
        out.append(text[pos : sp["start"]])
        pos = sp["end"]
    out.append(text[pos:])
    return "".join(out)


def leak_check(rows: list[dict]) -> dict:
    """동결 평가셋 + 기존 모든 보강 템플릿과 스켈레톤·이름 중복을 검사한다."""

    frozen_skel: set[str] = set()
    jsonl_paths = [
        ROOT / "apps/api/app/shared/privacy/eval/style_family_realistic/realistic_family.jsonl",
        ROOT / "apps/api/app/shared/privacy/eval/style_family_realistic/diag_structures.jsonl",
        ROOT / "apps/api/app/shared/privacy/eval/style_family_realistic/r1_heldout_eval.jsonl",
        ROOT / "apps/api/app/shared/privacy/eval/style_family_realistic/r2_heldout_eval.jsonl",
        ROOT / "apps/api/app/shared/privacy/eval/style_family_realistic/r1b_heldout_eval.jsonl",
        ROOT / "apps/api/app/shared/privacy/eval/style_v4/evalset_v4.jsonl",
        ROOT / "apps/api/app/shared/privacy/eval/style_v5/locked_test_v5.jsonl",
        ROOT / "apps/api/app/shared/privacy/eval/baseline_v2/train.jsonl",
        ROOT / "apps/api/app/shared/privacy/eval/baseline_v2/val.jsonl",
        ROOT / "apps/api/app/shared/privacy/eval/baseline_v2/test.jsonl",
    ]
    for p in jsonl_paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            frozen_skel.add(skeleton_from_spans(row["text"], row.get("spans", [])))

    new_skel = {skeleton_from_spans(r["text"], r["spans"]) for r in rows}
    overlap_frozen = new_skel & frozen_skel

    import importlib.util

    spec = importlib.util.spec_from_file_location("_synth_check", ROOT / "ml/ner/data/synth.py")
    synth_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(synth_mod)
    existing_tpl_skel = {
        t.replace("{P}", "").replace("{G}", "").replace("{NK}", "").replace("{AL}", "")
        for _c, t in synth_mod.AUG_TEMPLATES + synth_mod.AUG_TEMPLATES_2 + synth_mod.AUG_TEMPLATES_3
    }
    bare_tpl_skel = set()
    for tpl in (
        GENERAL_TEMPLATES
        + UNUSUAL_TEMPLATES
        + NICKNAME_TEMPLATES
        + LOCSUF_SURFACE_TEMPLATES
        + LOCCTX_TEMPLATES
        + BOUNDARY_TEMPLATES
    ):
        bare = tpl
        for key in ("{N}", "{N2}", "{INST}", "{i_ga}", "{eun_neun}", "{a_ya}", "{eul_reul}"):
            bare = bare.replace(key, "")
        bare_tpl_skel.add(bare)
    overlap_tpl = bare_tpl_skel & existing_tpl_skel

    all_names_used = set()
    for r in rows:
        for sp in r["spans"]:
            all_names_used.add(r["text"][sp["start"] : sp["end"]])
    overlap_names = all_names_used & PROBE_ONLY_NAMES

    dup_sentences = len(rows) - len({r["text"] for r in rows})

    return {
        "n_rows": len(rows),
        "n_person_spans": sum(len(r["spans"]) for r in rows),
        "overlap_with_frozen_skeleton": sorted(overlap_frozen),
        "overlap_with_existing_templates": sorted(overlap_tpl),
        "overlap_with_prior_names": sorted(overlap_names),
        "duplicate_sentence_count": dup_sentences,
        "unique_person_names": sorted(all_names_used),
    }


def main() -> None:
    rows: list[dict] = []
    rid = 0

    r, rid = gen_single_name(GENERAL_NAMES, GENERAL_TEMPLATES, "일반_메신저", rid)
    rows += r
    r, rid = gen_single_name(UNUSUAL_NAMES, UNUSUAL_TEMPLATES, "낯선_이름", rid)
    rows += r
    r, rid = gen_nicknames(NICKNAME_PAIRS, NICKNAME_TEMPLATES, "애칭_줄임말", rid)
    rows += r
    r, rid = gen_locsuffix(LOCSUF_SURFACE, LOCSUF_SURFACE_TEMPLATES, "ORGLOC_표면형", rid)
    rows += r
    r, rid = gen_locctx(LOCCTX_NAMES, LOCCTX_INSTITUTIONS, LOCCTX_TEMPLATES, "ORGLOC_문맥형", rid)
    rows += r
    r, rid = gen_boundary(BOUNDARY_NAMES, BOUNDARY_TEMPLATES, "조사_경계건강검진", rid)
    rows += r
    r, rid = gen_negatives(NEG_SENTENCES, "비인명", rid)
    rows += r

    report = leak_check(rows)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "unique_person_names"},
            ensure_ascii=False,
            indent=1,
        )
    )
    print("카테고리별 PERSON span 수:")
    from collections import Counter

    cat_spans = Counter()
    cat_sent = Counter()
    for r in rows:
        cat_spans[r["category"]] += len(r["spans"])
        cat_sent[r["category"]] += 1
    for cat in cat_spans:
        print(f"  {cat}: 문장 {cat_sent[cat]}, span {cat_spans[cat]}")
    print(f"총 문장 {len(rows)}, 총 PERSON span {sum(len(r['spans']) for r in rows)}")

    out_path = HERE / "blind_v6.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("저장:", out_path)

    report_path = HERE / "blind_v6_generation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("생성 보고서:", report_path)


if __name__ == "__main__":
    main()
