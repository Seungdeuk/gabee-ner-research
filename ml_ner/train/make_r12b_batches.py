"""R12b 데이터 작성 배치 사양·브리프 생성(사전 등록 NER_R12B_PREREGISTRATION.md §3). 풀(data_r12b/pools.json)에서 배치마다 겹치지 않는 슬라이스를 배정한다.
R12 와 달리: (1) 낱말 뒤 형태 단서 균형 규칙, (2) 장르 프로필 순환, (3) 작성 모델 4종 순환(models.json).
결과: data_r12b/batches/{id}.spec.json(검증기용), {id}.brief.md(작성자용), models.json, 작성 결과는 data_r12b/raw/{id}.jsonl.
"""

import json
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\venvs\ner-runs\data_r12b")
POOLS = json.loads((ROOT / "pools.json").read_text(encoding="utf-8"))
OUT = ROOT / "batches"
RAW = ROOT / "raw"
OUT.mkdir(exist_ok=True)
RAW.mkdir(exist_ok=True)
CHECKER = r"C:\Users\<user>\Documents\gabee-ner\ml\ner\train\check_r12b_marked.py"
PY = r"C:\Users\<user>\venvs\ner-runs\verify-py312\Scripts\python.exe"
COMPOUND = ["남궁", "황보", "제갈", "선우", "독고"]
SCEN = "일정·약속, 학교·학원·가정통신문, 가족 안부·심부름, 정산·돈·구매, 사진·영상 공유, 병원·건강, 여행·이동·집 방문, 잡담·감정 표현, 모임·행사 준비, 택배·배달, 청소·집안일, 식사·장보기"
GENRES = ["가족 단체 채팅방(여러 사람이 번갈아 말하는 짧은 줄들 중 한 줄)", "엄마·아빠가 아이에게/아이가 부모에게 보내는 1:1 메시지", "메모·일정 정리·할 일 목록 한 줄", "질문과 답이 이어지는 한 줄(질문 또는 답)",
          "학교·학원·어린이집 공지/가정통신문 문어체 한 줄", "혼잣말·일기·상태 메시지", "사진·영상 공유 캡션과 댓글", "전화·음성메시지를 글로 옮긴 듯한 구어체"]
MODELS = ["sonnet", "opus", "haiku", "fable"]

COMMON = """## 공통 규칙 (반드시 지키세요)
- 당신은 한국어 **가족 메신저 대화** 문장을 직접 씁니다. 문장 하나하나를 **서로 다른 문장 구조(틀)** 로 쓰세요. 단어나 이름만 바꾼 같은 틀 반복, 템플릿 치환 금지. 앞 두 어절과 끝 어절이 같은 틀은 한 배치에서 3번 넘게 쓰지 마세요.
- **이 배치의 장르 프로필**(문장마다 골고루 섞어 쓰세요): {genres}. 장면은 이 중에서 고르게: {scen}. 길이 8~150자(60자 이상 문장을 10% 이상), 짧은 단답·불완전 문장·`ㅋㅋ`·`ㅠㅠ`·이모티콘·줄임말·띄어쓰기 오류·조사 생략도 일부 섞으세요. 문장 위치(주어·목적어·문장 끝·호격)도 다양하게.
- 모든 문장은 합성입니다(실제 인물·유명인·실제 연락처·욕설·성적/폭력적 내용·정치 금지).
- 정답 표기: 사람 이름은 `[[PERSON|값]]`, 실제 별명·애칭은 `[[PERSON_ALIAS|값]]`, 지명·기관은 `[[LOCATION|값]]`·`[[ORGANIZATION|값]]`(없어도 됨). 값은 이름 자체만: **조사(가/는/은/를/을/한테/랑/아/야 …)·호칭(씨·님·선생님)·따옴표·공백은 태그 밖**. 호칭만(엄마·아빠·할머니·이모)은 태그하지 않고 `이름+가족호칭`이면 이름만 태그합니다.
- **애칭 `이`**: 받침 있는 낱말·이름 뒤에 `이`가 붙어 애칭이 되고 그 뒤에 또 조사(가·는·를·한테·랑·도·에게·야)가 오면 `이`는 애칭 접미사이므로 **값에 포함**합니다(예: `[[PERSON|값이]]가`). 받침 없는 이름 뒤 `가/는/를` 은 이름만 태그합니다.
{ambig}- 1글자 이름·반려동물 이름·사람 별명인지 구분이 안 되는 모호 문장·당신도 라벨을 확정하지 못하는 문장은 쓰지 마세요. 한 문장의 모든 사람 이름을 빠짐없이 태그하고, 음성 문장에는 숨은 이름이 없어야 합니다.
- 이 파일 밖의 어떤 데이터·코드·평가셋·저장소 파일도 열람하지 마세요(이 브리프와 체커만 사용).

## 출력 형식 (JSONL, 한 줄 = 한 문장, UTF-8)
```
{{"id": "{batch}-001", "category": "일정", "marked": "…[[PERSON|값]]…", "style": "반말대화", "scenario": "학원 픽업", "batch": "{batch}", "group": "{group}", "primary_stress_type": "유형코드"{pairfields}}}
```
- `id` 는 `{batch}-001` 부터 연속. `batch` 와 `group` 은 위 값 그대로. `primary_stress_type` 은 이 브리프의 유형 코드 중 하나.
{pairrule}
## 작업 순서
1. 문장을 직접 작성해 `{rawpath}` 에 저장합니다(Write 도구).
2. 검증기를 실행합니다: `{py} {checker} --spec {specpath} --data {rawpath}`
3. 오류가 0 이 될 때까지 고쳐 다시 저장·검증합니다(경고는 줄이세요). 오류 0 이면 마지막 응답에 **"OK <batch> rows=N"** 한 줄만 쓰세요(문장 원문을 응답에 붙이지 마세요).
"""

AMBIG_PAIR = ""  # 쌍 배치에서는 아래 PAIRRULE 이 `단어+이` 규칙을 따로 정한다
AMBIG_OTHER = "- 낱말 바로 뒤에 조사 `이`만 오는 모호 문장(주격 표지인지 애칭인지 불분명)은 쓰지 마세요.\n"

PAIRRULE = """- **대조쌍**: 배정된 단어마다 사람 문장 2개와 명사 문장 2개를 만들고, 첫 번째 사람/명사 문장은 `pair_id` 가 `{batch}-wNN-k1`, 두 번째는 `{batch}-wNN-k2`. 필드: `pair_id`, `role`(`person`|`noun`), `pair_word`(그 단어 그대로). 사람 문장 `primary_stress_type`=`D1P`, 명사 문장=`D1`.
  - 사람 문장: 그 단어가 **사람 이름이나 별명**으로 쓰이고 `[[PERSON|…]]`(또는 `PERSON_ALIAS`)로 태그(값은 단어 자체 또는 `단어+이`). 다른 사람 이름은 쓰지 마세요(가족 호칭은 허용).
  - 명사 문장: 같은 단어가 **일반 명사**로 쓰인 개인정보 없는 문장. **태그 없음**.
  - **가장 중요 — 조사·띄어쓰기만 보고 사람/명사를 맞힐 수 없게 쓰세요.** 낱말 바로 뒤 형태를 ⓐ조사 없음(공백·쉼표·문장 끝) ⓑ이/가 ⓒ은/는 ⓓ을/를 ⓔ랑·하고·와·과 ⓕ한테·에게·께 ⓖ아·야 ⓗ그 밖 으로 나눕니다. 검증기가 강제하는 규칙:
    1. 사람 문장들, 명사 문장들 각각에서 한 형태(예: ⓐ)가 60% 를 넘으면 안 됩니다.
    2. **각 단어의 k1 쌍(사람 k1, 명사 k1)은 낱말 뒤 형태가 같아야 합니다**(예: 둘 다 ⓑ, 또는 둘 다 ⓒ). 사람인지 사물인지는 앞뒤 낱말·서술어·장면으로만 가려져야 합니다. k2 쌍은 자유.
    3. 명사 문장의 10% 이상은 ⓕ·ⓖ 형태(예: 사물에 말 거는 `~야`, 의인화·비유 아닌 자연스러운 `~한테 물어보다`), 사람 문장의 20% 이상은 ⓐ(조사 없음) 형태(`단어 오면`, `단어, 오늘`, `…단어` 문장 끝).
  - **받침 있는 낱말의 `단어+이`**: 사람 문장에서 `[[PERSON|단어이]] 왔어`(이 뒤에 다른 조사 없이 공백)처럼 쓰는 것을 **허용·권장**합니다(태그에는 `이` 포함). 같은 낱말 명사 문장의 주격 `단어이 맑다` 와 겉모습이 같고 서술어·장면으로만 가려지게 쓰세요(이 경우 두 문장 모두 ⓑ).
  - 사람 문장의 **30% 이상은 `이름표·별명·이름이·불러·애칭` 같이 이름임을 직접 말하는 낱말 없이** 행동·관계·호칭·서술어로만 사람임이 드러나게 쓰세요.
  - 두 문장은 장면·길이·문장 위치·틀이 서로 달라야 하고 사람 문장이 항상 문두 주어가 되면 안 됩니다. 당신도 가르기 어려운 문장은 만들지 마세요.
"""

CAT = {
    "pairs": {"desc": "D1P/D1 대조쌍 (일반 명사와 같은 모양의 사람 이름)"},
    "p2": {"desc": "성 없는 2글자 이름·전체 3글자 이름 + 조사·호격", "types": {"A1": "성 없는 2글자 이름(배정 '성 없는 이름'만), 조사 있음·없음·호격(~아/야)·문말·붙여쓰기 섞음", "B2": "성+이름 3글자(배정 '전체 이름'만) + 조사(이가·은·를·한테·랑 …)·호격이 붙은 형태"}},
    "p3": {"desc": "별명·애칭", "types": {"ALIAS": "실제 별명·애칭 자체를 `PERSON_ALIAS` 로 태그(별명은 직접 만드세요: 가족끼리 부르는 애칭, 줄임말, 수식어형 별명 등). 일부(30% 이내)는 배정 '전체 이름'과 한 문장에 함께(이름은 PERSON, 별명은 PERSON_ALIAS)"}},
    "p4": {"desc": "인접·나열 다인물", "types": {"C2": "이름 둘이 **바로 인접**: 공백 없이 붙임·쉼표/중점 나열·`랑/하고/와` 접속·이름+별명(배정 이름만, 모두 태그)", "D3": "한 문장에 사람 2~3명(서로 다른 관계), 모두 태그(배정 이름만)"}},
    "p5": {"desc": "경계·끝글자 취약 이름", "types": {"B1": "3~4글자 이름 뒤에 **조사가 없는** 위치(쉼표·문장 끝·괄호·따옴표·`씨`/`님` 앞·가족 호칭 앞)", "BND": "이름 끝 글자가 조사·어미처럼 보이는 이름(…아/…야/…이/…가/…도/…랑/…은/…는 로 끝나는 배정 이름)이 문장 속에서 쓰인 경우, 이름 전체를 태그(조사는 밖)"}},
    "p6": {"desc": "가족 채팅 자연문", "types": {"FAM_POS": "사람 이름이 들어간 자연스러운 가족 채팅(배정 이름만, 모두 태그, 다양한 장면)", "FAM_NEG": "개인정보 없는 자연스러운 가족 채팅: 가족 호칭만 나오고 이름이 없는 메시지, 일상 대화, 날짜·금액, 이름처럼 보이지만 이름이 아닌 일반어·활용형·어구, 지명·기관명이 이름처럼 보이는 문장(지명·기관만 태그). **PERSON 태그 없음**, 숨은 이름 없음"}},
}
CAT_TYPE = {}
for c in CAT.values():
    CAT_TYPE.update(c.get("types", {}))
CAT_TYPE.update({"D1P": "일반 명사와 같은 모양의 **사람 이름** 문장(대조쌍의 사람 문장)", "D1": "같은 단어가 **일반 명사**로 쓰인 개인정보 없는 문장(대조쌍의 명사 문장)"})

models = {}
counter = {"i": 0}


def mk(bid, group, count, mix, names=None, words=None, per_word=None, desc=""):
    spec = {"batch": bid, "group": group, "count": count, "mix": mix}
    if names:
        spec.update(names)
    if words:
        spec["words"] = words
        spec["per_word"] = per_word
    (OUT / f"{bid}.spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    i = counter["i"]
    counter["i"] += 1
    models[bid] = MODELS[i % len(MODELS)]
    g = [GENRES[(i * 3 + k) % len(GENRES)] for k in range(3)]
    pairfields = ', "pair_id": "…", "role": "person|noun", "pair_word": "…"' if words else ""
    head = f"# R12b 데이터 작성 브리프 {bid} ({group})\n\n**유형·수량: {desc}** (총 {count}문장)\n\n"
    body = []
    if mix:
        body.append("### 유형 코드와 수량\n" + "\n".join(f"- `{t}` {n}문장" + (f": {CAT_TYPE.get(t, '')}" if CAT_TYPE.get(t) else "") for t, n in mix.items()))
    if names:
        if names.get("full_names"):
            body.append("### 배정 '전체 이름'(성+이름 3글자, PERSON 값은 이 목록의 이름만, 같은 이름 최대 2번)\n" + " ".join(names["full_names"]))
        if names.get("bare_names"):
            body.append("### 배정 '성 없는 이름'(2글자, PERSON 값은 이 목록의 이름만, 같은 이름 최대 2번)\n" + " ".join(names["bare_names"]))
        if names.get("long_names"):
            body.append("### 배정 '4글자 이름'(복성+이름)\n" + " ".join(names["long_names"]))
    if words:
        body.append(f"### 배정 단어({len(words)}개, 단어마다 사람 {per_word['person']}문장 + 명사 {per_word['noun']}문장)\n" + " ".join(words))
    common = COMMON.format(genres="; ".join(g), scen=SCEN, batch=bid, group=group, pairfields=pairfields, pairrule=PAIRRULE.format(batch=bid) if words else "",
                           ambig=AMBIG_PAIR if words else AMBIG_OTHER, rawpath=RAW / f"{bid}.jsonl", py=PY, checker=CHECKER, specpath=OUT / f"{bid}.spec.json")
    (OUT / f"{bid}.brief.md").write_text(head + "\n\n".join(body) + "\n\n" + common, encoding="utf-8")


batches = []
for group in ("train", "val", "dev"):
    W, F, B = POOLS["words"][group], POOLS["full_names"][group], POOLS["bare_names"][group]
    BND = POOLS["bnd_names"][group]
    longs = [COMPOUND[i % len(COMPOUND)] + B[(len(B) - 1 - i) % len(B)] for i in range(100)]
    fp = bp = lp = 0

    def take(seq_name, n, _g=group):
        global fp, bp, lp
        if seq_name == "full":
            s = F[fp:fp + n]
            fp += n
        elif seq_name == "bare":
            s = B[bp:bp + n]
            bp += n
        else:
            s = longs[lp:lp + n]
            lp += n
        return s

    g0 = group[0].upper()
    if group == "train":
        for i in range(12):
            bid = f"T-PAIR-{i + 1:02d}"
            mk(bid, group, 100, {"D1P": 50, "D1": 50}, words=W[i * 25:(i + 1) * 25], per_word={"person": 2, "noun": 2}, desc="D1P/D1 대조쌍")
            batches.append(bid)
        for i in range(2):
            bid = f"T-P2-{i + 1:02d}"
            mk(bid, group, 140, {"A1": 70, "B2": 70}, names={"full_names": take("full", 70), "bare_names": take("bare", 70)}, desc=CAT["p2"]["desc"])
            batches.append(bid)
        bid = "T-P3-01"
        mk(bid, group, 100, {"ALIAS": 100}, names={"full_names": take("full", 30)}, desc=CAT["p3"]["desc"])
        batches.append(bid)
        for i in range(2):
            bid = f"T-P4-{i + 1:02d}"
            mk(bid, group, 100, {"C2": 60, "D3": 40}, names={"full_names": take("full", 80), "bare_names": take("bare", 50)}, desc=CAT["p4"]["desc"])
            batches.append(bid)
        for i in range(2):
            bid = f"T-P5-{i + 1:02d}"
            mk(bid, group, 80, {"B1": 50, "BND": 30}, names={"full_names": take("full", 60) + BND[i * 20:(i + 1) * 20], "long_names": take("long", 25)}, desc=CAT["p5"]["desc"])
            batches.append(bid)
        for i in range(2):
            bid = f"T-P6-{i + 1:02d}"
            mk(bid, group, 170, {"FAM_POS": 60, "FAM_NEG": 110}, names={"full_names": take("full", 40), "bare_names": take("bare", 40)}, desc=CAT["p6"]["desc"])
            batches.append(bid)
    else:
        for i in range(2):
            bid = f"{g0}-PAIR-{i + 1:02d}"
            mk(bid, group, 120, {"D1P": 60, "D1": 60}, words=W[i * 30:(i + 1) * 30], per_word={"person": 2, "noun": 2}, desc="D1P/D1 대조쌍")
            batches.append(bid)
        for i in range(2):
            bid = f"{g0}-MIX-{i + 1:02d}"
            mix = {"A1": 8, "B2": 8, "ALIAS": 5, "C2": 8, "D3": 7, "B1": 7, "BND": 5, "FAM_POS": 20, "FAM_NEG": 32}
            mk(bid, group, sum(mix.values()), mix, names={"full_names": take("full", 50) + BND[i * 6:(i + 1) * 6], "bare_names": take("bare", 40), "long_names": take("long", 12)}, desc="이름 형태 혼합(유형별 수량은 아래)")
            batches.append(bid)
json.dump(batches, open(OUT / "index.json", "w", encoding="utf-8"), ensure_ascii=False)
json.dump(models, open(ROOT / "models.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
tot = {}
for b in batches:
    s = json.loads((OUT / f"{b}.spec.json").read_text(encoding="utf-8"))
    tot[s["group"]] = tot.get(s["group"], 0) + s["count"]
print("배치", len(batches), "그룹별 문장 수", tot, "모델", {m: list(models.values()).count(m) for m in MODELS})
