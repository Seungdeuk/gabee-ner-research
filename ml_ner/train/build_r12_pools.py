"""R12 데이터 작성용 풀 배정(사전 등록 NER_R12_PREREGISTRATION.md §3). 학습·검증·개발의 D1P 단어와 이름 풀을 서로 겹치지 않게 미리 나눈다.

- D1P 단어: 일반 명사이면서 사람 이름·별명으로도 쓰일 수 있는 2~3글자 낱말 후보에서 R11 데이터의 pair_word 와 겹치는 것을 뺀 뒤 train/val/dev 로 나눈다.
- 이름 풀: NVIDIA Nemotron-Personas-Korea(CC BY 4.0)에서 뽑은 전체 이름(성+이름 3자)과 성 없는 이름(2자) 중 모델 어휘의 통단어와 겹치지 않고
  R11 개발·검증 정답 이름과도 겹치지 않는 것만 쓴다. 풀은 세트 사이에 겹치지 않는다.
- 결과는 data_r12/pools.json (Git 비추적). 사용: python build_r12_pools.py
"""

import collections
import json
import random
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\venvs\ner-runs")
DATA11 = ROOT / "data_r11"
OUT = ROOT / "data_r12"
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\81d4ea6423f385fa\58fea5dc-3059-42a5-9d4f-5a126dc2262a\scratchpad")
TOK = ROOT / "r11" / "export" / "r11-B-s21-ep5" / "tokenizer.json"
SEED = 20261126

WORDS = """
하늘 바다 구름 노을 햇살 이슬 새벽 봄비 여름 가을 겨울 단풍 눈꽃 안개 서리 무지개 소나기 바람 산들 여울 물결 파도 모래 호수 들판 언덕 달빛 별빛 해님 달님 은하 우주 새싹
꽃잎 들꽃 민들레 국화 장미 백합 개나리 진달래 목련 벚꽃 수국 라일락 동백 해바라기 코스모스 튤립 앵두 포도 딸기 자두 복숭아 사과 귤 유자 모과 호두 밤톨 도토리 대추 보리 쌀알 콩알
팥죽 옥수수 감자 고구마 호박 가지 오이 당근 마늘 토마토 상추 쑥갓 냉이 달래 미나리 고사리 도라지 더덕 버섯 송이 표고 은행 잣나무 소나무 대나무 버드 단감 홍시 곶감 수박 참외 멜론 키위 망고
만두 호떡 붕어빵 찹쌀 떡볶이 김밥 라면 우동 국수 쿠키 마카롱 초코 푸딩 젤리 사탕 캔디 솜사탕 팝콘 호빵 꿀떡 인절미 약과 식혜 두부 콩떡 와플 케이크 머핀 도넛 크림 요거트 우유 꿀벌 모찌 단팥 크루아상 에그 샌드
나비 참새 제비 종달 병아리 오리 거북 토끼 다람쥐 고래 돌고래 여우 사슴 판다 코알라 기린 사자 호랑 늑대 부엉이 올빼미 까치 두루미 학 공작 꾀꼬리 뻐꾸기 개구리 올챙이 잠자리 반딧불 무당벌레 나방 매미 귀뚜라미 메뚜기
우산 지우개 리본 풍선 연필 가방 모자 장갑 목도리 단추 열쇠 거울 시계 구슬 방울 종이 실타래 바늘 가위 붓 물감 크레용 공책 책갈피 달력 지도 나침반 등불 촛불 횃불 사다리 징검다리 대문 마당 지붕 창문 우체통
행복 사랑 소망 희망 기쁨 믿음 지혜 보람 다짐 온유 평화 자유 용기 정직 성실 우정 미소 웃음 눈물 약속 추억 기억 다행 축복 감사 은총 자비 인내 겸손 정의 진실 순수 청춘 열정 열매
가람 나래 다솜 라온 보미 새롬 하람 해솔 늘봄 한결 소담 아라 아리 이레 이랑 마루 누리 미르 도담 두리 소리 노래 가락 장단 어울 한울 푸른 초롱 샛별 별똥 햇귀 윤슬 아침 한낮 저녁노을
산호 진주 보석 호박돌 수정 비취 청자 백자 옥빛 은빛 금빛 쪽빛 연두 하양 노랑 파랑 초록 보라 분홍 주황 갈색 회색 살구 개나리빛 자주 연분홍
버들 갈대 억새 이끼 솔잎 댓잎 연꽃 연잎 부들 창포 매화 난초 국수나무 감나무 밤나무 은행나무 느티 팽나무 향기 꽃향기 솔향 숲길 오솔길 샛길 둘레길 언덕길
구슬비 안개꽃 눈사람 눈송이 함박눈 서릿발 보름달 초승달 그믐달 별자리 북극성 은행잎 낙엽 솔방울 밤송이 풀잎 풀꽃 꽃송이 꽃다발 꽃씨 꽃길 별빛달 달무리
복덩 복실 보배 보물 알밤 알콩 달콩 깜찍 똘망 오롱 조롱 토실 말랑 몽실 몽글 뭉치 동글 반짝 반달 해달 햇볕 햇병아리 새내기 막내둥이 귀염 사랑둥이 잎새 이삭 씨앗 열쇠고리 나무늘보
소금 설탕 후추 간장 된장 고추장 참기름 들기름 식초 깨소금 물엿 조청 누룽지 숭늉 수제비 칼국수 비빔밥 잡채 전골 부침개 파전 오징어 새우 조개 멸치 고등어 갈치 꽁치 명태 대구살 장어 문어 낙지
기러기 갈매기 독수리 비둘기 칠면조 타조 앵무 카나리아 햄스터 다마스 청설모 두더지 고슴도치 너구리 오소리 멧돼지 수달 해달이 물개 바다표범 펭귄 북극곰 하마 코끼리 원숭이 침팬지 얼룩말
연필꽂이 필통 책가방 도시락 물병 수저 젓가락 접시 사발 주전자 냄비 프라이팬 국자 뚝배기 항아리 독 바구니 소쿠리 돗자리 방석 이불 베개 담요 수건 손수건 양말 신발 운동화 슬리퍼 장화
시냇물 샘물 약수 폭포 계곡 개울 강물 나루 포구 항구 등대 섬마을 갯벌 해변 백사장 몽돌 바위 절벽 동굴 고개 재 오솔 골목 대로 광장 다리목 정류장 나들목
햇감자 풋고추 청양 홍당무 열무 얼갈이 배추 양배추 브로콜리 파프리카 아보카도 블루베리 체리 레몬 라임 자몽 오렌지 석류 무화과 대봉 단호박 밤호박 완두 강낭콩 녹두 수수 조 기장 율무
소원 다정 상냥 영롱 청아 고운 맑음 밝음 따뜻 포근 아늑 든든 푸근 느긋 명랑 쾌활 영특 총명 씩씩 의젓 늠름 단아 우아 해맑 순둥 개구리왕 꼬마 아가 새댁
"""


def rows_of(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def pair_words_in(files):
    out = set()
    for f in files:
        p = DATA11 / f
        if p.exists():
            for r in rows_of(p):
                if r.get("pair_word"):
                    out.add(r["pair_word"])
    return out


def gold_names(files):
    out = set()
    for f in files:
        p = DATA11 / f
        if p.exists():
            for r in rows_of(p):
                for s in r.get("spans", []):
                    if (s.get("label") or s.get("mapped_kind")) in ("PERSON", "PERSON_ALIAS"):
                        out.add(r["text"][s["start"]:s["end"]])
    return out


def main():
    rng = random.Random(SEED)
    words = []
    for w in WORDS.split():
        if 2 <= len(w) <= 3 and w not in words:
            words.append(w)
    r11_train_words = pair_words_in(["r11_B_extra.jsonl", "r11_Dprime_train.jsonl"])
    r11_devval_words = pair_words_in(["r11_dev.jsonl", "r11_val.jsonl"])
    r11_words = r11_train_words | r11_devval_words
    fresh = [w for w in words if w not in r11_words]  # R11 의 어떤 데이터에도 없던 새 단어
    train_only = [w for w in words if w in r11_train_words and w not in r11_devval_words]  # R11 학습에만 있던 단어(학습 풀에만 허용)
    rng.shuffle(fresh)
    rng.shuffle(train_only)
    ntr, nv, nd = 280, 60, 60
    # 검증·개발 단어는 R11 어느 데이터와도 겹치지 않는 새 단어만, 남은 새 단어 + R11 학습 전용 단어로 학습 풀을 채운다
    if len(fresh) < nv + nd + 1 or len(fresh) - nv - nd + len(train_only) < ntr:
        raise SystemExit(f"단어가 모자란다: 새 단어 {len(fresh)}, 학습 풀 가능 {len(fresh) - nv - nd + len(train_only)} < {ntr} (단어 목록을 늘려야 한다)")
    val_w, dev_w = fresh[:nv], fresh[nv:nv + nd]
    train_w = (fresh[nv + nd:] + train_only)[:ntr]
    wp = {"train": train_w, "val": val_w, "dev": dev_w}
    vocab = {t for t in json.loads(TOK.read_text(encoding="utf-8"))["model"]["vocab"] if not t.startswith(("##", "[")) and all("가" <= c <= "힣" for c in t)}
    nv_full = json.loads((SCRATCH / "nv_names_full.json").read_text(encoding="utf-8"))["full"]
    seen = gold_names(["r11_dev.jsonl", "r11_val.jsonl", "baseline_val.jsonl", "synth_val.jsonl"])
    seen_given = {n[1:] for n in seen if len(n) == 3} | {n for n in seen if len(n) == 2}
    allwords = set(words)
    fulls = sorted(n for n, c in nv_full.items() if c >= 2 and n not in vocab and n not in seen and len(n) == 3 and n[1:] not in vocab and n[1:] not in allwords)
    given = collections.Counter()
    for n, c in nv_full.items():
        if len(n) == 3:
            given[n[1:]] += c
    bares = sorted(g for g, c in given.items() if c >= 3 and g not in vocab and g not in seen_given and g not in allwords and len(g) == 2)
    rng.shuffle(fulls)
    rng.shuffle(bares)
    fp = {"train": fulls[:700], "val": fulls[700:850], "dev": fulls[850:1000]}
    bp = {"train": bares[:600], "val": bares[600:750], "dev": bares[750:900]}
    for k in ("train", "val", "dev"):
        assert not (set(fp[k]) & set(w for ws in wp.values() for w in ws))
    for a in ("train", "val", "dev"):
        for b in ("train", "val", "dev"):
            if a < b:
                assert not (set(fp[a]) & set(fp[b])) and not (set(bp[a]) & set(bp[b])) and not (set(wp[a]) & set(wp[b]))
    OUT.mkdir(exist_ok=True)
    (OUT / "pools.json").write_text(json.dumps({"words": wp, "full_names": fp, "bare_names": bp, "r11_pair_words_excluded": len(r11_words)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"단어 후보 {len(words)} → R11 과 겹쳐 제외 {len(words) - len(fresh)} → 새 단어 {len(fresh)} | train {len(wp['train'])} val {len(wp['val'])} dev {len(wp['dev'])}")
    print(f"전체 이름 후보 {len(fulls)}(배정 {sum(len(v) for v in fp.values())}), 성 없는 이름 후보 {len(bares)}(배정 {sum(len(v) for v in bp.values())})")


if __name__ == "__main__":
    main()
