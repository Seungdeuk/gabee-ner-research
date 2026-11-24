"""실험 B(병합 완화)가 인접한 "서로 다른" 개체를 잘못 하나로 합치는지 스트레스 테스트한다 — 읽기
전용. 안전한 기존 이름(GIVEN2/NICKS)만 쓴다. ner.py·merge_ner 미수정(실험 B 쪽은 평가 스크립트
안에서만 디코딩을 바꾼 사본을 쓴다), 모델·DB·설정 미수정, 외부 LLM 미호출.

문장은 두 그룹: (a) 공백으로 정상 분리된 통제 문장(둘 다 안전해야 정상) (b) 쉼표·무공백 등
일부러 구분을 없앤 스트레스 문장(실험 B가 두 개체를 하나로 잘못 합치는지 보는 용도).
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from app.shared.config import settings  # noqa: E402
from app.shared.privacy import detectors, ner  # noqa: E402
from app.shared.privacy.types import Kind  # noqa: E402

import experiment_b_relaxed_merge as eb  # noqa: E402

CASES = [
    # --- 통제(공백/쉼표로 정상 분리, 전부 안전해야 함) ---
    ("control_space", "민준이랑 서윤이가 같이 놀았어", ["민준", "서윤"]),
    ("control_space2", "민준이가 서윤이 손을 잡았어", ["민준", "서윤"]),
    ("comma_nospace", "민준,서윤이가 같이 왔어", ["민준", "서윤"]),
    ("control_end_position", "오늘 유치원에서 민준이랑 서윤이를 만났어", ["민준", "서윤"]),
    ("control_three_people", "민준이랑 서윤이랑 지호가 같이 놀았어", ["민준", "서윤", "지호"]),
    ("control_particle_hante", "민준이가 서윤이한테 전화했어", ["민준", "서윤"]),
    ("control_particle_wa", "민준이와 서윤이가 같이 왔어", ["민준", "서윤"]),
    ("control_particle_object", "선생님이 민준이와 서윤이를 칭찬했어", ["민준", "서윤"]),
    ("control_family_msg_template", "민준이가 서윤이 손 잡고 유치원 갔어", ["민준", "서윤"]),
    ("control_alias_then_name", "민준이형이 서윤이를 데려다줬어", ["민준이형", "서윤"]),
    # --- 스트레스(일부러 무공백/붙여씀, 과잉병합 위험 확인용) ---
    ("concat_nospace", "민준서윤이가 같이 왔어", ["민준", "서윤"]),
    ("concat_nospace2", "지호민준이가 같이 놀았어", ["지호", "민준"]),
    ("concat_nospace_end", "오늘 민준서윤이를 만났어", ["민준", "서윤"]),
    ("alias_concat", "민준이형서윤이가 같이 왔어", ["민준이형", "서윤"]),
    ("alias_space_ctrl", "민준이형 서윤이가 같이 왔어", ["민준이형", "서윤"]),
    ("nick_concat", "콩이민준이가 같이 놀았어", ["콩이", "민준"]),
    ("nick_space_ctrl", "콩이 민준이가 같이 놀았어", ["콩이", "민준"]),
    ("alias_alias_concat", "민준이형지호형이 같이 왔어", ["민준이형", "지호형"]),
    ("three_people_nospace", "민준서윤지호가 같이 놀았어", ["민준", "서윤", "지호"]),
]


def run(model_dir: str, expect_fp12: str, use_relaxed: bool):
    ner.reset()
    settings.privacy_ner_mode = "required"
    settings.privacy_ner_model_dir = model_dir
    settings.privacy_ner_threads = 2
    status = ner.ner_status()
    assert status["fingerprint"][:12] == expect_fp12, status["fingerprint"][:12]
    ner._cache.clear()
    insp = eb.TokenInspector(model_dir) if use_relaxed else None

    out = []
    for name, text, entities in CASES:
        if use_relaxed:
            toks = insp.token_predictions(text)
            relaxed = eb.decode_relaxed(toks)
            ner_spans = eb.to_nerspans(relaxed, text)
            rule = detectors.detect(text, ())
            final = ner.merge_ner(text, rule, ner_spans)
        else:
            final = ner.detect_with_ner(text, ())
        spans = [(s.kind.name, s.start, s.end, text[s.start:s.end]) for s in final]
        # 두 개체가 하나의 span 으로 잘못 합쳐졌는지: 두 엔티티 이름이 모두 "부분 문자열"로만 들어있는
        # 하나의 PERSON span 이 있고, 그 span 길이가 두 엔티티 전부를 포함할 만큼 과도하게 긴 경우를 본다
        over_merged = any(
            s.kind is Kind.PERSON and sum(1 for e in entities if text[s.start:s.end].find(e) >= 0) >= 2
            for s in final
        )
        out.append({"case": name, "text": text, "entities": entities, "final_spans": spans, "over_merged_guess": over_merged})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--r6-candidate", required=True)
    ap.add_argument("--r7a-candidate", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    out = {
        "R6_normal": run(a.r6_candidate, "c59b174ba126", False),
        "R6_B": run(a.r6_candidate, "c59b174ba126", True),
        "A_normal": run(a.r7a_candidate, "38a93e174acd", False),
        "A_B": run(a.r7a_candidate, "38a93e174acd", True),
    }
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for cfg, cases in out.items():
        print(f"== {cfg} ==")
        for c in cases:
            flag = " <<< 과잉병합 의심" if c["over_merged_guess"] else ""
            print(f"  {c['case']}: {c['final_spans']}{flag}")
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
