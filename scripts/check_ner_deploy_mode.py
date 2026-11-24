"""배포 전 NER 모드 게이트(읽기 전용): 운영 설정이 `PRIVACY_NER_MODE=required` 가 아니면 종료 코드 1 로 배포를 막는다.

사용 위치: 배포 담당자가 새 리비전을 만들기 전(또는 만든 직후 트래픽 전환 전)에 실행한다. 이 스크립트는 아무것도 배포·변경하지 않는다.
- 입력 1(설정): `gcloud run services describe <서비스> --format=json`(또는 revisions describe) 결과 파일. 여기서 `PRIVACY_NER_*` 환경변수 **이름과 값만** 읽고,
  다른 환경변수·시크릿 값은 읽거나 출력하지 않는다.
- 입력 2(이미지 실제 동작): 새 이미지에서 `python -c "from app.shared.privacy import ner; import json; print(json.dumps(ner.ner_status()))"` 의 출력 파일.
- 규칙(하나라도 어기면 실패):
  1) 설정에 `PRIVACY_NER_MODE` 가 있으면 정확히 `required` 여야 한다(`auto`·`off`·오타·빈 값 모두 실패. 앱은 오타를 required 로 처리하지만 게이트는 의도하지 않은 값을 막는다).
  2) 설정에 `PRIVACY_NER_MODE` 가 없으면 이미지 기본값에 맡기는 것이므로 입력 2 가 반드시 필요하고 mode=required 여야 한다.
  3) 설정이 `PRIVACY_NER_MODEL_DIR`·`PRIVACY_NER_MODEL_FILE` 을 덮어쓰면 기대값과 같아야 한다(다르면 옛 모델 폴더를 가리킬 수 있다).
  4) 입력 2 가 있으면 mode=required, active=true, fingerprint=기대 지문이어야 한다.
  5) 값이 비밀 참조(valueFrom)면 내용을 확인할 수 없으므로 실패.
사용: python check_ner_deploy_mode.py --service-json svc.json [--image-status-json status.json] [--expected-fingerprint 3831601b7ffb]
"""

import argparse
import json
import sys
from pathlib import Path

EXPECT_DIR = "/srv/data/models/ner-koelectra-small-r11b-s21"
EXPECT_FILE = "model.onnx"
EXPECT_FP = "3831601b7ffb"  # R11 B-new seed 21 ONNX sha256 앞 12자


def ner_env(doc):
    """Cloud Run 서비스/리비전 JSON 에서 PRIVACY_NER_* 환경변수만 뽑는다. 값이 비밀 참조면 '<secret-ref>' 로 표시."""
    out = {}
    containers = []
    spec = doc.get("spec") or {}
    tmpl = ((spec.get("template") or {}).get("spec") or {})
    containers = tmpl.get("containers") or spec.get("containers") or []
    for c in containers[:1]:
        for e in c.get("env") or []:
            name = str(e.get("name", ""))
            if not name.startswith("PRIVACY_NER_"):
                continue
            out[name] = "<secret-ref>" if "valueFrom" in e else str(e.get("value", ""))
    return out


def check(env, image_status=None, expect_fp=EXPECT_FP):
    """반환: (통과 여부, 이유 목록)."""
    problems = []
    if "PRIVACY_NER_MODE" in env:
        v = env["PRIVACY_NER_MODE"]
        if v == "<secret-ref>":
            problems.append("PRIVACY_NER_MODE 가 비밀 참조라 값을 확인할 수 없다")
        elif v != "required":
            problems.append(f"PRIVACY_NER_MODE 설정값이 required 가 아니다: {v!r}")
    elif image_status is None:
        problems.append("설정에 PRIVACY_NER_MODE 가 없고 이미지 실제 동작(--image-status-json)도 없어 확인할 수 없다")
    d = env.get("PRIVACY_NER_MODEL_DIR")
    if d is not None and d != EXPECT_DIR:
        problems.append("PRIVACY_NER_MODEL_DIR 가 기대 폴더와 다르다(옛 모델을 가리킬 수 있다)")
    f = env.get("PRIVACY_NER_MODEL_FILE")
    if f is not None and f != EXPECT_FILE:
        problems.append("PRIVACY_NER_MODEL_FILE 이 model.onnx 가 아니다")
    if image_status is not None:
        if image_status.get("mode") != "required":
            problems.append(f"이미지 실제 mode 가 required 가 아니다: {image_status.get('mode')!r}")
        if image_status.get("active") is not True:
            problems.append("이미지에서 NER 엔진이 active 가 아니다")
        if image_status.get("fingerprint") != expect_fp:
            problems.append("이미지 모델 지문이 기대값과 다르다")
    return (not problems), problems


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if str(p) != "-" else json.loads(sys.stdin.read())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--service-json", help="gcloud run services/revisions describe --format=json 결과(없으면 이미지 검사만)")
    ap.add_argument("--image-status-json")
    ap.add_argument("--expected-fingerprint", default=EXPECT_FP)
    a = ap.parse_args(argv)
    env = ner_env(load_json(a.service_json)) if a.service_json else {}
    status = load_json(a.image_status_json) if a.image_status_json else None
    ok, problems = check(env, status, a.expected_fingerprint)
    print(json.dumps({"ok": ok, "ner_env_names_seen": sorted(env), "problems": problems}, ensure_ascii=False, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
