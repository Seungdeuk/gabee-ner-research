"""컨테이너 안(실제 이미지·실제 환경변수)에서 required 의 실패 차단을 확인한다. 합성 이름만 쓴다.
- 소켓 connect 를 세어 '외부 전송 시도' 가 0 인지 본다(--network none 과 함께 쓰면 시도 자체가 불가능하므로 시도 횟수로 확인).
- 가명화 입구(prepare_transcript)가 PrivacyUnavailable 로 막히는지 본다.
사용: docker run --rm -v <시험 폴더>:/srv/data/models/<NER 폴더>:ro --network none ... python /check/container_failclosed.py <label>
"""
import json
import socket
import sys

attempts = {"n": 0}
_orig = socket.socket.connect


def counting_connect(self, *a, **k):
    attempts["n"] += 1
    return _orig(self, *a, **k)


socket.socket.connect = counting_connect

from app.shared.privacy import ner  # noqa: E402
from app.shared.privacy.gateway import Row, prepare_transcript  # noqa: E402
from app.shared.privacy.policy import ANALYSIS  # noqa: E402

label = sys.argv[1] if len(sys.argv) > 1 else "case"
out = {"label": label, "mode": ner.current_mode()}
try:
    st = ner.ner_status()
    out["status"] = st
except Exception as e:  # noqa: BLE001
    out["status_error"] = type(e).__name__
rows = [Row(1, "정하율", "고은채한테 010-1234-5678 로 연락해 줘")]
try:
    with prepare_transcript(rows, [("정하율",), ("고은채",)], ANALYSIS) as prep:
        out["blocked"] = False
        out["note"] = "가명화 통과(모델 정상)"
        out["pseudonymized_has_raw_name"] = any(x in prep.lines[0].text for x in ("고은채", "010-1234-5678"))
except Exception as e:  # noqa: BLE001
    out["blocked"] = True
    out["exception"] = type(e).__name__
    out["status_code"] = getattr(e, "status_code", None)
    out["code"] = getattr(e, "code", None)
out["socket_connect_attempts"] = attempts["n"]
print(json.dumps(out, ensure_ascii=False))
