#!/usr/bin/env bash
# 비운영 로컬 Docker: R11 이미지 기동 -> S0 이미지로 롤백(교체) 시험. 운영과 무관.
docker rm -f gabee-nonprod-r11 gabee-nonprod-s0 >/dev/null 2>&1
now() { python -c "import time;print(time.time())"; }
diff() { python -c "print(round($2-$1,1))"; }
wait_health() {
  for i in $(seq 1 90); do
    c=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:18000/health 2>/dev/null)
    [ "$c" = "200" ] && break
    sleep 1
  done
  echo "$c"
}
t0=$(now)
docker run -d --name gabee-nonprod-r11 -p 18000:8000 gabee-api:r11b-s21-nonprod >/dev/null
c=$(wait_health)
t1=$(now)
echo "R11 health=$c start_to_health_s=$(diff $t0 $t1)"
sleep 4
docker logs gabee-nonprod-r11 2>&1 | grep -E "NER 엔진|Uvicorn running" | cut -c1-200 | head -4
docker exec gabee-nonprod-r11 sh -c 'echo MODE=$PRIVACY_NER_MODE; sha256sum /srv/data/models/ner-koelectra-small-r11b-s21/model.onnx | cut -c1-64'
docker exec gabee-nonprod-r11 python -c "import json;from app.shared.privacy import ner;print(json.dumps(ner.ner_status()))"
echo "--- rollback to S0 image"
r0=$(now)
docker rm -f gabee-nonprod-r11 >/dev/null
docker run -d --name gabee-nonprod-s0 -p 18000:8000 gabee-api:s0-rollback-nonprod >/dev/null
c=$(wait_health)
r1=$(now)
echo "S0 health=$c switch_time_s=$(diff $r0 $r1)"
sleep 4
docker logs gabee-nonprod-s0 2>&1 | grep -E "NER 엔진" | cut -c1-200 | head -3
docker exec gabee-nonprod-s0 sh -c 'sha256sum /srv/data/models/ner-koelectra-small-v2b/model.onnx | cut -c1-64'
docker exec gabee-nonprod-s0 python -c "import json;from app.shared.privacy import ner;print(json.dumps(ner.ner_status()))"
docker rm -f gabee-nonprod-s0 >/dev/null
echo "remaining nonprod containers: $(docker ps -a --format '{{.Names}}' | grep -c nonprod)"
