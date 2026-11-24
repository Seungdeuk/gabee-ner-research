#!/usr/bin/env bash
# S2b 컨테이너 측정 드라이버: 같은 자원 제한에서 S0, R11, R11, S0, S0, R11 순서로 교차 측정한다(g12 와 같은 순서).
# 사용: bash s2b_container.sh <라벨> <CPUS> <MEM(예 2g)> <THREADS> <CONCURRENCY>
#   CPUS·MEM·THREADS·CONCURRENCY 는 운영 담당자가 확인한 값만 쓴다. 임의 값으로 돌리면 결과를 S2b 로 인정하지 않는다("PLUMBING-ONLY" 표기).
# 모델 이미지: gabee-api:s0-rollback-nonprod(S0), gabee-api:r11b-s21-nonprod(R11). 네트워크 없음.
set -u
LABEL="$1"; CPUS="$2"; MEM="$3"; THREADS="$4"; CONC="$5"
export MSYS_NO_PATHCONV=1
H='C:\Users\<user>\Documents\gabee-ner-evalset-ops-handoff\ops_replace'
OUT="$H\\s2b_$LABEL"
mkdir -p "/c/Users/<user>/Documents/gabee-ner-evalset-ops-handoff/ops_replace/s2b_$LABEL"
DATA='C:\Users\<user>\venvs\ner-runs\data_r11'
i=0
for who in s0 r11 r11 s0 s0 r11; do
  i=$((i+1))
  if [ "$who" = "s0" ]; then IMG=gabee-api:s0-rollback-nonprod; MD=/srv/data/models/ner-koelectra-small-v2b; else IMG=gabee-api:r11b-s21-nonprod; MD=/srv/data/models/ner-koelectra-small-r11b-s21; fi
  docker run --rm --network none --cpus "$CPUS" --memory "$MEM" -v "$H:/check:ro" -v "$DATA:/data:ro" -v "$OUT:/out" \
    -e PRIVACY_NER_MODEL_DIR=$MD -e PYTHONPATH=/srv --entrypoint python "$IMG" /check/s2b_container_latency.py \
    --sentences /data/r11_val.jsonl --threads "$THREADS" --concurrency "$CONC" --out "/out/${i}_${who}.json" 2>&1 | tail -1 | sed "s/^/$i $who /"
done
