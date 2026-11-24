#!/usr/bin/env bash
# 비운영 검증용 Docker 빌드 컨텍스트 조립(저장소 밖). 빌드·푸시·배포는 하지 않는다. 기존 build-context 폴더는 건드리지 않는다.
#
# 사용: bash stage_build_context_nonprod.sh r11|s0 [git-ref]      ref 는 필수(기본값 없음): release/1.4 가 생기면 origin/release/1.4
#   r11: 코드(ref) + 모델 3개 폴더(doc-classifier, e5-small-ft, ner-koelectra-small-r11b-s21) + Dockerfile 패치(폴더 이름, PRIVACY_NER_MODE=required)
#   s0 : 코드(ref) + 모델 3개 폴더(doc-classifier, e5-small-ft, ner-koelectra-small-v2b) + 현재 Dockerfile 그대로(롤백 시험용)
# 출력: C:\Users\<user>\Documents\gabee-deploy-prep\build-context-nonprod-<r11|s0>
set -euo pipefail

KIND="${1:?r11 또는 s0}"
REF="${2:?git ref 가 필요하다 (release/1.4 가 생기면 origin/release/1.4; 1.3 기준으로 R11 이미지를 만들지 않는다)}"
REPO="/c/Users/<user>/Documents/gabee-ner"
OTHER_SRC="/c/Users/<user>/Documents/gabee/apps/api/data/models"
R11_SRC="/c/Users/<user>/venvs/ner-runs/r11/export/r11-B-s21-ep5"
S0_SRC="/c/Users/<user>/Documents/gabee-model-delivery/ner-koelectra-small-v2b"
OUT="/c/Users/<user>/Documents/gabee-deploy-prep/build-context-nonprod-$KIND"
MARK=".gabee-build-context-nonprod"
fail() { echo "중단: $*" >&2; exit 1; }

case "$KIND" in
  r11) NER_SRC="$R11_SRC"; NER_DIR="ner-koelectra-small-r11b-s21"; WANT_SHA="3831601b7ffb6093a8d1b8c832e46ad140712c6e402f17b3054c3ec0f9ae4c94" ;;
  s0)  NER_SRC="$S0_SRC";  NER_DIR="ner-koelectra-small-v2b";      WANT_SHA="46210943c45f01b6a9fa500894fb9821b1c130be63de8d8f7ef603d2d87ccf8e" ;;
  *) fail "KIND 는 r11 또는 s0" ;;
esac

if [ -e "$OUT" ]; then
  [ -f "$OUT/$MARK" ] || fail "$OUT 에 표식이 없어 지우지 않는다"
  rm -r "$OUT"
fi
git -C "$REPO" rev-parse --verify --quiet "$REF^{commit}" >/dev/null || fail "ref 없음: $REF"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
git -C "$REPO" archive --format=tar "$REF" apps/api | tar -x -C "$TMP"
mv "$TMP/apps/api" "$OUT"
touch "$OUT/$MARK"

if [ "$KIND" = "r11" ]; then
  # Dockerfile 패치: dockerfile_r11_ops.patch 와 같은 3줄 변경(줄바꿈은 LF 로 통일된 git ref 기준)
  sed -i 's#ner-koelectra-small-v2b)를 두고#ner-koelectra-small-r11b-s21)를 두고#' "$OUT/Dockerfile"
  sed -i 's#PRIVACY_NER_MODEL_DIR=/srv/data/models/ner-koelectra-small-v2b$#PRIVACY_NER_MODEL_DIR=/srv/data/models/ner-koelectra-small-r11b-s21 \\\n    PRIVACY_NER_MODE=required#' "$OUT/Dockerfile"
  grep -q "PRIVACY_NER_MODE=required" "$OUT/Dockerfile" || fail "패치 적용 실패"
  grep -q "ner-koelectra-small-r11b-s21" "$OUT/Dockerfile" || fail "패치 적용 실패(폴더 이름)"
fi

mkdir -p "$OUT/data/models/$NER_DIR"
cp -r "$OTHER_SRC/doc-classifier" "$OUT/data/models/doc-classifier"
cp -r "$OTHER_SRC/e5-small-ft" "$OUT/data/models/e5-small-ft"
for f in model.onnx tokenizer.json labels.json model_card.json; do cp "$NER_SRC/$f" "$OUT/data/models/$NER_DIR/$f"; done

got="$(sha256sum "$OUT/data/models/$NER_DIR/model.onnx" | cut -d' ' -f1)"
[ "$got" = "$WANT_SHA" ] || fail "NER model.onnx 해시 불일치: $got"
dirs="$(cd "$OUT/data/models" && ls | sort | tr '\n' ' ')"
[ "$dirs" = "doc-classifier e5-small-ft $NER_DIR " ] || fail "모델 폴더 구성이 다르다: $dirs"
bad="$(find "$OUT" -path "$OUT/data" -prune -o \( -name '*.onnx' -o -name '*.safetensors' -o -name '.env' -o -name '.env.*' \) -print | grep -v '\.env\.example$' || true)"
[ -z "$bad" ] || fail "코드 폴더에 모델·환경 파일: $bad"
echo "준비 완료($KIND): $OUT  ref=$REF@$(git -C "$REPO" rev-parse --short "$REF^{commit}")  NER=$NER_DIR sha=${got:0:12}"
du -sh "$OUT"/data/models/* | sed "s#$OUT/data/models/#  #"
