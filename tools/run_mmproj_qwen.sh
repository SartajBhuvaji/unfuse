#!/usr/bin/env bash
# mmproj + mtmd test only (after quant/smoke/perplexity).

set -euo pipefail

ROOT="${UNFUSE_ROOT:-/root/unfuse}"
GGUF_DIR="$ROOT/notebooks/qwen3.8-27b/gguf"
OUT="${OUT:-$HOME/gguf}"
MODELS="${MODELS:-$HOME/models}"
NAME=Qwen3.8-27B-text
Q=Q4_K_M
UPLOAD="${UPLOAD:-0}"

cd "$ROOT"
source /root/venv/bin/activate
: "${HF_TOKEN:?export HF_TOKEN in this shell}"

LLAMA=~/llama.cpp
BIN="$LLAMA/build/bin"
QUANT="$OUT/$NAME-$Q.gguf"
LOG="$GGUF_DIR/mmproj_run.log"
exec > >(tee -a "$LOG") 2>&1

echo "== mmproj start $(date -Is) =="
hf download Qwen/Qwen3.8-27B --local-dir "$MODELS/Qwen3.8-27B"
MMP="$OUT/mmproj-Qwen3.8-27B-F16.gguf"
python "$LLAMA/convert_hf_to_gguf.py" "$MODELS/Qwen3.8-27B" --mmproj \
  --outtype f16 --outfile "$MMP" 2>&1 | tee "$GGUF_DIR/mmproj_f16.log"
IMG="$GGUF_DIR/sample.jpg"
curl -fsSL -o "$IMG" \
  "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
"$BIN/llama-mtmd-cli" -m "$QUANT" --mmproj "$MMP" --image "$IMG" \
  -p "What animal is in this image? One word." -n 32 -ngl 99 -st \
  < /dev/null 2>&1 | tee "$GGUF_DIR/mmproj_mtmd_test.txt"
echo "== mmproj end $(date -Is) =="
