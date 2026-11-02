#!/usr/bin/env bash
# Quantize, validate and (optionally) publish the Qwen3.8-27B-text GGUF.
# Run ON the GPU box. Every step skips itself if its output already exists,
# so rerunning after a failure picks up where it stopped.
#
#   tools/qwen_gguf.sh
#   QUANTS="Q4_K_M Q8_0" tools/qwen_gguf.sh     # cut more quants
#   SKIP_SMOKE=1 tools/qwen_gguf.sh             # skip the llama.cpp smoke test
#   UPLOAD=1 HF_TOKEN=... tools/qwen_gguf.sh    # push to PRIVATE Hub repos
#
# Needs: the text checkpoint in $MODELS/Qwen3.8-27B-text, and llama.cpp built in
# ~/llama.cpp with llama-quantize, llama-perplexity and llama-mtmd-cli (build_gguf.sh
# builds them). The bf16 GGUF is converted here if missing or truncated.
#
# Output goes to notebooks/qwen3.8-27b/gguf/. Never writes a token to disk.

set -euo pipefail

ROOT="${UNFUSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GGUF_DIR="$ROOT/notebooks/qwen3.8-27b/gguf"
MODELS="${MODELS:-$HOME/models}"
OUT="${OUT:-$HOME/gguf}"
LLAMA="${LLAMA:-$HOME/llama.cpp}"
QUANTS="${QUANTS:-Q4_K_M}"
UPLOAD="${UPLOAD:-0}"
NAME=Qwen3.8-27B-text
BIN="$LLAMA/build/bin"
BASE="$OUT/$NAME-BF16.gguf"
SRC="$MODELS/$NAME"
MMP="$OUT/mmproj-Qwen3.8-27B-F16.gguf"
MIN_BF16_BYTES=$((50 * 1024 * 1024 * 1024))   # a truncated copy has bitten us once

cd "$ROOT"
# shellcheck disable=SC1091
source "${VENV:-$HOME/venv}/bin/activate"
mkdir -p "$GGUF_DIR/smoke-logs" "$OUT"
exec > >(tee -a "$GGUF_DIR/qwen_gguf.log") 2>&1
echo "== qwen_gguf start $(date -Is) =="

[ -x "$BIN/llama-quantize" ] || { echo "FATAL: no llama.cpp build at $BIN"; exit 1; }
[ -f "$GGUF_DIR/env.json" ] || bash tools/gpu_env.sh --json > "$GGUF_DIR/env.json"

# -- bf16 base ----------------------------------------------------------------
size() { stat -c%s "$1" 2>/dev/null || echo 0; }
if [ "$(size "$BASE")" -lt "$MIN_BF16_BYTES" ]; then
  echo "== bf16 GGUF missing or truncated: converting =="
  rm -f "$BASE"
  python "$LLAMA/convert_hf_to_gguf.py" "$SRC" --outfile "$BASE" --outtype bf16 \
    2>&1 | tee "$GGUF_DIR/convert_bf16.log"
fi

# -- RoPE: the converter's warning and what actually got written --------------
grep -i rope "$GGUF_DIR/convert_bf16.log" 2>/dev/null > "$GGUF_DIR/rope_warning.txt" || true
python "$LLAMA/gguf-py/gguf/scripts/gguf_dump.py" --no-tensors "$BASE" \
  | grep -Ei 'rope|mrope|qwen35' > "$GGUF_DIR/gguf_rope_metadata.txt" || true

# -- quants + known-answer smoke test ----------------------------------------
for Q in $QUANTS; do
  F="$OUT/$NAME-$Q.gguf"
  if [ ! -s "$F" ]; then
    echo "== quantize $Q =="
    "$BIN/llama-quantize" "$BASE" "$F" "$Q" "$(nproc)"
  fi
  if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    echo "== smoke $Q =="
    GGUF="$F" LOGS="$GGUF_DIR/smoke-logs" bash tools/smoke_inference.sh \
      "sartajbhuvaji/unfuse-$NAME" llamacpp | tee "$GGUF_DIR/smoke-logs/smoke-$Q.txt"
  fi
done

# -- perplexity: bf16 and each quant, same text, same context ------------------
# A 256 KiB slice keeps this to minutes. It is a sanity check, not a benchmark.
EVAL="$(mktemp)"; trap 'rm -f "$EVAL"' EXIT
head -c 262144 "$GGUF_DIR/wikitext2-test.txt" > "$EVAL"
{
  echo "| quant | perplexity |"; echo "|-------|------------|"
  for label in BF16 $QUANTS; do
    f="$OUT/$NAME-$label.gguf"; [ "$label" = BF16 ] && f="$BASE"
    plog="$GGUF_DIR/perplexity_$label.log"
    "$BIN/llama-perplexity" -m "$f" -f "$EVAL" -c 512 -ngl 99 > "$plog" 2>&1 || true
    val=$(grep -oE 'Final estimate: PPL = [0-9.]+' "$plog" | sed 's/.*= //' | tail -1)
    echo "| $label | ${val:-?} |"
  done
} | tee "$GGUF_DIR/perplexity_table.md"

# -- mmproj comes from the ORIGINAL repo; the text-only one has no tower -------
if [ ! -s "$MMP" ]; then
  [ -d "$MODELS/Qwen3.8-27B" ] || hf download Qwen/Qwen3.8-27B --local-dir "$MODELS/Qwen3.8-27B"
  python "$LLAMA/convert_hf_to_gguf.py" "$MODELS/Qwen3.8-27B" --mmproj --outtype f16 \
    --outfile "$MMP" 2>&1 | tee "$GGUF_DIR/mmproj_f16.log"
fi
IMG="$GGUF_DIR/sample.jpg"
[ -s "$IMG" ] || curl -fsSL -o "$IMG" \
  "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
FIRSTQ="${QUANTS%% *}"
"$BIN/llama-mtmd-cli" -m "$OUT/$NAME-$FIRSTQ.gguf" --mmproj "$MMP" --image "$IMG" \
  -p "What animal is in this image? One word." -n 32 -ngl 99 \
  < /dev/null 2>&1 | tee "$GGUF_DIR/mmproj_mtmd_test.txt"

# -- Hub: private repos only; the public flip is a separate, deliberate step ---
if [ "$UPLOAD" = "1" ]; then
  : "${HF_TOKEN:?export HF_TOKEN for UPLOAD=1}"
  FILES=(); for Q in $QUANTS; do FILES+=("$OUT/$NAME-$Q.gguf"); done
  python tools/upload_gguf_hub.py --private \
    --repo-id "sartajbhuvaji/unfuse-$NAME-GGUF" \
    --readme docs/model-cards/qwen3.8-27b-text-gguf.md "${FILES[@]}"
  python tools/upload_gguf_hub.py --private \
    --repo-id sartajbhuvaji/unfuse-Qwen3.8-27B-mmproj-GGUF \
    --readme docs/model-cards/qwen3.8-27b-mmproj-gguf.md "$MMP"
else
  echo "== Hub upload skipped (UPLOAD=1 to push, private) =="
fi
echo "== qwen_gguf end $(date -Is) =="
