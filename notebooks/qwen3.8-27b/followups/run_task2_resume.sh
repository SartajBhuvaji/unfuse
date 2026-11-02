#!/usr/bin/env bash
# Resume task2 after Q8_0 (skip existing quants).
set -euo pipefail
ROOT="${UNFUSE_ROOT:-/root/unfuse}"
FUP="$ROOT/notebooks/qwen3.8-27b/followups"
GGUF_DIR="$ROOT/notebooks/qwen3.8-27b/gguf"
OUT="${OUT:-$HOME/gguf}"
LLAMA="${LLAMA:-$HOME/llama.cpp}"
BIN="$LLAMA/build/bin"
NAME=Qwen3.8-27B-text
BASE="$OUT/$NAME-BF16.gguf"
EVAL="$GGUF_DIR/wikitext2-test.txt"
LOG="$FUP/task2_quants.log"

cd "$ROOT"
source /root/venv/bin/activate
mkdir -p "$FUP/smoke-logs"
exec > >(tee -a "$LOG") 2>&1
echo "== task2 resume $(date -Is) =="

for Q in Q5_K_M; do
  F="$OUT/$NAME-$Q.gguf"
  if [ ! -s "$F" ]; then
    echo "== quantize $Q =="
    "$BIN/llama-quantize" "$BASE" "$F" "$Q" "$(nproc)"
  fi
  echo "== smoke $Q =="
  GGUF="$F" LOGS="$FUP/smoke-logs" bash tools/smoke_inference.sh \
    "sartajbhuvaji/unfuse-$NAME" llamacpp | tee "$FUP/smoke-logs/smoke-$Q.txt"
done

{
  echo "| quant | perplexity | log |"
  echo "|-------|------------|-----|"
  for label in BF16 Q8_0 Q5_K_M Q4_K_M; do
    f="$OUT/$NAME-$label.gguf"
    [ "$label" = BF16 ] && f="$BASE"
    plog="$FUP/perplexity_$label.log"
    if [ "$label" = BF16 ] && [ -s "$FUP/perplexity_BF16.log" ]; then
      val=$(grep -oE 'Final estimate: PPL = [0-9.]+' "$FUP/perplexity_BF16.log" | sed 's/.*= //' | tail -1)
      echo "| $label | ${val:-?} | perplexity_$label.log |"
      continue
    fi
    echo "== perplexity $label full wikitext =="
    "$BIN/llama-perplexity" -m "$f" -f "$EVAL" -c 512 -ngl 99 > "$plog" 2>&1 || true
    val=$(grep -oE 'Final estimate: PPL = [0-9.]+' "$plog" | sed 's/.*= //' | tail -1)
    echo "| $label | ${val:-?} | perplexity_$label.log |"
  done
} | tee "$FUP/perplexity_full_table.md"

echo "== task2 end $(date -Is) =="
