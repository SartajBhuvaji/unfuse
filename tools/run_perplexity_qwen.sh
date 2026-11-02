#!/usr/bin/env bash
# Re-run perplexity only (fixed parser + 256 KiB eval slice).

set -euo pipefail

ROOT="${UNFUSE_ROOT:-/root/unfuse}"
GGUF_DIR="$ROOT/notebooks/qwen3.8-27b/gguf"
OUT="${OUT:-$HOME/gguf}"
NAME=Qwen3.8-27B-text
Q=Q4_K_M

cd "$ROOT"
LLAMA=~/llama.cpp
BIN="$LLAMA/build/bin"
BASE="$OUT/$NAME-BF16.gguf"
WIKI="$GGUF_DIR/wikitext2-test.txt"

cp "$WIKI" /tmp/wikitext2-test.txt
head -c 262144 /tmp/wikitext2-test.txt > /tmp/wikitext2-eval.txt

{
  echo "| quant | perplexity |"
  echo "|-------|------------|"
  for label in BF16 "$Q"; do
    f="$OUT/$NAME-${label}.gguf"
    [ "$label" = "BF16" ] && f="$BASE"
    plog="$GGUF_DIR/perplexity_${label}.log"
    "$BIN/llama-perplexity" -m "$f" -f /tmp/wikitext2-eval.txt -c 512 -ngl 99 > "$plog" 2>&1 || true
    val=$(grep -oE 'Final estimate: PPL = [0-9.]+' "$plog" | sed 's/Final estimate: PPL = //' | tail -1)
    [ -z "$val" ] && val="?"
    echo "| $label | $val |"
  done
} | tee "$GGUF_DIR/perplexity_table.md"
