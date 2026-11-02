#!/usr/bin/env bash
# Task 1: run tools/qwen_gguf_rope_check.py (official) + capture RoPE warning text.
set -euo pipefail
ROOT="${UNFUSE_ROOT:-/root/unfuse}"
FUP="$ROOT/notebooks/qwen3.8-27b/followups"
GGUF="$HOME/gguf/Qwen3.8-27B-text-BF16.gguf"
HF="$HOME/models/Qwen3.8-27B-text"
LLAMA="$HOME/llama.cpp/build/bin/llama-cli"
LOG="$FUP/task1_rope.log"

cd "$ROOT"
source /root/venv/bin/activate
mkdir -p "$FUP"
exec > >(tee -a "$LOG") 2>&1
echo "== task1 start $(date -Is) =="

cp -f "$ROOT/notebooks/qwen3.8-27b/gguf/rope_warning.txt" "$FUP/rope_warning.txt" 2>/dev/null || true
cp -f "$ROOT/notebooks/qwen3.8-27b/gguf/gguf_rope_metadata.txt" "$FUP/gguf_rope_metadata.txt" 2>/dev/null || true

python tools/qwen_gguf_rope_check.py \
  --hf-dir "$HF" \
  --gguf "$GGUF" \
  --llama-bin "$LLAMA" \
  --out "$FUP/rope_check_official.json"

echo "== task1 end $(date -Is) =="
