#!/usr/bin/env bash
# Upload Q4 text GGUF + mmproj to private Hub repos (runs on GPU box).

set -euo pipefail

ROOT="${UNFUSE_ROOT:-/root/unfuse}"
OUT="${OUT:-$HOME/gguf}"
: "${HF_TOKEN:?export HF_TOKEN}"

cd "$ROOT"
source /root/venv/bin/activate

LOG="$ROOT/notebooks/qwen3.8-27b/gguf/hub_upload.log"
exec > >(tee -a "$LOG") 2>&1

echo "== hub upload start $(date -Is) =="

python tools/upload_gguf_hub.py --private \
  --repo-id sartajbhuvaji/unfuse-Qwen3.8-27B-text-GGUF \
  --readme docs/model-cards/qwen3.8-27b-text-gguf.md \
  "$OUT/Qwen3.8-27B-text-Q4_K_M.gguf"

python tools/upload_gguf_hub.py --private \
  --repo-id sartajbhuvaji/unfuse-Qwen3.8-27B-mmproj-GGUF \
  --readme docs/model-cards/qwen3.8-27b-mmproj-gguf.md \
  "$OUT/mmproj-Qwen3.8-27B-F16.gguf"

echo "== hub upload end $(date -Is) =="
