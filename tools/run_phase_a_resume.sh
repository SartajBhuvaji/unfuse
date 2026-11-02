#!/usr/bin/env bash
# Continue Phase A after quant + RoPE (e.g. smoke failed on missing +x).

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
chmod +x tools/*.sh

LLAMA=~/llama.cpp
BIN="$LLAMA/build/bin"
BASE="$OUT/$NAME-BF16.gguf"
QUANT="$OUT/$NAME-$Q.gguf"
LOG="$GGUF_DIR/phase_a_resume.log"
exec > >(tee -a "$LOG") 2>&1

echo "== phase A resume $(date -Is) =="

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  LOGS="$GGUF_DIR/smoke-logs"
  mkdir -p "$LOGS"
  GGUF="$QUANT" LOGS="$LOGS" ./tools/smoke_inference.sh \
    sartajbhuvaji/unfuse-Qwen3.8-27B-text llamacpp \
    | tee "$LOGS/smoke-$Q.txt"
else
  echo "== smoke skipped (SKIP_SMOKE=1) =="
fi

WIKI="${ROOT}/notebooks/qwen3.8-27b/gguf/wikitext2-test.txt"
if [ ! -s /tmp/wikitext2-test.txt ]; then
  cp "$WIKI" /tmp/wikitext2-test.txt
fi
wc -c /tmp/wikitext2-test.txt
{
  echo "| quant | perplexity |"
  echo "|-------|------------|"
  for label in BF16 "$Q"; do
    f="$OUT/$NAME-${label}.gguf"
    [ "$label" = "BF16" ] && f="$BASE"
    ppl=$("$BIN/llama-perplexity" -m "$f" -f /tmp/wikitext2-test.txt -c 512 -ngl 99 2>&1 | tail -1)
    val=$(echo "$ppl" | grep -oE '[0-9]+\.[0-9]+' | tail -1 || echo "?")
    echo "| $label | $val |"
  done
} | tee "$GGUF_DIR/perplexity_table.md"

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

if [ "$UPLOAD" = "1" ]; then
  OUT="$OUT" python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
out = Path(os.environ["OUT"])
for f, repo in [
    (out / "Qwen3.8-27B-text-Q4_K_M.gguf", "sartajbhuvaji/unfuse-Qwen3.8-27B-text-GGUF"),
    (out / "mmproj-Qwen3.8-27B-F16.gguf", "sartajbhuvaji/unfuse-Qwen3.8-27B-mmproj-GGUF"),
]:
    api.create_repo(repo, exist_ok=True, private=True)
    api.upload_file(str(f), path_in_repo=f.name, repo_id=repo, repo_type="model")
    print("uploaded", f.name)
PY
fi

echo "== phase A resume end $(date -Is) =="
