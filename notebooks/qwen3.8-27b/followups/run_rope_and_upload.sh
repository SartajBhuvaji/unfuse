#!/usr/bin/env bash
# RoPE via llama-server + Hub upload Q8/Q5 (files only). Run on GPU box.
set -euo pipefail
ROOT="${UNFUSE_ROOT:-/root/unfuse}"
FUP="$ROOT/notebooks/qwen3.8-27b/followups"
GGUF="$HOME/gguf/Qwen3.8-27B-text-BF16.gguf"
LLAMA="$HOME/llama.cpp"
BIN="$LLAMA/build/bin"
LOG="$FUP/rope_server_job.log"

cd "$ROOT"
source /root/venv/bin/activate
if [ -f "$ROOT/.env" ]; then set -a; source "$ROOT/.env"; set +a; fi
bash "$FUP/ensure_qwen35_kernels.sh"
exec > >(tee -a "$LOG") 2>&1
echo "== job start $(date -Is) =="

# Build llama-server if missing
if [ ! -x "$BIN/llama-server" ]; then
  echo "== building llama-server =="
  cmake --build "$LLAMA/build" --target llama-server -j "$(nproc)"
fi

pkill -9 -f llama-server 2>/dev/null || true
sleep 5
nohup "$BIN/llama-server" -m "$GGUF" --port 8080 -ngl 99 -c 8192 \
  --reasoning-budget 0 > "$FUP/llama-server.log" 2>&1 &
echo "waiting for llama-server..."
python3 - <<'PY'
import time, urllib.request
for _ in range(300):
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
        print("server up")
        break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("server timeout")
PY

python "$FUP/rope_server_compare.py" --phase collect-gguf --gguf-cache "$FUP/rope_gguf_cache.json"
pkill -9 -f llama-server 2>/dev/null || true
sleep 15
python "$FUP/rope_server_compare.py" \
  --phase run-hf \
  --gguf-cache "$FUP/rope_gguf_cache.json" \
  --out-json "$FUP/rope_server_results.json" \
  --out-md "$FUP/ROPE.md"

if [ "${SKIP_HUB_UPLOAD:-0}" = 1 ]; then
  echo "SKIP_HUB_UPLOAD=1; skipping Hub upload"
  echo "== job end $(date -Is) =="
  exit 0
fi

: "${HF_TOKEN:?HF_TOKEN required for upload}"
HUB_README="$FUP/.hub_readme_snapshot.md"
python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download

p = hf_hub_download(
    repo_id="sartajbhuvaji/unfuse-Qwen3.8-27B-text-GGUF",
    filename="README.md",
    repo_type="model",
    token=os.environ["HF_TOKEN"],
)
open("/root/unfuse/notebooks/qwen3.8-27b/followups/.hub_readme_snapshot.md", "w").write(
    open(p).read()
)
print("snapshot README from hub")
PY
python "$ROOT/tools/upload_gguf_hub.py" \
  --repo-id sartajbhuvaji/unfuse-Qwen3.8-27B-text-GGUF \
  --readme "$HUB_README" \
  --private \
  "$HOME/gguf/Qwen3.8-27B-text-Q8_0.gguf" \
  "$HOME/gguf/Qwen3.8-27B-text-Q5_K_M.gguf"

echo "== job end $(date -Is) =="
