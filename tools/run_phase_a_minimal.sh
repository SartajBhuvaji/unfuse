#!/usr/bin/env bash
# Phase A — minimal GGUF validation (see docs/RUNBOOK-qwen-gguf-minimal.md).

set -euo pipefail

ROOT="${UNFUSE_ROOT:-/root/unfuse}"
GGUF_DIR="$ROOT/notebooks/qwen3.8-27b/gguf"
MODELS="${MODELS:-$HOME/models}"
OUT="${OUT:-$HOME/gguf}"
NAME=Qwen3.8-27B-text
Q=Q4_K_M
LOG="$GGUF_DIR/phase_a.log"
UPLOAD="${UPLOAD:-0}"

cd "$ROOT"
source /root/venv/bin/activate
: "${HF_TOKEN:?export HF_TOKEN in this shell}"
chmod +x tools/*.sh 2>/dev/null || true

mkdir -p "$GGUF_DIR" "$OUT"
exec > >(tee -a "$LOG") 2>&1

LLAMA=~/llama.cpp
BIN="$LLAMA/build/bin"
BASE="$OUT/$NAME-BF16.gguf"
QUANT="$OUT/$NAME-$Q.gguf"
SRC="$MODELS/$NAME"
MIN_BF16_BYTES=$((50 * 1024 * 1024 * 1024))

need_bf16_convert() {
  [ ! -f "$BASE" ] && return 0
  local sz
  sz=$(stat -c%s "$BASE" 2>/dev/null || stat -f%z "$BASE")
  [ "$sz" -lt "$MIN_BF16_BYTES" ]
}

if need_bf16_convert; then
  echo "== BF16 GGUF missing or truncated — re-converting from safetensors =="
  rm -f "$BASE"
  python "$LLAMA/convert_hf_to_gguf.py" "$SRC" \
    --outfile "$BASE" --outtype bf16 2>&1 | tee "$GGUF_DIR/convert_bf16.log"
fi

if [ ! -f "$BASE" ]; then
  echo "FATAL: still no $BASE after convert"
  exit 1
fi

if [ ! -x "$BIN/llama-quantize" ]; then
  echo "FATAL: missing llama.cpp build at $BIN"
  exit 1
fi

echo "== phase A start $(date -Is) =="
[ -f "$GGUF_DIR/env.json" ] || bash tools/gpu_env.sh --json > "$GGUF_DIR/env.json"

echo "== quantize $Q only =="
"$BIN/llama-quantize" "$BASE" "$QUANT" "$Q" "$(nproc)"

echo "== RoPE (warning + metadata) =="
grep -i rope "$GGUF_DIR/convert_bf16.log" 2>/dev/null | tee "$GGUF_DIR/rope_warning.txt" || true
python "$LLAMA/gguf-py/gguf/scripts/gguf_dump.py" --no-tensors "$BASE" \
  | grep -Ei 'rope|mrope|qwen35' | tee "$GGUF_DIR/gguf_rope_metadata.txt" || true

echo "== smoke (llamacpp, $Q) =="
LOGS="$GGUF_DIR/smoke-logs"
mkdir -p "$LOGS"
GGUF="$QUANT" LOGS="$LOGS" ./tools/smoke_inference.sh \
  sartajbhuvaji/unfuse-Qwen3.8-27B-text llamacpp \
  | tee "$LOGS/smoke-$Q.txt"

echo "== perplexity (BF16 + $Q, wikitext-2 test, ctx 512) =="
WIKI="${ROOT}/notebooks/qwen3.8-27b/gguf/wikitext2-test.txt"
if [ ! -s /tmp/wikitext2-test.txt ]; then
  cp "$WIKI" /tmp/wikitext2-test.txt
fi
head -c 262144 /tmp/wikitext2-test.txt > /tmp/wikitext2-eval.txt
wc -c /tmp/wikitext2-eval.txt
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

echo "== mmproj (F16) + mtmd smoke =="
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
  echo "== Hub upload (Q4 + mmproj F16 only, private) =="
  OUT="$OUT" python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
out = Path(os.environ["OUT"])
files = [
    out / "Qwen3.8-27B-text-Q4_K_M.gguf",
    out / "mmproj-Qwen3.8-27B-F16.gguf",
]
repos = {
    files[0].name: "sartajbhuvaji/unfuse-Qwen3.8-27B-text-GGUF",
    files[1].name: "sartajbhuvaji/unfuse-Qwen3.8-27B-mmproj-GGUF",
}
for f in files:
    if not f.is_file():
        raise SystemExit(f"missing {f}")
    repo = repos[f.name]
    api.create_repo(repo, exist_ok=True, private=True)
    api.upload_file(
        path_or_fileobj=str(f),
        path_in_repo=f.name,
        repo_id=repo,
        repo_type="model",
        commit_message=f"Add {f.name}",
    )
    print("uploaded", f.name, "->", repo)
PY
else
  echo "== Hub upload skipped (set UPLOAD=1 to push Q4 + mmproj F16 only) =="
fi

echo "== phase A end $(date -Is) =="
