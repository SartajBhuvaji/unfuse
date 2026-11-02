#!/usr/bin/env bash
# DEPRECATED — use tools/run_phase_a_minimal.sh (docs/RUNBOOK-qwen-gguf-minimal.md).
# Task 1 from docs/RUNBOOK-qwen-gguf-and-sweeps.md — run ON the GPU box.
# Requires HF_TOKEN in the environment. Logs under notebooks/qwen3.8-27b/gguf/.

set -euo pipefail

ROOT="${UNFUSE_ROOT:-/root/unfuse}"
GGUF_DIR="$ROOT/notebooks/qwen3.8-27b/gguf"
MODELS="${MODELS:-$HOME/models}"
OUT="${OUT:-$HOME/gguf}"
NAME=Qwen3.8-27B-text
LOG="$GGUF_DIR/conversion.log"

cd "$ROOT"
source /root/venv/bin/activate
: "${HF_TOKEN:?set HF_TOKEN}"

mkdir -p "$GGUF_DIR" "$MODELS" "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "== task1 start $(date -Is) =="
bash tools/gpu_env.sh --json > "$GGUF_DIR/env.json"
bash tools/gpu_env.sh

echo "== download text checkpoint =="
hf download sartajbhuvaji/unfuse-Qwen3.8-27B-text --local-dir "$MODELS/Qwen3.8-27B-text"

echo "== build GGUF ladder =="
cd ~
if [ -d llama.cpp/.git ]; then
  git -C llama.cpp fetch --depth 1 origin master && git -C llama.cpp checkout master && git -C llama.cpp pull --ff-only
else
  git clone https://github.com/ggml-org/llama.cpp
fi
LLAMA_REV=$(git -C ~/llama.cpp rev-parse HEAD)
echo "llama.cpp commit: $LLAMA_REV" | tee "$GGUF_DIR/llama_cpp_rev.txt"

sudo apt-get update -qq
sudo apt-get install -y -qq cmake build-essential libcurl4-openssl-dev
cmake -S ~/llama.cpp -B ~/llama.cpp/build \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build ~/llama.cpp/build -j"$(nproc)" --target llama-quantize llama-cli llama-perplexity llama-mtmd-cli

SRC="$MODELS/Qwen3.8-27B-text" BASE=bf16 QUANTS=(Q8_0 Q6_K Q5_K_M Q4_K_M)
BASE_UP=BF16
echo "== convert $BASE_UP ==" | tee "$GGUF_DIR/convert_bf16.log"
python ~/llama.cpp/convert_hf_to_gguf.py "$SRC" \
  --outfile "$OUT/$NAME-$BASE_UP.gguf" --outtype bf16 2>&1 | tee -a "$GGUF_DIR/convert_bf16.log"

for Q in "${QUANTS[@]}"; do
  echo "== quantize $Q =="
  ~/llama.cpp/build/bin/llama-quantize "$OUT/$NAME-$BASE_UP.gguf" "$OUT/$NAME-$Q.gguf" "$Q" "$(nproc)"
done

echo "== gguf metadata (rope) =="
python ~/llama.cpp/gguf-py/gguf/scripts/gguf_dump.py --no-tensors "$OUT/$NAME-$BASE_UP.gguf" \
  | grep -E 'qwen|rope|mrope|general\.' | tee "$GGUF_DIR/gguf_rope_metadata.txt" || true

echo "== rope + logit check (transformers vs bf16 GGUF) =="
python "$ROOT/tools/qwen_gguf_rope_check.py" \
  --hf-dir "$SRC" --gguf "$OUT/$NAME-$BASE_UP.gguf" \
  --out "$GGUF_DIR/rope_check.json" --llama-bin ~/llama.cpp/build/bin/llama-cli

echo "== smoke (llamacpp, known answers) =="
LOGS="$GGUF_DIR/smoke-logs" mkdir -p "$LOGS"
for Q in BF16 "${QUANTS[@]}"; do
  GGUF="$OUT/$NAME-$Q.gguf" LOGS="$LOGS" \
    GGUF="$GGUF" ./tools/smoke_inference.sh sartajbhuvaji/unfuse-Qwen3.8-27B-text llamacpp \
    | tee "$LOGS/smoke-$Q.txt" || echo "SMOKE FAIL $Q"
done

echo "== perplexity (wikitext-2 test) =="
python - <<'PY' | tee "$GGUF_DIR/perplexity_wikitext2.txt"
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(t for t in ds["text"] if t.strip())
open("/tmp/wikitext2-test.txt", "w").write(text)
print("wikitext2 chars", len(text))
PY
PPL_BIN=~/llama.cpp/build/bin/llama-perplexity
for Q in BF16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
  echo "--- $Q ---"
  "$PPL_BIN" -m "$OUT/$NAME-$Q.gguf" -f /tmp/wikitext2-test.txt -c 512 -ngl 99 2>&1 | tail -5
done | tee -a "$GGUF_DIR/perplexity_wikitext2.txt"

echo "== mmproj from original Qwen3.8-27B =="
hf download Qwen/Qwen3.8-27B --local-dir "$MODELS/Qwen3.8-27B"
python ~/llama.cpp/convert_hf_to_gguf.py "$MODELS/Qwen3.8-27B" --mmproj \
  --outtype f16 --outfile "$OUT/mmproj-Qwen3.8-27B-F16.gguf" 2>&1 | tee "$GGUF_DIR/mmproj_f16.log" || true
python ~/llama.cpp/convert_hf_to_gguf.py "$MODELS/Qwen3.8-27B" --mmproj \
  --outtype q8_0 --outfile "$OUT/mmproj-Qwen3.8-27B-Q8_0.gguf" 2>&1 | tee "$GGUF_DIR/mmproj_q8_0.log" || true

echo "== mmproj smoke (Q4 text + F16 mmproj) =="
# One real image: use a small public sample
IMG="$GGUF_DIR/sample.jpg"
curl -fsSL -o "$IMG" "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
~/llama.cpp/build/bin/llama-mtmd-cli -m "$OUT/$NAME-Q4_K_M.gguf" \
  --mmproj "$OUT/mmproj-Qwen3.8-27B-F16.gguf" \
  --image "$IMG" -p "What animal is in this image? One word." -n 32 -ngl 99 -st \
  < /dev/null 2>&1 | tee "$GGUF_DIR/mmproj_mtmd_test.txt" || true

echo "== private Hub upload (GGUF only, new private repos) =="
OUT="$OUT" python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
gguf = Path(os.environ["OUT"])
text_files = sorted(gguf.glob("Qwen3.8-27B-text-*.gguf"))
mmproj_files = sorted(gguf.glob("mmproj-Qwen3.8-27B-*.gguf"))
text_repo = "sartajbhuvaji/unfuse-Qwen3.8-27B-text-GGUF"
mm_repo = "sartajbhuvaji/unfuse-Qwen3.8-27B-mmproj-GGUF"
for repo, files in [(text_repo, text_files), (mm_repo, mmproj_files)]:
    if not files:
        print("skip empty", repo)
        continue
    print("upload", repo, len(files), "files")
    api.create_repo(repo, exist_ok=True, private=True)
    for f in files:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo,
            repo_type="model",
            commit_message=f"Add {f.name}",
        )
PY

echo "== task1 end $(date -Is) =="
