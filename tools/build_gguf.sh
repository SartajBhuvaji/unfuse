#!/usr/bin/env bash
# Build llama.cpp and produce the GGUF quant ladder. Run ON the instance.
#
# Quantization is CPU work -- the GPU is idle throughout. A CPU-only llama.cpp
# build is fine and much faster to compile.
#
# NOTE: `-no-cnv` was REMOVED from current llama.cpp. Without `-st/--single-turn`,
# llama-cli drops into interactive mode and spins on empty stdin, burning CPU and
# emitting nothing. That failure looks exactly like slow inference. Hence `-st`
# and `< /dev/null` in the smoke test below.

set -euo pipefail

SRC="${SRC:-$HOME/models/GLM-4.6-Flash-text}"
OUT="${OUT:-$HOME/gguf}"
NAME="${NAME:-GLM-4.6-Flash-text}"
# Base precision the quants are cut from. bf16 for models trained in bf16: an f16
# base can overflow on activations/weights outside f16 range. GLM was built at f16.
BASE="${BASE:-f16}"; BASE_UP="${BASE^^}"
QUANTS=("${@:-Q8_0 Q6_K Q5_K_M Q4_K_M}")

sudo apt-get update -qq
sudo apt-get install -y -qq cmake build-essential libcurl4-openssl-dev

cd ~
[ -d llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build --config Release -j"$(nproc)" --target llama-quantize llama-cli llama-perplexity llama-mtmd-cli >/dev/null

source ~/venv/bin/activate
pip install -q gguf sentencepiece protobuf
mkdir -p "$OUT"

echo "== $BASE_UP =="
python convert_hf_to_gguf.py "$SRC" --outfile "$OUT/$NAME-$BASE_UP.gguf" --outtype "$BASE"

for Q in ${QUANTS[*]}; do
  echo "== $Q =="
  ./build/bin/llama-quantize "$OUT/$NAME-$BASE_UP.gguf" "$OUT/$NAME-$Q.gguf" "$Q" "$(nproc)"
done

echo "== smoke test =="
for Q in $BASE_UP ${QUANTS[*]}; do
  printf '%-10s ' "$Q"
  timeout 600 ./build/bin/llama-cli -m "$OUT/$NAME-$Q.gguf" \
    -p "The capital of France is" -n 24 -t "$(nproc)" --temp 0 -st --no-warmup \
    < /dev/null 2>/dev/null | tail -3 | tr '\n' ' '
  echo "[exit $?]"
done

ls -lh "$OUT"
echo "GGUF_DONE"
