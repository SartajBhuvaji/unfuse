#!/usr/bin/env bash
# Benchmark Qwen3.8-27B against its text-only split: three bench_pair runs.
# Run ON the GPU box, after tools/qwen_gguf.sh if you want both. Output goes to
# notebooks/qwen3.8-27b/bench/sweeps/. Read the header of tools/bench_pair.sh first.
#
#   tools/qwen_sweeps.sh
#   VLLM_MAX_SEQS=256 tools/qwen_sweeps.sh

set -euo pipefail

ROOT="${UNFUSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SW="$ROOT/notebooks/qwen3.8-27b/bench/sweeps"
ORIG="Qwen/Qwen3.8-27B"
DERV="sartajbhuvaji/unfuse-Qwen3.8-27B-text"
LOG="$SW/qwen_sweeps.log"

cd "$ROOT"
# shellcheck disable=SC1091
source "${VENV:-$HOME/venv}/bin/activate"
mkdir -p "$SW"
# Two sweeps at once kill each other's servers (it happened). The lock lives in /tmp.
exec 9>/tmp/qwen_sweeps.lock
if ! flock -n 9; then
  echo "FATAL: another qwen_sweeps.sh is running"
  exit 1
fi
exec > >(tee -a "$LOG") 2>&1

echo "== sweeps start $(date -Is) =="

# vLLM defaults max_num_seqs=1024; Qwen3.8 hybrid Mamba cache caps near 334 blocks.
# Parent refuses to start above that (see RESULTS.md). Pin vLLM only; leave SGLang unpinned
# on baseline/longctx unless MAX_RUNNING is set for a sweep row.
VLLM_MAX_SEQS="${VLLM_MAX_SEQS:-256}"

OUT="$SW/baseline-512" IN_LEN=512 OUT_LEN=256 CONC="1 16" CTX=32768 \
  MAX_SEQS="$VLLM_MAX_SEQS" MAX_RUNNING='' \
  tools/bench_pair.sh "$ORIG" "$DERV" vllm sglang

OUT="$SW/longctx-in4096" IN_LEN=4096 OUT_LEN=256 CONC="1 16" CTX=8192 NPROMPTS=32 \
  MAX_SEQS="$VLLM_MAX_SEQS" MAX_RUNNING='' \
  tools/bench_pair.sh "$ORIG" "$DERV" vllm sglang

OUT="$SW/conc-c32" IN_LEN=512 OUT_LEN=256 CONC="32" CTX=32768 NPROMPTS=64 \
  MAX_SEQS=32 MAX_RUNNING=32 \
  tools/bench_pair.sh "$ORIG" "$DERV" vllm sglang

echo "== sweeps end $(date -Is) =="
