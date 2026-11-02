#!/usr/bin/env bash
# Phase B — three bench_pair runs (see docs/RUNBOOK-qwen-gguf-minimal.md).

set -euo pipefail

ROOT="${UNFUSE_ROOT:-/root/unfuse}"
SW="$ROOT/notebooks/qwen3.8-27b/bench/sweeps"
ORIG="Qwen/Qwen3.8-27B"
DERV="sartajbhuvaji/unfuse-Qwen3.8-27B-text"
LOG="$SW/phase_b.log"

cd "$ROOT"
source /root/venv/bin/activate
: "${HF_TOKEN:?export HF_TOKEN in this shell}"
mkdir -p "$SW"
exec > >(tee -a "$LOG") 2>&1

echo "== phase B start $(date -Is) =="

OUT="$SW/baseline-512" IN_LEN=512 OUT_LEN=256 CONC="1 16" CTX=32768 \
  MAX_SEQS= MAX_RUNNING= \
  tools/bench_pair.sh "$ORIG" "$DERV" vllm sglang

OUT="$SW/longctx-in4096" IN_LEN=4096 OUT_LEN=256 CONC="1 16" CTX=8192 NPROMPTS=32 \
  MAX_SEQS= MAX_RUNNING= \
  tools/bench_pair.sh "$ORIG" "$DERV" vllm sglang

OUT="$SW/conc-c32" IN_LEN=512 OUT_LEN=256 CONC="32" CTX=32768 NPROMPTS=64 \
  MAX_SEQS=32 MAX_RUNNING=32 \
  tools/bench_pair.sh "$ORIG" "$DERV" vllm sglang

echo "== phase B end $(date -Is) =="
