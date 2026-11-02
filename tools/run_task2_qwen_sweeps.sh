#!/usr/bin/env bash
# DEPRECATED — use tools/run_phase_b_minimal.sh.
# Task 2 benchmark sweeps from docs/RUNBOOK-qwen-gguf-and-sweeps.md

set -euo pipefail

ROOT="${UNFUSE_ROOT:-/root/unfuse}"
SW="$ROOT/notebooks/qwen3.8-27b/bench/sweeps"
ORIG="Qwen/Qwen3.8-27B"
DERV="sartajbhuvaji/unfuse-Qwen3.8-27B-text"
LOG="$SW/sweeps.log"

cd "$ROOT"
source /root/venv/bin/activate
: "${HF_TOKEN:?set HF_TOKEN}"
mkdir -p "$SW"
exec > >(tee -a "$LOG") 2>&1

run_sweep() {
  local tag=$1
  shift
  local out="$SW/$tag"
  mkdir -p "$out"
  echo "== sweep $tag $(date -Is) env: $* =="
  OUT="$out" "$@" tools/bench_pair.sh "$ORIG" "$DERV" vllm sglang
}

echo "== baseline pair (512/256 c=1 16) =="
OUT="$SW/baseline-512" IN_LEN=512 OUT_LEN=256 CONC="1 16" CTX=32768 \
  tools/bench_pair.sh "$ORIG" "$DERV" vllm sglang

for IN in 4096 8192 32768; do
  CTX=40960
  NP=32
  [ "$IN" = "32768" ] && NP=16
  OUT="$SW/longctx-in${IN}" IN_LEN="$IN" OUT_LEN=256 CONC="1 16" CTX="$CTX" NPROMPTS="$NP" \
    tools/bench_pair.sh "$ORIG" "$DERV" vllm sglang || echo "FAIL longctx $IN"
done

for C in 32 64 128; do
  OUT="$SW/conc-c${C}" IN_LEN=512 OUT_LEN=256 CONC="$C" CTX=32768 NPROMPTS=64 \
    MAX_SEQS="$C" MAX_RUNNING="$C" \
    tools/bench_pair.sh "$ORIG" "$DERV" vllm sglang || echo "FAIL conc $C"
done

echo "== task2 end $(date -Is) =="
