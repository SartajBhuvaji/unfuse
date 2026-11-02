#!/usr/bin/env bash
# Task 4: vLLM long context 8k and 32k, c=1 and c=16, CTX=40960.
set -euo pipefail
ROOT="${UNFUSE_ROOT:-/root/unfuse}"
FUP="$ROOT/notebooks/qwen3.8-27b/followups"
ORIG="Qwen/Qwen3.8-27B"
DERV="sartajbhuvaji/unfuse-Qwen3.8-27B-text"
LOG="$FUP/task4_longctx.log"
VLLM_MAX_SEQS="${VLLM_MAX_SEQS:-256}"

cd "$ROOT"
source /root/venv/bin/activate
exec 9>/tmp/qwen_followups_task4.lock
flock -n 9 || { echo "task4 lock held"; exit 1; }
exec > >(tee -a "$LOG") 2>&1
echo "== task4 start $(date -Is) =="

for IN in 8192 32768; do
  tag="in${IN}"
  OUT="$FUP/longctx-$tag"
  mkdir -p "$OUT"
  # Fewer prompts at 32k so the row finishes inside the time box.
  NP=16
  [ "$IN" -eq 32768 ] && NP=8
  echo "== IN_LEN=$IN NPROMPTS=$NP =="
  OUT="$OUT" IN_LEN="$IN" OUT_LEN=256 CONC="1 16" CTX=40960 NPROMPTS="$NP" \
    MAX_SEQS="$VLLM_MAX_SEQS" MAX_RUNNING='' \
    tools/bench_pair.sh "$ORIG" "$DERV" vllm
done
echo "== task4 end $(date -Is) =="
