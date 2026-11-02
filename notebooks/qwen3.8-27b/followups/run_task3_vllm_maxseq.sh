#!/usr/bin/env bash
# Task 3: vLLM baseline 512/256 at max-num-seqs 64, 128, 256, 320 (parent may refuse).
set -euo pipefail
ROOT="${UNFUSE_ROOT:-/root/unfuse}"
FUP="$ROOT/notebooks/qwen3.8-27b/followups"
ORIG="Qwen/Qwen3.8-27B"
DERV="sartajbhuvaji/unfuse-Qwen3.8-27B-text"
LOG="$FUP/task3_maxseq.log"

cd "$ROOT"
source /root/venv/bin/activate
exec 9>/tmp/qwen_followups_task3.lock
flock -n 9 || { echo "task3 lock held"; exit 1; }
exec > >(tee -a "$LOG") 2>&1
echo "== task3 start $(date -Is) =="

for MS in 64 128 256 320; do
  OUT="$FUP/maxseq-$MS"
  mkdir -p "$OUT"
  echo "== MAX_SEQS=$MS =="
  if ! OUT="$OUT" IN_LEN=512 OUT_LEN=256 CONC="16" CTX=32768 NPROMPTS=64 \
    MAX_SEQS="$MS" MAX_RUNNING='' \
    tools/bench_pair.sh "$ORIG" "$DERV" vllm; then
    echo "FAILED maxseq=$MS" | tee "$OUT/bench_failed.txt"
  fi
done
echo "== task3 end $(date -Is) =="
