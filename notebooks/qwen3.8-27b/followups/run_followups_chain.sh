#!/usr/bin/env bash
# Run followups 2→4 after task 1 (or skip 1 if ROPE_JSON already exists).
set -euo pipefail
ROOT="${UNFUSE_ROOT:-/root/unfuse}"
FUP="$ROOT/notebooks/qwen3.8-27b/followups"
LOG="$FUP/followups_chain.log"
exec > >(tee -a "$LOG") 2>&1

wait_task1() {
  while pgrep -f followups/rope_check.py >/dev/null; do
    echo "waiting for followups rope_check.py..."
    sleep 30
  done
  while pgrep -f qwen_gguf_rope_check.py >/dev/null; do
    echo "waiting for tools rope check..."
    sleep 30
  done
}

if [ ! -s "$FUP/rope_check.json" ]; then
  wait_task1
fi

if ! grep -q "task2 end" "$FUP/task2_quants.log" 2>/dev/null; then
  if [ -s "$HOME/gguf/Qwen3.8-27B-text-Q8_0.gguf" ]; then
    bash "$FUP/run_task2_resume.sh"
  else
    bash "$FUP/run_task2_quants_perplexity.sh"
  fi
fi
bash "$FUP/run_task3_vllm_maxseq.sh"
bash "$FUP/run_task4_vllm_longctx.sh"
echo "== chain end $(date -Is) =="
