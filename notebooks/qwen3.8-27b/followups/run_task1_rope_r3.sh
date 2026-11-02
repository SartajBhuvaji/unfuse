#!/usr/bin/env bash
# Re-run followups/rope_check.py after parser/reasoning fix (GPU; ~30-60 min).
set -euo pipefail
ROOT="${UNFUSE_ROOT:-/root/unfuse}"
FUP="$ROOT/notebooks/qwen3.8-27b/followups"
LOG="$FUP/task1_rope_r3.log"
cd "$ROOT"
source /root/venv/bin/activate
exec > >(tee -a "$LOG") 2>&1
echo "== rope r3 $(date -Is) =="
python "$FUP/rope_check.py"
echo "== rope r3 end $(date -Is) =="
