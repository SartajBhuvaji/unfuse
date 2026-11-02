#!/usr/bin/env bash
# From your laptop: sync this repo to a DO GPU box and run setup_instance_do.sh.
#
# Usage:
#   HF_TOKEN=hf_... SSH_KEY=~/.ssh/id_ed25519 ./tools/bootstrap_do_gpu.sh 162.243.66.238
#
# HF_TOKEN is passed in the remote shell environment only — not written to .env on
# the droplet. Requires root SSH.

set -euo pipefail

IP="${1:?usage: $0 <droplet-ip>}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519}"
USER="${SSH_USER:-root}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/unfuse}"

if [ ! -f "$KEY" ]; then
  echo "FATAL: SSH private key not found at $KEY (set SSH_KEY=...)"
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ] && [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source <(sed 's/\r$//' "${REPO_ROOT}/.env")
  set +a
fi
: "${HF_TOKEN:?set HF_TOKEN in the environment (or in local .env for this bootstrap only)}"

SSH=(ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${USER}@${IP}")
RSYNC=(rsync -az --delete-delay
  --exclude .git/
  --exclude .venv/
  --exclude venv/
  --exclude __pycache__/
  --exclude .tox/
  --exclude '*.gguf'
  --exclude models/
  -e "ssh -i '${KEY}' -o BatchMode=yes -o StrictHostKeyChecking=accept-new")

echo "== checking SSH to ${USER}@${IP} =="
"${SSH[@]}" "whoami && nvidia-smi -L"

echo "== syncing repo to ${REMOTE_ROOT} =="
"${SSH[@]}" "mkdir -p ${REMOTE_ROOT}"
"${RSYNC[@]}" "${REPO_ROOT}/" "${USER}@${IP}:${REMOTE_ROOT}/"

echo "== running setup_instance_do.sh =="
tok=$(printf %q "$HF_TOKEN")
"${SSH[@]}" "export HF_TOKEN=${tok} && cd ${REMOTE_ROOT} && bash tools/setup_instance_do.sh"

echo "== gpu provenance =="
"${SSH[@]}" "PYTHON=/root/venv/bin/python bash ${REMOTE_ROOT}/tools/gpu_env.sh"

echo "BOOTSTRAP_DONE ssh -i ${KEY} ${USER}@${IP}"
