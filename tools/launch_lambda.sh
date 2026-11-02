#!/usr/bin/env bash
# Provision a Lambda Cloud GPU instance for unfuse work.
#
# Reads LAMBDA_API_KEY from .env. Base URL is cloud.lambda.ai (NOT lambdalabs.com).
#
# THE TRAP THIS SCRIPT EXISTS TO AVOID:
#   POST /ssh-keys with public_key=null does not error. Lambda silently generates
#   its own keypair and returns the private half in the response body. Discard that
#   response and you have an instance you can never SSH into -- the only fix is
#   terminate + relaunch, because ssh_key_names is fixed at launch time.
#   Hence verify_key() below. Do not remove it.

set -euo pipefail

API="https://cloud.lambda.ai/api/v1"
# NOT lambda-glm. That name is registered to the keypair Lambda generated for
# itself during the incident described above -- its private half was in an API
# response nobody kept, so any instance launched with it is unreachable. The
# account still carries it; verify_key is what stops you using it by accident.
KEY_NAME="${KEY_NAME:-lambda-glm2}"
KEY_PATH="${KEY_PATH:-$HOME/.ssh/lambda_glm}"
INSTANCE_TYPE="${INSTANCE_TYPE:-gpu_1x_a100_sxm4}"   # A100-SXM4 40GB, $1.99/hr
REGION="${REGION:-us-west-2}"
IMAGE_FAMILY="${IMAGE_FAMILY:-lambda-stack-24-04}"   # ships torch, NOT transformers
NAME="${NAME:-unfuse}"

# A .env saved from Windows carries CRLF, which leaves a trailing \r on the key
# and makes a correct key fail as `global/invalid-api-key`. Strip it on read
# rather than trusting the file to be clean.
[ -f .env ] && { set -a; . <(sed 's/\r$//' .env); set +a; }
: "${LAMBDA_API_KEY:?set LAMBDA_API_KEY in .env}"

api() { curl -sS -u "$LAMBDA_API_KEY:" "$@"; }

capacity() {
  echo "== capacity =="
  api "$API/instance-types" | python3 -c "
import json,sys
d=json.load(sys.stdin)['data']
for k,v in sorted(d.items()):
    r=[x['name'] for x in v['regions_with_capacity_available']]
    if r: print(f\"  {k:<26} \${v['instance_type']['price_cents_per_hour']/100:>6.2f}/hr  {r}\")"
}

ensure_key() {
  [ -f "$KEY_PATH" ] || ssh-keygen -t ed25519 -f "$KEY_PATH" -N "" -C "$KEY_NAME" >/dev/null
  # Env assignment MUST precede the command. `python3 -c "..." P=x` passes P= as a
  # script ARGUMENT, not an env var -- that is exactly how public_key became null.
  P="$(cat "$KEY_PATH.pub")" python3 -c "
import json,os; print(json.dumps({'name':'$KEY_NAME','public_key':os.environ['P'].strip()}))" > /tmp/key.json
  api -H "Content-Type: application/json" --data @/tmp/key.json "$API/ssh-keys" >/dev/null || true
}

verify_key() {
  local stored local_b64
  stored=$(api "$API/ssh-keys" | python3 -c "
import json,sys
ks=[k for k in json.load(sys.stdin)['data'] if k['name']=='$KEY_NAME']
print(ks[0]['public_key'].split()[1] if ks else '')")
  local_b64=$(cut -d' ' -f2 < "$KEY_PATH.pub")
  [ -n "$stored" ] && [ "$stored" = "$local_b64" ] \
    || { echo "FATAL: stored key != local key. Lambda generated its own."; exit 1; }
  echo "key verified: stored matches $KEY_PATH.pub"
}

launch() {
  api -H "Content-Type: application/json" -d "{
    \"region_name\":\"$REGION\",\"instance_type_name\":\"$INSTANCE_TYPE\",
    \"ssh_key_names\":[\"$KEY_NAME\"],\"name\":\"$NAME\",
    \"image\":{\"family\":\"$IMAGE_FAMILY\"}}" \
    "$API/instance-operations/launch" | python3 -c "
import json,sys
r=json.load(sys.stdin)
if 'error' in r: sys.exit('launch failed: %s' % r['error'])
print(r['data']['instance_ids'][0])"
}

wait_active() {
  local iid=$1 ip=""
  for _ in $(seq 1 60); do
    read -r st ip < <(api "$API/instances/$iid" | python3 -c "
import json,sys; d=json.load(sys.stdin)['data']; print(d['status'], d.get('ip') or '-')")
    echo "  $st $ip"
    [ "$st" = "active" ] && break
    sleep 15
  done
  # status flips to active a few seconds before sshd accepts the key
  for _ in $(seq 1 20); do
    ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
        -o BatchMode=yes "ubuntu@$ip" true 2>/dev/null && break
    sleep 15
  done
  echo "$ip"
}

case "${1:-launch}" in
  capacity) capacity ;;
  launch)
    capacity; ensure_key; verify_key
    IID=$(launch); echo "launched $IID"
    IP=$(wait_active "$IID" | tail -1)
    echo
    echo "  instance : $IID"
    echo "  ip       : $IP"
    echo "  ssh      : ssh -i $KEY_PATH ubuntu@$IP"
    echo "  terminate: $0 terminate $IID"
    ;;
  terminate)
    api -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$2\"]}" \
      "$API/instance-operations/terminate" | python3 -m json.tool ;;
  *) echo "usage: $0 {capacity|launch|terminate <id>}"; exit 1 ;;
esac
