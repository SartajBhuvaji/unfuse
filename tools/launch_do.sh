#!/usr/bin/env bash
# Provision a DigitalOcean GPU Droplet for unfuse work, then (optionally)
# hand off to bootstrap_do_gpu.sh.
#
# Reads DIGITALOCEAN_ACCESS_TOKEN from .env. API base is api.digitalocean.com/v2.

set -euo pipefail

API="https://api.digitalocean.com/v2"
KEY_NAME="${KEY_NAME:-do-unfuse}"
KEY_PATH="${KEY_PATH:-$HOME/.ssh/do_unfuse}"
SIZE="${SIZE:-gpu-h100x1-80gb}"
REGION="${REGION:-}"
IMAGE="${IMAGE:-gpu-h100x1-base}"
NAME="${NAME:-unfuse}"

[ -f .env ] && { set -a; . <(sed 's/\r$//' .env); set +a; }
: "${DIGITALOCEAN_ACCESS_TOKEN:?set DIGITALOCEAN_ACCESS_TOKEN in .env}"

api() { curl -sS -H "Authorization: Bearer $DIGITALOCEAN_ACCESS_TOKEN" -H "Content-Type: application/json" "$@"; }

sizes() {
  echo "== GPU sizes and where $SIZE has capacity =="
  api "$API/sizes?per_page=200" | python3 -c "
import json,sys
d=json.load(sys.stdin)['sizes']
gpu=[s for s in d if s['slug'].startswith('gpu-')]
for s in sorted(gpu, key=lambda s: s['slug']):
    mark = ' <-- SIZE' if s['slug'] == '$SIZE' else ''
    print(f\"  {s['slug']:<22} \${s['price_hourly']:>6.2f}/hr  available={s['available']}  regions={s['regions']}{mark}\")"
}

ensure_key() {
  [ -f "$KEY_PATH" ] || ssh-keygen -t ed25519 -f "$KEY_PATH" -N "" -C "$KEY_NAME" >/dev/null
  existing=$(api "$API/account/keys?per_page=200" | python3 -c "
import json,sys
ks=[k for k in json.load(sys.stdin)['ssh_keys'] if k['name']=='$KEY_NAME']
print(ks[0]['fingerprint'] if ks else '')")
  [ -n "$existing" ] && return 0
  P="$(cat "$KEY_PATH.pub")" python3 -c "
import json,os; print(json.dumps({'name':'$KEY_NAME','public_key':os.environ['P'].strip()}))" > /tmp/do_key.json
  api --data @/tmp/do_key.json "$API/account/keys" >/dev/null
}

verify_key() {
  local stored local_b64
  stored=$(api "$API/account/keys?per_page=200" | python3 -c "
import json,sys
ks=[k for k in json.load(sys.stdin)['ssh_keys'] if k['name']=='$KEY_NAME']
print(ks[0]['public_key'].split()[1] if ks else '')")
  local_b64=$(cut -d' ' -f2 < "$KEY_PATH.pub")
  [ -n "$stored" ] && [ "$stored" = "$local_b64" ] \
    || { echo "FATAL: stored key != local key."; exit 1; }
  echo "key verified: stored matches $KEY_PATH.pub"
}

key_fingerprint() {
  api "$API/account/keys?per_page=200" | python3 -c "
import json,sys
ks=[k for k in json.load(sys.stdin)['ssh_keys'] if k['name']=='$KEY_NAME']
print(ks[0]['fingerprint'])"
}

launch() {
  [ -n "$REGION" ] || { echo "FATAL: set REGION (run '$0 sizes')"; exit 1; }
  local fp; fp=$(key_fingerprint)
  api -d "{
    \"name\":\"$NAME\",\"region\":\"$REGION\",\"size\":\"$SIZE\",\"image\":\"$IMAGE\",
    \"ssh_keys\":[\"$fp\"],\"ipv6\":false,\"tags\":[\"unfuse\"]}" \
    "$API/droplets" | python3 -c "
import json,sys
r=json.load(sys.stdin)
if 'id' not in r.get('droplet', {}): sys.exit('launch failed: %s' % r)
print(r['droplet']['id'])"
}

wait_active() {
  local did=$1 ip=""
  for _ in $(seq 1 60); do
    read -r st ip < <(api "$API/droplets/$did" | python3 -c "
import json,sys
d=json.load(sys.stdin)['droplet']
v4=[n for n in d['networks']['v4'] if n['type']=='public']
print(d['status'], v4[0]['ip_address'] if v4 else '-')")
    echo "  $st $ip"
    [ "$st" = "active" ] && [ "$ip" != "-" ] && break
    sleep 15
  done
  for _ in $(seq 1 20); do
    ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
        -o BatchMode=yes "root@$ip" true 2>/dev/null && break
    sleep 15
  done
  echo "$ip"
}

case "${1:-launch}" in
  sizes) sizes ;;
  launch)
    sizes; ensure_key; verify_key
    DID=$(launch); echo "launched $DID"
    IP=$(wait_active "$DID" | tail -1)
    echo
    echo "  droplet   : $DID"
    echo "  ip        : $IP"
    echo "  ssh       : ssh -i $KEY_PATH root@$IP"
    echo "  bootstrap : HF_TOKEN=... SSH_KEY=$KEY_PATH ./tools/bootstrap_do_gpu.sh $IP"
    echo "  terminate : $0 terminate $DID"
    if [ "${BOOTSTRAP:-0}" = "1" ]; then
      echo "== bootstrapping (BOOTSTRAP=1) =="
      SSH_KEY="$KEY_PATH" ./tools/bootstrap_do_gpu.sh "$IP"
    fi
    ;;
  terminate)
    : "${2:?usage: $0 terminate <droplet-id>}"
    api -X DELETE "$API/droplets/$2" ;;
  *) echo "usage: $0 {sizes|launch|terminate <id>}"; exit 1 ;;
esac
