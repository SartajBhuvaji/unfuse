#!/usr/bin/env bash
# List unfuse-tagged DigitalOcean GPU Droplets and estimated running cost.

set -euo pipefail

API="https://api.digitalocean.com/v2"
TAG="${TAG:-unfuse}"

[ -f .env ] && { set -a; . <(sed 's/\r$//' .env); set +a; }
: "${DIGITALOCEAN_ACCESS_TOKEN:?set DIGITALOCEAN_ACCESS_TOKEN in .env}"

api() { curl -sS -H "Authorization: Bearer $DIGITALOCEAN_ACCESS_TOKEN" -H "Content-Type: application/json" "$@"; }

prices() {
  api "$API/sizes?per_page=200" | python3 -c "
import json,sys
d=json.load(sys.stdin)['sizes']
print(json.dumps({s['slug']: s['price_hourly'] for s in d}))"
}

status() {
  local price_json; price_json=$(prices)
  api "$API/droplets?tag_name=$TAG&per_page=200" | PRICES="$price_json" python3 -c "
import json, os, sys
from datetime import datetime, timezone

droplets = json.load(sys.stdin)['droplets']
prices = json.loads(os.environ['PRICES'])

if not droplets:
    print(f'no Droplets tagged \"$TAG\" are running')
    sys.exit(0)

total = 0.0
print(f'{\"id\":<12} {\"name\":<16} {\"size\":<16} {\"status\":<10} {\"ip\":<16} {\"elapsed\":<10} {\"cost so far\"}')
for d in droplets:
    created = datetime.fromisoformat(d['created_at'].replace('Z', '+00:00'))
    elapsed_h = (datetime.now(timezone.utc) - created).total_seconds() / 3600
    rate = prices.get(d['size_slug'], 0.0)
    cost = elapsed_h * rate
    total += cost
    v4 = [n for n in d['networks']['v4'] if n['type'] == 'public']
    ip = v4[0]['ip_address'] if v4 else '-'
    print(f\"{d['id']:<12} {d['name']:<16} {d['size_slug']:<16} {d['status']:<10} {ip:<16} \"
          f\"{elapsed_h:>7.2f}h  \${cost:>7.2f} (\${rate:.2f}/hr)\")

print(f'\ntotal running cost so far: \${total:.2f}')
"
}

killall() {
  local ids; ids=$(api "$API/droplets?tag_name=$TAG&per_page=200" | python3 -c "
import json,sys
print('\n'.join(str(d['id']) for d in json.load(sys.stdin)['droplets']))")
  if [ -z "$ids" ]; then
    echo "no Droplets tagged \"$TAG\" to kill"
    return 0
  fi
  echo "about to destroy:"
  while read -r id; do echo "  $id"; done <<< "$ids"
  if [ "${FORCE:-0}" != "1" ]; then
    read -r -p "destroy all of these? [y/N] " confirm
    [ "$confirm" = "y" ] || { echo "aborted"; return 1; }
  fi
  echo "$ids" | while read -r id; do
    echo "destroying $id"
    api -X DELETE "$API/droplets/$id" >/dev/null
  done
}

case "${1:-status}" in
  status) status ;;
  killall) killall ;;
  *) echo "usage: $0 {status|killall}  (killall: FORCE=1 skips confirmation)"; exit 1 ;;
esac
