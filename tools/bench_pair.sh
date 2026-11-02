#!/usr/bin/env bash
# Benchmark a multimodal parent and its text-only derivative against each other,
# on one GPU, through one client. Run ON the instance.
#
#   tools/bench_pair.sh ORIGINAL DERIVED [ENGINE...]
#   ENGINE   vllm sglang   (default: both)
#
# Writes to $OUT (default ~/bench-pair):
#   env.json                     the machine, from gpu_env.sh
#   serve-<engine>-<role>.json   memory + scheduling facts scraped from startup
#   bench-<engine>-<role>-warm.json    the row that counts
#   bench-<engine>-<role>-cold.json    the discarded first pass, kept as evidence
#   <engine>-<role>.log          the raw startup log behind the scraped numbers
#
# THE THREE THINGS THAT DECIDE WHETHER THE RESULT MEANS ANYTHING:
#
#   1. One client. Not `vllm bench serve` against `sglang.bench_serving`. Those
#      two disagree on prompt sampling, warmup and what "duration" means, and on
#      the first GLM run they produced a 3x TTFT gap that reversed the moment a
#      shared client measured both. bench_openai.py is that shared client.
#
#   2. Two passes, first discarded. On Qwen3.8-27B an unwarmed first pass read
#      457.9 tok/s where the warmed pass read 554.7 -- 21%, larger than any
#      effect being reported. Both passes are written out so a warm row can
#      never be quietly compared against a cold one.
#
#   3. The scheduler's own concurrency limit, recorded per model. SGLang handed
#      the Qwen parent max_running_requests=5 and the derivative 16, and the
#      2.79x that fell out of that got as far as a draft model card before
#      anyone checked. serve_stats.py scrapes it so the two rows can be
#      compared, or marked as not comparable.
#
# MAX_SEQS / MAX_RUNNING pin the limit equal across both models. Leave them
# unset for the first pass -- you want to see what the engine picks on its own
# before you overrule it.

set -uo pipefail

ORIG="${1:?usage: $0 ORIGINAL DERIVED [vllm|sglang ...]}"
DERV="${2:?usage: $0 ORIGINAL DERIVED [vllm|sglang ...]}"
shift 2
ENGINES=("${@:-vllm sglang}")
[ "${#ENGINES[@]}" -eq 1 ] && read -ra ENGINES <<< "${ENGINES[0]}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HOME/bench-pair}"; mkdir -p "$OUT"
VENVS="${VENVS:-$HOME/venvs}"
CTX="${CTX:-32768}"
IN_LEN="${IN_LEN:-512}"; OUT_LEN="${OUT_LEN:-256}"; NPROMPTS="${NPROMPTS:-64}"
CONC="${CONC:-1 16}"
VLLM_VER="${VLLM_VER:-0.27.1}"; SGL_VER="${SGL_VER:-0.5.18}"
MAX_SEQS="${MAX_SEQS:-}"; MAX_RUNNING="${MAX_RUNNING:-}"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

mkvenv() { local n=$1; shift
  [ -d "$VENVS/$n" ] || uv venv "$VENVS/$n" --python 3.12 -q
  VIRTUAL_ENV="$VENVS/$n" uv pip install -q "$@" 2>&1 | tail -2; }

wait_http() { local url=$1 max=${2:-1800} i=0
  while [ "$i" -lt "$max" ]; do
    curl -sf "$url" >/dev/null 2>&1 && { echo "$i"; return 0; }
    # A server that died is not a server that is still loading. Without this the
    # loop burns the full 30 min timeout on a stack trace written 20 s in.
    pgrep -f "$SERVER_PAT" >/dev/null || return 1
    sleep 5; i=$((i+5))
  done
  return 1; }

bench2() {  # <port> <engine> <role> -- cold pass, then the warm one that counts
  local port=$1 eng=$2 role=$3
  read -ra levels <<< "$CONC"
  for pass in cold warm; do
    echo "  $pass pass"
    python3 "$HERE/bench_openai.py" --port "$port" --label "$eng-$role-$pass" \
      --input-len "$IN_LEN" --output-len "$OUT_LEN" --num-prompts "$NPROMPTS" \
      --concurrency "${levels[@]}" --json-out "$OUT/bench-$eng-$role-$pass.json" >/dev/null
  done; }

run_vllm() {  # <model> <role>
  local model=$1 role=$2 log="$OUT/vllm-$role.log"
  SERVER_PAT="vllm serve"
  # vLLM's compile path shells out to `ninja` by bare name. Running
  # $VENVS/vllm/bin/vllm does NOT put $VENVS/vllm/bin on PATH -- only activating
  # the venv does -- so the ninja sitting right next to that binary stays
  # invisible and the failure surfaces as a bare FileNotFoundError three frames
  # deep inside flashinfer's JIT, under a RuntimeError that names nothing.
  export PATH="$VENVS/vllm/bin:$PATH"
  pkill -f "vllm serve" 2>/dev/null; sleep 8
  local args=(--max-model-len "$CTX" --port 8000 --reasoning-parser glm45)
  [ -n "$MAX_SEQS" ] && args+=(--max-num-seqs "$MAX_SEQS")
  nohup "$VENVS/vllm/bin/vllm" serve "$model" "${args[@]}" > "$log" 2>&1 &
  local t; t=$(wait_http http://127.0.0.1:8000/health) || {
    echo "  DID NOT START. last lines:"; tail -25 "$log"; return 1; }
  echo "  ready in ${t}s"
  python3 "$HERE/serve_stats.py" "$log" --engine vllm --engine-version "$VLLM_VER" \
    --label "vllm-$role" --model "$model" --json-out "$OUT/serve-vllm-$role.json"
  bench2 8000 vllm "$role"
  pkill -f "vllm serve" 2>/dev/null; sleep 10; }

run_sglang() {  # <model> <role>
  local model=$1 role=$2 log="$OUT/sglang-$role.log"
  SERVER_PAT="sglang.launch_server"
  export PATH="$VENVS/sglang/bin:$PATH"
  pkill -f sglang.launch_server 2>/dev/null; sleep 8
  local args=(--context-length "$CTX" --host 127.0.0.1 --port 30000 --reasoning-parser glm45)
  [ -n "$MAX_RUNNING" ] && args+=(--max-running-requests "$MAX_RUNNING")
  nohup "$VENVS/sglang/bin/python" -m sglang.launch_server --model-path "$model" \
    "${args[@]}" > "$log" 2>&1 &
  local t; t=$(wait_http http://127.0.0.1:30000/health) || {
    echo "  DID NOT START. last lines:"; tail -25 "$log"; return 1; }
  echo "  ready in ${t}s"
  python3 "$HERE/serve_stats.py" "$log" --engine sglang --engine-version "$SGL_VER" \
    --label "sglang-$role" --model "$model" --json-out "$OUT/serve-sglang-$role.json"
  bench2 30000 sglang "$role"
  pkill -f sglang.launch_server 2>/dev/null; sleep 10; }

bash "$HERE/gpu_env.sh" --json > "$OUT/env.json"
bash "$HERE/gpu_env.sh"

for eng in "${ENGINES[@]}"; do
  case "$eng" in
    vllm)   mkvenv vllm "vllm==$VLLM_VER" ninja ;;
    sglang) mkvenv sglang "sglang[all]==$SGL_VER" ;;
    *) echo "unknown engine: $eng"; exit 1 ;;
  esac
  # Each engine brings its own torch. env.json records the ambient one, which
  # is not what served anything, so capture the venv that actually ran.
  PYTHON="$VENVS/$eng/bin/python" bash "$HERE/gpu_env.sh" --json > "$OUT/env-$eng.json"
  # Original first, every time. If the box is going to fall over it should fall
  # over on the harder model before four other rows have been paid for.
  for pair in "original:$ORIG" "derived:$DERV"; do
    role="${pair%%:*}"; model="${pair#*:}"
    say "$eng / $role / $model"
    "run_$eng" "$model" "$role"
  done
done

say "concurrency limits the engines chose"
python3 - "$OUT" <<'PY'
import glob, json, os, sys
by = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "serve-*.json"))):
    d = json.load(open(f))
    by.setdefault(d["engine"], {})[os.path.basename(f).split("-")[-1][:-5]] = d
for eng, roles in by.items():
    lim = {r: d.get("concurrency_limit") for r, d in roles.items()}
    same = len(set(lim.values())) == 1
    print(f"  {eng:8} {lim}  {'comparable' if same else 'NOT COMPARABLE -- pin them equal and rerun'}")
PY

say "results"
python3 - "$OUT" <<'PY'
import glob, json, os, sys
hdr = ("engine", "model", "c", "out tok/s", "per stream", "TTFT p50", "TTFT p99", "TPOT")
print("| " + " | ".join(hdr) + " |"); print("|" + "|".join(["---"] * len(hdr)) + "|")
for f in sorted(glob.glob(os.path.join(sys.argv[1], "bench-*-warm.json"))):
    d = json.load(open(f)); eng, role = d["label"].split("-")[:2]
    for r in d["runs"]:
        print(f"| {eng} | {role} | {r['concurrency']} | {r['output_tok_s']} | "
              f"{r['per_stream_tok_s']} | {r['ttft_ms_p50']} ms | {r['ttft_ms_p99']} ms | "
              f"{r['tpot_ms_mean']} ms |")
PY
