#!/usr/bin/env bash
# Smoke-test AND benchmark one checkpoint across every stack we publish for.
# Run ON a GPU instance. Needs uv (installed automatically) and, for llamacpp,
# nvcc + cmake.
#
#   tools/smoke_inference.sh MODEL [BACKEND...]
#   MODEL      HF repo id or local path
#   BACKEND    transformers vllm sglang llamacpp   (default: all four)
#   GGUF=...   local .gguf path; required by the llamacpp backend
#
# Writes $LOGS/results.tsv (pass/fail) and $LOGS/bench-<backend>.json (metrics),
# then prints both as tables ready to paste into a model card.
#
# WHY THIS CHECKS ANSWERS AND NOT EXIT CODES:
#   The worst failure in this repo's history loaded cleanly, served happily, and
#   emitted "same dePK singleale dePK singleale". A W8A16 checkpoint hit vLLM's
#   W8A8-only FP8 path; because the weight shapes matched, nothing raised. Exit
#   status proved nothing. So every backend here must produce a known-correct
#   token -- "Paris", "126" -- or it is recorded FAIL however cleanly it exited.
#
# TRAPS THIS SCRIPT EXISTS TO AVOID (each cost real GPU-hours to find):
#   1. vLLM's compile path shells out to `ninja` and dies with a bare
#      FileNotFoundError three frames deep in EngineCore. Installing ninja is not
#      enough: running $VENV/bin/vllm does NOT put $VENV/bin on PATH -- only
#      activation does -- so ninja stays invisible. Hence `export PATH` below.
#   2. vllm and sglang pin different torch builds. One shared venv means whichever
#      you installed second silently downgrades the first. Always separate.
#   3. /health returns 200 well after the process starts and well before it can
#      decode. Poll for it; never sleep-and-hope.
#   4. Reasoning models spend the whole budget thinking. max_tokens=160 returns a
#      truncated monologue and no answer, which reads exactly like a broken model.
#      Budget 900, and probe enable_thinking=false as its own case.
#   5. llama.cpp ships no Linux CUDA binary -- only Windows. Linux means source.
#   6. llama-cli without `-st` drops into interactive mode and blocks on stdin
#      forever. That looks identical to a hung GPU.

set -uo pipefail

MODEL="${1:?usage: $0 MODEL [BACKEND...]}"; shift
BACKENDS=("${@:-transformers vllm sglang llamacpp}")
[ "${#BACKENDS[@]}" -eq 1 ] && read -ra BACKENDS <<< "${BACKENDS[0]}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$HERE/bench_openai.py"
VENVS="${VENVS:-$HOME/venvs}"
LOGS="${LOGS:-$HOME/logs}"; mkdir -p "$LOGS"
CTX="${CTX:-32768}"
GGUF="${GGUF:-}"
CONC="${CONC:-1 16}"          # concurrency levels to benchmark
IN_LEN="${IN_LEN:-512}"; OUT_LEN="${OUT_LEN:-256}"; NPROMPTS="${NPROMPTS:-64}"
RESULTS="$LOGS/results.tsv"; : > "$RESULTS"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
pass() { printf 'PASS\t%s\t%s\n' "$1" "${2:-}" >> "$RESULTS"; printf '  \033[32mPASS\033[0m %s %s\n' "$1" "${2:-}"; }
fail() { printf 'FAIL\t%s\t%s\n' "$1" "${2:-}" >> "$RESULTS"; printf '  \033[31mFAIL\033[0m %s %s\n' "$1" "${2:-}"; }

mkvenv() { local n=$1; shift
  [ -d "$VENVS/$n" ] || uv venv "$VENVS/$n" --python 3.12 -q
  VIRTUAL_ENV="$VENVS/$n" uv pip install -q "$@" 2>&1 | tail -3; }

wait_http() { local url=$1 max=${2:-900} i=0
  while [ "$i" -lt "$max" ]; do
    curl -sf "$url" >/dev/null 2>&1 && { echo "$i"; return 0; }
    sleep 5; i=$((i+5)); done
  return 1; }

read -r -d '' CASES <<'JSON'
[
 ["What is the capital of France? Answer in one word.", "paris"],
 ["What is 84 * 3 / 2? Reply with just the number.", "126"],
 ["Translate to German: The weather is cold today.", "kalt"]
]
JSON

check_openai() {  # <port>
  CASES="$CASES" python3 - "$1" <<'PY'
import json, os, sys, time, urllib.request
port = sys.argv[1]
url = f"http://127.0.0.1:{port}/v1/chat/completions"
model = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models"))["data"][0]["id"]

def ask(prompt, **extra):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 900, "temperature": 0, **extra}
    r = urllib.request.Request(url, json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r))
    return d["choices"][0]["message"], d.get("usage", {})

ok = True
for prompt, anchor in json.loads(os.environ["CASES"]):
    t = time.time(); m, u = ask(prompt); dt = time.time() - t
    # vLLM calls it `reasoning`; SGLang calls it `reasoning_content`. Same data,
    # different key -- client code hardcoding one silently drops the other.
    body = (m.get("content") or "") + (m.get("reasoning") or m.get("reasoning_content") or "")
    hit = anchor.lower() in body.lower(); ok &= hit
    print(f"  [{'ok' if hit else 'MISS'}] {u.get('completion_tokens',0):>4} tok {dt:5.1f}s  {prompt[:44]!r}")
    if not hit: print(f"        wanted {anchor!r}, got {(m.get('content') or '')[:200]!r}")
try:
    m, u = ask("What is the capital of France? Answer in one word.",
               chat_template_kwargs={"enable_thinking": False})
    print(f"  no-think: {u.get('completion_tokens')} tok -> {(m.get('content') or '').strip()[:60]!r}")
except Exception as e:
    print(f"  no-think: unsupported ({type(e).__name__})")
sys.exit(0 if ok else 1)
PY
}

bench_openai() {  # <port> <label>
  read -ra levels <<< "$CONC"
  python3 "$BENCH" --port "$1" --label "$2" --input-len "$IN_LEN" \
    --output-len "$OUT_LEN" --num-prompts "$NPROMPTS" \
    --concurrency "${levels[@]}" --json-out "$LOGS/bench-$2.json" > /dev/null
}

run_transformers() {
  say "transformers"
  mkvenv hf torch transformers accelerate
  CASES="$CASES" OUT="$LOGS/bench-transformers.json" \
  "$VENVS/hf/bin/python" - "$MODEL" <<'PY'
import json, os, sys, threading, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
M = sys.argv[1]
t0 = time.time()
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="auto")
load_s = time.time() - t0
print(f"  load {load_s:.0f}s  {sum(p.numel() for p in model.parameters()):,} params")

ok = True
for prompt, anchor in json.loads(os.environ["CASES"]):
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(model.device)
    t = time.time(); out = model.generate(**enc, max_new_tokens=900, do_sample=False)
    n = out.shape[1] - enc["input_ids"].shape[1]; dt = time.time() - t
    txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    hit = anchor.lower() in txt.lower(); ok &= hit
    print(f"  [{'ok' if hit else 'MISS'}] {n:>4} tok {dt:5.1f}s {n/dt:5.1f} tok/s  {prompt[:40]!r}")
    if not hit: print(f"        wanted {anchor!r}, got {txt[:200]!r}")

# Single-stream TTFT needs a streamer: generate() only returns when finished, so
# without one, prefill and decode are indistinguishable.
enc = tok(("the quick brown fox jumps over the lazy dog. " * 52), return_tensors="pt").to(model.device)
streamer = TextIteratorStreamer(tok, skip_prompt=True)
kw = dict(**enc, max_new_tokens=256, do_sample=False, streamer=streamer)
t0 = time.perf_counter()
# NOT daemon=True. A daemon thread still inside generate() when the interpreter
# starts tearing down CUDA aborts the process with "terminate called without an
# active exception" -- AFTER every check has passed and the metrics are written.
# The run is recorded FAIL on exit status alone, which is precisely the mistake
# this script exists to avoid. Join it.
th = threading.Thread(target=model.generate, kwargs=kw)
th.start()
ttft, n = None, 0
for _ in streamer:
    if ttft is None: ttft = time.perf_counter() - t0
    n += 1
total = time.perf_counter() - t0
th.join()
row = {"label": "transformers", "runs": [{
    "concurrency": 1, "requests": 1, "failed": 0, "duration_s": round(total, 2),
    "output_tok_s": round(n / total, 1), "per_stream_tok_s": round(n / total, 1),
    "ttft_ms_mean": round(ttft * 1000, 1), "ttft_ms_p50": round(ttft * 1000, 1),
    "ttft_ms_p99": round(ttft * 1000, 1),
    "tpot_ms_mean": round((total - ttft) * 1000 / max(n - 1, 1), 2)}],
    "load_s": round(load_s, 1)}
json.dump(row, open(os.environ["OUT"], "w"), indent=2)
print(f"  c=1   {row['runs'][0]['output_tok_s']:>8.1f} out tok/s | "
      f"TTFT p50 {row['runs'][0]['ttft_ms_p50']:>7.1f} ms | TPOT {row['runs'][0]['tpot_ms_mean']:>6.2f} ms")
sys.exit(0 if ok else 1)
PY
  [ $? -eq 0 ] && pass transformers || fail transformers
}

# Ask the engine which parsers it has by handing it one it does not. More honest
# than grepping a version-specific registry that only populates lazily.
probe_parsers() { local v=$1 f=$2
  export PATH="$VENVS/$v/bin:$PATH"
  timeout 150 vllm serve "$MODEL" "$f" __nope__ --port 8899 2>&1 \
    | grep -oE "Available parsers: .*|chose from \{[^}]*\}" | head -1; }

run_vllm() {
  say "vLLM"
  mkvenv vllm vllm ninja           # ninja: trap 1
  export PATH="$VENVS/vllm/bin:$PATH"
  local ver; ver=$("$VENVS/vllm/bin/python" -c 'import vllm;print(vllm.__version__)' 2>/dev/null)
  echo "  vllm $ver"
  local rp; rp=$(probe_parsers vllm --reasoning-parser)
  pkill -f "vllm serve" 2>/dev/null; sleep 5
  local args=(--max-model-len "$CTX" --port 8000)
  # Only pass a parser this build actually has -- an unknown name is a hard KeyError.
  grep -q "glm45" <<< "$rp" && args+=(--reasoning-parser glm45 --tool-call-parser glm45 --enable-auto-tool-choice)
  nohup vllm serve "$MODEL" "${args[@]}" > "$LOGS/vllm.log" 2>&1 &
  local t; t=$(wait_http http://127.0.0.1:8000/health) \
    || { fail vllm "never came up -- see $LOGS/vllm.log"; return; }
  echo "  ready in ${t}s"
  if check_openai 8000; then bench_openai 8000 vllm; pass vllm "${ver} ready=${t}s"
  else fail vllm "$ver"; fi
  pkill -f "vllm serve" 2>/dev/null; sleep 8
}

run_sglang() {
  say "SGLang"
  mkvenv sglang "sglang[all]"
  local ver; ver=$("$VENVS/sglang/bin/python" -c 'import sglang;print(sglang.__version__)' 2>/dev/null)
  echo "  sglang $ver"
  pkill -f sglang.launch_server 2>/dev/null; sleep 5
  nohup "$VENVS/sglang/bin/python" -m sglang.launch_server --model-path "$MODEL" \
    --context-length "$CTX" --reasoning-parser auto --tool-call-parser auto \
    --host 127.0.0.1 --port 30000 > "$LOGS/sglang.log" 2>&1 &
  local t; t=$(wait_http http://127.0.0.1:30000/health) \
    || { fail sglang "never came up -- see $LOGS/sglang.log"; return; }
  echo "  ready in ${t}s"
  if check_openai 30000; then bench_openai 30000 sglang; pass sglang "${ver} ready=${t}s"
  else fail sglang "$ver"; fi
  pkill -f sglang.launch_server 2>/dev/null; sleep 8
}

run_llamacpp() {
  say "llama.cpp"
  [ -n "$GGUF" ] || { fail llamacpp "set GGUF=/path/to/model.gguf"; return; }
  local B="$HOME/llama.cpp/build/bin"
  if [ ! -x "$B/llama-cli" ]; then
    git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$HOME/llama.cpp" 2>/dev/null
    cmake -S "$HOME/llama.cpp" -B "$HOME/llama.cpp/build" -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-80}" -DLLAMA_CURL=ON \
      -DCMAKE_BUILD_TYPE=Release > "$LOGS/llamacpp_build.log" 2>&1
    cmake --build "$HOME/llama.cpp/build" -j"$(nproc)" --target llama-cli llama-bench \
      >> "$LOGS/llamacpp_build.log" 2>&1 \
      || { fail llamacpp "build failed -- see $LOGS/llamacpp_build.log"; return; }
  fi
  local ok=1
  while IFS=$'\t' read -r prompt anchor; do
    # -st and </dev/null: without both, llama-cli waits on stdin forever (trap 6).
    local out; out=$("$B/llama-cli" -m "$GGUF" -p "$prompt" -n 900 -ngl 99 -st --temp 0 \
      2>/dev/null < /dev/null)
    if grep -qi -- "$anchor" <<< "$out"; then echo "  [ok] ${prompt:0:44}"
    else ok=0; echo "  [MISS] ${prompt:0:44} -- wanted $anchor"; fi
  done < <(CASES="$CASES" python3 -c "
import json,os
for p,a in json.loads(os.environ['CASES']): print(p+'\t'+a)")
  # llama-bench is the standard here: pp = prefill t/s, tg = decode t/s.
  "$B/llama-bench" -m "$GGUF" -ngl 99 -p "$IN_LEN" -n "$OUT_LEN" -o json \
    > "$LOGS/bench-llamacpp.json" 2>/dev/null
  python3 - "$LOGS/bench-llamacpp.json" <<'PY'
import json, sys
try: rows = json.load(open(sys.argv[1]))
except Exception: sys.exit(0)
for r in rows:
    kind = "prefill" if r.get("n_prompt") else "decode"
    print(f"  {kind:8} {r['avg_ts']:8.1f} tok/s  (n={r.get('n_prompt') or r.get('n_gen')})")
PY
  [ "$ok" -eq 1 ] && pass llamacpp || fail llamacpp
}

for b in "${BACKENDS[@]}"; do
  case "$b" in
    transformers) run_transformers ;;
    vllm) run_vllm ;;
    sglang) run_sglang ;;
    llamacpp) run_llamacpp ;;
    *) echo "unknown backend: $b" ;;
  esac
done

say "correctness  ($MODEL)"
column -t -s$'\t' "$RESULTS"

say "metrics"
python3 - "$LOGS" <<'PY'
import glob, json, os, sys
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "bench-*.json"))):
    try: d = json.load(open(f))
    except Exception: continue
    if not isinstance(d, dict): continue          # llama-bench writes a list
    for r in d.get("runs", []):
        rows.append((d.get("label") or os.path.basename(f), r))
if not rows:
    print("  (none)"); sys.exit()
hdr = ("engine", "conc", "out tok/s", "per-stream", "TTFT p50 ms", "TTFT p99 ms", "TPOT ms")
print("| " + " | ".join(hdr) + " |")
print("|" + "|".join(["---"] * len(hdr)) + "|")
for label, r in rows:
    print(f"| {label} | {r['concurrency']} | {r['output_tok_s']} | {r['per_stream_tok_s']} "
          f"| {r['ttft_ms_p50']} | {r['ttft_ms_p99']} | {r['tpot_ms_mean']} |")
PY
! grep -q '^FAIL' "$RESULTS"
