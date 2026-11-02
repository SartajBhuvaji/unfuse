# Hard-won gotchas

Each of these cost real time. They are documented at their point of use.

- **Lambda `POST /ssh-keys` with a null `public_key` does not error.** It silently
  generates its own keypair and returns the private half in the response body.
  `ssh_key_names` is fixed at launch, so a wrong key means terminate + relaunch.
  Root cause was a shell bug: `python3 -c "..." P="$V"` passes `P=` as a script
  *argument*, not an env var. Env assignments go *before* the command.
- **`-no-cnv` was removed from llama.cpp.** Without `-st/--single-turn`, `llama-cli`
  drops into interactive mode and spins on empty stdin — burning CPU, emitting
  nothing, looking exactly like slow inference.
- **nbformat requires trailing newlines on every `source` line.** Splitting with
  `.split("\n")` produces a notebook VS Code renders fine (it rejoins with newlines)
  but `nbconvert` collapses to a single line, failing instantly with a `SyntaxError`.
- **`Glm4Config` does not populate `architectures`.** Without it `AutoModelForCausalLM`
  has nothing to dispatch to, and the checkpoint only loads via the explicit class.
- **The Hub quant-picker widget needs root-level GGUFs in a repo with no safetensors.**
  A `gguf/` subfolder still indexes metadata but renders no widget.
- **`total_size` in `model.safetensors.index.json` is not always bytes.** For
  GLM-4.6V-Flash it reports the parameter count. Verify against real file sizes.
- **A `.env` edited on Windows silently breaks every API key.** WSL reads the CRLF, so
  the key carries a trailing `\r` and the service reports `invalid-api-key` for a key
  that is correct. `file .env` says `CRLF line terminators`; `sed -i 's/\r$//' .env` fixes it.
- **Only Gemma 4 *12B* is encoder-free.** E2B/E4B/26B-A4B/31B are `model_type: gemma4`
  with conventional SigLIP towers. Check `model_type` before applying any recipe.
- **transformers 5.15.0 breaks every Gemma 4 31B on vLLM.** It made `head_dim` a
  per-layer attribute and raises `AmbiguousGlobalPerLayerAttributeError` on a global
  read; vLLM reads it globally. Measured: 5.5.3 through 5.14.1 return `head_dim = 256`,
  5.15.0 and 5.15.1 raise. Pin `transformers==5.14.1` to serve. The guard is correct,
  not a regression — Gemma 4 31B really has two head_dims, 256 on the 50 sliding layers
  and 512 on the 10 full-attention ones, which is also why it has 230 attention
  projections rather than 240 (the full layers share K and V, so no `v_proj`).
- **NVFP4 does not require Blackwell.** vLLM selects `MarlinNvFp4LinearKernel` and runs
  it from SM75 up. Blackwell provides native FP4 tensor cores; without them Marlin
  repacks to W4A16, so the *stored* saving is not a *runtime* saving — a 23 GB NVFP4
  checkpoint OOM'd a 40 GB A100. Separately, the FP8 KV cache needs SM89+, so on older
  cards pass `--kv-cache-dtype bfloat16`.
- **vLLM's ModelOpt FP8 path is W8A8-only, and it fails silently.**
  `ModelOptFp8LinearMethod` registers `input_scale` as required and calls
  `layer.input_scale.max()`. A weight-only FP8 (W8A16) checkpoint has no `input_scale`,
  but FP8 and BF16 weights have identical shapes, so the model loads, generates, and
  emits nonsense rather than erroring. Two related traps in the same config: vLLM routes
  to its mixed path only when `quant_algo` is exactly `MIXED_PRECISION`, and
  `quantized_layers` keys are matched by exact dict membership with **no glob
  expansion** — `model.layers.0.self_attn*` matches nothing, you need one key per
  projection.
- **Each engine's own benchmark tool disagrees with every other one.** `vllm bench serve`
  and `sglang.bench_serving` sample prompts differently, warm up differently, and define
  "duration" differently. Run against those, SGLang appeared to beat vLLM on TTFT by 3×;
  run against one shared client over plain HTTP, vLLM won by 1.7×. Cross-engine numbers
  are only meaningful from a single harness — `tools/bench_openai.py` is that harness.
  Pin output length with `ignore_eos`, or a reasoning model's chattiness becomes your
  throughput measurement.
- **Installing a tool into a venv does not put it on `PATH`.** vLLM shells out to `ninja`
  and dies with a bare `FileNotFoundError` surfaced three frames deep as
  `Engine core initialization failed`. Running `$VENV/bin/vllm` does *not* add
  `$VENV/bin` to `PATH` — only activation does — so a correctly installed `ninja` stays
  invisible. `export PATH="$VENV/bin:$PATH"`.
- **vllm and sglang pin different torch builds.** Installed into one venv, the second
  silently downgrades the first. Always give them separate environments.
- **A daemon thread holding CUDA aborts the process at exit, after everything passed.**
  `TextIteratorStreamer` work handed to `Thread(..., daemon=True)` can still be inside
  `generate()` when the interpreter tears down CUDA: `terminate called without an active
  exception`, non-zero exit, every check already green and the metrics already written.
  Non-daemon plus `join()`.
- **The reasoning field has two names.** vLLM 0.27.1 returns thinking as
  `message.reasoning`; SGLang returns `message.reasoning_content`. Client code that reads
  one silently drops it on the other. Read both.
- **llama.cpp publishes no Linux CUDA binary.** The release assets carry
  `llama-*-bin-win-cuda-*.zip` and nothing equivalent for Linux, so a GPU build there
  means compiling from source with `-DGGML_CUDA=ON`.
