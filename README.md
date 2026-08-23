# unfuse

Split a multimodal model into its parts, losslessly.

A model that reads images is two things glued together: an encoder that turns pixels into
vectors, and an ordinary language model that consumes them. `unfuse` takes them apart —
keep the language model, keep the encoder, or keep both separately. It's a tensor filter
and a rename. No retraining, no distillation, no quality loss.

The split itself is easy. The work is **proving the result is unchanged**, and every
extraction here is verified against the original before it ships.

## Models

| Model | Encoder | Removable | Text-only | Encoder-only | Notes |
|---|---|---|---|---|---|
| [GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) | 24-layer ViT + merger | 892,498,432 (8.671%) | done | done | bit-exact, published |
| [Gemma 4 12B](https://huggingface.co/google/gemma-4-12B) | none — 52.4M input embedder | 52,379,904 (0.438%) | scripted | n/a | encoder-free; not worth splitting for size |
| [Gemma 4 31B NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) | 27-layer SigLIP | ~576M (1.9%) | scripted | — | quantized; the split is the small half of the work |

Gemma 4 is in the table as a warning, not a target. It has no vision *tower* at all — Google
replaced the encoder with a single matmul, so there is almost nothing to remove and
quantization is a far bigger lever. Worth checking before you assume a model is worth
splitting. Details in `docs/FINDINGS.md`.

## Splitting is not always the lever

`nvidia/Gemma-4-31B-IT-NVFP4` is in the table because it makes the point. Its vision tower
is real and removable, but at ~1.15 GB it is the *smallest* of three obvious wins. Reading
the checkpoint's own quantization metadata shows why:

```bash
unfuse inspect ~/models/gemma-4-31b-nvfp4 --quant
```

NVFP4 is a 4-bit weight and activation format, but 63 modules are excluded — and 60 of
them are the attention blocks, listed one layer at a time. Only the MLPs were quantized,
which leaves **unquantized attention as the largest single block in a 32.67 GB file**.
Removing the encoder saves 1.15 GB; taking attention to FP8 saves 7.9 GB.

So that model gets treated as a quantization problem with a split attached, rather than
the reverse. `notebooks/gemma-4-31b-nvfp4/` does all three — vision removal, FP8
attention, and pairing the unpaired MTP drafter — and measures each separately.

Adding a model means answering three questions: where does the encoder live in the tensor
names, does a standalone class exist for the half you want, and what does the config need
to become. The two notebooks in `notebooks/glm-4.6v-flash/` walk through both directions
end to end.

## Published

- [GLM-4.6-Flash-text](https://huggingface.co/sartajbhuvaji/GLM-4.6-Flash-text) — bf16 text-only, 9.4B params
- [GLM-4.6-Flash-text-GGUF](https://huggingface.co/sartajbhuvaji/GLM-4.6-Flash-text-GGUF) — F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M
- [GLM-4.6V-Flash-mmproj-GGUF](https://huggingface.co/sartajbhuvaji/GLM-4.6V-Flash-mmproj-GGUF) — the vision half, F32/BF16/F16/Q8_0

The last two recombine into the original multimodal model under llama.cpp.

## Layout

```
src/unfuse/   the model-agnostic core: inspect a checkpoint, verify two match, score a
              quantized one against the build already shipping
notebooks/    one directory per model, each with its notebooks and diagram
tools/        Lambda provisioning, GGUF build, HF upload, notebook guard, FP8
              requantization, speculative-decoding benchmark, cross-engine
              serving smoke test + benchmark
tests/        unit tests for the core (no GPU, no network)
docs/         findings, a study guide, the guide for adding a model, model cards
```

```bash
pip install -e ".[notebook,gguf,dev]"
unfuse inspect ~/models/some-model --prefix model.visual.
```

Only inspection and verification are library code. The surgery lives in the
notebooks, because every model needs different treatment and hiding that behind one
function would produce a registry of special cases pretending to be an abstraction.
See `docs/ADDING-A-MODEL.md`.

New to any of this? `docs/STUDY_GUIDE.md` lists every topic the repo assumes — number
formats, NVFP4's two-level scaling, compressed-tensors, KV cache arithmetic, speculative
decoding, and the verification methodology — with what each is, why it matters here, and
where it shows up. `docs/FINDINGS.md` is the narrative of what was tried, what failed, and
what the failures taught.

Both notebooks are committed **with their execution outputs** — those outputs are the
evidence for the bit-exact claim below and cannot be reproduced without a GPU and a
20 GB download. Don't strip them.

## Two bars, deliberately not shared

Extraction does no arithmetic on the weights, so it either reproduces the original bits or
it is broken. `unfuse verify` holds that line and a test pins it there.

Quantization is the opposite case: it rounds every weight it touches, so a non-zero
difference is not a bug, it is the operation. `unfuse agreement` scores that — but not
against a threshold, because no one can justify the number and any threshold that exists
will eventually be tuned until the checkpoint passes.

The bar is comparative instead. Whoever shipped the quantized build published benchmark
scores with it, so they have already decided how much damage is acceptable. Measure their
checkpoint against the original, measure yours the same way, require yours to be no worse:

```bash
unfuse agreement google/gemma-4-31B-it ~/models/mine --baseline nvidia/Gemma-4-31B-IT-NVFP4
```

The two modules share no code, and merging them would break both: the exact bar would
reject every correct quantization, and the tolerant bar would let a broken extraction ship.

## The headline result

The GLM-4.6V-Flash extraction is **bit-exact on text**. Logits compared against the
original model:

```
max|d| = 0.000e+00  across six prompts, argmax_match=True on all
max|d| = 0.000e+00  at 1,207 tokens
```

That mattered because the text config uses multimodal RoPE (`mrope_section: [8,12,12]`).
For text-only input all three axes carry the same position, so mRoPE *should* collapse
to standard RoPE — but that was an argument from the config, not a measurement. Now it
is measured, at depth.

## Running it

```bash
pip install -e ".[notebook,gguf,dev]"
cp .env.example .env                      # LAMBDA_API_KEY, HUGGINGFACE_TOKEN
```

**Keeping the vision tower** runs on a laptop GPU — 4.66 GB download (the vision
tensors sit in one shard), 1.78 GB result:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/glm-4.6v-flash/keep_vision.ipynb
```

**Removing vision** needs ~21 GB of VRAM for the bit-exact check, so it wants a rented
GPU:

```bash
./tools/launch_lambda.sh                  # verifies the SSH key actually registered
./tools/setup_instance.sh                 # on the instance
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/glm-4.6v-flash/remove_vision.ipynb
./tools/build_gguf.sh                     # llama.cpp + quant ladder + smoke test
HF_TOKEN=... python tools/upload_hf.py --repo user/Model --model-dir ... --gguf-dir ...
```

**Before publishing**, prove the checkpoint actually serves. `smoke_inference.sh` runs
vLLM, SGLang, transformers and llama.cpp against it, checks each one returns
known-correct answers rather than merely exiting 0, and benchmarks the two servers with
a single shared client so their numbers are comparable:

```bash
GGUF=~/gguf/Model-Q4_K_M.gguf ./tools/smoke_inference.sh user/Model   # on the instance
```

It prints a pass/fail table and a metrics table ready to paste into a model card.

## Hard-won gotchas

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

## Next

- End-to-end test the mmproj against the text GGUF. Pairing is verified from GGUF
  metadata only; the build machine had 7 GiB of RAM and could not hold both.
- Run the Gemma 4 ablation — the unpublished experiment described in
  `docs/FINDINGS.md`. Those scripts are written but never executed, and live outside
  this repo until they have results worth showing.
- Re-run `notebooks/gemma-4-31b-nvfp4/` if its numbers are revisited. It has been executed
  and carries its outputs; an 80 GB card is enough, since the BF16 expansion is 61 GB and
  NVFP4 itself runs on Ampere through Marlin.
- Decide what to do about FP8 attention and vLLM. The W8A16 choice is the right one on
  accuracy grounds and has no vLLM serving path, because the ModelOpt FP8 path is W8A8.
  Either calibrate a W8A8 variant, or wait for FP8 weight-only support upstream. Until
  then the published checkpoint runs only via `tools/dequantize.py`.
