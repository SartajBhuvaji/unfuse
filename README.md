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

Gemma 4 is in the table as a warning, not a target. It has no vision *tower* at all — Google
replaced the encoder with a single matmul, so there is almost nothing to remove and
quantization is a far bigger lever. Worth checking before you assume a model is worth
splitting. Details in `docs/FINDINGS.md`.

Adding a model means answering three questions: where does the encoder live in the tensor
names, does a standalone class exist for the half you want, and what does the config need
to become. The two notebooks in `glm4v/` walk through both directions end to end.

## Published

- [GLM-4.6-Flash-text](https://huggingface.co/sartajbhuvaji/GLM-4.6-Flash-text) — bf16 text-only, 9.4B params
- [GLM-4.6-Flash-text-GGUF](https://huggingface.co/sartajbhuvaji/GLM-4.6-Flash-text-GGUF) — F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M
- [GLM-4.6V-Flash-mmproj-GGUF](https://huggingface.co/sartajbhuvaji/GLM-4.6V-Flash-mmproj-GGUF) — the vision half, F32/BF16/F16/Q8_0

The last two recombine into the original multimodal model under llama.cpp.

## Layout

```
src/unfuse/   the model-agnostic core: inspect a checkpoint, verify two match
notebooks/    one directory per model, each with its notebooks and diagram
tools/        Lambda provisioning, GGUF build, HF upload, notebook guard
tests/        unit tests for the core (no GPU, no network)
docs/         findings, the guide for adding a model, published model cards
```

```bash
pip install -e ".[notebook,gguf,dev]"
unfuse inspect ~/models/some-model --prefix model.visual.
```

Only inspection and verification are library code. The surgery lives in the
notebooks, because every model needs different treatment and hiding that behind one
function would produce a registry of special cases pretending to be an abstraction.
See `docs/ADDING-A-MODEL.md`.

```
glm4v/        the two notebooks, with outputs, plus the architecture diagram
publishing/   HF upload script and the three model cards
infra/        Lambda provisioning, instance setup, GGUF build
docs/         findings and measured numbers
```

Both notebooks are committed **with their execution outputs** — those outputs are the
evidence for the bit-exact claim below and cannot be reproduced without a GPU and a
20 GB download. Don't strip them.

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
./infra/01_launch_lambda.sh               # verifies the SSH key actually registered
./infra/02_setup_instance.sh              # on the instance
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/glm-4.6v-flash/remove_vision.ipynb
./infra/03_build_gguf.sh                  # llama.cpp + quant ladder + smoke test
HF_TOKEN=... python publishing/upload.py --repo user/Model --model-dir ... --gguf-dir ...
```

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
- **Only Gemma 4 *12B* is encoder-free.** E2B/E4B/26B-A4B/31B are `model_type: gemma4`
  with conventional SigLIP towers. Check `model_type` before applying any recipe.

## Next

- End-to-end test the mmproj against the text GGUF. Pairing is verified from GGUF
  metadata only; the build machine had 7 GiB of RAM and could not hold both.
- Run the Gemma 4 ablation — the unpublished experiment described in
  `docs/FINDINGS.md`. Those scripts are written but never executed, and live outside
  this repo until they have results worth showing.
