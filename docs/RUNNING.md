# Running it

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
