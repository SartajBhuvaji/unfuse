# Runbook: Qwen3.8-27B GGUF, and the benchmark sweeps

Work for a machine with a GPU. Everything here was prepared without one, so the
conversion is only dry-run checked and nothing below has been executed.

Branch: `qwen-gguf-and-sweeps`. Commit results there. Do not touch `main`.

## What is already established (don't redo it)

- `sartajbhuvaji/unfuse-Qwen3.8-27B-text` is the text-only checkpoint, bf16,
  54.64 GB, architecture `Qwen3_5ForCausalLM`, 866 tensors including a 15-tensor MTP
  head (`mtp.*`).
- llama.cpp master (`1b6935b`, 2026-11-02) **can convert it**. This used to be
  blocked upstream and is not any more. `convert_hf_to_gguf.py --remote --dry-run
  --outtype bf16` plans 866 tensors / 54.6 GB, and the MTP head lands as block 64
  (`blk.64.nextn.*`). The converter now lives in `conversion/qwen.py`.
- The same converter makes a 334-tensor, 927.6 MB f16 mmproj from the original
  `Qwen/Qwen3.8-27B` with `--mmproj`.
- One warning on conversion is unexplained: `Unknown RoPE type: default`. It is
  probably harmless (the config already carries `mrope_section: [11, 11, 10]`) but
  that is a guess. Task 1 settles it.

## Rules

- **Hardware.** Bf16 work needs one 80 GB card (H100 80 GB). A 40 GB card cannot
  load the model. GGUF conversion and quantization are CPU work. Budget roughly
  200 GB of disk: 55 GB checkpoint, 55 GB bf16 GGUF, ~40 GB of quants, 56 GB parent.
- **Tokens** come from the environment (`HF_TOKEN`). Never write one to a file on
  the machine and never commit one.
- **Hub.** Create any new repo explicitly **private**. Do not make anything public
  and do not edit the existing public repos. Publishing is a separate decision.
- **Commits.** Author as `Sartaj Bhuvaji <sbhuvaji@seattleu.edu>`. Plain, human
  commit messages. No AI attribution or co-author trailers anywhere, in commits,
  files or logs you add. Write prose in the results files the way a person would
  write it: direct, specific, no filler.
- **Do not commit weights or GGUFs.** Commit JSON, logs and the results write-up.
- **Report failures as failures.** A step that did not run is "not run", not
  "skipped" or "passed".

## Task 1: Qwen3.8-27B GGUF (about 1-2 h)

1. Record the machine: `bash tools/gpu_env.sh --json >
   notebooks/qwen3.8-27b/gguf/env.json`.
2. Download the checkpoint, then build the ladder from a **bf16** base:

   ```bash
   hf download sartajbhuvaji/unfuse-Qwen3.8-27B-text --local-dir ~/models/Qwen3.8-27B-text
   SRC=~/models/Qwen3.8-27B-text NAME=Qwen3.8-27B-text BASE=bf16 \
     tools/build_gguf.sh Q8_0 Q6_K Q5_K_M Q4_K_M
   ```

   `BASE=bf16` is new in this branch. Don't use f16 as the base for this model.
   Note the llama.cpp commit you actually built and put it in the results.
3. **Resolve the RoPE warning.** Grab the exact warning text from the conversion
   and read the GGUF's `qwen35.rope.*` metadata (`gguf-py` has `gguf_dump.py`).
   Then compare against transformers on the same prompts: greedy-decode 64 tokens
   on at least six prompts, and compare logits of the first generated token. Do
   it at a long prompt (>=1,000 tokens) too, since RoPE errors grow with position.
   bf16 GGUF against the bf16 safetensors should agree closely; report the real
   max logit difference and argmax agreement, not a verdict.
4. **Smoke test** with known answers, not exit codes. `build_gguf.sh` runs a quick
   one; for the real check use `tools/smoke_inference.sh` with the `llamacpp`
   backend and `GGUF=` set. A quant that exits 0 and emits nonsense is a FAIL.
5. **Perplexity.** The GLM GGUFs never had one, so this is new. Run
   `llama-perplexity` on the same text (wikitext-2 test split, same context
   length) for BF16, Q8_0, Q6_K, Q5_K_M and Q4_K_M. Put them in one table.
6. **mmproj.** From the *original* repo, not the text-only one:

   ```bash
   hf download Qwen/Qwen3.8-27B --local-dir ~/models/Qwen3.8-27B
   python ~/llama.cpp/convert_hf_to_gguf.py ~/models/Qwen3.8-27B --mmproj \
     --outtype f16 --outfile ~/gguf/mmproj-Qwen3.8-27B-F16.gguf
   ```

   Then test the recombination: run `llama-mtmd-cli` with the Q4_K_M text GGUF plus
   this mmproj on one real image with a question whose answer you can verify.
   Also make a Q8_0 mmproj if the converter supports it. If it fails, say exactly
   how.
7. **Upload privately** to `sartajbhuvaji/unfuse-Qwen3.8-27B-text-GGUF` (root-level
   GGUFs, no safetensors, see `tools/upload_hf.py` for why) and
   `sartajbhuvaji/unfuse-Qwen3.8-27B-mmproj-GGUF`. Create both private. Do not
   write model cards, they will be written from your results.

Optional, only if time remains: export the MTP head alone with `--mtp` and confirm
the GGUF loads.

## Task 2: benchmark sweeps (about 1-1.5 h)

Same machine, run after Task 1. Use `tools/bench_pair.sh`: one shared client, two
passes with the first discarded, and the scheduler's own concurrency limit recorded
per model. Read its header before running; those three rules exist because earlier
numbers were wrong without them.

```bash
tools/bench_pair.sh Qwen/Qwen3.8-27B sartajbhuvaji/unfuse-Qwen3.8-27B-text vllm sglang
```

Everything published so far is 512 in / 256 out at concurrency 1 and 16. Do these
two sweeps, keeping 256 out:

1. **Long context.** `IN_LEN=4096`, `8192`, `32768`. `CTX` must exceed input plus
   output; use `CTX=40960` for the 32k run. Lower `NPROMPTS` (for example 32) at the
   long lengths so the run finishes. Hypothesis to test, not assume: the split's
   extra KV headroom matters more as context grows, because cache is the binding
   constraint at long context and weights are not. For every run, record the KV
   capacity in tokens for each model from `serve-*.json`, since that is the number
   the hypothesis is about.
2. **Concurrency.** `CONC="32 64 128"` at 512/256. Find where the parent stops
   scaling and the split keeps going.

Traps that already cost time on this model:

- vLLM refused `--max-num-seqs 1024` with exactly 362 available Mamba cache blocks.
  The published card says to pass `--max-num-seqs 256`. If a run is refused, record
  the refusal and the number it reports rather than silently lowering it.
- SGLang gave the parent `max_running_requests=5` and the split 16, which produced a
  fake 2.79x gap. Set `MAX_SEQS` / `MAX_RUNNING` equal across both models for any
  row you want to compare, and also keep the unpinned first pass.
- On vLLM the split was 1.15x *slower* at c=16 (554.7 vs 635.8 tok/s) despite more KV
  cache (301,465 vs 277,435 tokens). This was never explained. Watch whether the
  long-context rows show the same direction.
- A cold pass reads ~21% low. Never compare a warm row to a cold one.

Tensor parallel needs 2 GPUs and is out of scope here.

## Task 3: only if budget remains

- Work out what the 362 Mamba cache blocks is a function of (`gpu_memory_utilization`,
  layer count, hidden size?) so the `--max-num-seqs` advice can become a rule.
- Chase the 1.15x slowdown. vLLM logged `Padding mamba page size by 0.13%` for the
  split; suspect different page-size padding between the two configs.

Time-box this to an hour. A clear "could not explain it, here is what I ruled out"
is a good result.

## Deliverables

Commit to `qwen-gguf-and-sweeps` and push:

```
notebooks/qwen3.8-27b/gguf/       env.json, conversion log, rope check, smoke
                                  results, perplexity table, mmproj test
notebooks/qwen3.8-27b/bench/sweeps/   every bench-*.json, serve-*.json, startup log
notebooks/qwen3.8-27b/RESULTS.md  what ran, what it showed, what failed
```

`RESULTS.md` should lead with the findings, then give the numbers, then list
anything that did not run or did not pass and why. Include the llama.cpp commit,
the vLLM and SGLang versions, and the exact commands. Don't edit the README,
the model cards, or `docs/FINDINGS.md`. Those get updated from your results.

When finished, say which Hub repos you created (all private), and what you left
running.
