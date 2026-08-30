# Findings

Measured numbers and the literature behind them. Everything here was verified against
live checkpoints, not read off model cards.

## GLM-4.6V-Flash — done, published

`zai-org/GLM-4.6V-Flash`, `model_type: glm4v`, `Glm4vForConditionalGeneration`.

**Parameter count is 10,292,777,472, not the ~5.1B you get from dividing the index's
`total_size` by 2.** That field reports parameter count for this repo, not bytes. Actual
shard files total 20,585,644,048 bytes, which matches 10.29B at bf16 and matches the sum
of every tensor shape.

| Group | Params | Share |
|---|---|---|
| `visual.blocks.0–23` (ViT, 1536d, 12 heads) | 679,550,976 | 6.602% |
| `visual.merger` (proj + gate/up/down + norm) | 185,081,856 | 1.798% |
| `visual.downsample` (Conv2d, spatial_merge 2) | 25,169,920 | 0.245% |
| `visual.patch_embed.proj` (Conv3d 14×14×2) | 1,807,872 | 0.018% |
| `visual.embeddings.position_embedding` (576 × 1536) | 884,736 | 0.009% |
| `visual.{post_conv,post}_layernorm` | 3,072 | 0.000% |
| **DELETE — `model.visual.*`, 181 tensors** | **892,498,432** | **8.671%** |
| `language_model.layers.0–39` (203,969,024 each) | 8,158,760,960 | 79.267% |
| `language_model.embed_tokens` (151552 × 4096) | 620,756,992 | 6.031% |
| `lm_head` (**untied**) | 620,756,992 | 6.031% |
| `language_model.norm` | 4,096 | 0.000% |
| **KEEP — 523 tensors** | **9,400,279,040** | **91.329%** |

### The surgery

1. Drop every key under `model.visual.` (181 tensors)
2. Rename `model.language_model.` → `model.`
3. Keep `lm_head.weight` — `tie_word_embeddings: false`
4. Rebuild config via `Glm4Config`, dropping `mrope_section` and `vision_config`
5. **Set `architectures = ["Glm4ForCausalLM"]`** — `Glm4Config` won't

Vocabulary needs no work: 151552 with image/video ids (151363/151364) inside it.

### Verification

The real risk was RoPE. `mrope_section: [8, 12, 12]` sums to 32 = `head_dim 128 ×
partial_rotary_factor 0.5 ÷ 2`, so mRoPE spans the full rotary budget split across
temporal/height/width axes. For text-only input all three axes carry the same position,
so it should reduce to standard RoPE.

**It does, exactly.** `max|d| = 0.000e+00` on six prompts with `argmax_match=True`, and
again at 1,207 tokens. Zero divergence, not near-zero.

### GGUF

Built with llama.cpp at master (`GLM4` architecture supported). All five load and
generate coherently. **No perplexity comparison against bf16 was run.**

| Quant | Size (GB) |
|---|---|
| F16 | 18.81 |
| Q8_0 | 10.00 |
| Q6_K | 8.27 |
| Q5_K_M | 7.05 |
| Q4_K_M | 6.17 |

### Most of the memory win is the profiler, not the weights

Benchmarked the GLM-4.6V-Flash parent against its text-only split on one A100-SXM4-40GB,
which the card had never done: it only ever measured the derivative. Under vLLM 0.27.1 at
`--max-model-len 32768` the split gets 479,280 KV tokens against the parent's 310,512,
+54%. The vision tower is only 1.79 GB of 20.59, so that number cannot be weights.

vLLM's own profile says where it goes:

| | parent | parent, `video: 0` | split |
|---|---|---|---|
| weights + non-torch | 19.67 GiB | 19.67 GiB | 17.85 GiB |
| peak activation | 4.82 GiB | 0.97 GiB | 0.21 GiB |
| KV cache | 11.85 GiB | 15.70 GiB | 18.28 GiB |
| KV tokens | 310,512 | 411,552 | 479,280 |
| engine init | 116 s | 23 s | 11 s |

The parent's 4.82 GiB peak activation is vLLM profiling the multimodal path with one
maximum-size *video* — not an image — and sizing a 66,242-token encoder cache off it. That
memory is reserved for the life of the server whether or not anything multimodal is ever
sent, and it costs 105 s of startup as well.

Which means 60% of the headline gain (101,040 of 168,768 tokens) is available on the parent
with `--limit-mm-per-prompt '{"video": 0}'` and no surgery at all. The split is worth +16%
over a parent configured that way, plus 1.73 GiB of weights that no flag reaches. A card
claiming the full +54% against a default-configured parent is not wrong, but it is
comparing against a misconfiguration, so both numbers belong in it.

Do not set `'{"video": 0, "image": 0}'`. vLLM 0.27.1 then skips loading the tower and dies
in profiling with `AttributeError: 'NoneType' object has no attribute 'size'`.

Both checkpoints store KV at an identical 40,960 bytes per token (40 layers, 2 KV heads,
head_dim 128, bf16), which is the arithmetic check on every scraped figure above: tokens
must equal GiB × 26,214.4. It also means none of the gain is a cheaper cache.

### Throughput did not move, and that is the result

Eight rows, two checkpoints, two engines, concurrency 1 and 16. Every c=1 row lands between
61.8 and 63.7 tok/s and every c=16 row between 962.7 and 994.9 — a 3% spread. Both models
read the same 18.80 GB per token, because decode never touches the resident vision tower,
so both sit against the same 82.7 tok/s ceiling on the A100's 1.555 TB/s and both measure
~77% of it.

The trap in the bandwidth arithmetic is charging the parent for its whole 20.59 GB
checkpoint. That lowers its ceiling by 9% and manufactures an advantage for the split that
the measurements do not show. Pass the bytes decode actually reads.

Two comparability checks that had to be made before any of this could be reported. vLLM
gave both models `max_num_seqs=256`, clamped from `max_num_batched_tokens=2048`, but prints
it only at DEBUG, so it took a separate start with identical flags to establish; two
unlogged values are unverified, not equal. SGLang derived 3,085 for the parent and 4,096
for the split, unequal but both ~200× the tested concurrency, so nothing was throttled —
unlike the Qwen run, where the parent got 5 against an offered 16 and the resulting 2.79×
nearly shipped.

And warmup earned its keep on exactly one row of eight: SGLang on the split read 50.7 tok/s
cold against 63.3 warm, +24.9%, with TPOT identical at 15.58 ms. Compared against SGLang's
warmed parent at 63.7, the cold figure would have published the split as 20% slower than
the model it was derived from.

## Keeping only the vision tower — done, ran locally

The inverse operation. `Glm4vVisionModel` already exists in transformers and its
submodules (`embeddings`, `patch_embed`, `blocks`, `merger`, `downsample`,
`post_conv_layernorm`, `post_layernorm`) map exactly onto what sits under
`model.visual.`, so stripping that prefix is the whole surgery. No architecture
retarget, no RoPE question — much simpler than the text side.

**Two facts make this runnable on a laptop:**

1. All 181 vision tensors live in **shard 4 only** (verified via the index's
   `weight_map`), so the download is 4.66 GB rather than 20.59 GB.
2. The result is 1.78 GB.

Executed on an RTX 2060 (6.44 GB, sm_75 — no bf16, so float32), 7 GiB RAM:

```
181 tensors, 892,498,432 params, 1.785 GB
byte-compared 181 tensors against the source shard -> all bit-identical
max|d| = 0.000e+00   (64, 4096)
448x448 image -> 1024 patches -> 256 soft tokens x 4096d
```

Verification here is deliberately weaker than the removal notebook's. That one
compared against the full multimodal model; this compares a directly-loaded reference
against the `from_pretrained` checkpoint, because 20.6 GB will not fit locally. It
guards the plumbing — config reconstruction, key naming, round-trip — not the
arithmetic, which is identical by construction given the byte comparison.

**Note on WSL:** the run reported peak VRAM of 7.32 GB on a 6.44 GB card. WSL2's WDDM
driver lets CUDA spill into shared system memory, confirmed by a deliberate 7 GB
allocation succeeding. The same code would OOM on native Linux with that card.

The output is a **feature extractor, not a model** — no language head, no classifier.
It emits embeddings aligned to GLM-4's input space. This asymmetry matters: removing
vision leaves a complete working model, because a decoder with an `lm_head` stands
alone. Removing text leaves a component.

## mmproj GGUF — vision towers do not take K-quants

`convert_hf_to_gguf.py --mmproj` accepts only `f32`, `bf16`, `f16`, `q8_0`.
`llama-quantize` and the K-quant ladder apply to the **text** GGUF. Confirmed against
what publishers actually ship:

| Publisher | mmproj formats |
|---|---|
| unsloth (Gemma 3, Gemma 4) | BF16, F16, F32 |
| ggml-org (Qwen2.5-VL) | Q8_0, F16 |

**Q8_0 at 1.03 GB is the floor** for this tower — there is no Q4_K_M projector.

`--mmproj` needs the *full* multimodal architecture and cannot read a stripped
`Glm4vVisionModel` checkpoint. The workaround: stage a directory with the original
`config.json` plus an index whose `weight_map` references only shard 4. No extra
download.

llama.cpp has **no separate `GLM4V` model architecture** — only `CHATGLM`, `GLM4`,
`GLM4_MOE`, with `glm4v` existing as a `VisionProjectorType`. So GLM-4V's text tower
and our text-only extraction both convert to arch `glm4`, and the mmproj should pair
with either. Verified from GGUF metadata; **not** verified by running end-to-end
multimodal inference, which needs more RAM than the build machine had.


## Gemma 4 12B — scripts written, unrun

`model_type: gemma4_unified` — the **encoder-free** variant. There is no vision tower.
Google replaced the 27-layer SigLIP with a single matmul: 48×48 raw pixel patches →
`LayerNorm → Linear(6912→3840) → LayerNorm` plus factorized X/Y coordinate embeddings.
Audio is 40ms raw waveform frames → `Linear(640→3840)`. No mel spectrogram, no conformer.

| Tensor | Params |
|---|---|
| `vision_embedder.patch_dense.weight` `[3840, 6912]` | 26,542,080 |
| `vision_embedder.pos_embedding` `[1120, 2, 3840]` | 8,601,600 |
| `vision_embedder.{patch_ln1,patch_ln2,pos_norm}` | 29,952 |
| `embed_vision.embedding_projection` `[3840, 3840]` | 14,745,600 |
| `embed_audio.embedding_projection` `[3840, 640]` | 2,457,600 |
| **Total** | **52,379,904** |

**0.438% of 11,959,730,224 params — 105 MB of 23.9 GB.** Cross-checked against
llama.cpp's `mmproj-F32` for this model: 209.5 MB ÷ 4 bytes = 52.4M.

Removing it saves essentially nothing. Quantizing to Q4_K_M takes 23.9 GB → 7.1 GB,
roughly 160× the saving. The real reason to extract is dodging
[transformers#45200](https://github.com/huggingface/transformers/issues/45200), where
fine-tuning on pure text raises `ValueError: mm_token_type_ids is required`.

**Only the 12B is encoder-free.** E2B/E4B/26B-A4B/31B are `model_type: gemma4` with
conventional SigLIP towers (~576M on the 31B) and USM conformer audio encoders.

## The open research question

An experiment that has not been published. The script exists but has never been run,
so it is kept outside this repo until it has results.

[MINER](https://arxiv.org/abs/2410.04819) found ~2% of FFN neurons become modality-specific
in encoder-**based** MLLMs, concentrated in shallow layers — deactivating that 2% collapses
multimodal performance (Qwen2-VL 0.56→0.24). It never ran the converse: ablate the
vision-specific neurons, then measure **text-only** performance. And nobody has run it on
an encoder-free model at all — every published modality-pruning result studies LLaVA,
InternVL, or Bunny, all of which have separable vision towers.

The script scores FFN neurons by activation ratio on image vs text tokens, zeroes the top
slice, and watches both NLLs. Vision NLL rising while text NLL holds would mean
vision-serving capacity emerged in the shared FFN despite the architecture — making the
0.438% ceiling wrong. Roughly one GPU-day.

## Qwen3.8-27B — split, verified, published

`Qwen/Qwen3.8-27B`, `model_type: qwen3_5`, `Qwen3_5ForConditionalGeneration`.
2.95M downloads/month and 12,691 likes at the time of writing — roughly 15x the demand
of GLM-4.6V-Flash. That is the reason to do it. It is not because the split saves space.

**The whole checkpoint, measured by reading all 18 shard headers over HTTP range
requests — about 200 KB fetched, no weights downloaded.**

| Group | Params | Share | GB |
|---|---|---|---|
| `model.language_model.*` (64 layers) | 25,624,600,064 | 92.236% | 51.25 |
| `lm_head` (**untied**) | 1,271,398,400 | 4.576% | 2.54 |
| `mtp.*` (fused MTP head, 15 tensors) | ~424,694,784 | ~1.529% | 0.85 |
| **DELETE — `model.visual.*`, 333 tensors** | **460,730,096** | **1.658%** | **0.92** |
| **Total** | **27,781,427,952** | 100% | **55.56** |

**The tower is 1.658%.** GLM-4.6V-Flash was 8.671%. Removing it saves 0.92 GB out of
55.56. This is the Gemma 4 lesson again, and the reason to run `inspect` before assuming
a model is worth splitting: the win here is packaging — loads as a plain causal LM, no
vision dependencies, no image preprocessing — not size.

### `deepstack_visual_indexes` is a false alarm

`vision_config.deepstack_visual_indexes` reads like it should complicate the split: it
names the tower layers whose features are injected into the language model, which sounds
like vision reaching into the LLM at several depths. It does not.

**Zero tensors carry a deepstack name.** The only trees in the checkpoint are
`model.visual.*`, `model.language_model.*`, `lm_head`, and `mtp.*` — 1199 tensors, all
four accounted for, nothing unclassified. The indices are resolved inside the visual
tree. The split is clean.

Worth recording because the config *looks* like a problem and the answer cost one probe.

### It is a hybrid, not a transformer

`layer_types` is 48 `linear_attention` + 16 `full_attention`, `full_attention_interval:
4` — every fourth layer is full attention, the rest are linear. With
`linear_conv_kernel_dim: 4`, `linear_num_key_heads: 16`, `linear_num_value_heads: 48`,
and `mamba_ssm_dtype: float32`.

Do not carry over vanilla-transformer assumptions: KV cache sizing, GGUF conversion, and
engine support all behave differently for the hybrid stack.

### The MTP head has no home in transformers

`mtp.*` is a fused multi-token-prediction speculative head — 15 tensors, ~424.7M params,
`mtp_num_hidden_layers: 1`. There is **no `Qwen3_5` MTP class in transformers 5.16**
(`Qwen3_5ForCausalLM`, `Qwen3_5TextConfig`, `Qwen3_5VisionModel` all exist; no MTP), so
`Qwen3_5ForCausalLM` reports these as unexpected keys.

Kept by default anyway. Dropping them silently removes a capability, and the MTP-bearing
community GGUFs of this model are among its most downloaded builds. `--drop-mtp` is
opt-in.

### The surgery

1. Drop every key under `model.visual.` (333 tensors)
2. Rename `model.language_model.` → `model.`
3. Keep `lm_head.weight` — `tie_word_embeddings: false`
4. Build the config from `text_config` via `Qwen3_5TextConfig`, dropping `image_token_id`,
   `video_token_id`, `vision_start_token_id`, `vision_end_token_id`, `language_model_only`
5. **Set `architectures = ["Qwen3_5ForCausalLM"]`** — `Qwen3_5TextConfig` will not, the
   same trap as `Glm4Config` in the GLM split

All six non-default keys (`attn_output_gate`, `full_attention_interval`,
`mamba_ssm_dtype`, `mtp_num_hidden_layers`, `mtp_use_dedicated_embeddings`,
`output_gate_type`) survive the round-trip through `Qwen3_5TextConfig` — verified.

### Engine support is arriving

sglang commit `01db8a7` (PR #35297, 2026-08-25) registers **Qwen3_5 text-only
architectures** in the mamba radix cache whitelists, adding "Text-only entries of the
same hybrid stack (`models/qwen3_5_text`...)". An engine is building the landing pad for
this checkpoint shape.

### Verified three times, on two machines

The split was run locally and twice on a rented H100, each time from a fresh download of
the parent. All three produced **27,320,697,856 params / 54.64 GB**, and all three passed
the byte-level check on all 866 kept tensors. Reproduction is not assumed here either.

Verification is **tensor hashes, not logits**. `unfuse.verify.compare_logits` is right when
the surgery could change arithmetic; this one only drops and renames. For a copy-and-rename
byte equality over every kept tensor proves *every* input produces identical output, where
matching logits proves it for the prompts you tried. It is also the only option on a box
that cannot hold a 27B model in RAM.

Two implementation notes that cost real time:

- **numpy has no bfloat16.** `tensor.numpy()` raises `unsupported ScalarType BFloat16` on
  exactly the dtype the checkpoint is stored in, so the verifier could not check the one
  thing it existed to check. Hash `t.contiguous().view(torch.uint8).numpy().tobytes()`
  instead — the claim is byte equality, not numeric equality, so this also works for fp8.
- **Do not mirror the input sharding.** Read-shard-N-write-shard-N holds a whole ~3 GB
  shard in RAM and dies partway on a small machine, leaving a checkpoint that still looks
  plausible on disk. Buffer by bytes and flush on a budget; output sharding then does not
  match input sharding, which is fine because the index is the map.

### The MTP head is real, kept, and currently unusable

Loading the split reports **26,895,998,464** parameters against **27,320,697,856** in the
checkpoint. The difference is **424,699,392** — exactly the MTP head. transformers loads
the model and silently ignores those tensors, because there is no `Qwen3_5` MTP class.

vLLM 0.27.1 *does* ship `model_executor/models/qwen3_5_mtp.py` and registers
`"Qwen3_5MTP"`, so this looked recoverable. It is not, and both paths were tested:

| config `model_type` | result |
|---|---|
| `qwen3_5_text` (ours) | `NotImplementedError: Unsupported speculative method: 'mtp'` |
| `qwen3_5` (forced) | resolves `Qwen3_5MTP`, then `AssertionError: Attempted to load weight (torch.Size([5120])) into parameter (torch.Size([4096]))` |

The cause is in `vllm/config/speculative.py:516` — the MTP auto-detect fires only for
`model_type in ("qwen3_5", "qwen3_5_moe")`, and that path then reads the *multimodal*
config layout, expecting a nested `text_config`. A flattened text-only config makes it
fall back to a default `hidden_size` of 4096 against our real 5120.

So the head costs 0.85 GB and does nothing today. Kept anyway: it breaks nothing, and
the alternative is deleting a capability that cannot be recovered without re-downloading
55 GB. `--drop-mtp` exists for anyone who wants the clean load.

### Decode is bandwidth-bound, and the numbers say so

Worth computing before attributing a single-stream result to anything else. An H100 SXM5
has 3.35 TB/s of HBM3. A dense model at batch 1 reads every weight once per token, so:

| | Weights | Floor per token | Ceiling | Best measured | Of ceiling |
|---|---|---|---|---|---|
| Qwen3.8-27B | 55.56 GB | 16.59 ms | 60.3 tok/s | 48.8 tok/s | ~81% |
| text-only split | 54.64 GB | 16.31 ms | 61.3 tok/s | 47.2 tok/s | ~77% |

This is the actual explanation for the flattest result in the benchmark table: all four
c=1 rows land between 46.6 and 48.8 tok/s regardless of engine or checkpoint, because
every one of them is moving the same ~55 GB over the same bus. Engine choice cannot matter
until concurrency amortises those reads.

It also bounds the value of further tuning — there is ~20–25% of headroom at c=1, and the
only levers that reach past it are more bandwidth or fewer bytes (quantization), not a
different server or a better flag.

Approximate by construction: weight traffic only, no KV-cache reads, and the hybrid stack's
48 linear-attention layers do not access memory like the 16 full-attention ones. Quote it
as a band, not a figure.

### Serving traps

**vLLM's default `max_num_seqs` will not start this model.**

```
ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (362).
Each decode sequence requires one Mamba cache block, so CUDA graph capture cannot proceed.
```

This is the hybrid stack asserting itself: the 48 linear-attention layers need one Mamba
cache block *per concurrent sequence*, a limit entirely separate from KV cache. Pass
`--max-num-seqs 256`. Any copy-paste command for this model that omits it does not work.

**The config still carries mRoPE.** `rope_parameters` retains `mrope_section` and
`mrope_interleaved` from the multimodal parent; vLLM logs
`Unrecognized keys in rope_parameters`. Harmless so far — but the GLM split explicitly
dropped `mrope_section` and this one should too.

### What removing 1.658% actually buys

Same H100 80GB, same flags, vLLM 0.27.1:

| | Parent | Text-only | Δ |
|---|---|---|---|
| Model weights | 51.10 GiB | 50.13 GiB | −0.97 GiB (−1.9%) |
| KV cache available | 18.27 GiB | 19.84 GiB | **+1.57 GiB (+8.6%)** |
| KV cache tokens | 277,435 | 301,465 | **+8.7%** |
| Max concurrency @ 32k | 8.47× | 9.20× | **+8.6%** |

The weights saving is small; it converts almost entirely into cache headroom, which is
the number worth quoting.

### The benchmark result that nearly got published wrong

512-in / 256-out, greedy, `ignore_eos`, one shared OpenAI-compatible client, H100 80GB.

| Engine | Model | c=1 tok/s | c=16 tok/s | c=16 TTFT p50 |
|---|---|---|---|---|
| vLLM 0.27.1 | parent | 47.4 | **635.8** | 657 ms |
| vLLM 0.27.1 | text-only | 46.6 | 554.7 | 654 ms |
| SGLang 0.5.18 | parent | 48.8 | 225.3 | 10,585 ms |
| SGLang 0.5.18 | text-only | 47.2 | **628.5** | 715 ms |

Three things to read off this, in order of how badly the first draft got them wrong:

1. **Warmup moves the number by 21%.** The first vLLM run on the split gave 457.9 tok/s;
   the same server and flags with one discarded warmup pass gave 554.7. Every row above
   is warmed. An unwarmed row is not comparable to a warmed one, and mixing them silently
   flips conclusions.
2. **"The text-only split is faster" is false.** On vLLM the *parent* is 1.15× faster at
   c=16. On SGLang the split is 2.79× faster — but only because SGLang's planner allocated
   the parent `max_running_requests=5` versus the split's `16` and then starved it. That
   is an engine memory-planning artifact, not a property of either model.
3. **The real result is variance.** Across the two engines the parent's c=16 throughput
   spans **2.82×** (635.8 vs 225.3); the split's spans **1.13×** (554.7 vs 628.5) on the
   same hardware. The split is not faster. It is *predictable*, and predictability is
   what you actually get for dropping the vision tower.

At c=1 all four runs land between 46.6 and 48.8 tok/s. The same 64 text layers do the
work in every case, and single-stream decode cannot tell the models apart — which is the
sanity check that says the surgery did not change the compute path.

## Why not to prune for size

- [Structural Pruning of LVLMs](https://arxiv.org/abs/2604.24380) (IJCV 2026): at 15%
  backbone compression, multimodal performance drops **9.13%** while text-only drops
  **1.07%**. Vision-language alignment degrades first, unprompted. Don't hunt vision
  neurons surgically — width-prune generically and let vision die.
- [Minitron](https://arxiv.org/abs/2408.11796) is the only credible path to a genuinely
  smaller model, and it demonstrated exactly 12B→8B (Mistral NeMo → MN-Minitron-8B,
  MMLU 69.5 vs the teacher's ~68.0). Width-prune, never depth — layer-dropping destroys
  GSM8K/HumanEval while leaving MMLU looking fine. Target ≤30%; 45% is the tipping point.
  Cost: ~18k H100-hours at the 94B-token tier (~$40k), or ~$150–300k for full replication.
- [LinguDistill](https://arxiv.org/abs/2604.00829): multimodal adaptation *suppresses*
  text ability (HellaSwag −19.5%, ARC-C −13.4%) and it "is not easily reversed through
  lightweight fine-tuning." There is no dormant text-only model hiding under the weights.

## Hardware constraints

| Format | Needs | A100 (sm_80) |
|---|---|---|
| bf16 | sm_80 | yes |
| GGUF | anything, CPU included | yes |
| FP8 | sm_89+ | **no** |
| NVFP4 | sm_100+ to quantize *and* serve | **no** |

NVFP4 is NVIDIA's Blackwell format — E2M1 (1 sign, 2 exponent, 1 mantissa), 16-element
blocks, FP8 E4M3 per-block scale plus FP32 per-tensor scale. **It is not a GGUF quant.**
Producing it needs a B200 and
[LLM Compressor](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/).
Below SM100 vLLM silently degrades to weight-only and skips activation quantization.

## Gemma 4 31B NVFP4 — the whole story

Built and measured on a Lambda H100 PCIe on 2026-08-22. This section is deliberately
written as a narrative rather than a result sheet, because most of the value was in the
wrong turns.

**Outcome: 32.63 GB → 23.00 GB (−29.5%)**, made of NVIDIA's NVFP4 MLPs copied verbatim,
230 attention projections requantized to FP8, and no vision tower, retargeted onto
`Gemma4ForCausalLM`. Generation quality is preserved.

### What we set out to do, and why

`nvidia/Gemma-4-31B-IT-NVFP4` advertises a 4-bit weight *and* activation format. Reading
`hf_quant_config.json` tells a different story: 63 excluded modules, of which **60 are the
attention blocks**, listed one layer at a time.

| Component | Params | State | Size |
|---|---|---|---|
| MLPs (60 layers) | 20.8B | NVFP4, group 16 | 41.6 → ~11.7 GB |
| Attention (60 layers) | 7.93B | **untouched, BF16** | 15.85 GB |
| Embeddings (tied = `lm_head`) | 1.41B | untouched | 2.82 GB |
| Vision tower (SigLIP, 27 layers) | ~576M | untouched | ~1.15 GB |

~31.5 GB against a real 32.63 GB, so the accounting holds. Only the MLPs were quantized,
which leaves **unquantized attention as the largest single block in the file**.

Leaving it BF16 is defensible — NVFP4 quantizes activations too, and attention activations
are the outlier-heavy ones that SmoothQuant and AWQ exist to handle. It is also a deviation
from NVIDIA's own practice: their DeepSeek-V4 and Qwen3-Omni builds ship FP4 experts
alongside **FP8** attention. `mtq.NVFP4_MLP_WEIGHT_ONLY_CFG` is a named recipe in ModelOpt,
so MLP-only is a deliberate configuration rather than an oversight.

Three changes followed, in this order: strip the vision tower (pure deletion, verifiable
exactly), FP8 the attention weights (the real size win), pair the unpaired MTP drafter
(free throughput, and an open question).

### Decision: weights-only FP8, not W8A8

FP8 has two independent halves with very different risk. The weight half is a pure function
of the weights — no calibration set, no forward passes, nothing to get wrong beyond the
scale arithmetic. The activation half needs calibration data and is exactly where
attention's outliers cause trouble.

Weights-only still takes attention from 15.85 GB to 7.93 GB and still speeds up batch-1
decode, which is memory-bandwidth-bound rather than compute-bound. What it gives up is a
prefill win, because BF16 activations cannot feed FP8 tensor cores.

Chosen because it is the version that definitely ships, and because the headline
measurement (drafter acceptance) wanted only one variable moving at a time. W8A8 remains a
sensible follow-up.

### Decision: modify NVIDIA's checkpoint rather than requantize from BF16

The alternative was starting from `google/gemma-4-31B-it` and doing one mixed-precision
ModelOpt run. Rejected: it needs a 62 GB download and a calibration pass, and the result is
no longer comparable to NVIDIA's published benchmark numbers, because any accuracy delta
could come from our calibration set rather than from the FP8 attention. Working in place
keeps their calibrated NVFP4 MLPs bit-identical, so exactly one thing changes.

### What did not work: loading the checkpoint

**transformers cannot load this checkpoint at all.** `quant_method: modelopt` is not among
the methods it supports (`fp8`, `mxfp4`, `compressed-tensors`, `quark`, and others are).
It does not error on the unknown method. It ignores the quantization config entirely, loads
the packed `uint8` tensors as dense weights, and dies on a shape mismatch:

```
ckpt: torch.Size([21504, 2688])  vs  model: torch.Size([21504, 5376])
```

Exactly 2× on the input dimension — which is itself the tell that NVFP4 packs two 4-bit
values per byte, and the clue that made the unpacker straightforward to write.

### The fix that unblocked everything: verify by expansion

**Weight-only quantization is exactly simulatable.** An FP4 or FP8 kernel does not multiply
in 4 or 8 bits; it dequantizes the stored weight to a wide type and multiplies against the
activations. So a checkpoint whose weights have been round-tripped through the format
computes bit-identical results to the real one, and accuracy can be measured on an expanded
BF16 copy.

Two consequences:

- **NVFP4 accuracy work does not need Blackwell.** Only throughput does. This overturned
  the earlier plan, which had assumed scoring required sm_100 hardware.
- It sidesteps the missing loader entirely.

What expansion does *not* simulate is NVFP4's **activation** quantization, so an expanded
NVFP4 checkpoint is very slightly optimistic. Both sides of any comparison are expanded the
same way, so the comparison stays fair for the weight change under test — but it is not a
claim about end-to-end deployed accuracy.

NVFP4's layout, confirmed against the real tensors:

```
weight          [out, in/2]   uint8       two e2m1 values per byte, low nibble first
weight_scale    [out, in/16]  float8_e4m3 one per group of 16
weight_scale_2  scalar        float32     one per tensor
w = e2m1(nibble) * weight_scale * weight_scale_2
```

### Verifying the verifier

Before trusting a single downstream number, two checks:

```
model.layers.{0,30,59}.self_attn.q_proj.weight    max|d| = 0.0000   bit-identical
model.layers.*.mlp.gate_proj.weight               rel     = 4-10%   correct for e2m1
weight_scale.max() * weight_scale_2 * 6 = 0.2617 == the original tensor's absmax
```

Attention was never quantized by NVIDIA, so after stripping and renaming it **must** be
bit-identical — and is. The second identity is the cheapest possible check that a
dequantizer is right: the largest e2m1 magnitude (6.0) times the largest block scale must
reproduce the original range exactly. Run it before trusting anything else.

### Five bugs the real checkpoint found that local tests did not

1. **`*self_attn*` matches `q_norm` and `k_norm`.** An early run quantized all 120 of them.
   They are 1-D RMSNorm scales: no size to win, and among the most precision-sensitive
   weights in the model. Caught only because the converted count came to 350 where
   projections should be 230. Fixed by targeting `*_proj` *and* refusing any 1-D tensor
   structurally, since the pattern had already proven easy to get wrong.
2. **The quantization schema is a hybrid.** `quantization_config` carries modelopt keys
   (`quant_algo`, `quant_method`) *and* compressed-tensors keys (`config_groups`, `ignore`,
   `kv_cache_scheme`) in one block. Reading only the modelopt half silently loses the group
   size (nested inside `config_groups`) and the KV cache format (differently named).
3. **Tied embeddings defeat name matching.** The exclusion list names `lm_head`, but the
   tensor ships as `embed_tokens`, so matching on names alone reports BF16 weights as
   quantized. The dtype in the header is the authority; the config is a claim.
4. **Renaming a prefix invalidates the exclusion patterns.** Moving
   `model.language_model.*` to `model.*` leaves every `ignore` entry pointing at a path that
   no longer exists, so they match nothing and the next reader concludes attention was
   quantized when it was not.
5. **Patterns matched full tensor names, not modules.** Any user-supplied pattern without a
   `.weight` suffix would silently convert nothing at all.

`230`, incidentally, is not a typo for 240. Gemma 4 sets `attention_k_eq_v`, so **10 layers
have no `v_proj`** — K and V are shared. Check the architecture before assuming a count is
wrong.

### What did not work: scoring on raw prompts

The first scoring run produced clean-looking numbers that said our checkpoint **beat**
NVIDIA's, and passed the "no worse than baseline" gate. It was entirely noise.

The tell was in the generations: the BF16 **reference itself** was emitting
`France is France is France is`. Scoring against a broken yardstick makes every number
meaningless.

The control experiment cleared the pipeline: the **untouched original** produces the
identical degenerate output on that prompt, and through the chat template both it and our
copy answer `Paris`. So it was never the extraction — `gemma-4-31B-it` is
instruction-tuned, and it was being fed raw completion prompts.

Scoring in the degenerate regime inflates divergence by roughly an order of magnitude and
lets small perturbations move results in arbitrary directions, which is precisely how a
*more* quantized checkpoint came out closer to the reference.

**Decision:** `agreement` grew a `--chat` mode and a `CHAT_PROMPTS` set. `verify` kept its
raw prompts unchanged, because an extraction must be bit-exact on *any* input, degenerate
or not. The two bars stay separate.

The general lesson is worth more than the fix: *a plausible result that favours your own
work is the one to distrust most.*

### Decision: two verification bars that share no code

`verify.compare_logits` demands identical bits. Right for extraction, impossible for
quantization — rounding every weight *is* the operation.

An absolute tolerance is not a fix. Nobody can justify the threshold, and the moment one
exists it will be tuned until the checkpoint passes. So `agreement` uses the shipped
quantized build as the baseline.

But a strict "no worse than baseline" gate is the right test only when a change is not
supposed to cost anything. **Ours adds quantization on top of NVIDIA's**, so it cannot pass
by construction, and reporting it as pass/fail would be rigged. `Agreement.marginal_cost`
reports the increment instead.

Measured through the chat template against the BF16 reference:

| | worst top1 | worst KL |
|---|---|---|
| B  NVIDIA NVFP4 | 0.7826 | 1.448e-01 |
| C  ours, +FP8 attention | 0.9130 | 1.716e-01 |

C adds 1.19× the baseline's KL divergence. Its top-1 came out *higher* than B's on all six
prompts, which should not happen for a strictly more quantized model — six prompts and ~150
token positions is a small sample, and that pattern is the sample saying so. **Treat both
rows as indicative and re-run on a larger prompt set before quoting them.**

### Do not read KL as benchmark accuracy

A KL of 0.145 looks alarming next to NVIDIA's published GPQA of 85.35% vs 85.80%, which
*is* near-lossless. There is no contradiction: KL over a 262k vocabulary is far more
sensitive than multiple-choice accuracy, and 9.5% relative weight error at 4 bits can leave
benchmark scores essentially unchanged. This nearly triggered a false alarm about the
dequantizer. Judge a quantization against another quantization, never against an absolute
divergence.

### The headline: quantizing the target costs speculative acceptance

The open question. The drafter (`google/gemma-4-31B-it-assistant`, 0.47B) was trained
against the BF16 target; quantizing the target moves its distribution away from what the
drafter learned to predict. Speculative decoding stays **exact by construction** — the
target verifies every drafted token and discards what it would not have produced — so this
cannot change output quality. It changes throughput, and throughput is entirely acceptance.

Measured as tokens per target forward pass, the arithmetic ceiling on speedup and a
property of the two distributions rather than of any kernel:

| target | tok/fwd | vs BF16 |
|---|---|---|
| R  base BF16 | 2.220 | — |
| B  NVIDIA NVFP4 | 1.655 | −25.4% |
| C  ours, +FP8 attention | 1.247 | −43.8% |

**Monotonic in the amount of quantization**, which is what makes it credible where the
top-1 numbers were not. A published "up to 3×" for the BF16 pairing does not transfer to a
quantized target: NVIDIA's own build surrenders a quarter of the benefit before anyone adds
anything.

Caveats that matter:

- **Ignore the wall-clock column.** The BF16 reference measured a 0.97× "speedup" —
  speculative decoding running *slower* than plain generation — because transformers'
  assisted-generation overhead dominates at this scale. A harness property, not a model
  property, and exactly why tok/fwd is what gets reported.
- Uses `assistant_model` draft-model speculation, not necessarily the native MTP path the
  drafter ships for.
- Four prompts, 96 new tokens each. Directionally clear, not a precise coefficient.

Practical reading: if throughput is why you want speculative decoding, quantizing the
target is not free, and the trade against 30% smaller weights is a real one to weigh.

### The KV cache — and a 2× error in the first estimate

The initial analysis assumed uniform layers and claimed 80.0 KB/token, 21.47 GB at full
context, "as large as the weights". That was wrong.

Gemma 4's layers are **heterogeneous**. transformers raises
`AmbiguousGlobalPerLayerAttributeError` on a global read of `num_key_value_heads`, which is
what surfaced the mistake — a library refusing to answer an ill-posed question:

```
 10 x full_attention     kv_heads=4   head_dim=512   -> 40.0 KB/token
 50 x sliding_attention  kv_heads=16  head_dim=256   -> capped at ~0.4 GB by the window
```

The counter-intuitive part: the **sliding** layers carry four times the KV heads of the
global ones, but their 1024-token window caps them at a fixed ~0.4 GB however long the
context runs. Only the 10 global layers grow:

```
   32,768 tokens ->   1.75 GB
  131,072 tokens ->   5.79 GB
  262,144 tokens ->  11.16 GB
```

So at the full window the cache is roughly **half** the quantized weights, not equal to
them. Worth knowing before optimising further, but it does not outrank the weight savings,
and the stronger claim was an artifact of assuming a homogeneous model.

### Two things reached the published repo before being caught

Both are the same bug wearing different clothes: metadata that describes the model
*before* the surgery.

**Loose files travel with a checkpoint.** `_copy_rest` copied everything non-safetensors
forward, which meant NVIDIA's `README.md` (their model card, describing their build) and
`.quant_summary.txt` (a 2,155-line ModelOpt quantizer dump listing the vision tower and
reporting attention as unquantized) both landed in the upload. Neither is true of the
result. There is now a `STALE_METADATA` skip set, with a test asserting the tokenizer and
chat template still travel -- the guard must not become an excuse to drop files the model
needs.

**Promoting a nested config exposes fields to new validators.** Gemma 4's `text_config`
carries `num_experts: null`, `expert_intermediate_size: null` and `top_k_experts: null`,
inert because `enable_moe_block` is False. Nested inside `text_config` nobody checks them.
The extraction promotes `text_config` to top level, and the Hugging Face config parser then
rejects `"num_experts" must be a number`.

The fix was to drop the three inert keys -- but the check that made it safe was
constructing `Gemma4TextConfig` both ways and asserting every resolved field matched.
Dropping keys from a config on the grounds that they "look unused" is otherwise a guess.

### vLLM and this checkpoint — three wrong answers before the right one

The published card told people to run `vllm serve --quantization modelopt`. That was
written from what the format's documentation implies, never run, and it does not work.
Correcting it took two rented boxes and about $7, mostly because the first two
explanations were both wrong and both *felt* conclusive.

The final answer: **the checkpoint cannot be served by vLLM at all today**, for a reason
that has nothing to do with Gemma 4 or with vLLM's Gemma support, and everything to do with
the weights-only FP8 decision recorded above.

#### The first wrong answer: "vLLM doesn't support Gemma 4"

The observed error was:

```
AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute
```

and the conclusion drawn was an upstream architecture-support gap. It cost nothing to
check and was false. A file listing of the *release tag* — no GPU, no download, one
`curl` — shows `gemma4.py`, `gemma4_mm.py` and `gemma4_mtp.py` all present in v0.27.1, and
`git log` dates the first of them to 2026-04-02, four months before that release.

*Lesson: before renting a GPU to explain a failure, check whether the cheapest possible
observation already falsifies the explanation.*

#### The second wrong answer: "it's a transformers version skew"

Falsifying a cause is not finding one. The next hypothesis came from two pieces of
evidence in the error itself:

- `AmbiguousGlobalPerLayerAttributeError` is raised by **transformers**, not by vLLM. The
  exception's home library is evidence about which side changed.
- vLLM 0.27.1 pins `transformers >= 5.5.3` with **no upper bound**, and was released
  2026-08-11. `unfuse` requires `transformers>=5.15.1`, released 2026-08-19 — so vLLM was
  necessarily running against a transformers newer than it was ever tested with.

That is testable on the config alone, so it reproduces with no weights and no GPU. A
version sweep bisected it exactly:

```
5.5.3, 5.8.0, 5.11.0, 5.13.1, 5.14.0, 5.14.1   head_dim = 256    global read OK
5.15.0, 5.15.1                                  AmbiguousGlobalPerLayerAttributeError
```

Pinning `transformers==5.14.1` does clear the config error. It is a real and necessary
finding — it affects *any* Gemma 4 31B on vLLM, Google's included — but it was not the
blocker. It only exposed the next failure, an `AssertionError` deep in the QKV loader.

*Lesson: fixing the first error tells you the second error, not that you were right.*

The transformers guard is also **correct**, not a regression. Gemma 4 31B genuinely has two
head_dims — 256 on the 50 sliding layers, 512 on the 10 full-attention layers — so a global
`head_dim` read is meaningless. vLLM 0.27.1 reads it globally and gets 512, which is wrong
for 50 of 60 layers.

#### The control experiment that inverted the conclusion

The question worth answering was not "why does this fail" but "**is this ours?**". The
control is to run the *unmodified upstream artifact* through the identical path:

```
nvidia/Gemma-4-31B-IT-NVFP4                       ->  SERVED OK
sartajbhuvaji/Gemma-4-31B-IT-NVFP4-FP8attn-text   ->  AssertionError in load_qkv_weight
```

Upstream serves — on an **A100**, not even Blackwell. So vLLM can serve Gemma 4 31B, the
transformers pin is sufficient for that, and the blocker is something this build
introduced. Everything believed up to that point was pointing the wrong way.

*Lesson: a control on the unmodified upstream artifact is what separates "their bug" from
"ours", and it is worth its cost early rather than late.*

#### Instrumenting instead of guessing

The assertion carried no shapes, so the loader was monkeypatched to print them at the point
of failure rather than reasoning about what they might be:

```
shard=k   param=(16384, 2688)   loaded=(4096, 5376)
```

`2688 = 5376/2` is the tell — the same 2× that identifies FP4 packing elsewhere in this
document. vLLM had allocated the attention parameter **NVFP4-packed** while the stored
tensor is full-width FP8. So vLLM was applying NVFP4 to attention, which meant it had never
read the `quantized_layers` mapping that says otherwise.

Reading the dispatch source rather than the documentation found why — two exact-match gates:

```python
# ModelOptMixedPrecisionConfig.override_quantization_method
if algo is not None and algo == "MIXED_PRECISION":
    return "modelopt_mixed"
```

```python
# _quantized_layer_prefix_candidates: exact dict membership, no glob expansion
for candidate in self._quantized_layer_prefix_candidates(prefix):
    if candidate in self.quantized_layers:
```

Two real defects in this checkpoint's config, both invisible from the outside:

1. `quant_algo` is `"NVFP4"`. The mixed path is reachable only on the literal string
   `"MIXED_PRECISION"`, so it was never entered.
2. `quantized_layers` keys are globs — `model.layers.0.self_attn*`. The lookup is dict
   membership, so they match nothing. Correct keys name each projection:
   `model.layers.0.self_attn.q_proj`, 230 FP8 entries beside 180 NVFP4 MLP entries.

*Lesson: when a config key appears to be ignored, read the dispatch code. The gate is
often a literal string comparison, and no amount of plausible-looking JSON will pass it.*

#### The real blocker, and why it is the worst kind

With both defects patched the model **loaded, ran, and generated fluent nonsense**:

```
>>> What is the capital of France? Answer in one word.
same dePK singleale dePK singleale dePK singleale ...
```

`ModelOptFp8LinearMethod.create_weights` registers `input_scale` as a required per-tensor
parameter and `process_weights_after_loading` calls `layer.input_scale.max()`. vLLM's
ModelOpt FP8 path is **W8A8**. This checkpoint is deliberately **W8A16** — FP8 weights,
BF16 activations, chosen for exactly the accuracy reasons recorded above — so there is no
`input_scale` to load, and the scales stay at their initialised sentinels.

Nothing errors because **FP8 and BF16 weights have identical shapes**. Every shape check
passes; only the arithmetic is wrong. This is strictly worse than the crash it replaced: a
crash is self-reporting, and a checkpoint that serves confident garbage is not.

So the decision that is right on accuracy grounds is the one with no serving path, and
fixing defects 1 and 2 in isolation would convert a loud failure into a silent one. They
are worth fixing for metadata correctness, but `quant_algo` should stay `NVFP4` until
either a calibrated W8A8 variant exists or vLLM supports FP8 weight-only in the ModelOpt
path.

*Lesson: prefer the failure mode that announces itself. A dtype swap that preserves shape
will pass every structural check a loader makes.*

#### What actually runs, tested

Stock released vLLM — not a nightly, because "install the nightly" is a materially
different instruction to publish:

```bash
pip install vllm==0.27.1 'transformers==5.14.1'
python tools/dequantize.py ./ckpt ./ckpt-bf16
vllm serve ./ckpt-bf16
```

H100 PCIe 80 GB: engine up in 103 s, 36.5 tok/s on a batch of 5, eager. `Paris`; `408`
with correct working; `Das Wetter ist heute kalt.`; Rayleigh scattering; `my_string[::-1]`.
A vLLM nightly serves the same expansion at 42 s and 48.8 tok/s. Expansion costs the size
saving — 23 GB becomes 61 GB — so it wants an 80 GB card.

#### Corrections to claims this repo had already published

- **"NVFP4 executes only on Blackwell (sm_100+)" is false under vLLM.** It selects
  `MarlinNvFp4LinearKernel`, and `ModelOptMixedPrecisionConfig.get_min_capability()` is 75,
  its comment recording validation on a T4 (SM75) and an A100 (SM80). Blackwell provides
  native FP4 tensor cores; without them Marlin repacks to W4A16, so the *stored* saving is
  not a *runtime* saving — a 40 GB A100 ran out of memory loading this 23 GB checkpoint.
- **The FP8 KV cache needs SM89+**, not Blackwell. Below that vLLM refuses with
  `FP8 KV cache is not supported by the Triton attention backend` and names the fix:
  `--kv-cache-dtype bfloat16`.
- **vLLM 0.27.1 has no `quantized_layers` support at all** — it parses only
  `exclude_modules` (legacy `hf_quant_config.json`) or `ignore`. NVIDIA's build serves on it
  because its 60 attention blocks sit in `exclude_modules`, which 0.27.1 does read. Mixed
  precision cannot be expressed to that release in any form.

#### The meta-lesson

Two of the three explanations along the way were wrong, and each was confidently written
down before the next observation demolished it. The published card asserted the first one.
The pattern that eventually worked was: cheapest falsifying observation first, then a
control against the unmodified upstream, then instrumentation at the point of failure, then
the library's own source — and only then a claim, tested on the exact versions named in it.

### Environment problems that cost real time

Four from the Qwen3.8-27B run, all of which produced a *silent* wrong state rather than
an error at the point of the mistake:

- **`pip install` does not put the script on PATH.** `pip install huggingface_hub[cli]`
  warns that it installed to `~/.local/bin`, which is not on PATH on a fresh Lambda image.
  `nohup hf download ... &` then fails with `nohup: failed to run command 'hf'` *inside the
  log*, while the launching shell happily echoes "download started". Twenty minutes of a
  rented GPU went to a download that never began. Use absolute paths for anything
  backgrounded, and verify bytes on disk rather than trusting an echo. Same family as the
  `ninja` PATH trap already recorded above.
- **`pkill -f <pattern>` matches your own SSH session.** The remote command string is in
  the remote shell's `argv`, so `pkill -f pipeline.sh` issued over ssh kills the shell
  running it — connection drops, exit 255, and whatever was mid-flight dies with it. This
  bit three times. The `[p]attern` bracket trick only helps when the pattern is not also
  present verbatim elsewhere in the command; a heredoc containing `launch_server` puts
  that string in argv too. Kill by PID, or write the script to a file and scp it.
- **`hf upload` creates the repo PUBLIC by default.** There is no prompt. A 54.64 GB
  checkpoint began publishing to a public repo before its model card existed. Create the
  repo explicitly, or check `model_info(...).private` immediately and flip it.
- **A backgrounded job that outlives the SSH session needs `setsid`.** Plain `nohup ... &`
  still died when its parent session was killed; `setsid nohup ... < /dev/null & disown`
  survives.

Two more from building the daily-intel routine, worth recording because they are network
policy rather than code:

- `reddit.com` returns **403** to datacenter IPs on `.json`, `.rss`, and `old.reddit`
  alike, as does `x.com`. Neither is reachable programmatically from a cloud runner.
- arXiv's API works over **https** and fails over plain `http`.

### Environment problems that cost real time

- **A `.env` edited on Windows kills every API key.** Files under `/mnt/c` get CRLF, so
  sourcing leaves a trailing `\r` and the service reports `invalid-api-key` for a key that
  is correct. `file .env` diagnoses it; `sed -i 's/\r$//' .env` fixes it. The launch script
  now strips CRLF on read rather than trusting the file.
- **`--system-site-packages` mixes apt and pip packages badly.** The venv inherited apt's
  torchvision 0.22 while pip installed torch 2.13, so `torchvision::nms` failed to
  register — surfacing as a completely misleading
  `ModuleNotFoundError: Could not import module 'Gemma4Config'`. pip then *skipped*
  installing torchvision because the apt copy "satisfied" the requirement. Then apt's scipy
  collided with the venv's numpy. The fix was a clean venv with no system packages and
  torch+torchvision from the same cu130 index.
- **A shell script without a shebang runs under `/bin/sh`.** The setup script used
  `exec > >(tee ...)`, a bash-only process substitution, so it died instantly *and left no
  log*, because the redirect that would have recorded the error was itself the error.
- **The registered SSH key was the wrong one.** `launch_lambda.sh` defaulted to
  `KEY_NAME=lambda-glm`, which is the keypair Lambda silently generated for itself during
  an earlier incident — its private half was in an API response nobody kept. The
  `verify_key` guard caught it. Default is now `lambda-glm2`.

Every one of these fails *silently or misleadingly*, which is the common thread: the error
message pointed at the wrong layer in three of the four cases.
