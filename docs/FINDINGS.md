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
