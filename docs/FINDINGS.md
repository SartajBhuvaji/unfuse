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
