# Study guide

Everything you need to understand to read this repository and extend it, ordered roughly
from foundations to specifics. Each entry says **what** it is, **why it matters here**, and
where it shows up, so you can skip what you already know.

Nothing here is optional trivia. Every topic cost real time in this project, either because
not knowing it produced a bug, or because knowing it produced a shortcut.

If you only read four things: **§3.5 NVFP4**, **§3.9 simulated quantization**,
**§9.3 verification by expansion**, and **§9.8 the degenerate-regime trap**. Those carry
most of the load. A suggested reading order is at the end.

---

## 1. Mathematical foundations

### 1.1 Matrix multiplication, shapes, and layout
`[out, in] @ [in]`, batching, and why a weight stored `[out, in/2]` is an immediate signal
that something is packed. Contiguity and strides matter once you start reshaping nibbles.

*Why here:* the 2× shape mismatch that revealed NVFP4's packing was a pure shape argument.

### 1.2 Softmax, temperature, and logits
What a logit is, why softmax is shift-invariant, and what temperature does to a
distribution.

*Why here:* every comparison in this repo is over logits, not text.

### 1.3 Log-space computation and the logsumexp trick
Why `log_softmax` is not the same as `log(softmax(x))` in floating point, and why softmaxing
a 262,144-wide vector directly overflows.

*Why here:* mandatory in `_score`. Get it wrong and KL comes out `nan` or silently biased.

### 1.4 Entropy, cross-entropy, perplexity
The standard trio, and how perplexity relates to loss.

*Why here:* perplexity is the metric people reach for first; §9.5 explains why this repo
does not use it as the bar.

### 1.5 KL divergence — forward, reverse, and asymmetry
`KL(P‖Q) ≠ KL(Q‖P)`. Which one you compute encodes which model you treat as truth.

*Why here:* `KL(reference ‖ candidate)` — the reference is truth, and the candidate is
penalised for putting low probability where the reference put high.

### 1.6 Other distances, and why they were not used
Jensen-Shannon (symmetric, bounded), total variation, cosine similarity on weights,
top-k overlap.

*Why here:* top-k overlap *is* used alongside KL, because a distribution can drift while the
ranking survives — and ranking is what greedy decoding consumes.

### 1.7 Floating-point non-associativity
`(a+b)+c ≠ a+(b+c)`. Summation order changes results, so identical math in a different
order gives different bits.

*Why here:* it is exactly why bit-exactness is a strong claim. An extraction that reorders
nothing must reproduce bits; if it does not, the computation changed.

### 1.8 Sources of nondeterminism on GPUs
Atomics, split-k reductions, TF32, cuDNN/cuBLAS autotuning, batch-size-dependent kernel
selection.

*Why here:* if you ever fail to reproduce a bit-exact result, check these before assuming
the extraction broke.

### 1.9 Catastrophic cancellation and error accumulation
Subtracting near-equal numbers destroys precision; quantization error compounds across 60
layers rather than staying local.

*Why here:* why per-layer error looks tiny and end-to-end divergence does not.

---

## 2. Number formats

### 2.1 IEEE 754 anatomy
Sign, exponent, mantissa; bias; subnormals; how many representable values a format has and
how they are spaced (denser near zero, sparser far away).

### 2.2 bfloat16 vs float16
Same 16 bits split differently. bf16 keeps float32's 8 exponent bits and sacrifices
mantissa; fp16 keeps mantissa in a narrower range. bf16 trades precision for range and
rarely overflows.

*Why here:* every checkpoint is bf16 at rest and every expanded copy targets bf16.

### 2.3 float32, TF32, float64
What each is for, and why TF32 silently reduces precision on Ampere+ unless disabled.

### 2.4 FP8 — E4M3 and E5M2
E4M3: 4 exponent, 3 mantissa bits, **saturates at 448.0**. E5M2: more range, less precision.
Weights conventionally use E4M3, gradients E5M2.

*Why here:* `FP8_MAX = 448.0`, and the scale is chosen so the largest weight lands exactly
there — anything less wastes the format's range.

### 2.5 FP4 — E2M1
1 sign, 2 exponent, 1 mantissa bit. Eight magnitudes: `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`.
Small enough to memorise, and memorising it makes FP4 code readable at a glance.

*Why here:* the lookup table at the heart of the unpacker.

### 2.6 Nibble packing and unpacking
Two 4-bit values per byte; which nibble holds the earlier element (low, here); masking and
shifting; interleaving back to full width.

*Why here:* get the order backwards and weights scramble within pairs — the model still
loads and still produces plausible-looking garbage.

### 2.7 Integer formats
INT8/INT4, symmetric vs asymmetric (zero-point), and why weights are usually symmetric while
activations may not be.

### 2.8 Rounding modes
Round-to-nearest-even vs truncation vs stochastic rounding, and why the default matters at
4 bits where every rounding decision is large.

### 2.9 Saturation vs overflow
Clamping to the format maximum rather than wrapping or producing inf.

*Why here:* the `.clamp(-FP8_MAX, FP8_MAX)` in `quantize_tensor` is not decorative.

---

## 3. Quantization

### 3.1 The basic operation
`q = round(w / s)`, `w' = q * s`. Everything else is a choice about how `s` is computed and
how finely it varies.

### 3.2 Granularity
Per-tensor (one scale), per-channel (one per output row), per-group/block (one per N inputs
along a row). Finer granularity costs storage and buys accuracy.

*Why here:* `--per-channel`, and NVFP4's group size of 16.

### 3.3 Choosing the scale
absmax, percentile clipping, MSE-optimal search. absmax is simplest and is what this repo
uses for weight-only FP8.

### 3.4 The two independent axes: weights and activations
`W8A16` quantizes only weights; `W8A8` also quantizes activations. **These are separate
decisions** and conflating them is the most common confusion in the area.

*Why here:* NVFP4 in this checkpoint is W4A4 but only inside the MLPs; our FP8 attention is
W8A16 by deliberate choice.

### 3.5 NVFP4 — **read this one**
NVIDIA's 4-bit format with **two levels of scale**:

```
weight          [out, in/2]   uint8         two e2m1 values per byte
weight_scale    [out, in/16]  float8_e4m3   one per group of 16
weight_scale_2  scalar        float32       one per tensor
w = e2m1(nibble) * weight_scale * weight_scale_2
```

The block scale is *itself* quantized to FP8, which is exactly why a second global scale
exists — it restores the range the FP8 block scales cannot span.

*Why here:* it is what the checkpoint is made of. The identity
`weight_scale.max() * weight_scale_2 * 6 == original absmax` (6.0 being the largest e2m1
magnitude) is the cheapest correctness check for any FP4 dequantizer.

### 3.6 Double quantization elsewhere
bitsandbytes NF4 also quantizes its scales. Same idea, different constants — useful
contrast for understanding why NVFP4 is shaped the way it is.

### 3.7 MXFP4 and OCP microscaling
Group size 32, a shared **power-of-two** exponent instead of an FP8 scale. Knowing the
difference stops you applying an NVFP4 recipe to an MXFP4 checkpoint.

### 3.8 PTQ vs QAT, and calibration
Post-training quantization runs data through to pick activation scales; QAT fine-tunes
through fake-quant ops. Calibration set choice measurably changes results.

*Why here:* avoiding a calibration pass is why we modified NVIDIA's checkpoint instead of
requantizing from BF16 — it keeps their MLPs bit-identical so exactly one variable moves.

### 3.9 Simulated / fake quantization — **read this one**
Quantize then immediately dequantize, keeping a wide dtype. For **weight-only**
quantization this is not an approximation: the real kernel dequantizes before multiplying
against wide activations anyway.

*Why here:* it is why NVFP4 accuracy work needs no Blackwell GPU. It does **not** extend to
activation quantization.

### 3.10 Activation outliers
A few channels carry values orders of magnitude above the rest; one scale for the tensor
destroys everything else. This is the entire reason SmoothQuant and AWQ exist.

*Why here:* it justifies both NVIDIA leaving attention alone and our taking only the weight
half.

### 3.11 The algorithm landscape
GPTQ (Hessian-based error compensation), AWQ (activation-aware channel scaling),
SmoothQuant (migrating difficulty from activations to weights), LLM.int8() (outlier
decomposition), HQQ, QuIP/QuIP#, SpQR.

*Why here:* background for reading other checkpoints, and for knowing what you would reach
for if W8A8 attention turned out to hurt.

### 3.12 Sensitivity and mixed precision
Not all layers tolerate quantization equally. Which layers to leave alone is a real design
decision, and NVIDIA's exclusion list is one answer to it.

### 3.13 KV cache quantization
Quantizing K and V separately from weights; FP8 KV is common, FP4 KV is emerging.

*Why here:* `kv_cache_scheme` is already FP8 in this checkpoint, and §5.6 explains why that
lever is the big one at long context.

### 3.14 What never to quantize
Norms, biases, embeddings (usually), and anything 1-D. No size to win, high sensitivity.

*Why here:* the norm bug. The tool now refuses 1-D tensors **structurally**, because the
pattern had already proven easy to get wrong.

### 3.15 Packing layouts and kernel expectations
Serving engines expect specific packed layouts and scale shapes. A scale nobody applies is
silently a large error.

### 3.16 Dequant kernels and fusion
Fused dequantize-matmul, Marlin/Machete/CUTLASS. Why weight-only quantization can still be
fast at batch 1 but does not help prefill.

---

## 4. Transformer architecture

### 4.1 The block
Attention + MLP + residuals + normalization, and the tensor names: `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

*Why here:* every pattern in this repo selects modules by these names.

### 4.2 Attention mechanics
QK product, scaling, softmax, value aggregation, causal masking.

### 4.3 RMSNorm vs LayerNorm
RMSNorm drops mean-centering. Its weight is a **1-D** per-feature scale.

*Why here:* `q_norm`/`k_norm` are 1-D RMSNorm weights, and quantizing them was the bug.

### 4.4 Gated MLPs / SwiGLU
Why there are three MLP projections rather than two.

*Why here:* the 180 NVFP4 tensors are exactly 3 × 60 layers.

### 4.5 MHA, MQA, GQA
How many key/value heads are shared across query heads; `num_key_value_heads` is the knob.

*Why here:* it sets KV cache size.

### 4.6 head_dim and hidden_size
Their relationship, and why `head_dim` is not always `hidden_size / num_heads` in modern
models.

*Why here:* Gemma 4's global layers use `head_dim=512` with `hidden_size=5376` — assuming
the division would give the wrong cache size.

### 4.7 Sliding-window vs global attention
Local attention sees the last N tokens; global sees everything; hybrids interleave them.

*Why here:* Gemma 4 runs 5 sliding : 1 global.

### 4.8 The KV cache
What it stores and the size formula:
`layers × kv_heads × head_dim × 2 (K and V) × bytes × tokens`.

*Why here:* the thing that can dwarf the weights at long context — and assuming uniform
layers produced a 2× error in this project.

### 4.9 Position encodings
Absolute, learned, ALiBi, RoPE; base/theta and context extension.

### 4.10 mRoPE
The multimodal variant splitting rotation across axes (`mrope_section`).

*Why here:* the GLM extraction turned on proving mRoPE collapses to standard RoPE for
text-only input — measured, not assumed.

### 4.11 Tied embeddings
`embed_tokens` and `lm_head` sharing one tensor.

*Why here:* it defeats name-based matching. The exclusion list says `lm_head`; the tensor
ships as `embed_tokens`.

### 4.12 Stabilisation tricks
Logit softcapping, QK-norm, attention sinks, residual scaling.

*Why here:* Gemma 4 uses several; you will meet these config keys.

### 4.13 Mixture of experts
Routing, top-k, experts, shared experts, load balancing.

*Why here:* not used by this model, but `enable_moe_block` appears in its config, and the
DeepSeek-V4 precedent (FP4 experts, FP8 attention) is the basis for our layout.

---

## 5. Gemma 4 specifically

### 5.1 The family is not uniform
`gemma-4-12B` is `gemma4_unified` and **encoder-free** — its whole vision path is a 52M
input embedder. The 31B is `model_type: gemma4` with a real 27-layer SigLIP tower.

*Why here:* **always check `model_type` before applying any recipe.**

### 5.2 Heterogeneous per-layer configs
Layers differ. transformers exposes `config.per_layer_config[i]` and raises
`AmbiguousGlobalPerLayerAttributeError` on a global `num_key_value_heads` read.

*Why here:* the 10 global layers use `kv_heads=4, head_dim=512`; the 50 sliding layers use
`kv_heads=16, head_dim=256`. Assuming uniformity doubles the computed cache. A library
refusing to answer an ill-posed question is what caught this.

### 5.3 `attention_k_eq_v`
Some layers share K and V, so they have no separate `v_proj`.

*Why here:* the FP8 count is **230, not 240**, and 230 is correct.

### 5.4 Model classes and tensor naming
`Gemma4ForConditionalGeneration` names tensors `model.language_model.*`;
`Gemma4ForCausalLM` expects `model.*`.

*Why here:* the extraction is a prefix rename — and the rename invalidates every `ignore`
pattern referencing the old path.

### 5.5 Config surface
`text_config` vs `vision_config`, `layer_types`, `sliding_window`,
`max_position_embeddings`, `hidden_size_per_layer_input`, `final_logit_softcapping`.

### 5.6 The long-context regime
262,144-token window. At full length the cache is ~11 GB against 23 GB of weights.

---

## 6. Multimodal models

### 6.1 Vision encoders
ViT and SigLIP: patch embedding, transformer over patches, pooling. SigLIP's sigmoid loss
vs CLIP's contrastive softmax.

### 6.2 Projectors and soft tokens
How image features become tokens the LM consumes; fixed vs variable token budgets; pixel
shuffle / merger layers.

*Why here:* `vision_soft_tokens_per_image`, and the three prefixes the strip removes.

### 6.3 Image preprocessing
Patch size, resolution, tiling/cropping for high resolution, and the processor/feature
extractor split.

### 6.4 Separable vs fused encoders
Whether the encoder is a distinct tower (removable) or fused into an embedder (not worth
removing).

*Why here:* the entire premise of `unfuse`.

### 6.5 Audio and video paths
Same structure, different front end. Gemma 4's unified variants carry both.

### 6.6 Encoder-only extraction — the other direction
Keeping the tower and discarding the language model. The result is an `mmproj`-style
artifact that pairs back with a text-only GGUF to reconstitute the original multimodal
model under llama.cpp.

*Why here:* `notebooks/glm-4.6v-flash/keep_vision.ipynb` does exactly this, and it is the
cheap direction — the vision half of GLM-4.6V-Flash is 1.78 GB and runs on a laptop GPU,
while removing vision needs ~21 GB of VRAM for the bit-exact check.

---

## 7. Speculative decoding

### 7.1 Draft and verify
A small drafter proposes k tokens; the target verifies them in one forward pass and
discards anything it would not have produced.

### 7.2 Why it is exact
The rejection-sampling rule makes the output distribution **identical** to the target's.
Worth reading the proof once — it is short, and it is why this is free.

### 7.3 Acceptance rate is the entire speedup
Tokens per target forward pass is the arithmetic ceiling.

*Why here:* it is the metric the headline experiment reports, precisely because it is
hardware-independent.

### 7.4 Draft length tuning
Longer drafts win more when acceptance is high and waste more when it is not.

### 7.5 The variants
MTP heads, EAGLE-1/2/3, Medusa, tree attention, lookahead decoding, prompt-lookup/n-gram
drafting. They are different mechanisms with different acceptance behaviour.

*Why here:* the drafter ships as MTP but was measured through transformers'
`assistant_model` draft-model path — a real caveat on the number.

### 7.6 Distribution match matters
The drafter is trained against a specific target. Change the target and acceptance falls.

*Why here:* the headline result — −25.4% for NVIDIA's build, −43.8% for ours.

---

## 8. Formats, libraries, and ecosystem

### 8.1 safetensors
Header-only metadata reads, dtype strings, sharding,
`model.safetensors.index.json`, `weight_map`, and `metadata.total_size` — which publishers
do not always fill with bytes.

*Why here:* inspection reads headers only, so a 33 GB checkpoint inspects in a second.

### 8.2 Why not pickle
Arbitrary code execution on load. safetensors exists to avoid it.

### 8.3 Hugging Face Hub
Repos, revisions, LFS/Xet storage, gated repos and license acceptance, model cards,
`snapshot_download`, `hf download`, and the HTTP API for checking access without downloading.

### 8.4 transformers loading machinery
`AutoConfig`/`AutoModel` dispatch via `model_type`, the `architectures` field,
`from_pretrained`, `device_map`, `low_cpu_mem_usage`, `tie_weights`, and
**`output_loading_info=True`** — which reveals silently randomly-initialized weights.

*Why here:* a model with missing weights loads happily and generates garbage.

Also: **`architectures` is never populated for you.** A config class you construct in Python
leaves it empty, and without it `AutoModelForCausalLM` has nothing to dispatch on — the
checkpoint then loads only via the explicit class. Set it when writing a config.

### 8.5 The `quantization_config` registry
transformers dispatches on `quant_method`. An **unsupported** method is not an error — the
config is ignored, the checkpoint loads as dense, and it fails later and confusingly.

*Why here:* `quant_method: modelopt` is unsupported, which is why transformers cannot load
this checkpoint at all.

### 8.6 compressed-tensors — the niche one worth knowing
vLLM's quantization description format: `config_groups` (each with `weights`,
`input_activations`, `targets`), a global `ignore` list, `kv_cache_scheme`, and a `format`
string like `nvfp4-pack-quantized`. Schemes are described **structurally**
(`num_bits`, `type`) rather than by format name, so you derive the format from the numbers.

*Why here:* this checkpoint's `quantization_config` is a **hybrid** — modelopt keys *and*
compressed-tensors keys in one block. A parser reading one half silently loses the group
size and KV format.

### 8.7 NVIDIA ModelOpt
`mtq` configs (including the named `NVFP4_MLP_WEIGHT_ONLY_CFG`), `export_hf_checkpoint`,
and the legacy `hf_quant_config.json`.

### 8.8 vLLM
`--quantization modelopt`, per-layer algorithms under `quantized_layers`, PagedAttention,
continuous batching, prefix caching. Mixed FP4/FP8 checkpoints are a **standard shipping
layout** (DeepSeek-V4, Qwen3-Omni), not an exotic hack.

**Quantization support and architecture support are different questions.** vLLM 0.27.1
cannot load Gemma 4 31B at all — quantized or not, ours or Google's — because it reads
`config.head_dim` globally on a model whose layers are heterogeneous. Check that the engine
supports the *architecture* before assuming a quantization format is the obstacle. That
check is cheap: it fails on the config alone, with no weights downloaded.

### 8.9 TensorRT-LLM and SGLang
Engine building and its constraints; alternative serving stacks with different quantization
support matrices.

### 8.10 GGUF and llama.cpp
The quant ladder (Q8_0, Q6_K, Q5_K_M, Q4_K_M), `mmproj` files for vision towers, and the
fact that vision towers do not take K-quants.

### 8.11 Tokenizers
BPE/SentencePiece, special tokens, vocabulary size and its effect on embedding size and on
KL computation cost.

### 8.12 Chat templates and instruction tuning
`apply_chat_template`, generation prompts, and the difference between base and IT models.

*Why here:* see §9.8. This is not cosmetic.

### 8.13 generate() and cache classes
Greedy vs sampling vs beam; `DynamicCache`/`StaticCache`; `attn_implementation`
(eager/sdpa/flash_attention_2) and its effect on both speed and exact numerics.

### 8.14 PyTorch mechanics used here
`torch.float8_e4m3fn`, dtype casting rules, `repeat_interleave`, `topk`, `isin`,
forward pre-hooks, `torch.cuda.synchronize`, `empty_cache`, and `no_grad`.

*Why here:* the benchmark counts target forward passes with a hook rather than reading
engine internals, because a hook is portable across engines.

### 8.15 Downloading only the shard you need
`model.safetensors.index.json` maps every tensor to its shard, so before downloading 20 GB
you can ask whether the part you want happens to live in one file.

*Why here:* `shards_for_prefix` in `inspect.py`. GLM-4.6V-Flash's vision tensors sit in a
single shard, which turned a 20 GB download into 4.66 GB and moved that whole workflow onto
a laptop.

### 8.16 The GGUF pipeline
`convert_hf_to_gguf.py` to F16, then `llama-quantize` for each rung of the ladder. Building
llama.cpp CPU-only is correct here — quantization never touches the GPU — and much faster to
compile.

*Why here:* `tools/build_gguf.sh`, plus a gotcha worth memorising: **`-no-cnv` was removed
from llama.cpp**. Without `-st/--single-turn`, `llama-cli` drops into interactive mode and
spins on empty stdin, burning CPU and emitting nothing — a failure that looks exactly like
slow inference.

### 8.17 Publishing, and the Hub's quant-picker widget
Two repos by convention: `<repo>` for safetensors plus the card, `<repo>-GGUF` for
quantizations **at the root**.

The split is not cosmetic. The Hub's hardware-compatibility widget renders only for
root-level GGUF files in a repo containing no safetensors. Put them in a subfolder and the
metadata still indexes, but the widget vanishes and `llama-cli -hf` / `ollama run hf.co/...`
will not resolve them. `upload_hf.py` asserts those three conditions out loud after
uploading, because they are easy to break silently.

Also worth knowing: **model cards are resolved at display time, not commit time.** Fixing an
identity or a card is retroactive across existing commits; you rarely need to rewrite
anything.

---

## 9. Verification methodology

### 9.1 Logits, not text
Generated strings hide differences; logits score every token in the vocabulary.

### 9.2 Bit-exactness as a bar
Extraction does no arithmetic, so a correct extraction reproduces bits exactly. A *tiny*
non-zero difference is the worst outcome: it looks like rounding, ships, and misbehaves
later.

### 9.3 Verification by expansion — **read this one**
Weight-only quantization is exactly simulatable (§3.9), so accuracy can be measured on a
dequantized copy on any hardware. Only throughput needs the target hardware.

### 9.4 Verifying the verifier
Before trusting a dequantizer, check that untouched tensors come back **bit-identical** and
that the scale arithmetic reproduces the original range exactly.

### 9.5 Why not perplexity
Insensitive, dataset-dependent, and easy to look fine while behaviour changed. Distribution
agreement on representative prompts is sharper.

### 9.6 Metrics that complement each other
Worst-case top-1 agreement, top-k overlap, KL, max absolute logit delta. A model can shave
average divergence while flipping more greedy choices — report both.

### 9.7 Comparative bars, not absolute thresholds
Nobody can justify "KL < 0.004", and any threshold that exists will be tuned until the
checkpoint passes. Compare against a build someone already shipped and benchmarked. When a
change *adds* quantization, a strict "no worse" gate is rigged — report the increment.

### 9.8 The degenerate-regime trap — **read this one**
An instruction-tuned model fed a raw completion prompt can collapse into repetition. Scored
there, divergences inflate ~10× and small perturbations move results arbitrarily — a *more*
quantized model can look closer to the reference.

*Why here:* this produced a clean, plausible, entirely wrong result that happened to favour
our own checkpoint.

### 9.9 Control experiments
When output looks wrong, test the **untouched original** the same way before blaming your
pipeline. One check separated "my extraction is broken" from "my prompt is wrong".

### 9.10 Sample size and significance
Six prompts is ~150 token positions — enough for a direction, not a coefficient. A result
that is non-monotonic where physics says it should be monotonic is usually the sample
saying so.

### 9.11 Confounded metrics
Wall-clock throughput mixes acceptance rate, kernel efficiency, and harness overhead.
Choose the metric that isolates the thing under test.

### 9.12 Benchmarks vs divergence
KL over a 262k vocabulary is far more sensitive than multiple-choice accuracy. A large KL
and a near-unchanged GPQA are not contradictory, and reading one as the other nearly
triggered a false alarm here.

### 9.13 Regression tests for bugs found in the field
Every bug in `docs/FINDINGS.md` has a test pinning it. Tests that encode *why* a rule exists
survive refactors better than the rule alone — `test_fp8_attention.py` says in its docstring
that an early run quantized 120 norms, so nobody re-widens the pattern later.

---

## 10. Hardware and systems

### 10.1 CUDA compute capability
`sm_80` (Ampere, bf16), `sm_89`/`sm_90` (Ada/Hopper, FP8), `sm_100` (Blackwell, FP4).
**NVFP4 executes only on Blackwell.**

### 10.2 Tensor cores and what engages them
Both operands must be in the low-precision format. Weight-only quantization does not engage
FP8 tensor cores.

### 10.3 Memory-bandwidth vs compute bound
Decode is bandwidth-bound (one token, all weights read); prefill is compute-bound.

*Why here:* why weight-only FP8 still speeds up decode but not prefill.

### 10.4 GPU memory accounting
Weights + KV cache + activations + fragmentation. Loading models one at a time when two
will not co-reside.

### 10.5 Streaming large checkpoints
Shard-at-a-time processing so peak memory is one shard, not the whole model.

### 10.6 Interconnect and storage
PCIe vs SXM/NVLink, disk throughput, page cache, and why a 95 GB download dominates a short
experiment.

---

## 11. Engineering environment

Most of the failures below reported the **wrong layer**. That is the common thread, and it
is why each entry names the symptom as well as the cause.

### 11.1 Python venvs and `--system-site-packages`
Mixing apt and pip packages breaks ABI-coupled pairs. torch and torchvision must come from
the same build. pip will **skip** installing a package a system copy appears to satisfy.

*Symptom seen here:* `operator torchvision::nms does not exist`, surfacing as
`ModuleNotFoundError: Could not import module 'Gemma4Config'`.

### 11.2 Shell fundamentals
Shebangs — a script without one runs under `/bin/sh`, where bash process substitution is a
syntax error. `nohup`/`setsid` for surviving disconnects. Env assignments must **precede**
the command.

*Seen here:* `exec > >(tee ...)` in a shebang-less script died instantly and left no log,
because the redirect that would have recorded the error *was* the error.

### 11.3 Line endings
CRLF from Windows editors silently corrupts secrets read by shell scripts. `file` diagnoses
it.

### 11.4 Remote GPU workflow
Provisioning APIs, SSH key registration and **verifying it registered**, headless notebook
execution with `nbconvert --execute`, and background job supervision.

### 11.5 Quoting in nested heredocs
Escaping inside a heredoc inside an SSH command mangles f-strings. Write the file locally
and copy it.

### 11.6 Secrets on rented hardware
**Never write a token to disk on a rented machine.** That disk is recycled to the next
tenant, and `chmod 600` protects nothing once the volume is reassigned. Pass secrets through
the environment for the life of the process instead.

*Why here:* `upload_hf.py` states this rule in its own docstring, and it was still violated
during this project — `HF_TOKEN` was written to `~/.hftoken` on two rented instances for
convenience. The instances are terminated; the right response is to rotate the token, not to
assume the disk was wiped.

---

## 12. Repository conventions

### 12.1 Two bars that share no code
`verify` (bit-exact, extraction) and `agreement` (comparative, quantization). Merging them
breaks both: the exact bar rejects every correct quantization, the tolerant bar lets a
broken extraction ship.

### 12.2 Surgery lives in notebooks
Only inspection and verification are library code. Per-model surgery stays in notebooks
rather than becoming a registry of special cases pretending to be an abstraction.

### 12.3 Notebooks are evidence
Committed **with outputs**, because those outputs are the proof of claims that cannot be
regenerated without a large GPU. `tools/check_notebooks.py` fails CI if they are stripped.

### 12.4 Observed state beats declared state
Where a config and a tensor header disagree, the header wins. The config is a claim.

### 12.5 Metadata describes a model, not a directory
Loose files (`README.md`, quantizer dumps) and inherited config fields describe the model
they were written for. Copy them forward and the checkpoint documents something that no
longer exists. Promoting a nested config block to top level also exposes its fields to
validators that never saw them before.

*Why here:* both reached a published repo. A skip set handles the files; for config fields,
the safe test is constructing the config class both ways and asserting every resolved field
matches.

### 12.6 Notebook file structure
nbformat cells, `source` as a list of lines each ending in `\n` except the last, and
`execution_count` as the marker that a cell actually ran.

### 12.7 Never publish a claim you have not run
The model card shipped with `vllm serve --quantization modelopt` as its usage line. It was
written from what the format's documentation implies, not from anything observed, and it did
not work.

*Why here:* documentation is an artifact with the same standard of evidence as a number in a
table. If a command appears in a card, run it first — and if you cannot, say so in the card.

### 12.8 Commit identity and attribution
GitHub resolves commit authorship by **email**. An address not verified on any account
renders as a faceless second person, and a squash-merge whose author email differs from the
account's adds a `Co-authored-by:` trailer — showing one human as two.

*Why here:* the repo's local git config used an unlinked address. Check
`git log --format='%an <%ae>' -1` before pushing.

### 12.9 Project tooling
`ruff` (lint + format), `pytest`, `hatchling`, `pyproject.toml` extras, the `Makefile`
targets, and the CI split between fast checks and notebook validation.

---

## 13. Adjacent topics this repo touches but does not use

### 13.1 Pruning
Structured vs unstructured sparsity; Minitron-style **width** pruning (safe to ~30%) vs
depth pruning (much riskier).

### 13.2 Distillation
Recovering quality after aggressive compression, and why this repo avoids anything needing
retraining.

### 13.3 Modality-specific neurons
MINER (arXiv:2410.04819) found ~2% of FFN neurons are modality-specific but never measured
text-only performance after ablating them. That experiment is still open.

### 13.4 LoRA and adapters
Why quantized bases complicate adapter merging.

---

## Suggested order

1. §2 number formats → §3.1–3.5 quantization → §3.9 simulation
2. §4 transformer architecture → §5 Gemma 4's specifics
3. §8.1, §8.4–8.6 formats — especially compressed-tensors
4. §9 verification methodology — the most transferable part of this repo
5. §7 speculative decoding
6. §1 the maths, when a number looks wrong
7. §10–11 when something breaks

**Papers worth the time:** SmoothQuant and AWQ (activation outliers), GPTQ (error
compensation), the OCP microscaling spec and NVIDIA's NVFP4 material (formats), Leviathan et
al. on speculative decoding (why it is exact), EAGLE (why drafters differ), Minitron
(pruning), and MINER (the open question above).
