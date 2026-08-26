---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: finetune
pipeline_tag: text-generation
library_name: transformers
language:
  - en
  - zh
tags:
  - qwen
  - qwen3_5
  - text-generation
  - vision-removed
  - hybrid-attention
  - linear-attention
---

# Qwen3.8-27B-text

`Qwen/Qwen3.8-27B` with the vision stack surgically removed. **No fine-tuning, no distillation, no retraining** — the 333 vision tensors were deleted and the remaining 866 were re-keyed onto the `Qwen3_5ForCausalLM` architecture.

The text path is **bit-identical** to the original. Verified, not assumed — see [Verification](#verification).

|  | Original | This model |
|---|---|---|
| Architecture | `Qwen3_5ForConditionalGeneration` | `Qwen3_5ForCausalLM` |
| Parameters | 27,781,427,952 | **27,320,697,856** |
| Tensors | 1,199 | **866** |
| Size (bf16) | 55.56 GB | **54.64 GB** |
| Accepts images | yes | no |

![architecture](architecture.svg)

## Why this exists

Being blunt about the tradeoff, because the headline number is small: **the vision tower is only 1.658% of this checkpoint.** Removing it saves 0.92 GB out of 55.56. If you came for a smaller model, this is not it.

What you get instead, measured on one H100 80GB:

| vLLM 0.27.1 | Original | This model | Δ |
|---|---|---|---|
| Model weights in VRAM | 51.10 GiB | **50.13 GiB** | −0.97 GiB (−1.9%) |
| KV cache available | 18.27 GiB | **19.84 GiB** | **+1.57 GiB (+8.6%)** |
| KV cache tokens | 277,435 | **301,465** | **+8.7%** |
| Max concurrency @ 32k ctx | 8.47× | **9.20×** | **+8.6%** |

The weight saving is small and converts almost entirely into cache headroom. **+8.7% more concurrent requests on the same card** is the number worth caring about — that, plus a plain causal-LM config with no vision dependencies, no image preprocessing, and no multimodal code path to configure around.

## Architecture

Qwen3.8-27B is a **hybrid**, not a plain transformer. Its 64 decoder layers are 48 `linear_attention` and 16 `full_attention`, one full-attention layer every fourth (`full_attention_interval: 4`), with `linear_conv_kernel_dim: 4` and `mamba_ssm_dtype: float32`.

This matters for serving — see [the `--max-num-seqs` trap](#the-one-flag-you-must-pass).

| Vision component | Tensors | Params |
|---|---|---|
| `model.visual.blocks.0–26` (ViT, 1152d, 16 heads, patch 16) | — | — |
| `model.visual.patch_embed`, `merger` (1152 → 5120) | — | — |
| **Total deleted — `model.visual.*`** | **333** | **460,730,096** |

`deepstack_visual_indexes` in the original `vision_config` names several tower layers whose features are injected into the language model, which reads like the vision branch touches the text spine at multiple depths. **It does not.** Zero tensors carry a deepstack name; all 1,199 tensors fall cleanly into `model.visual.`, `model.language_model.`, `lm_head` and `mtp.`, with nothing unaccounted for. The indices resolve inside the visual tree, and the cut is a single edge.

### The MTP head is retained but currently unusable

The original ships a fused multi-token-prediction speculative head — `mtp.*`, 15 tensors, **424,699,392 params, 0.85 GB**. It is **kept** in this repo.

It does nothing today, and you should know that going in:

| Loader | Behaviour |
|---|---|
| transformers 5.15 | loads fine, silently ignores it — reports 26,895,998,464 params |
| vLLM 0.27.1, `model_type: qwen3_5_text` | `NotImplementedError: Unsupported speculative method: 'mtp'` |
| vLLM 0.27.1, `model_type` forced to `qwen3_5` | resolves `Qwen3_5MTP`, then fails on a shape mismatch (5120 vs 4096) |

vLLM's MTP auto-detect fires only for `model_type in ("qwen3_5", "qwen3_5_moe")`, and that path then reads the *multimodal* config layout expecting a nested `text_config`. A flattened text-only config cannot reach it either way.

Kept anyway: it costs 0.85 GB, breaks nothing, and cannot be recovered later without re-downloading 55 GB. If an engine gains support, it is already here.

## Verification

The split drops tensors and renames the survivors. It changes no arithmetic. So this repo is verified by **byte equality on every kept tensor**, not by comparing logits on sample prompts.

That is the stronger claim. Byte equality means *every* input produces identical output; matching logits means they matched on the prompts you tried.

```
checked 866 tensors, 54.64 GB
PASS — every kept tensor is byte-identical to its source.
```

Run three times — once locally, twice on a rented H100, each from a fresh download of the original. All three produced 27,320,697,856 params / 54.64 GB and passed.

Runtime, on H100 80GB:

```
LOADED 8.9s as Qwen3_5ForCausalLM
params: 26,895,998,464          # 424,699,392 short of the checkpoint: the ignored MTP head
vram_GB: 53.79

[PASS] What is the capital of France?          -> 'Paris'
[PASS] What is 84 * 3 / 2?                     -> '126'
[PASS] Translate to German: ...cold today.     -> 'Das Wetter ist heute kalt.'
[PASS] Python one-liner to reverse a string s. -> 's[::-1]'
CORRECTNESS 4/4
```

## Benchmarks

512 input / 256 output tokens, greedy, `ignore_eos`, driven through **one shared OpenAI-compatible streaming client** so both engines are measured identically. Every row is **warmed** — one benchmark pass discarded before measuring. Full JSON and the environment block are in the [unfuse repo](https://github.com/SartajBhuvaji/unfuse).

**1× NVIDIA H100 80GB HBM3** · sm_90 · driver 580.105.08 · CUDA 13.0 (nvcc 12.8) · 700 W · torch 2.7.0 · Ubuntu 24.04.4

| Engine | Model | c | Output tok/s | Per stream | TTFT p50 | TTFT p99 | TPOT |
|---|---|---|---|---|---|---|---|
| vLLM 0.27.1 | original | 1 | 47.4 | 47.4 | 69.0 ms | 69.3 ms | 20.42 ms |
| vLLM 0.27.1 | **this model** | 1 | 46.6 | 46.6 | 69.0 ms | 69.2 ms | 21.28 ms |
| vLLM 0.27.1 | original | 16 | **635.8** | 39.7 | 657.2 ms | 669.5 ms | 22.70 ms |
| vLLM 0.27.1 | **this model** | 16 | 554.7 | 34.7 | 654.1 ms | 673.7 ms | 26.04 ms |
| SGLang 0.5.18 | original | 1 | 48.8 | 48.8 | 63.1 ms | 63.2 ms | 20.25 ms |
| SGLang 0.5.18 | **this model** | 1 | 47.2 | 47.2 | 69.5 ms | 69.6 ms | 20.91 ms |
| SGLang 0.5.18 | original | 16 | 225.3 | 14.1 | 10,585 ms | 15,858 ms | 21.62 ms |
| SGLang 0.5.18 | **this model** | 16 | **628.5** | 39.3 | 714.8 ms | 719.3 ms | 22.94 ms |

### How to read this

**This model is not faster, and anyone telling you otherwise is quoting one engine.** On vLLM the original is 1.15× faster at concurrency 16. On SGLang this model is 2.79× faster — but only because SGLang's memory planner allocated the original `max_running_requests=5` against this model's `16`, then starved it. That is an engine artifact, not a property of either checkpoint.

What survives both engines is **variance**:

| | vLLM | SGLang | spread |
|---|---|---|---|
| Original @ c=16 | 635.8 tok/s | 225.3 tok/s | **2.82×** |
| **This model @ c=16** | 554.7 tok/s | 628.5 tok/s | **1.13×** |

The original's throughput swings 2.8× depending on which engine you pick. This model's swings 1.13×. **Predictability is what removing the vision tower actually buys you**, alongside the cache headroom above.

At concurrency 1 all four runs land between 46.6 and 48.8 tok/s. The same 64 text layers do the work in every case — which is the sanity check confirming the surgery did not touch the compute path.

⚠️ **Warmup moves these numbers by 21%.** An unwarmed first run of this model on vLLM gave 457.9 tok/s where the warmed run gave 554.7. If you benchmark this yourself, discard a pass first, and never compare a warmed row against an unwarmed one.

## The one flag you must pass

**vLLM will not start this model at default settings:**

```
ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (362).
Each decode sequence requires one Mamba cache block, so CUDA graph capture cannot proceed.
```

This is the hybrid stack asserting itself — the 48 linear-attention layers need one Mamba cache block **per concurrent sequence**, a limit entirely separate from KV cache. Pass `--max-num-seqs 256`. Every command below already does.

## Usage

| Stack | Status | Notes |
|---|---|---|
| vLLM 0.27.1 | ✅ tested | needs `--max-num-seqs` |
| SGLang 0.5.18 | ✅ tested | best throughput at concurrency here |
| transformers 5.15.1 | ✅ tested | single-GPU scripting, research |
| llama.cpp | ❌ not yet | no `Qwen3_5` in `convert_hf_to_gguf.py`; the hybrid stack is not convertible today |

Needs ~55 GB of VRAM in bf16 — one 80 GB card (H100/H200/A100-80GB), or shard it with `--tensor-parallel-size`.

### vLLM

```bash
pip install "vllm>=0.27.1"

vllm serve sartajbhuvaji/Qwen3.8-27B-text \
  --max-model-len 32768 \
  --max-num-seqs 256
```

Cold start on an H100 is ~75 s. On two smaller cards add `--tensor-parallel-size 2`.

### SGLang

```bash
pip install "sglang[all]>=0.5.18"

python -m sglang.launch_server \
  --model-path sartajbhuvaji/Qwen3.8-27B-text \
  --port 8001 \
  --context-length 32768 \
  --max-running-requests 256
```

### Calling either server

Both expose an OpenAI-compatible API, so the same client works — only the port changes (vLLM 8000, SGLang 8001).

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sartajbhuvaji/Qwen3.8-27B-text",
    "messages": [{"role": "user", "content": "Write a Python one-liner that reverses a string."}],
    "max_tokens": 256,
    "temperature": 0
  }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

resp = client.chat.completions.create(
    model="sartajbhuvaji/Qwen3.8-27B-text",
    messages=[{"role": "user", "content": "Explain a mutex vs a semaphore."}],
    max_tokens=512,
    temperature=0,
)
print(resp.choices[0].message.content)
```

### transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "sartajbhuvaji/Qwen3.8-27B-text"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")

enc = tok.apply_chat_template(
    [{"role": "user", "content": "What is 84 * 3 / 2? Reply with just the number."}],
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,  # transformers 5.x returns a BatchEncoding, not a tensor
).to("cuda")

out = model.generate(**enc, max_new_tokens=96, do_sample=False)
print(tok.decode(out[0][enc["input_ids"].shape[-1] :], skip_special_tokens=True))
```

Loading reports **26,895,998,464** parameters rather than the 27,320,697,856 on disk. That gap is exactly the MTP head, which transformers has no class for. It is expected and harmless.

## Reading the output

This model emits reasoning traces before its answer, closed by `</think>`:

```
We need answer just number. Compute 84*3/2 = 252/2=126. Final only 126.
</think>

126
```

Split on `</think>` and take the last part if you only want the answer.

## Provenance

- Original: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0)
- Surgery, verification and benchmark harness: [github.com/SartajBhuvaji/unfuse](https://github.com/SartajBhuvaji/unfuse)
- Method: drop `model.visual.*` (333 tensors), rename `model.language_model.` → `model.`, retain untied `lm_head`, rebuild the config from `text_config` via `Qwen3_5TextConfig` with `architectures = ["Qwen3_5ForCausalLM"]`

## Limitations

- **Not smaller in any way you will notice.** 54.64 GB against 55.56 GB. The gain is cache headroom and cross-engine predictability, not footprint.
- **Cannot see.** Send it an image and it will fail — use the original.
- **The MTP head is inert** on every loader tested. 0.85 GB of the download does nothing today.
- **No GGUF / llama.cpp path** until `convert_hf_to_gguf.py` learns `Qwen3_5`.
- **Benchmarks are one GPU, one shape.** 512-in/256-out on a single H100 80GB at concurrency 1 and 16. Different context lengths, batch shapes or hardware will rank the engines differently — that is the entire point of the variance section above.
- `rope_parameters` still carries `mrope_section` and `mrope_interleaved` inherited from the multimodal parent. vLLM logs `Unrecognized keys in rope_parameters`; harmless, and slated for removal.
