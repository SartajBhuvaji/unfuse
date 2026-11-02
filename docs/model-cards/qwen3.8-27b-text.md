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

`Qwen/Qwen3.8-27B` with the vision stack removed. No fine-tuning, no distillation, no retraining. The 333 vision tensors were deleted and the remaining 866 re-keyed onto `Qwen3_5ForCausalLM`.

The text path is bit-identical to the original. See [Verification](#verification).

|  | Original | This model |
|---|---|---|
| Architecture | `Qwen3_5ForConditionalGeneration` | `Qwen3_5ForCausalLM` |
| Parameters | 27,781,427,952 | 27,320,697,856 |
| Tensors | 1,199 | 866 |
| Size (bf16) | 55.56 GB | 54.64 GB |
| Accepts images | yes | no |

![architecture](architecture.svg)

## What you get

The vision tower is 1.658% of this checkpoint. Removing it saves 0.92 GB out of 55.56. If you want a smaller model, this isn't one.

The useful part is what that GB turns into. Measured on one H100 80GB under vLLM 0.27.1:

| | Original | This model | Δ |
|---|---|---|---|
| Model weights in VRAM | 51.10 GiB | 50.13 GiB | −0.97 GiB (−1.9%) |
| KV cache available | 18.27 GiB | 19.84 GiB | +1.57 GiB (+8.6%) |
| KV cache tokens | 277,435 | 301,465 | +8.7% |
| Max concurrency @ 32k ctx | 8.47× | 9.20× | +8.6% |

So: about 8.7% more concurrent requests on the same card, plus a plain causal-LM config with no vision dependencies, no image preprocessing, and no multimodal code path to work around.

## Architecture

Qwen3.8-27B is a hybrid, not a plain transformer. Its 64 decoder layers are 48 `linear_attention` and 16 `full_attention`, one full-attention layer every fourth (`full_attention_interval: 4`), with `linear_conv_kernel_dim: 4` and `mamba_ssm_dtype: float32`. This affects serving, see [Required flags](#required-flags).

Deleted: `model.visual.*`, 333 tensors, 460,730,096 params. That covers the 27-layer ViT (1152d, 16 heads, patch 16), the patch embedding, and the merger that projects 1152 → 5120.

`deepstack_visual_indexes` in the original `vision_config` names several tower layers whose features get injected into the language model, which looks like the vision branch touches the text spine at several depths. It doesn't. No tensor carries a deepstack name, and all 1,199 tensors fall into `model.visual.`, `model.language_model.`, `lm_head` and `mtp.` with nothing left over. The indices resolve inside the visual tree and the cut is a single edge.

### MTP head

The original carries a fused multi-token-prediction speculative head: `mtp.*`, 15 tensors, 424,699,392 params, 0.85 GB. It's kept here, but nothing can currently use it.

| Loader | Behaviour |
|---|---|
| transformers 5.15 | loads, ignores it, reports 26,895,998,464 params |
| vLLM 0.27.1, `model_type: qwen3_5_text` | `NotImplementedError: Unsupported speculative method: 'mtp'` |
| vLLM 0.27.1, `model_type` forced to `qwen3_5` | resolves `Qwen3_5MTP`, then fails on a shape mismatch (5120 vs 4096) |

vLLM's MTP auto-detect only fires for `model_type in ("qwen3_5", "qwen3_5_moe")`, and that path reads the multimodal config layout, which expects a nested `text_config`. A flattened text-only config can't reach it either way.

Kept because it costs 0.85 GB, breaks nothing, and can't be added back later without re-downloading 55 GB.

## Verification

The split drops tensors and renames the survivors. No arithmetic changes. So it's verified by byte equality on every kept tensor rather than by comparing logits on sample prompts: identical bytes means identical output for any input, where matching logits only covers the prompts you tried.

```
checked 866 tensors, 54.64 GB
PASS - every kept tensor is byte-identical to its source.
```

Run three times, once locally and twice on a rented H100, each from a fresh download of the original. All three gave 27,320,697,856 params / 54.64 GB and passed.

Runtime check on H100 80GB:

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

512 input / 256 output tokens, greedy, `ignore_eos`, both engines driven through the same OpenAI-compatible streaming client. Every row is warmed: one pass discarded before measuring. Raw JSON and the environment capture are in the [unfuse repo](https://github.com/SartajBhuvaji/unfuse).

1× NVIDIA H100 80GB HBM3, sm_90, driver 580.105.08, CUDA 13.0 (nvcc 12.8), 700 W, torch 2.7.0, Ubuntu 24.04.4.

| Engine | Model | c | Output tok/s | Per stream | TTFT p50 | TTFT p99 | TPOT |
|---|---|---|---|---|---|---|---|
| vLLM 0.27.1 | original | 1 | 47.4 | 47.4 | 69.0 ms | 69.3 ms | 20.42 ms |
| vLLM 0.27.1 | this model | 1 | 46.6 | 46.6 | 69.0 ms | 69.2 ms | 21.28 ms |
| vLLM 0.27.1 | original | 16 | 635.8 | 39.7 | 657.2 ms | 669.5 ms | 22.70 ms |
| vLLM 0.27.1 | this model | 16 | 554.7 | 34.7 | 654.1 ms | 673.7 ms | 26.04 ms |
| SGLang 0.5.18 | original | 1 | 48.8 | 48.8 | 63.1 ms | 63.2 ms | 20.25 ms |
| SGLang 0.5.18 | this model | 1 | 47.2 | 47.2 | 69.5 ms | 69.6 ms | 20.91 ms |
| SGLang 0.5.18 | original | 16 | 225.3 | 14.1 | 10,585 ms | 15,858 ms | 21.62 ms |
| SGLang 0.5.18 | this model | 16 | 628.5 | 39.3 | 714.8 ms | 719.3 ms | 22.94 ms |

### Notes on the numbers

This model is not faster than the original, and any claim that it is has been measured on one engine only. On vLLM the original wins by 1.15× at concurrency 16. On SGLang this model wins by 2.79×, but only because SGLang's memory planner gave the original `max_running_requests=5` against this model's `16` and then starved it. The planner caps both by the size of the Mamba state cache (28 slots against 48), so this is the default behaviour of the original on SGLang, not a difference in speed per request. Setting the flag to 32 on the original does not lift the cap.

A later session on the same GPU class did not reproduce the vLLM gap. With `--max-num-seqs` at 64, 128, 256 and 320, vLLM read 360 to 402 tok/s at concurrency 16 for both models, with the two level at every setting, and the 555 to 636 tok/s above did not come back. Why that earlier run read differently was not found, so treat the 1.15× vLLM gap as unconfirmed. Long inputs showed no throughput difference either: 97.0 against 95.0 tok/s at 8k and 26.3 against 26.4 at 32k, concurrency 16, even though this model holds 2% to 9% more KV cache there.

What holds across both engines is the spread:

| | vLLM | SGLang | spread |
|---|---|---|---|
| Original @ c=16 | 635.8 tok/s | 225.3 tok/s | 2.82× |
| This model @ c=16 | 554.7 tok/s | 628.5 tok/s | 1.13× |

The original's throughput moves by 2.8× depending on which engine you run. This model's moves by 1.13×. That predictability, plus the cache headroom above, is the practical case for the split.

At concurrency 1 all four runs land between 46.6 and 48.8 tok/s, which is expected: the same 64 text layers do the work in every case.

One warning if you benchmark this yourself. Warmup moves the result by 21%: an unwarmed first run of this model on vLLM gave 457.9 tok/s where the warmed run gave 554.7. Discard a pass first, and don't compare a warmed row against an unwarmed one.

### Bandwidth limit

At concurrency 1 a dense model is memory-bandwidth bound. Every weight is read from HBM once per token, so the bus sets a ceiling no engine can beat.

| | Weights | Bytes/token ÷ 3.35 TB/s | Ceiling | Best measured | Of ceiling |
|---|---|---|---|---|---|
| Original | 55.56 GB | 16.59 ms | 60.3 tok/s | 48.8 tok/s | ~81% |
| This model | 54.64 GB | 16.31 ms | 61.3 tok/s | 47.2 tok/s | ~77% |

An H100 SXM5 has 3.35 TB/s of HBM3. Reading 54.64 GB of weights costs at least 16.31 ms per decoded token, a floor of 61.3 tok/s, and the measured TPOT of 20.91 ms sits at 78% of that.

Which means single-stream decode here is already close to the hardware limit. There's roughly 20-25% of headroom and no server flag will find more than that; faster single-stream generation needs more bandwidth or fewer bytes, i.e. a quantized build. It also explains the flat c=1 column above: every configuration is moving a near-identical number of bytes over the same bus, and the engines only diverge once concurrency amortises those reads.

Treat this as ±10% rather than a precise efficiency figure. It counts weight traffic only, ignores KV-cache reads, and the 48 linear-attention layers don't access memory the way the 16 full-attention ones do.

## Required flags

vLLM will not start this model at default settings:

```
ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (362).
Each decode sequence requires one Mamba cache block, so CUDA graph capture cannot proceed.
```

The 48 linear-attention layers need one Mamba cache block per concurrent sequence, which is a separate limit from KV cache. Pass `--max-num-seqs 256`. The commands below already do.

## Usage

| Stack | Status | Notes |
|---|---|---|
| vLLM 0.27.1 | tested | needs `--max-num-seqs` |
| SGLang 0.5.18 | tested | best throughput at concurrency here |
| transformers 5.15.1 | tested | single-GPU scripting, research |
| llama.cpp | tested | GGUF quants in [unfuse-Qwen3.8-27B-text-GGUF](https://huggingface.co/sartajbhuvaji/unfuse-Qwen3.8-27B-text-GGUF); vision projector in [unfuse-Qwen3.8-27B-mmproj-GGUF](https://huggingface.co/sartajbhuvaji/unfuse-Qwen3.8-27B-mmproj-GGUF) |

Needs ~55 GB of VRAM in bf16, so one 80 GB card (H100/H200/A100-80GB), or shard it with `--tensor-parallel-size`.

### vLLM

```bash
pip install "vllm>=0.27.1"

vllm serve sartajbhuvaji/unfuse-Qwen3.8-27B-text \
  --max-model-len 32768 \
  --max-num-seqs 256
```

Cold start on an H100 is about 75 s. On two smaller cards add `--tensor-parallel-size 2`.

### SGLang

```bash
pip install "sglang[all]>=0.5.18"

python -m sglang.launch_server \
  --model-path sartajbhuvaji/unfuse-Qwen3.8-27B-text \
  --port 8001 \
  --context-length 32768 \
  --max-running-requests 256
```

### Calling either server

Both expose an OpenAI-compatible API, so the same client works against either. Only the port changes (vLLM 8000, SGLang 8001).

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sartajbhuvaji/unfuse-Qwen3.8-27B-text",
    "messages": [{"role": "user", "content": "Write a Python one-liner that reverses a string."}],
    "max_tokens": 256,
    "temperature": 0
  }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

resp = client.chat.completions.create(
    model="sartajbhuvaji/unfuse-Qwen3.8-27B-text",
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

model_id = "sartajbhuvaji/unfuse-Qwen3.8-27B-text"
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

Loading reports 26,895,998,464 parameters rather than the 27,320,697,856 on disk. That gap is the MTP head, which transformers has no class for. Expected, and harmless.

## Output format

The model emits reasoning before its answer, closed by `</think>`:

```
We need answer just number. Compute 84*3/2 = 252/2=126. Final only 126.
</think>

126
```

Split on `</think>` and take the last part if you only want the answer.

## Provenance

- Original: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0)
- Surgery, verification and benchmark harness: [github.com/SartajBhuvaji/unfuse](https://github.com/SartajBhuvaji/unfuse)
- Method: drop `model.visual.*` (333 tensors), rename `model.language_model.` to `model.`, keep the untied `lm_head`, rebuild the config from `text_config` via `Qwen3_5TextConfig` with `architectures = ["Qwen3_5ForCausalLM"]`

## Limitations

- Barely smaller: 54.64 GB against 55.56 GB. The gain is cache headroom and cross-engine predictability, not footprint.
- No vision. Send it an image and it will fail. Use the original.
- The MTP head is inert on every loader tested, so 0.85 GB of the download does nothing today.
- GGUF quants (Q8_0, Q5_K_M, Q4_K_M) are in a separate repo; see the table above. Perplexity there is within 0.7% of bf16.
- Benchmarks cover one GPU and one shape: 512-in/256-out on a single H100 80GB at concurrency 1 and 16. Other context lengths, batch shapes or hardware will rank the engines differently.
- `rope_parameters` still carries `mrope_section` and `mrope_interleaved` from the multimodal parent. vLLM logs `Unrecognized keys in rope_parameters`. Harmless, will be removed.
