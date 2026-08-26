---
license: mit
base_model: zai-org/GLM-4.6V-Flash
base_model_relation: finetune
pipeline_tag: text-generation
library_name: transformers
language:
  - en
  - zh
tags:
  - glm
  - glm4
  - text-generation
  - vision-removed
  - pruning
  - gguf
---

# GLM-4.6-Flash-text

`zai-org/GLM-4.6V-Flash` with the vision stack removed. No fine-tuning, no distillation, no retraining. The 181 vision tensors were deleted and the remaining 523 re-keyed onto `Glm4ForCausalLM`.

The text path is bit-identical to the original. See [Verification](#verification).

|  | Original | This model |
|---|---|---|
| Architecture | `Glm4vForConditionalGeneration` | `Glm4ForCausalLM` |
| Parameters | 10,292,777,472 | 9,400,279,040 |
| Tensors | 704 | 523 |
| Size (bf16) | 20.59 GB | 18.80 GB |
| Accepts images | yes | no |

Removed: 892,498,432 params, 8.671% of the model, 1.785 GB.

## Architecture

![Forward path and the cut](architecture.svg)

GLM-4.6V-Flash is a bolted-on vision tower: a 24-layer ViT feeding soft tokens into the decoder's embedding stream via `masked_scatter`. Text tokens never touch a vision weight, so deleting the branch removes one edge from the graph and leaves the text computation alone.

| Vision component | Params |
|---|---|
| `visual.blocks.0–23` (ViT, 1536d, 12 heads) | 679,550,976 |
| `visual.merger` (proj + gate/up/down + norm) | 185,081,856 |
| `visual.downsample` (Conv2d, spatial_merge 2) | 25,169,920 |
| `visual.patch_embed.proj` (Conv3d 14×14×2) | 1,807,872 |
| `visual.embeddings.position_embedding` (576 × 1536) | 884,736 |
| `visual.post_conv_layernorm`, `visual.post_layernorm` | 3,072 |
| Total deleted | 892,498,432 |

What remains is a standard GLM-4 decoder: 40 layers, 4096 hidden, 13696 intermediate, 32 attention heads with 2 KV heads (GQA 16:1), 151552 vocab, 131072 max positions, untied `lm_head`.

## Verification

The real risk here was RoPE. GLM-4.6V-Flash uses multimodal RoPE (`mrope_section: [8, 12, 12]`, summing to 32 = `head_dim 128 × partial_rotary_factor 0.5 ÷ 2`), which splits the rotary budget across temporal, height and width axes. For text-only input all three axes carry the same position index, so mRoPE should collapse to standard RoPE. That's an argument though, not a measurement.

Logits compared against the original on identical text input:

```
max|d|=0.000e+00  argmax_match=True  'The capital of France is'
max|d|=0.000e+00  argmax_match=True  'def fibonacci(n):'
max|d|=0.000e+00  argmax_match=True  'Explain why the sky appears blue, in one sentence.'
max|d|=0.000e+00  argmax_match=True  '1, 1, 2, 3, 5, 8, 13,'
max|d|=0.000e+00  argmax_match=True  'Translate to German: The weather is cold today.'
max|d|=0.000e+00  argmax_match=True  'The three laws of thermodynamics state that'

EXACT MATCH -- mRoPE collapsed to RoPE cleanly. Extraction is lossless.
```

Short prompts don't exercise RoPE at depth, which is where a position-encoding bug would show up, so the same check was run at length:

```
sequence length: 1207
long-context max|d| = 0.000e+00
PASS
```

Zero divergence at 1,207 tokens. On text this is the original model.

## Files

bf16 safetensors at the root, quantizations under `gguf/`.

| File | Format | Size | Notes |
|---|---|---|---|
| `model-0000{1..4}.safetensors` | bf16 | 18.80 GB | reference weights, bit-exact |
| `gguf/GLM-4.6-Flash-text-F16.gguf` | F16 | 18.81 GB | lossless GGUF, requantize from this |
| `gguf/GLM-4.6-Flash-text-Q8_0.gguf` | Q8_0 | 10.00 GB | near-lossless |
| `gguf/GLM-4.6-Flash-text-Q6_K.gguf` | Q6_K | 8.27 GB | very high quality |
| `gguf/GLM-4.6-Flash-text-Q5_K_M.gguf` | Q5_K_M | 7.05 GB | high quality |
| `gguf/GLM-4.6-Flash-text-Q4_K_M.gguf` | Q4_K_M | 6.17 GB | recommended, best size/quality tradeoff |

All five load and generate coherently under llama.cpp (`architecture: glm4`, 131072 context). Sizes are GB (10⁹ bytes) as the Hub reports them; `ls -h` will show smaller GiB numbers for the same files.

If you only want GGUF, the same quantizations sit at the repo root of [sartajbhuvaji/GLM-4.6-Flash-text-GGUF](https://huggingface.co/sartajbhuvaji/GLM-4.6-Flash-text-GGUF), where the Hub's quantization picker renders and `llama-cli -hf` / `ollama run hf.co/…` resolve directly. The `gguf/` copies here are identical, so use whichever is convenient.

There is no NVFP4 build. NVFP4 is NVIDIA's Blackwell format (E2M1, 16-element blocks, FP8 E4M3 block scales) and needs SM100+ hardware to quantize and serve. It is not a GGUF quant and can't be produced on Ampere. If you want one, run [LLM Compressor](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/) on a B200 with the bf16 weights here as input.

## Usage

Every command below was run end to end on a single A100-SXM4-40GB at the versions pinned in each block. Measured numbers are in [Benchmarks](#benchmarks).

| Stack | Status | Best at |
|---|---|---|
| vLLM 0.27.1 | tested | fastest here on both throughput and latency |
| SGLang 0.5.18 | tested | within a few percent, slightly faster single-stream decode |
| transformers 5.15.1 | tested | single-GPU scripting, research |
| llama.cpp (`b10595`) | tested | CPU/consumer GPU, quantized |

### vLLM

```bash
pip install vllm==0.27.1 ninja

vllm serve sartajbhuvaji/GLM-4.6-Flash-text \
  --served-model-name GLM-4.6-Flash-text \
  --max-model-len 32768 \
  --reasoning-parser glm45 \
  --tool-call-parser glm45 --enable-auto-tool-choice
```

That works as-is. It pulls only the 18 GB of safetensors and ignores the `gguf/` folder in this repo. Serving takes ~120 s from nothing on a fast link including download, ~45 s with weights cached and the compile cache warm. At `--max-model-len 32768` on a 40 GB card there's still a 479,280-token KV cache left, so raise the context freely.

If you hit `FileNotFoundError: ninja`: vLLM's compile path shells out to `ninja`, and the failure surfaces three frames deep as `RuntimeError: Engine core initialization failed`, with the real cause buried far above it in the log. `pip install ninja` fixes it, but only if `ninja` is on your `PATH`. Invoking `/path/to/venv/bin/vllm` directly does not put that venv's `bin` on `PATH`; only activating the venv does.

### SGLang

```bash
pip install "sglang[all]==0.5.18"

python -m sglang.launch_server \
  --model-path sartajbhuvaji/GLM-4.6-Flash-text \
  --served-model-name GLM-4.6-Flash-text \
  --context-length 32768 \
  --reasoning-parser glm45 --tool-call-parser glm45 \
  --host 0.0.0.0 --port 30000
```

`--reasoning-parser auto` also works, reading the parser choice off the chat template.

### Calling either server

Both expose the OpenAI API, so the same client works against either. Only the port differs (vLLM 8000, SGLang 30000).

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "GLM-4.6-Flash-text",
    "messages": [{"role": "user", "content": "What is a mixture-of-experts layer?"}],
    "max_tokens": 900
  }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
r = client.chat.completions.create(
    model="GLM-4.6-Flash-text",
    messages=[{"role": "user", "content": "What is a mixture-of-experts layer?"}],
    max_tokens=900,
)
print(r.choices[0].message.content)
```

Tool calling works on both with the `glm45` tool parser above. Pass `tools=[...]` and read `message.tool_calls` as usual.

### transformers

```bash
pip install torch==2.13.0 transformers==5.15.1 accelerate
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "sartajbhuvaji/GLM-4.6-Flash-text", dtype=torch.bfloat16, device_map="auto"
)
tok = AutoTokenizer.from_pretrained("sartajbhuvaji/GLM-4.6-Flash-text")

msgs = [{"role": "user", "content": "What is a mixture-of-experts layer?"}]
enc = tok.apply_chat_template(
    msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
out = model.generate(**enc, max_new_tokens=900)
print(tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True))
```

Needs a transformers version carrying `Glm4ForCausalLM` (v4.52+). Keep `max_new_tokens` around 900, see [Output format](#output-format).

### llama.cpp

Use the [GGUF repo](https://huggingface.co/sartajbhuvaji/GLM-4.6-Flash-text-GGUF), where the files sit at the root and `-hf` resolves them:

```bash
llama-cli -hf sartajbhuvaji/GLM-4.6-Flash-text-GGUF:Q4_K_M \
  -p "Explain gradient descent" -n 900 -ngl 99 -st
```

`-st` (`--single-turn`) matters for scripted use. Without it `llama-cli` drops into interactive mode and waits on stdin, which looks like a hung GPU. The older `-no-cnv` flag has been removed. `-ngl 99` offloads every layer to the GPU.

To use the copies in this repo instead, download by explicit path, since `-hf` only resolves root-level GGUFs:

```bash
hf download sartajbhuvaji/GLM-4.6-Flash-text \
  gguf/GLM-4.6-Flash-text-Q4_K_M.gguf --local-dir .
llama-cli -m gguf/GLM-4.6-Flash-text-Q4_K_M.gguf -p "Explain gradient descent" -n 900 -ngl 99 -st
```

There is no prebuilt Linux CUDA binary; llama.cpp publishes CUDA archives for Windows only. On Linux, build it:

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_CURL=ON
cmake --build build -j --target llama-cli llama-server
```

(`80` is A100, `89` for L40S/4090, `90` for H100.)

### Ollama

```bash
ollama run hf.co/sartajbhuvaji/GLM-4.6-Flash-text-GGUF:Q4_K_M
```

## Output format

This is a reasoning model. Its output has a few quirks.

It thinks first, at length, and often in Chinese. Every answer is preceded by a `<think>…</think>` block, frequently in Chinese no matter what language you prompted in. Budget for it: `max_tokens=160` reliably returns a truncated monologue with no answer in it, which looks like a broken model. 900 is a safe default.

The final answer is wrapped in `<|begin_of_box|>…<|end_of_box|>`, a GLM convention that survives into this checkpoint. No parser strips it, so strip it yourself:

```python
import re

answer = re.sub(r"<\|(begin|end)_of_box\|>", "", content).strip()
```

The reasoning field has two different names. With a reasoning parser enabled the thinking is split out of `content` into its own field, but vLLM 0.27.1 calls it `message.reasoning` while SGLang calls it `message.reasoning_content`. Client code that hardcodes one silently drops the reasoning on the other. Read both:

```python
m = r.choices[0].message
reasoning = getattr(m, "reasoning", None) or getattr(m, "reasoning_content", None)
```

Turning thinking off works on both servers via the chat template, and makes a large difference: the France question drops from 53 tokens to 2.

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

## Benchmarks

Measured on 1× A100-SXM4-40GB, 512-token prompts, 256-token outputs, greedy, with `ignore_eos` pinning every request to exactly 256 output tokens so the runs are comparable. vLLM and SGLang were driven by the same client, so the gap between them isn't a difference in benchmark tooling.

| Engine | Concurrency | Output tok/s | Per stream | TTFT p50 | TTFT p99 | TPOT |
|---|---|---|---|---|---|---|
| vLLM 0.27.1 | 1 | 61.9 | 61.9 | 40.3 ms | 40.6 ms | 16.06 ms |
| vLLM 0.27.1 | 16 | 996.0 | 62.2 | 87.3 ms | 100.9 ms | 15.77 ms |
| SGLang 0.5.18 | 1 | 63.4 | 63.4 | 68.2 ms | 68.8 ms | 15.55 ms |
| SGLang 0.5.18 | 16 | 981.4 | 61.3 | 90.9 ms | 97.3 ms | 16.00 ms |
| transformers 5.15.1 | 1 | 19.5 | 19.5 | 74.5 ms | — | 51.17 ms |

vLLM and SGLang land within 1.5% of each other on batched throughput. vLLM's edge is TTFT at low concurrency, about 1.7× faster off the line, while SGLang is marginally faster at single-stream decode (63.4 vs 61.9 tok/s). Either is a defensible choice; vLLM is the default recommendation on these numbers. transformers is ~3× slower per stream and does no continuous batching, which is fine for scripting and wrong for serving.

One caveat, because it changed the conclusion. Each engine ships its own benchmark tool, and run against those, SGLang appeared to win TTFT by 3×. They disagree on prompt sampling, warmup and what counts as "duration", so their numbers aren't comparable to each other. The table above comes from one client hitting both servers over the same HTTP path, which reversed the result.

llama.cpp is measured separately because it's a different quantization (`llama-bench`, Q4_K_M, all layers offloaded):

| Quant | Prefill | Decode |
|---|---|---|
| Q4_K_M | 4,097 tok/s | 128.7 tok/s |

Q4_K_M decodes about 2× faster than bf16 on the same card at a third of the memory, the usual quantization trade, and the reason it's the recommended file for single-stream use.

Cold startup (first run, no compile cache): vLLM ~140 s, SGLang ~140 s. Warm: ~45 s and ~55 s. transformers loads the weights in 5 s.

## Reproducing this

```python
from safetensors.torch import load_file, save_file

# 1. drop every tensor under model.visual.  (181 tensors, 892,498,432 params)
# 2. rename  model.language_model.*  ->  model.*
# 3. keep lm_head.weight -- it is UNTIED (tie_word_embeddings: false)
sd = {
    k.replace("model.language_model.", "model.", 1): v
    for k, v in load_file(shard).items()
    if not k.startswith("model.visual.")
}
```

Then rebuild the config through `Glm4Config`, dropping `mrope_section` and `vision_config`, and set `architectures = ["Glm4ForCausalLM"]`. `Glm4Config` does not populate that field, and without it `AutoModelForCausalLM` has nothing to dispatch to.

Vocabulary needs no work: `vocab_size` stays 151552, and the image/video token ids (151363/151364) remain in it. They're simply never emitted.

## Limitations

- No vision. Passing images does nothing, since the tokens have no embedder behind them. Use the [original model](https://huggingface.co/zai-org/GLM-4.6V-Flash) if you need multimodal.
- Reasoning model. It emits `<think>` blocks before answering, sometimes in Chinese regardless of prompt language. Budget `max_new_tokens` accordingly; 160 is not enough to get past the reasoning to an answer.
- Inherits everything else from GLM-4.6V-Flash, including its biases and knowledge cutoff. Text behaviour is bit-identical, so any evaluation of the original's text ability transfers exactly.
- Quantization is not verified bit-exact. The bit-exactness result above covers the bf16 weights only. The GGUF quants are lossy by construction; each was checked to load and generate coherent text under llama.cpp, but no perplexity or benchmark comparison against bf16 was run. If you need a measured quality delta, compute it yourself.

## Provenance

Derived from [zai-org/GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) (MIT). This model is MIT as well.

Conversion and verification were run on a single A100-SXM4-40GB with transformers 5.16.0.dev0 and torch 2.7.0. GGUF builds used llama.cpp at master with the `GLM4` architecture.
