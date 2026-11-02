# RoPE check (bf16 GGUF vs safetensors)

## Numbers

| # | prompt (head) | chat tokens | top-1 match | max |logp| diff (shared top-20) |
|---|---------------|-------------|-------------|----------------------------------|
| 1 | 'The capital of France is' | 57 | False | n/a |
| 2 | 'def fibonacci(n):' | 56 | False | n/a |
| 3 | 'Explain why the sky appears blue, in one sentence.' | 64 | False | n/a |
| 4 | '1, 1, 2, 3, 5, 8, 13,' | 73 | False | n/a |
| 5 | 'Translate to German: The weather is cold today.' | 62 | False | n/a |
| 6 | 'The three laws of thermodynamics state that' | 60 | False | n/a |
| 7 | 'The following is a technical description repeated for length' | 1259 | False | n/a |

- **GGUF bf16 PPL (llama-perplexity):** 6.6497
- **Safetensors PPL (HF, llama chunking):** 6.6516
- **Chunks / predictions:** 602 / 153510

## Protocol

- **GGUF:** `llama-server` on bf16 GGUF; `POST /completion` with `n_predict=1`, `temperature=0`, `n_probs=20`.
- **HF:** same user prompts via `apply_chat_template(..., add_generation_prompt=True)`; compare next-token top-20 ids/logprobs.
- **VRAM:** GGUF server stopped before loading HF (single 80GB GPU).
- **No llama-cli stdout parsing.**
- **PPL:** same `wikitext2-test.txt` as task 2; HF uses llama-perplexity non-stride chunking (512 ctx, score last 256 tokens per chunk). Residual gap vs GGUF may include tokenizer path (llama `common_tokenize` vs HF `tokenizer`).
