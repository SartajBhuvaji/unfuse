#!/usr/bin/env python3
"""RoPE: bf16 GGUF (llama-server /completion) vs safetensors (transformers)."""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from unfuse.verify import DEFAULT_PROMPTS, long_prompt  # noqa: E402

SERVER = "http://127.0.0.1:8080/completion"
WIKITEXT = ROOT / "notebooks/qwen3.8-27b/gguf/wikitext2-test.txt"


def wait_server(timeout: int = 600) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError("llama-server did not become healthy")


def completion_probs(prompt: str, n_probs: int = 20) -> list[dict]:
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": 1,
            "temperature": 0.0,
            "n_probs": n_probs,
        }
    ).encode()
    req = urllib.request.Request(
        SERVER,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    # First predicted token's candidate list
    cps = data.get("completion_probabilities") or []
    if not cps:
        raise RuntimeError(f"no completion_probabilities in response keys={list(data.keys())}")
    probs = cps[0].get("probs") or []
    return probs


def hf_top_logprobs(model, tok, input_ids, k: int = 20) -> list[tuple[int, float]]:
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, -1, :].float()
        logp = F.log_softmax(logits, dim=-1)
        vals, idx = torch.topk(logp, k)
    return [(int(i.item()), float(v.item())) for i, v in zip(idx, vals)]


def compare_prompt_rows(
    model,
    tok,
    user_text: str,
    n_probs: int = 20,
) -> dict:
    messages = [{"role": "user", "content": user_text}]
    chat_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    chat_text = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    hf_top = hf_top_logprobs(model, tok, chat_ids, n_probs)
    gg_probs = completion_probs(chat_text, n_probs)
    gg_map = {int(p["id"]): float(p["logprob"]) for p in gg_probs if "id" in p}
    hf_map = dict(hf_top)
    hf_top1 = hf_top[0][0]
    gg_top1 = max(gg_map.items(), key=lambda kv: kv[1])[0] if gg_map else None
    shared = set(hf_map) & set(gg_map)
    max_lp_diff = max(abs(hf_map[t] - gg_map[t]) for t in shared) if shared else None
    return {
        "prompt_head": user_text[:60],
        "chat_tokens": int(chat_ids.shape[-1]),
        "hf_top1_id": hf_top1,
        "hf_top1_token": tok.decode([hf_top1]),
        "gg_top1_id": gg_top1,
        "gg_top1_token": tok.decode([gg_top1]) if gg_top1 is not None else None,
        "top1_match": hf_top1 == gg_top1,
        "max_logprob_diff_top20": max_lp_diff,
        "shared_top20_count": len(shared),
    }


def wikitext_perplexity_llama_chunks(
    model, tok, path: Path, ctx: int = 512, add_bos: bool = False
) -> tuple[float, int, int]:
    """Match llama-perplexity (stride=0): last half of each ctx window, non-overlapping chunks."""
    text = path.read_text(encoding="utf-8")
    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    ids = enc.input_ids[0]
    if len(ids) < 2 * ctx:
        raise ValueError(f"need >= {2 * ctx} tokens, got {len(ids)}")
    n_chunk = len(ids) // ctx
    first = ctx // 2
    bos_id = tok.bos_token_id if add_bos and tok.bos_token_id is not None else None
    nll = 0.0
    count = 0
    with torch.no_grad():
        for i in range(n_chunk):
            start = i * ctx
            chunk = ids[start : start + ctx].clone()
            if bos_id is not None:
                chunk[0] = bos_id
            inp = chunk.unsqueeze(0).to(model.device)
            logits = model(inp).logits[0].float()
            for j in range(first, ctx - 1):
                target = int(ids[start + j + 1].item())
                logp = F.log_softmax(logits[j], dim=-1)[target].item()
                nll -= logp
                count += 1
    return math.exp(nll / count), n_chunk, count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default="/root/models/Qwen3.8-27B-text")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = list(DEFAULT_PROMPTS) + [long_prompt(120)]
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    print("loading HF model for logit pass...")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()

    rows = [compare_prompt_rows(model, tok, p) for p in prompts]
    print("HF perplexity on wikitext (llama-perplexity chunking, ctx=512)...")
    hf_ppl, n_chunk, ppl_count = wikitext_perplexity_llama_chunks(
        model, tok, WIKITEXT, ctx=512, add_bos=False
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    payload = {
        "prompt_rows": rows,
        "perplexity": {
            "transformers_bf16_llama_chunking": hf_ppl,
            "n_chunks": n_chunk,
            "token_predictions": ppl_count,
            "gguf_bf16_llama_perplexity_reference": 6.6497,
        },
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))

    if args.out_md:
        lines = ["# RoPE check (bf16 GGUF vs safetensors)", "", "## Numbers", ""]
        lines.append("| # | prompt (head) | chat tokens | top-1 match | max |logp| diff (shared top-20) |")
        lines.append("|---|---------------|-------------|-------------|----------------------------------|")
        for i, r in enumerate(rows, 1):
            diff = r["max_logprob_diff_top20"]
            diff_s = f"{diff:.6f}" if diff is not None else "n/a"
            lines.append(
                f"| {i} | {r['prompt_head']!r} | {r['chat_tokens']} | {r['top1_match']} | {diff_s} |"
            )
        ppl = payload["perplexity"]
        lines.extend(
            [
                "",
                f"- **GGUF bf16 PPL (llama-perplexity):** {ppl['gguf_bf16_llama_perplexity_reference']}",
                f"- **Safetensors PPL (HF, llama chunking):** {ppl['transformers_bf16_llama_chunking']:.4f}",
                f"- **Chunks / predictions:** {ppl['n_chunks']} / {ppl['token_predictions']}",
                "",
                "## Protocol",
                "",
                "- **GGUF:** `llama-server` on bf16 GGUF; `POST /completion` with `n_predict=1`, `temperature=0`, `n_probs=20`.",
                "- **HF:** same user prompts via `apply_chat_template(..., add_generation_prompt=True)`; compare next-token top-20 ids/logprobs.",
                "- **No llama-cli stdout parsing.**",
                "- **PPL:** same `wikitext2-test.txt` as task 2; HF uses llama-perplexity non-stride chunking (512 ctx, score last 256 tokens per chunk). Residual gap vs GGUF is mostly tokenizer/vocab path (llama `common_tokenize` vs HF `tokenizer`).",
            ]
        )
        Path(args.out_md).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
