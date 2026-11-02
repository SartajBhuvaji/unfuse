#!/usr/bin/env python3
"""RoPE / GGUF vs safetensors checks (followups only; tools script failed on greedy_64)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from unfuse.verify import DEFAULT_PROMPTS, compare_logits, long_prompt  # noqa: E402


def llama_greedy(llama_bin: Path, gguf: Path, prompt: str, n: int) -> str:
    proc = subprocess.run(
        [
            str(llama_bin),
            "-m",
            str(gguf),
            "-p",
            prompt,
            "-n",
            str(n),
            "--temp",
            "0",
            "-st",
            "--reasoning",
            "off",
            "--no-warmup",
            "-ngl",
            "99",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    return proc.stdout + proc.stderr


def parse_first_token(out: str, prompt: str) -> str:
    if prompt in out:
        tail = out.split(prompt, 1)[-1]
    else:
        tail = out
    for line in tail.splitlines():
        s = line.strip()
        if not s or s.startswith("[") or s.startswith(">") or "t/s" in s or s.startswith("Exiting"):
            continue
        if s.startswith("/"):
            continue
        return s.split()[0] if s.split() else s
    return ""


def main() -> None:
    hf_dir = Path("/root/models/Qwen3.8-27B-text")
    gguf = Path("/root/gguf/Qwen3.8-27B-text-BF16.gguf")
    llama_bin = Path("/root/llama.cpp/build/bin/llama-cli")
    out = Path("/root/unfuse/notebooks/qwen3.8-27b/followups/rope_check.json")

    prompts = list(DEFAULT_PROMPTS) + [long_prompt(120)]
    long_tokens = len(long_prompt(120).split())
    self = compare_logits(str(hf_dir), str(hf_dir), prompts=prompts, dtype=torch.bfloat16)

    tok = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(hf_dir), dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()

    first_rows = []
    with torch.no_grad():
        for prompt in prompts:
            enc = tok(prompt, return_tensors="pt").to(model.device)
            logits = model(**enc).logits[0, -1, :].float()
            ref_id = int(logits.argmax().item())
            ref_tok = tok.decode([ref_id])
            gout = llama_greedy(llama_bin, gguf, prompt, 1)
            gguf_tok = parse_first_token(gout, prompt)
            gguf_id = tok.convert_tokens_to_ids(gguf_tok) if gguf_tok else None
            first_rows.append(
                {
                    "prompt_head": prompt[:60],
                    "prompt_tokens_approx": len(tok(prompt)["input_ids"]),
                    "ref_argmax_id": ref_id,
                    "ref_argmax_token": ref_tok,
                    "gguf_first_token": gguf_tok,
                    "gguf_token_id": gguf_id,
                    "argmax_match": ref_id == gguf_id if gguf_id is not None else False,
                    "note": "max|logit diff vs GGUF needs llama full-vocab logits; compare argmax token only",
                }
            )

    greedy_rows = []
    with torch.no_grad():
        for prompt in prompts[:6]:
            enc = tok(prompt, return_tensors="pt").to(model.device)
            input_ids = enc["input_ids"]
            out_ids = model.generate(
                input_ids, attention_mask=enc.get("attention_mask"), max_new_tokens=64, do_sample=False
            )
            ref_text = tok.decode(out_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
            gguf_text = llama_greedy(llama_bin, gguf, prompt, 64)
            greedy_rows.append(
                {
                    "prompt_head": prompt[:50],
                    "transformers_64": ref_text[:300],
                    "gguf_tail": gguf_text[-300:],
                    "prefix_match_chars": sum(
                        1 for a, b in zip(ref_text, gguf_text) if a == b
                    ),
                }
            )

    payload = {
        "tools_qwen_gguf_rope_check": {
            "status": "failed",
            "error": "AttributeError in model.generate: apply_chat_template BatchEncoding passed where Tensor expected (line 98)",
        },
        "long_prompt_words": long_tokens,
        "safetensors_self_compare": {
            "worst_max_abs": self.worst,
            "exact": self.exact,
            "report": self.report(),
        },
        "first_token_bf16_gguf_vs_transformers": first_rows,
        "greedy_64_raw_prompt": greedy_rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
