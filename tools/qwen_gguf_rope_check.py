#!/usr/bin/env python3
"""Compare bf16 safetensors vs bf16 GGUF on RoPE-sensitive checks (runbook task 1)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from unfuse.verify import DEFAULT_PROMPTS, compare_logits, long_prompt


def first_token_from_llama(llama_bin: Path, gguf: Path, prompt: str) -> tuple[str, str]:
    """Greedy first continuation token text from llama-cli."""
    proc = subprocess.run(
        [
            str(llama_bin),
            "-m",
            str(gguf),
            "-p",
            prompt,
            "-n",
            "1",
            "--temp",
            "0",
            "-st",
            "--no-warmup",
            "-ngl",
            "99",
        ],
        capture_output=True,
        text=True,
        input="",
        timeout=600,
    )
    out = proc.stdout + proc.stderr
    # llama-cli prints generated text after the prompt marker
    m = re.search(r"(?s)prompt.*?\n(.*)", out)
    tok = (m.group(1).strip().split() or [""])[0] if m else ""
    return tok, out[-800:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama-bin", required=True)
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir)
    gguf = Path(args.gguf)
    llama_bin = Path(args.llama_bin)

    prompts = list(DEFAULT_PROMPTS) + [long_prompt(120)]
    comp = compare_logits(str(hf_dir), str(hf_dir), prompts=prompts, dtype=torch.bfloat16)
    # Self-check: should be exact; confirms transformers path works before GGUF compare.

    tok = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(hf_dir), dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()

    gguf_rows = []
    with torch.no_grad():
        for prompt in prompts:
            ids = tok(prompt, return_tensors="pt").to(model.device)
            ref_id = model(**ids).logits[:, -1, :].argmax(-1).item()
            ref_tok = tok.decode([ref_id])
            ll_tok, tail = first_token_from_llama(llama_bin, gguf, prompt)
            gguf_rows.append(
                {
                    "prompt_head": prompt[:60],
                    "ref_token_id": ref_id,
                    "ref_token": ref_tok,
                    "gguf_first_token": ll_tok,
                    "argmax_match": ref_tok.strip() == ll_tok.strip()
                    or ref_id == tok.convert_tokens_to_ids(ll_tok)
                    if ll_tok
                    else False,
                    "llama_tail": tail,
                }
            )

    greedy_64 = []
    for prompt in prompts[:6]:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        out = model.generate(ids, max_new_tokens=64, do_sample=False)
        ref_text = tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
        proc = subprocess.run(
            [
                str(llama_bin),
                "-m",
                str(gguf),
                "-p",
                prompt,
                "-n",
                "64",
                "--temp",
                "0",
                "-st",
                "--no-warmup",
                "-ngl",
                "99",
            ],
            capture_output=True,
            text=True,
            input="",
            timeout=900,
        )
        gguf_text = proc.stdout
        greedy_64.append(
            {
                "prompt_head": prompt[:50],
                "transformers_64": ref_text[:200],
                "gguf_64": gguf_text[-200:],
            }
        )

    payload = {
        "safetensors_self_compare": {
            "worst_max_abs": comp.worst,
            "exact": comp.exact,
            "report": comp.report(),
        },
        "first_token_gguf_vs_transformers": gguf_rows,
        "greedy_64_sample": greedy_64,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2)[:4000])


if __name__ == "__main__":
    main()
