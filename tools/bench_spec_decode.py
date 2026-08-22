"""Measure what a speculative-decoding drafter buys on a quantized target.

The question this answers
-------------------------
Speculative decoding is exact: the target verifies every drafted token and
discards what it would not have produced, so the output distribution is unchanged
no matter how bad the drafter is. What changes is *throughput*, and that depends
entirely on how often the target accepts a draft.

That is the interesting part here. `google/gemma-4-31B-it-assistant` was trained
against the BF16 target. Quantizing the target moves its output distribution, so
the drafter is now predicting a slightly different model than the one it learned.
Nobody has published how much acceptance that costs on an NVFP4 target -- and
since acceptance is the entire speedup, a drafter that pairs well with BF16 may
pair worse with FP4. Run this against each build to find out.

What it measures
----------------
Wall-clock tokens/sec, and **mean tokens per target forward pass**. The second is
the honest metric: it is the arithmetic ceiling on the speedup, it is independent
of hardware and batch size, and it is what actually degrades if quantization hurts
acceptance. Wall-clock alone confounds acceptance with kernel efficiency.

Target forward passes are counted with a hook rather than read out of the engine,
because no two engines report this the same way and a hook is portable.

NVFP4 needs Blackwell to execute. On Hopper the target either refuses to load or
falls back to a dequantized path, and the timings then mean nothing -- so this
records the GPU it ran on alongside every number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROMPTS = [
    "Write a Python function that merges two sorted lists.",
    "Explain the difference between a mutex and a semaphore.",
    "Summarise the causes of the 1873 financial panic.",
    "What is the time complexity of building a heap, and why?",
]


@dataclass
class Run:
    label: str
    gpu: str
    tokens: int
    seconds: float
    target_forwards: int

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0

    @property
    def tokens_per_forward(self) -> float:
        """1.0 without a drafter. Above 1.0 is exactly the acceptance win."""
        return self.tokens / self.target_forwards if self.target_forwards else 0.0


def _count_forwards(model) -> tuple[list[int], object]:
    counter = [0]

    def hook(*_args, **_kwargs):
        counter[0] += 1

    return counter, model.register_forward_pre_hook(hook)


def run_once(target, tokenizer, prompts, max_new_tokens, assistant=None, label="") -> Run:
    import torch

    counter, handle = _count_forwards(target)
    tokens = 0
    # Exclude compile/warmup from the timing: the first generate call pays for
    # kernel autotuning and would otherwise dominate a short benchmark.
    for prompt in prompts[:1]:
        ids = tokenizer(prompt, return_tensors="pt").to(target.device)
        target.generate(**ids, max_new_tokens=8, do_sample=False, assistant_model=assistant)

    counter[0] = 0
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").to(target.device)
        out = target.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=False, assistant_model=assistant
        )
        tokens += out.shape[-1] - ids["input_ids"].shape[-1]
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start
    handle.remove()

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return Run(label, gpu, tokens, elapsed, counter[0])


def main(argv: list[str] | None = None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("target", help="target checkpoint (path or hub id)")
    parser.add_argument("--drafter", default="google/gemma-4-31B-it-assistant")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--label", default=None)
    parser.add_argument("--json", type=Path, help="append results here as JSON lines")
    args = parser.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=None, device_map="cuda", low_cpu_mem_usage=True
    ).eval()

    label = args.label or args.target
    results = [run_once(target, tokenizer, PROMPTS, args.max_new_tokens, None, f"{label} (alone)")]

    if args.drafter and args.drafter.lower() != "none":
        drafter = AutoModelForCausalLM.from_pretrained(
            args.drafter, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True
        ).eval()
        results.append(
            run_once(target, tokenizer, PROMPTS, args.max_new_tokens, drafter, f"{label} + drafter")
        )

    print(f"\n{'run':<44} {'tok/s':>9} {'tok/fwd':>9} {'tokens':>8}")
    print("-" * 74)
    for r in results:
        print(
            f"{r.label[:44]:<44} {r.tokens_per_second:>9.2f} "
            f"{r.tokens_per_forward:>9.3f} {r.tokens:>8}"
        )
    print(f"\ngpu: {results[0].gpu}")

    if len(results) == 2:
        speedup = results[1].tokens_per_second / results[0].tokens_per_second
        print(f"speedup {speedup:.2f}x, acceptance ceiling {results[1].tokens_per_forward:.3f}x")

    if args.json:
        with args.json.open("a") as fh:
            for r in results:
                fh.write(json.dumps({**asdict(r), "tok_per_s": r.tokens_per_second}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
