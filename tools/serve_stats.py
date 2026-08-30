#!/usr/bin/env python3
"""Pull the memory and scheduling facts out of an engine's startup log.

The interesting numbers for a vision-removal split are not throughput. They are
what the freed GB turns into: KV cache bytes, KV cache tokens, and how many
32k-context requests fit on the card. Every engine prints those once, at
startup, and then never again -- they are not on /metrics and there is no API
for them. So they get scraped.

It also pulls the concurrency limit the engine chose for itself
(vLLM max_num_seqs, SGLang max_running_requests). That one is not decoration.
On the Qwen3.8-27B run SGLang gave the original model 5 and the derived model
16, and the resulting 2.79x "win" was a scheduler artifact reported as a
property of the checkpoint. Two rows are only comparable if this field matches.

Multimodal parents also allocate an encoder/vision profiling buffer that a
text-only derivative does not. That is a real serving cost and belongs in the
table, but it is a consequence of the code path, not of the weights, so it is
reported as its own field rather than folded into the weight delta.

Usage:
    serve_stats.py LOG [--engine auto|vllm|sglang] [--label L] [--json-out F]
"""

import argparse
import json
import re
import sys

# Every pattern is (field, regex, cast). A field that does not match is left
# absent and named in "missing" -- a scraper that silently emits null for a log
# format it did not recognise is worse than one that says so.
VLLM = [
    ("engine_version", r"vLLM API server version ([0-9][\w.+-]*)", str),
    ("weights_gib", r"Model loading took ([\d.]+) GiB", float),
    ("kv_cache_gib", r"Available KV cache memory: ([\d.]+) GiB", float),
    ("kv_cache_tokens", r"GPU KV cache size: ([\d,]+) tokens", "int_commas"),
    ("max_model_len", r"Maximum concurrency for ([\d,]+) tokens per request", "int_commas"),
    ("max_concurrency", r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", float),
    # vLLM prints the effective cap only when it was passed explicitly
    # (max_num_seqs=N in the config dump) or at DEBUG level, where the
    # usage-context default lands after being clamped to
    # min(max_num_seqs, max_num_batched_tokens). At INFO with no flag, it
    # prints nothing at all -- so an absent value here means "unverified",
    # never "the default".
    ("concurrency_limit", r"(?:max_num_seqs=|Defaulting max_num_seqs to )(\d+)", int),
    (
        "mm_encoder_budget_tokens",
        r"[Ee]ncoder cache will be initialized with a budget of (\d+)",
        int,
    ),
]

SGLANG = [
    ("weights_gib", r"Load weight end\..*?mem usage=([\d.]+) GB", float),
    # 0.5.18 interposes "dtype: torch.bfloat16," between the sentence and the
    # token count; earlier builds did not. Tolerate anything in between.
    ("kv_cache_tokens", r"KV Cache is allocated\..*?#tokens: ([\d,]+)", "int_commas"),
    ("kv_k_gib", r"KV Cache is allocated\..*?K size: ([\d.]+) GB", float),
    ("kv_v_gib", r"KV Cache is allocated\..*?V size: ([\d.]+) GB", float),
    ("max_model_len", r"context_len=(\d+)", int),
    ("concurrency_limit", r"max_running_requests=(\d+)", int),
]

# SGLang prints "GB" but divides by 1<<30, so its GB is a GiB. vLLM prints GiB
# and means it. Mixing the two in one table is a 7% error, so the unit each
# engine actually uses is recorded rather than assumed.
UNITS = {"vllm": "GiB", "sglang": "GiB (log says GB; SGLang divides by 2^30)"}


def detect(text):
    if "vLLM API server" in text or "vllm" in text.lower()[:4000]:
        return "vllm"
    if "sglang" in text.lower() or "max_total_num_tokens" in text:
        return "sglang"
    return ""


def scrape(text, patterns):
    out, missing = {}, []
    for field, pat, cast in patterns:
        # No DOTALL: every one of these facts is printed on a single line, and
        # letting `.` cross newlines lets a pattern stitch two unrelated lines
        # together into a plausible-looking wrong answer.
        m = re.search(pat, text)
        if not m:
            missing.append(field)
            continue
        raw = m.group(1)
        out[field] = int(raw.replace(",", "")) if cast == "int_commas" else cast(raw)
    return out, missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--engine", default="auto", choices=["auto", "vllm", "sglang"])
    p.add_argument("--engine-version", default="")
    p.add_argument("--label", default="")
    p.add_argument("--model", default="")
    p.add_argument("--json-out", default="")
    a = p.parse_args()

    with open(a.log, errors="replace") as f:
        text = f.read()
    engine = a.engine if a.engine != "auto" else detect(text)
    if not engine:
        sys.exit(f"{a.log}: cannot tell which engine wrote this log; pass --engine")

    stats, missing = scrape(text, VLLM if engine == "vllm" else SGLANG)
    if engine == "sglang" and "kv_k_gib" in stats and "kv_v_gib" in stats:
        stats["kv_cache_gib"] = round(stats.pop("kv_k_gib") + stats.pop("kv_v_gib"), 2)

    # vLLM states max concurrency directly. SGLang does not, but it prints both
    # halves of the division, so it is derived rather than left blank.
    if "max_concurrency" not in stats and {"kv_cache_tokens", "max_model_len"} <= stats.keys():
        stats["max_concurrency"] = round(stats["kv_cache_tokens"] / stats["max_model_len"], 2)
        stats["max_concurrency_derived"] = True

    out = {
        "label": a.label,
        "model": a.model,
        "engine": engine,
        "memory_unit": UNITS[engine],
        **({"engine_version": a.engine_version} if a.engine_version else {}),
        **stats,
        "missing": missing,
    }
    print(json.dumps(out, indent=2))
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(out, f, indent=2)
    if missing:
        print(f"  not found in log: {', '.join(missing)}", file=sys.stderr)


if __name__ == "__main__":
    main()
