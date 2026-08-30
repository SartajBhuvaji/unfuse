#!/usr/bin/env python3
"""Turn a bench_pair.sh output directory into the three model-card sections.

Hand-copying numbers out of JSON into a table is how a card ends up claiming a
figure no artifact supports. This reads the committed results and prints the
markdown, so the card and the raw data cannot drift apart: rerun it and diff.

    report_pair.py ~/bench-pair --gpu "NVIDIA A100-SXM4-40GB" \
        --orig-gb 20.586 --derived-gb 18.801 --text-gb 18.801
"""

import argparse
import json
import pathlib
import sys

# U+2212, to match the minus the existing cards use in their delta columns.
MINUS = "\u2212"

ROLES = ["original", "derived"]
NAMES = {"original": "original", "derived": "this model"}


def load(d, pat):
    f = d / pat
    return json.load(open(f)) if f.exists() else None


def num(x, dp):
    return f"{x:,.0f}" if dp == 0 else f"{x:.{dp}f}"


def gib_to_tokens(gib, bytes_per_token):
    return round(gib * (1 << 30) / bytes_per_token)


def memory_table(d, bytes_per_token):
    """The headline result: what the freed VRAM turns into.

    vLLM only. SGLang prints 'GB' while dividing by 2^30, and mixing its numbers
    with vLLM's real GiB is a silent 7% error.
    """
    s = {r: load(d, f"serve-vllm-{r}.json") for r in ROLES}
    if not all(s.values()):
        return "  (missing vLLM serve stats)\n"
    o, v = s["original"], s["derived"]
    rows = [
        ("Model weights in VRAM", "weights_gib", "GiB", 2),
        ("KV cache available", "kv_cache_gib", "GiB", 2),
        ("KV cache tokens", "kv_cache_tokens", "", 0),
        ("Max concurrency @ 32k ctx", "max_concurrency", "x", 2),
    ]
    out = ["| | Original | This model | Δ |", "|---|---|---|---|"]
    for label, key, unit, dp in rows:
        if key not in o or key not in v:
            continue
        a, b = o[key], v[key]
        delta = b - a
        pct = 100.0 * delta / a if a else 0.0
        u = f" {unit}" if unit == "GiB" else unit
        sign = "+" if delta >= 0 else MINUS
        out.append(
            f"| {label} | {num(a, dp)}{u} | {num(b, dp)}{u} | "
            f"{sign}{num(abs(delta), dp)}{u} ({sign}{abs(pct):.1f}%) |"
        )
    # A cache that costs the same per token in both models is the whole reason
    # the token count moved: it is freed VRAM, not a cheaper cache.
    for r in ROLES:
        got = s[r].get("kv_cache_tokens")
        want = gib_to_tokens(s[r].get("kv_cache_gib", 0), bytes_per_token)
        if got and abs(got - want) / got > 0.02:
            out.append(f"<!-- CHECK {r}: {got:,} tokens reported, {want:,} implied by GiB -->")
    if "mm_encoder_budget_tokens" in o:
        out.append(
            f"<!-- original also allocates a {o['mm_encoder_budget_tokens']:,}-token "
            "encoder cache; derived does not -->"
        )
    return "\n".join(out) + "\n"


def bench_table(d, versions):
    out = [
        "| Engine | Model | c | Output tok/s | Per stream | TTFT p50 | TTFT p99 | TPOT |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for eng in ["vllm", "sglang"]:
        for role in ROLES:
            b = load(d, f"bench-{eng}-{role}-warm.json")
            if not b:
                continue
            for r in b["runs"]:
                out.append(
                    f"| {versions.get(eng, eng)} | {NAMES[role]} | {r['concurrency']} "
                    f"| {r['output_tok_s']} | {r['per_stream_tok_s']} "
                    f"| {r['ttft_ms_p50']:,.1f} ms | {r['ttft_ms_p99']:,.1f} ms "
                    f"| {r['tpot_ms_mean']} ms |"
                )
    return "\n".join(out) + "\n"


def limits(d, tested_concurrency):
    """Verdict on whether two rows can be compared at the concurrency tested.

    Three outcomes, and conflating them is how the Qwen run nearly shipped a
    2.79x scheduler artifact as a model property:

      equal            the scheduler treated both checkpoints the same.
      both headroom    unequal, but both caps sit above the offered load, so
                       neither model was ever throttled. Still comparable.
      starved          a cap at or below the offered load. That row measures
                       the scheduler, not the checkpoint.
      unverified       the engine never printed it. Two unknowns are not a
                       match; this is the one case that must not read as "fine".
    """
    out = []
    for eng in ["vllm", "sglang"]:
        got = {r: (load(d, f"serve-{eng}-{r}.json") or {}).get("concurrency_limit") for r in ROLES}
        if any(v is None for v in got.values()):
            out.append(
                f"- {eng}: not reported by the engine, so UNVERIFIED "
                f"(pin --max-num-seqs / --max-running-requests to make it checkable)"
            )
            continue
        lo = min(got.values())
        if len(set(got.values())) == 1:
            verdict = "equal"
        elif lo > tested_concurrency:
            verdict = (
                f"unequal, but both exceed the tested concurrency of "
                f"{tested_concurrency}, so neither model is throttled"
            )
        else:
            verdict = f"STARVED -- {lo} is at or below the tested {tested_concurrency}"
        out.append(f"- {eng}: original={got['original']}, this model={got['derived']} ({verdict})")
    return "\n".join(out) + "\n"


def best_c1(d, role):
    vals = []
    for eng in ["vllm", "sglang"]:
        b = load(d, f"bench-{eng}-{role}-warm.json")
        if b:
            vals += [r["output_tok_s"] for r in b["runs"] if r["concurrency"] == 1]
    return max(vals) if vals else None


def cold_vs_warm(d):
    out = []
    for eng in ["vllm", "sglang"]:
        for role in ROLES:
            c, w = (
                load(d, f"bench-{eng}-{role}-cold.json"),
                load(d, f"bench-{eng}-{role}-warm.json"),
            )
            if not (c and w):
                continue
            for rc, rw in zip(c["runs"], w["runs"]):
                if rc["concurrency"] != rw["concurrency"]:
                    continue
                gap = 100.0 * (rw["output_tok_s"] - rc["output_tok_s"]) / rc["output_tok_s"]
                out.append(
                    f"- {eng} {role} c={rc['concurrency']}: cold {rc['output_tok_s']} -> "
                    f"warm {rw['output_tok_s']} tok/s ({gap:+.1f}%)"
                )
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--gpu", default="NVIDIA A100-SXM4-40GB")
    p.add_argument("--hbm", type=float, default=1.555)
    p.add_argument("--orig-gb", type=float, required=True)
    p.add_argument("--derived-gb", type=float, required=True)
    p.add_argument(
        "--text-gb",
        type=float,
        required=True,
        help="bytes decode actually reads in the ORIGINAL: the checkpoint minus the vision tower",
    )
    p.add_argument("--bytes-per-token", type=int, default=40960)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 16])
    a = p.parse_args()

    d = pathlib.Path(a.dir).expanduser()
    if not d.is_dir():
        sys.exit(f"{d}: no such directory")
    env = load(d, "env.json") or {}
    ver = {"vllm": "vLLM 0.27.1", "sglang": "SGLang 0.5.18"}

    print("### environment\n")
    cc = str(env.get("compute_capability", "")).replace(".", "")
    print(
        f"1x {env.get('gpu', a.gpu)}, sm_{cc}, driver {env.get('driver', '?')}, "
        f"CUDA {env.get('cuda_driver', '?')}, {env.get('power_limit', '?')}, "
        f"{env.get('os', '?')}, kernel {env.get('kernel', '?')}.\n"
    )
    # torch is per-engine, never ambient: each engine pins its own build into its
    # own venv, and the system torch on the box served nothing.
    for eng, label in ver.items():
        e = load(d, f"env-{eng}.json")
        if e:
            print(f"- {label}: torch {e.get('torch')}, python {e.get('python')}")
    print()
    print("### What you get\n")
    print(memory_table(d, a.bytes_per_token))
    print("### Scheduler limits\n")
    print(limits(d, max(a.concurrency)))
    print("### Benchmarks\n")
    print(bench_table(d, ver))
    print("### Warmup effect (cold pass discarded, kept as evidence)\n")
    print(cold_vs_warm(d))
    print("### Bandwidth limit\n")
    print(f"| | Weights read/token | / {a.hbm} TB/s | Ceiling | Best measured c=1 | Of ceiling |")
    print("|---|---|---|---|---|---|")
    for role, gb in [("original", a.text_gb), ("derived", a.derived_gb)]:
        floor = gb / a.hbm
        ceil = 1000.0 / floor
        m = best_c1(d, role)
        meas = f"{m} tok/s" if m else "-"
        pct = f"~{100.0 * m / ceil:.0f}%" if m else "-"
        print(f"| {NAMES[role]} | {gb} GB | {floor:.2f} ms | {ceil:.1f} tok/s | {meas} | {pct} |")
    print(
        f"\n<!-- original checkpoint on disk is {a.orig_gb} GB, but decode reads only "
        f"{a.text_gb} GB of it: the vision tower is resident, not streamed. -->"
    )


if __name__ == "__main__":
    main()
