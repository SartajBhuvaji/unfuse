#!/usr/bin/env python3
"""Express a measured decode rate as a fraction of what the memory bus allows.

A dense model at concurrency 1 reads every weight out of HBM once per token.
That sets a floor on time-per-token that no engine flag can go under:

    floor_seconds = checkpoint_bytes / hbm_bytes_per_second

Reporting tok/s without it makes an engine change look like the lever, when on
these models most of the headroom is already gone. It is also what explains a
flat concurrency-1 column across two checkpoints that differ by 9% of weights:
both are moving nearly the same bytes over the same bus.

Treat the result as +/-10%, not a precise efficiency. It counts weight traffic
only and ignores KV-cache reads, which grow with context and are not free.

PASS THE BYTES DECODE ACTUALLY READS, NOT THE SIZE OF THE CHECKPOINT.
A multimodal parent holds its vision tower resident in VRAM but does not stream
it over the bus on a text-only token. Billing the parent for a tower it never
reads lowers its ceiling, and the derived model then appears to gain headroom
the split did not give it. On GLM-4.6V-Flash that error is 8.7%, comfortably
larger than the effect being reported. The tower costs residency, not
bandwidth; residency belongs in the KV-cache table instead.

    bandwidth_ceiling.py --hbm 1.555 \
        --model original:20.59:63.4 --model derived:18.80:65.1
"""

import argparse
import json

# Advertised peak HBM bandwidth, TB/s (10^12 B/s), by GPU as nvidia-smi names it.
HBM = {
    "NVIDIA A100-SXM4-40GB": 1.555,
    "NVIDIA A100-SXM4-80GB": 2.039,
    "NVIDIA H100 80GB HBM3": 3.35,
    "NVIDIA H200": 4.8,
    "NVIDIA GH200 480GB": 4.0,
}


def row(name, gb, measured_tok_s, hbm_tb_s):
    """gb is 10^9 bytes, as the Hub reports checkpoint size."""
    floor_ms = gb / hbm_tb_s  # (gb*1e9 B) / (hbm*1e12 B/s) * 1e3 ms == gb/hbm
    ceiling = 1000.0 / floor_ms
    r = {
        "model": name,
        "weights_gb": gb,
        "floor_ms_per_token": round(floor_ms, 2),
        "ceiling_tok_s": round(ceiling, 1),
    }
    if measured_tok_s is not None:
        r["measured_tok_s"] = measured_tok_s
        r["pct_of_ceiling"] = round(100.0 * measured_tok_s / ceiling, 1)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hbm", type=float, help="peak HBM bandwidth in TB/s")
    p.add_argument("--gpu", default="", help=f"instead of --hbm; known: {', '.join(HBM)}")
    p.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME:GB[:TOK_S]",
        help="checkpoint size in GB (10^9 B) and optionally its best measured c=1 tok/s",
    )
    p.add_argument("--json-out", default="")
    a = p.parse_args()

    hbm = a.hbm
    if not hbm:
        if a.gpu not in HBM:
            p.error(f"--gpu {a.gpu!r} unknown; pass --hbm TB/s (known: {', '.join(HBM)})")
        hbm = HBM[a.gpu]

    rows = []
    for spec in a.model:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            p.error(f"--model {spec!r} should be NAME:GB or NAME:GB:TOK_S")
        rows.append(
            row(parts[0], float(parts[1]), float(parts[2]) if len(parts) == 3 else None, hbm)
        )

    out = {"gpu": a.gpu, "hbm_tb_s": hbm, "rows": rows}
    print(f"| | Weights | Bytes/token / {hbm} TB/s | Ceiling | Best measured | Of ceiling |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        meas = f"{r['measured_tok_s']} tok/s" if "measured_tok_s" in r else "-"
        pct = f"~{r['pct_of_ceiling']:.0f}%" if "pct_of_ceiling" in r else "-"
        print(
            f"| {r['model']} | {r['weights_gb']} GB | {r['floor_ms_per_token']} ms "
            f"| {r['ceiling_tok_s']} tok/s | {meas} | {pct} |"
        )
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
