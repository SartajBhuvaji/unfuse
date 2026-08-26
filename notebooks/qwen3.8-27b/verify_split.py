"""Prove the Qwen3.8-27B split is lossless, without loading either model.

Why not compare logits
----------------------
`unfuse.verify.compare_logits` is the right tool when the surgery could change
arithmetic -- requantization, dtype changes, a rebuilt attention path. This split
changes none of that. It drops tensors and renames the survivors.

For a pure copy-and-rename, a tensor-level comparison is strictly STRONGER evidence
than a logit comparison: identical bytes for every kept tensor means every possible
input produces identical output, whereas matching logits on N prompts means they
matched on N prompts. It is also the only option on a machine that cannot hold a 27B
model in RAM.

What this checks
----------------
1. Every kept source tensor appears in the output under its renamed key, exactly once.
2. Byte-for-byte equality of the raw tensor buffers (BLAKE2b over the numpy bytes),
   plus dtype and shape.
3. Nothing extra appears in the output.
4. No dropped tensor survives.
5. The index's total_size matches the bytes actually written.

Streams one tensor at a time, so peak memory is one tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

VISION_PREFIX = "model.visual."
TEXT_PREFIX = "model.language_model."


def rename(name: str) -> str:
    if name.startswith(TEXT_PREFIX):
        return "model." + name[len(TEXT_PREFIX) :]
    return name


def digest(t) -> str:
    a = t.numpy() if hasattr(t, "numpy") else t
    return hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest()


def load_index(d: Path) -> dict:
    return json.loads((d / "model.safetensors.index.json").read_text())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--drop-mtp", action="store_true")
    args = p.parse_args()

    from safetensors import safe_open

    src_map = load_index(args.src)["weight_map"]
    dst_index = load_index(args.dst)
    dst_map = dst_index["weight_map"]

    dropped = {n for n in src_map if n.startswith(VISION_PREFIX)}
    if args.drop_mtp:
        dropped |= {n for n in src_map if n.startswith("mtp.")}
    expected = {rename(n) for n in src_map if n not in dropped}

    print(f"source {len(src_map)} tensors, dropping {len(dropped)}")
    print(f"output {len(dst_map)} tensors, expected {len(expected)}")

    problems: list[str] = []
    if set(dst_map) != expected:
        missing = expected - set(dst_map)
        extra = set(dst_map) - expected
        for n in sorted(missing)[:10]:
            problems.append(f"MISSING from output: {n}")
        for n in sorted(extra)[:10]:
            problems.append(f"UNEXPECTED in output: {n}")
        if len(missing) > 10 or len(extra) > 10:
            problems.append(f"... {len(missing)} missing, {len(extra)} unexpected total")

    survivors = [n for n in dropped if rename(n) in dst_map]
    for n in survivors[:10]:
        problems.append(f"DROPPED TENSOR SURVIVED: {n}")

    # Group by source shard so each file opens once.
    by_shard: dict[str, list[str]] = {}
    for n in src_map:
        if n in dropped:
            continue
        by_shard.setdefault(src_map[n], []).append(n)

    dst_open: dict[str, object] = {}
    checked = 0
    total_bytes = 0
    for shard in sorted(by_shard):
        with safe_open(args.src / shard, framework="pt") as sf:
            for name in by_shard[shard]:
                new_name = rename(name)
                if new_name not in dst_map:
                    continue
                dshard = dst_map[new_name]
                if dshard not in dst_open:
                    dst_open[dshard] = safe_open(args.dst / dshard, framework="pt").__enter__()
                a = sf.get_tensor(name)
                b = dst_open[dshard].get_tensor(new_name)
                if a.shape != b.shape:
                    problems.append(f"SHAPE {name}: {a.shape} vs {b.shape}")
                elif a.dtype != b.dtype:
                    problems.append(f"DTYPE {name}: {a.dtype} vs {b.dtype}")
                else:
                    da, db = digest(a), digest(b)
                    if da != db:
                        problems.append(f"BYTES DIFFER {name}: {da} vs {db}")
                total_bytes += a.numel() * a.element_size()
                checked += 1
                if checked % 100 == 0:
                    print(f"  checked {checked}/{len(expected)}", flush=True)
                del a, b

    declared = dst_index.get("metadata", {}).get("total_size")
    if declared != total_bytes:
        problems.append(f"index total_size {declared} != actual {total_bytes}")

    print(f"\nchecked {checked} tensors, {total_bytes / 1e9:.2f} GB")
    if problems:
        print(f"\nFAIL — {len(problems)} problem(s):")
        for x in problems[:25]:
            print("  " + x)
        return 1
    print("\nPASS — every kept tensor is byte-identical to its source.")
    print("The split is lossless by construction, not by sampling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
