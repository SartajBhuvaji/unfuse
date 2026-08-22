"""Command line entry point.

Two verbs, both model-agnostic:

  unfuse inspect <checkpoint> [--prefix model.visual.]
  unfuse verify  <reference> <candidate>

`inspect` is what you run first against a model you have never split before. It
tells you where the encoder lives in the tensor names and how much of the model it
accounts for -- which is usually enough to decide whether splitting is worth doing
at all. On Gemma 4 12B the answer is no: its vision path is 0.438% of the weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from unfuse.inspect import (
    find_index,
    group_by_prefix,
    read_inventory,
    shards_for_prefix,
    split_by_prefix,
)


def _human(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def cmd_inspect(args: argparse.Namespace) -> int:
    inv = read_inventory(args.checkpoint)
    print(f"{len(inv)} tensors, {inv.params:,} parameters, {_human(inv.nbytes)}\n")

    if args.prefix:
        matched, rest = split_by_prefix(inv, args.prefix)
        if not len(matched):
            print(f"nothing matches prefix {args.prefix!r}")
            return 1
        share = 100 * matched.params / inv.params
        print(f"{args.prefix}*")
        print(
            f"  {len(matched):>4} tensors  {matched.params:>15,}  {share:6.3f}%  {_human(matched.nbytes)}"
        )
        print("rest")
        print(
            f"  {len(rest):>4} tensors  {rest.params:>15,}  {100 - share:6.3f}%  {_human(rest.nbytes)}"
        )

        index = find_index(args.checkpoint)
        if index:
            shards = shards_for_prefix(index, args.prefix)
            total = len({*index["weight_map"].values()})
            print(f"\nlives in {len(shards)} of {total} shards: {', '.join(shards)}")
            if len(shards) == 1:
                print("only that shard needs downloading")
        print()

    print(f"{'group':<44} {'tensors':>8} {'params':>16} {'size':>10}")
    for name, group in group_by_prefix(inv, args.depth).items():
        print(f"{name:<44} {len(group):>8} {group.params:>16,} {_human(group.nbytes):>10}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from unfuse.verify import compare_logits, long_prompt

    prompts = None
    if args.long:
        prompts = [long_prompt()]

    result = compare_logits(args.reference, args.candidate, prompts=prompts, device=args.device)
    print(result.report())
    return 0 if result.exact else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="unfuse", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="show what is inside a checkpoint")
    p_inspect.add_argument("checkpoint", type=Path)
    p_inspect.add_argument("--prefix", help="split on this tensor-name prefix, e.g. model.visual.")
    p_inspect.add_argument("--depth", type=int, default=2, help="name segments to group by")
    p_inspect.set_defaults(func=cmd_inspect)

    p_verify = sub.add_parser("verify", help="check two checkpoints give identical logits")
    p_verify.add_argument("reference")
    p_verify.add_argument("candidate")
    p_verify.add_argument("--device", default="cuda")
    p_verify.add_argument(
        "--long",
        action="store_true",
        help="use one long prompt instead, to exercise position encoding",
    )
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
