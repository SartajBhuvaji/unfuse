"""Command line entry point.

Two verbs, both model-agnostic:

  unfuse inspect   <checkpoint> [--prefix model.visual.] [--quant]
  unfuse verify    <reference> <candidate>
  unfuse agreement <reference> <candidate> [--baseline <shipped-quantized>]

`inspect` is what you run first against a model you have never split before. It
tells you where the encoder lives in the tensor names and how much of the model it
accounts for -- which is usually enough to decide whether splitting is worth doing
at all. On Gemma 4 12B the answer is no: its vision path is 0.438% of the weights.

`--quant` additionally reads the checkpoint's quantization metadata and reports
which parts were actually quantized. That is worth running before believing a
model card: NVIDIA's NVFP4 Gemma 4 31B leaves all 60 attention layers in BF16.

`verify` and `agreement` are the two bars. `verify` demands identical bits and is
for extraction; `agreement` scores a quantized checkpoint against a reference and
is passed only when it is no worse than an already-shipped quantized build.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from unfuse.inspect import (
    find_index,
    group_by_prefix,
    quant_report,
    read_inventory,
    read_quant_config,
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

    if args.quant:
        qc = read_quant_config(args.checkpoint)
        if qc is None:
            print("no quantization metadata -- checkpoint is unquantized\n")
        else:
            print(f"quantized by {qc.producer or 'unknown'}: {qc.algo}", end="")
            print(f", group_size {qc.group_size}" if qc.group_size else "")
            print(f"kv cache: {qc.kv_cache_algo or 'unquantized'}")
            print(f"{len(qc.exclude_modules)} modules excluded\n")
            print(f"{'group':<44} {'tensors':>8} {'params':>16} {'size':>10}  state")
            for name, row in quant_report(inv, qc, args.depth).items():
                print(
                    f"{name:<44} {row['tensors']:>8} {row['params']:>16,} "
                    f"{_human(row['nbytes']):>10}  {row['state']}"
                )
            return 0

    print(f"{'group':<44} {'tensors':>8} {'params':>16} {'size':>10}")
    for name, group in group_by_prefix(inv, args.depth).items():
        print(f"{name:<44} {len(group):>8} {group.params:>16,} {_human(group.nbytes):>10}")
    return 0


def cmd_agreement(args: argparse.Namespace) -> int:
    from unfuse.agreement import compare_distributions, long_prompt

    prompts = [long_prompt()] if args.long else None
    result = compare_distributions(
        args.reference,
        args.candidate,
        prompts=prompts,
        device=args.device,
        label="candidate",
        chat=args.chat,
    )
    baseline = None
    if args.baseline:
        baseline = compare_distributions(
            args.reference,
            args.baseline,
            prompts=prompts,
            device=args.device,
            label="baseline",
            chat=args.chat,
        )
    print(result.report(baseline))
    # Without a baseline there is nothing to pass or fail against, so report only.
    return 0 if baseline is None or result.is_no_worse_than(baseline) else 1


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
    p_inspect.add_argument(
        "--quant", action="store_true", help="report which parts are actually quantized"
    )
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

    p_agree = sub.add_parser("agreement", help="score a quantized checkpoint against a reference")
    p_agree.add_argument("reference", help="the unquantized model")
    p_agree.add_argument("candidate", help="the checkpoint under test")
    p_agree.add_argument(
        "--baseline",
        help="an already-shipped quantized build; the candidate must be no worse than it",
    )
    p_agree.add_argument("--device", default="cuda")
    p_agree.add_argument("--long", action="store_true", help="use one long prompt instead")
    p_agree.add_argument(
        "--chat",
        action="store_true",
        help="score through the chat template; required for instruction-tuned models",
    )
    p_agree.set_defaults(func=cmd_agreement)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
