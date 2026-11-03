#!/usr/bin/env python3
"""Upload GGUF files + README to a Hub model repo (no safetensors)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--readme", required=True, help="markdown → README.md at repo root")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("files", nargs="+", help="paths to .gguf files (repo root names = basenames)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN in the environment")

    api = HfApi(token=token)
    who = api.whoami()
    print(f"authenticated as {who['name']}")

    api.create_repo(args.repo_id, exist_ok=True, private=args.private, repo_type="model")
    print("repo:", args.repo_id, "private=", args.private)

    t0 = time.time()
    api.upload_file(
        path_or_fileobj=str(args.readme),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add draft model card",
    )
    print(f"README uploaded in {time.time() - t0:.0f}s")

    for path in args.files:
        p = Path(path)
        if not p.is_file():
            sys.exit(f"missing {p}")
        print(f"\n== uploading {p.name} ({p.stat().st_size / 1e9:.2f} GB) ==")
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=p.name,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Add {p.name}",
        )
        print(f"done in {time.time() - t0:.0f}s")

    files = api.list_repo_files(args.repo_id)
    root_gguf = [f for f in files if f.endswith(".gguf") and "/" not in f]
    print(f"\nroot gguf files: {root_gguf}")
    print(
        "quant widget ok:", bool(root_gguf) and not any(f.endswith(".safetensors") for f in files)
    )


if __name__ == "__main__":
    main()
