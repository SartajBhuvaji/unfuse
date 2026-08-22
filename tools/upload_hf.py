"""Publish a text-only checkpoint and its GGUF quants to the Hugging Face Hub.

Two repos, matching the convention every GGUF publisher follows:

  <repo>        bf16 safetensors + model card + architecture diagram
  <repo>-GGUF   quantizations AT THE ROOT

The split is not cosmetic. The Hub's hardware-compatibility / quant-picker widget
only renders for root-level GGUF files in a repo with no safetensors. Put the GGUFs
in a subfolder of the main repo and the metadata still indexes, but the widget does
not appear and `llama-cli -hf` / `ollama run hf.co/...` will not resolve them.

Token comes from HF_TOKEN in the environment. Never commit it, and never write it
to a file on a rented machine -- that disk is recycled to the next tenant.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. user/GLM-4.6-Flash-text")
    ap.add_argument("--model-dir", required=True, help="bf16 safetensors directory")
    ap.add_argument("--gguf-dir", help="directory of .gguf files; publishes <repo>-GGUF")
    ap.add_argument("--model-card", help="markdown file to upload as the main README.md")
    ap.add_argument("--gguf-card", help="markdown file to upload as the GGUF README.md")
    ap.add_argument("--diagram", help="svg/png referenced by the model card")
    ap.add_argument("--private", action="store_true", help="default is PUBLIC")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN in the environment (do not hardcode or write to disk)")

    api = HfApi(token=token)
    who = api.whoami()
    print(f"authenticated as {who['name']} (role: {who['auth']['accessToken']['role']})")

    model_dir = Path(args.model_dir)
    if args.model_card:
        shutil.copy(args.model_card, model_dir / "README.md")
    if args.diagram:
        shutil.copy(args.diagram, model_dir / Path(args.diagram).name)

    print(f"\n== {args.repo} ==")
    print("repo:", api.create_repo(args.repo, exist_ok=True, private=args.private))
    t0 = time.time()
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo,
        repo_type="model",
        commit_message="Add weights, model card, architecture diagram",
    )
    print(f"uploaded in {time.time() - t0:.0f}s")

    if args.gguf_dir:
        gguf_repo = f"{args.repo}-GGUF"
        gguf_dir = Path(args.gguf_dir)
        if args.gguf_card:
            shutil.copy(args.gguf_card, gguf_dir / "README.md")

        print(f"\n== {gguf_repo} ==")
        print("repo:", api.create_repo(gguf_repo, exist_ok=True, private=args.private))
        t0 = time.time()
        api.upload_folder(
            folder_path=str(gguf_dir),
            repo_id=gguf_repo,
            repo_type="model",
            commit_message="Add GGUF quantizations at repo root",
        )
        print(f"uploaded in {time.time() - t0:.0f}s")

        _check_widget(api, gguf_repo)


def _check_widget(api: HfApi, repo: str) -> None:
    """The widget's requirements are easy to break silently; assert them out loud."""
    files = api.list_repo_files(repo)
    root = [f for f in files if f.endswith(".gguf") and "/" not in f]
    sub = [f for f in files if f.endswith(".gguf") and "/" in f]
    safet = any(f.endswith(".safetensors") for f in files)
    print(f"\n  gguf at root : {len(root)}")
    print(f"  gguf in subdir: {len(sub)}")
    print(f"  safetensors   : {safet}")
    print("  quant picker will render:", bool(root) and not safet and not sub)


if __name__ == "__main__":
    main()
