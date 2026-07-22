"""Phase 4 — publish the fine-tuned model to the Hugging Face Hub.

Uploads `models/beto-final/` (weights + tokenizer + metrics JSONs) and
`docs/HUB_MODEL_CARD.md` (as the Hub repo's README, with the
"not a truth detector" disclaimer, metrics, and BETO/ColombiaCheck
attribution) to a model repo.

Auth: one-time `hf auth login` with a **Write** token. The token stays in the
local HF cache — this script never reads or prints it.

Usage:
    python -m fake_news_co.push_to_hub --dry-run          # check plan + login
    python -m fake_news_co.push_to_hub --private          # first (private) push
    python -m fake_news_co.push_to_hub                    # public push
    python -m fake_news_co.push_to_hub --repo-id USER/name
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

from fake_news_co.paths import MODELS, ROOT

DEFAULT_REPO_ID = "polux89/beto-colombiacheck"  # HF username is lowercase
FINAL_DIR = MODELS / "beto-final"
HUB_CARD = ROOT / "docs" / "HUB_MODEL_CARD.md"

# training_args.bin is a torch pickle only useful to resume training — not
# needed for inference and not something a Hub consumer should unpickle.
IGNORE = ["training_args.bin", "checkpoint-*"]


def _preflight(api: HfApi) -> str:
    """Validate local artifacts + login. Returns the authenticated username."""
    missing = [
        str(p)
        for p in (FINAL_DIR / "model.safetensors", FINAL_DIR / "tokenizer.json", HUB_CARD)
        if not Path(p).exists()
    ]
    if missing:
        sys.exit(f"missing artifacts: {missing}. Train/evaluate first.")
    try:
        user = api.whoami()["name"]
    except Exception:
        sys.exit(
            "Not logged in to Hugging Face. Run `hf auth login` with a Write "
            "token (Settings -> Access Tokens), then retry."
        )
    return user


def push(repo_id: str = DEFAULT_REPO_ID, private: bool = False,
         dry_run: bool = False) -> None:
    api = HfApi()
    user = _preflight(api)
    files = sorted(p.name for p in FINAL_DIR.iterdir() if p.is_file()
                   and p.name not in ("training_args.bin",))
    print(f"logged in as : {user}")
    print(f"target repo  : {repo_id}  ({'private' if private else 'PUBLIC'})")
    print(f"model folder : {FINAL_DIR}  -> {files}")
    print(f"model card   : {HUB_CARD.name} -> README.md")
    if dry_run:
        print("dry run — nothing uploaded.")
        return

    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=FINAL_DIR,
        repo_id=repo_id,
        ignore_patterns=IGNORE,
        commit_message="Upload fine-tuned BETO verdict classifier",
    )
    api.upload_file(
        path_or_fileobj=HUB_CARD,
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card",
    )
    print(f"done -> https://huggingface.co/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--private", action="store_true",
                    help="create/keep the Hub repo private")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate artifacts and login without uploading")
    args = ap.parse_args()
    push(repo_id=args.repo_id, private=args.private, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
