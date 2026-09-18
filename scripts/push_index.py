"""Upload the built index to a Hugging Face dataset repo.

Streamlit Cloud gives roughly 1 GB of RAM and a small ephemeral disk, and the
built artifact is larger than GitHub's 100 MB per-file limit. Hosting it as a
HF dataset keeps the git repo small and lets the app pull the index at startup
behind st.cache_resource, decoupled from code deploys.

NOT run as part of the build -- it publishes data to a public repo and needs a
token. Run it yourself when you are ready:

    huggingface-cli login
    python scripts/push_index.py --repo <your-username>/recipe-rag-index

Then, in the Streamlit app:

    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id="<your-username>/recipe-rag-index",
                             repo_type="dataset")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag import config


# What the running app actually reads.
UPLOAD_PATTERNS = ["chroma_db/*", "chroma_db/**/*", "parents.parquet", "build_report.json"]


def uploaded_files(root: Path):
    from fnmatch import fnmatch

    for child in sorted(root.rglob("*")):
        if not child.is_file():
            continue
        relative = child.relative_to(root).as_posix()
        if any(fnmatch(relative, pattern) for pattern in UPLOAD_PATTERNS):
            yield child, relative


def directory_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="e.g. username/recipe-rag-index")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be uploaded and exit")
    args = parser.parse_args()

    if not config.PARENTS_PARQUET.exists():
        print(f"No index at {config.ARTIFACT_DIR}. Run build_index.py first.",
              file=sys.stderr)
        return 1

    files = list(uploaded_files(config.ARTIFACT_DIR))
    total = sum(child.stat().st_size for child, _ in files) / 1024 / 1024
    print(f"artifact dir : {config.ARTIFACT_DIR}")
    print(f"uploading    : {len(files)} files, {total:.1f} MB")
    print(f"(whole dir is {directory_size_mb(config.ARTIFACT_DIR):.1f} MB; the rest is "
          f"logs, evaluation output and the browsable copy)")
    for child, relative in files:
        print(f"  {relative}  {child.stat().st_size / 1024 / 1024:.1f} MB")

    if args.dry_run:
        print("\ndry run: nothing uploaded")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private,
                    exist_ok=True)
    api.upload_folder(
        folder_path=str(config.ARTIFACT_DIR),
        repo_id=args.repo,
        repo_type="dataset",
        # Only what the app needs to answer. The browsable SQLite copy is 87 MB
        # of convenience, and the logs and evaluation output are this machine's
        # history, not part of the product.
        allow_patterns=UPLOAD_PATTERNS,
        commit_message="Recipe RAG index: Chroma children + Parquet parents",
    )
    print(f"\nuploaded to https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
