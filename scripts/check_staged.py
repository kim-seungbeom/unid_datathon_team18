"""Reject private paths; optionally inspect one explicitly approved clean notebook."""

import argparse
import json
import subprocess
from pathlib import Path, PurePosixPath

CLEAN_NOTEBOOK = "image_features/01_extract_features.ipynb"


def clean_notebook(content: bytes) -> bool:
    """Validate staged bytes, not the working copy. Never include contents in errors."""
    try:
        notebook = json.loads(content)
        if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
            return False
        if set(notebook.get("metadata", {})) - {"kernelspec", "language_info"}:
            return False
        for cell in notebook["cells"]:
            if cell.get("cell_type") not in {"code", "markdown"}:
                return False
            if cell.get("outputs") or cell.get("attachments") or cell.get("metadata"):
                return False
            if cell.get("execution_count") is not None:
                return False
            if "data:image/" in "".join(cell.get("source", [])):
                return False
        return True
    except (ValueError, TypeError, AttributeError):
        return False


BLOCKED_DIRS = {
    "open",
    "data",
    "datasets",
    "outputs",
    "runs",
    "logs",
    "checkpoints",
    "artifacts",
    "weights",
    ".venv",
    "venv",
    ".ssh",
    ".aws",
    ".codex",
}
BLOCKED_SUFFIXES = {
    ".csv",
    ".tsv",
    ".parquet",
    ".jsonl",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".joblib",
    ".log",
    ".ipynb",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".zip",
    ".tar",
    ".gz",
    ".pem",
    ".key",
}


def blocked_path(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(set(path.parts) & BLOCKED_DIRS)
        or bool(set(path.suffixes) & BLOCKED_SUFFIXES)
        or (path.name.startswith(".env") and path.name != ".env.example")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow_final_weights",
        action="store_true",
        help="Final release only: allow exactly weights/model.pt; block other artifacts",
    )
    parser.add_argument(
        "--allow_clean_notebook",
        action="store_true",
        help="Allow only the approved feature notebook after checking staged bytes for outputs",
    )
    args = parser.parse_args()
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=True,
    )
    names = [
        name.decode("utf-8", errors="surrogateescape")
        for name in result.stdout.split(b"\0")
        if name
    ]
    count = 0
    for name in names:
        if not blocked_path(name):
            continue
        if args.allow_final_weights and name == "weights/model.pt":
            continue
        if args.allow_clean_notebook and name == CLEAN_NOTEBOOK:
            staged = subprocess.run(
                ["git", "show", f":{name}"],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                check=True,
            )
            if clean_notebook(staged.stdout):
                continue
        count += 1
    if count:
        raise SystemExit(f"Blocked {count} private/artifact path(s). Unstage and review locally.")
    print(f"Checked {len(names)} staged paths. Review staged contents manually as well.")


if __name__ == "__main__":
    main()
