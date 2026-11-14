"""Reject known private-data paths in the Git index without opening their contents."""

import argparse
import subprocess
from pathlib import Path, PurePosixPath

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
    count = sum(
        blocked_path(name) and not (args.allow_final_weights and name == "weights/model.pt")
        for name in names
    )
    if count:
        raise SystemExit(f"Blocked {count} private/artifact path(s). Unstage and review locally.")
    print(f"Checked {len(names)} staged paths. Review staged contents manually as well.")


if __name__ == "__main__":
    main()
