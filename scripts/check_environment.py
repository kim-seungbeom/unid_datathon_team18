"""Print software/environment metadata only; never inspect competition files."""

import argparse
import importlib.metadata
import platform
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require_cuda", action="store_true")
    args = parser.parse_args()
    print(f"Python: {platform.python_version()}")
    print(f"Interpreter: {sys.executable}")
    print(f"OS: {platform.platform()}")
    missing = []
    for package in ("torch", "numpy", "pandas", "Pillow"):
        try:
            print(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        raise SystemExit("Missing dependencies: " + ", ".join(missing))
    import torch

    version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    if sys.version_info < (3, 10) or version < (2, 7):
        raise SystemExit("This starter requires Python >=3.10 and torch >=2.7.")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA runtime: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: check the organizer environment; do not replace torch.")


if __name__ == "__main__":
    main()
