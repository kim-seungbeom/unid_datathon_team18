"""Internal image processing. Console output contains progress/shape only, never rows."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch
from encoders import DIMENSIONS, FILL, MEAN, STD, build_encoder, preprocess
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ExtractionError(RuntimeError):
    pass


@dataclass(repr=False, frozen=True)
class Row:
    sample_id: str
    image: Path


def read_manifest(csv_path: Path, image_root: Path) -> list[Row]:
    root = Path(image_root).resolve()
    rows, seen = [], set()
    try:
        with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"sample_id", "image_path"}.issubset(reader.fieldnames or []):
                raise ExtractionError("Manifest requires sample_id and image_path.")
            for item in reader:
                # Labels and sensor values are neither retained nor used by the encoder.
                sid = item["sample_id"]
                rel = Path(item["image_path"] or "")
                if not sid or sid in seen:
                    raise ExtractionError("Empty or duplicate sample IDs.")
                if rel.is_absolute() or ".." in rel.parts:
                    raise ExtractionError("Image path must stay under image_root.")
                candidates = [(root / rel).resolve(), (root / rel.name).resolve()]
                image = next(
                    (
                        p
                        for p in candidates
                        if root in p.parents and p.is_file() and p.suffix.lower() == ".png"
                    ),
                    None,
                )
                if image is None:
                    raise ExtractionError("Missing or invalid image. Check the manifest locally.")
                rows.append(Row(sid, image))
                seen.add(sid)
    except ExtractionError:
        raise
    except Exception:
        raise ExtractionError(
            "Could not read the input manifest; raw details suppressed."
        ) from None
    if not rows:
        raise ExtractionError("Manifest is empty.")
    return rows


class Images(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        try:
            with Image.open(self.rows[index].image) as image:
                value = preprocess(image)
        except Exception:
            raise ExtractionError(
                "Image decoding failed; raw file information suppressed."
            ) from None
        return value, index


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


@torch.inference_mode()
def extract(
    *,
    model_name,
    csv_path,
    image_root,
    output_dir,
    batch_size=32,
    workers=4,
    device="cuda",
    l2_normalize=False,
    _encoder=None,
):
    """Returns aggregate metadata. Outputs are local/private; no training or fitting."""
    if model_name not in DIMENSIONS or batch_size < 1 or workers < 0:
        raise ExtractionError("Invalid model, batch size, or worker count.")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ExtractionError("CUDA unavailable; select the competition server's Python kernel.")
    rows = read_manifest(Path(csv_path), Path(image_root))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)  # No overwriting a completed or partial extraction.
    metadata = {
        "status": "running",
        "model": model_name,
        "rows": len(rows),
        "dimension": DIMENSIONS[model_name],
        "dtype": "float32",
        "device": str(device),
        "batch_size": batch_size,
        "workers": workers,
        "l2_normalize": l2_normalize,
        "preprocessing": {
            "size": 224,
            "resize": "bilinear_letterbox",
            "fill": FILL,
            "mean": MEAN,
            "std": STD,
            "augmentation": "none",
        },
        "row_order": "same_as_input_csv",
        "trained": False,
    }
    write_json(output / "metadata.json", metadata)
    started = time.perf_counter()
    features = encoder = None
    try:
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        encoder = (
            (_encoder if _encoder is not None else build_encoder(model_name)).to(device).eval()
        )
        if encoder.name != model_name or encoder.dimension != DIMENSIONS[model_name]:
            raise ExtractionError("Encoder does not match the requested model.")
        metadata["weights"] = encoder.source
        metadata["packages"] = {}
        for package in ("torch", "torchvision", "transformers", "numpy", "pillow"):
            try:
                metadata["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                metadata["packages"][package] = "not_installed"
        loader = DataLoader(
            Images(rows),
            batch_size=batch_size,
            num_workers=workers,
            shuffle=False,
            drop_last=False,
            pin_memory=device.type == "cuda",
        )
        features = np.lib.format.open_memmap(
            output / "features.partial.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), encoder.dimension),
        )
        completed = 0
        for step, (images, indices) in enumerate(loader):
            value = encoder(images.to(device, non_blocking=True)).float()
            if not torch.isfinite(value).all():
                raise ExtractionError("Encoder produced non-finite features.")
            if l2_normalize:
                value = torch.nn.functional.normalize(value, dim=1)
            features[indices.numpy()] = value.cpu().numpy()
            completed += len(indices)
            if step % 50 == 0 or completed == len(rows):
                print(f"{model_name}: {completed}/{len(rows)} images", flush=True)
        features.flush()
        del features
        features = None
        with (output / "sample_ids.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id"])
            writer.writerows((row.sample_id,) for row in rows)
        digest = hashlib.sha256()
        for row in rows:
            digest.update(row.sample_id.encode("utf-8") + b"\0")
        metadata.update(
            status="complete",
            seconds=time.perf_counter() - started,
            sample_order_sha256=digest.hexdigest(),
        )
        (output / "features.partial.npy").replace(output / "features.npy")
        write_json(output / "metadata.json", metadata)
    except BaseException as error:
        metadata.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
        )
        write_json(output / "metadata.json", metadata)
        if isinstance(error, (KeyboardInterrupt, ExtractionError)):
            raise
        raise ExtractionError(
            f"Extraction failed ({type(error).__name__}); use a new output folder after fixing it."
        ) from None
    finally:
        if features is not None:
            features.flush()
        del features, encoder
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return metadata


def parser(model_name=None):
    p = argparse.ArgumentParser(description="Frozen image feature extraction; no model training")
    if model_name is None:
        p.add_argument("--model_name", choices=list(DIMENSIONS), required=True)
    p.add_argument("--input", dest="csv_path", type=Path, required=True)
    p.add_argument(
        "--image_root", type=Path, required=True, help="Split folder or its images folder"
    )
    p.add_argument(
        "--output", dest="output_dir", type=Path, required=True, help="New output directory"
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--l2_normalize", action="store_true")
    return p


def main(model_name=None):
    args = vars(parser(model_name).parse_args())
    if model_name is not None:
        args["model_name"] = model_name
    result = extract(**args)
    print(json.dumps({k: result[k] for k in ("status", "model", "rows", "dimension", "seconds")}))


if __name__ == "__main__":
    main()
