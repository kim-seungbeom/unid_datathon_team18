"""Torch-only visual IO: frozen vectors or the shared current-image loader."""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import BatteryDataset

ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS = {"resnet18": 512, "dinov2": 384, "convnext_tiny": 768}


def embedding_contract(meta):
    keys = ("model", "weights", "dimension", "preprocessing", "l2_normalize", "trained")
    contract = {key: meta.get(key) for key in keys}
    if (
        contract["model"] not in DIMENSIONS
        or contract["dimension"] != DIMENSIONS[contract["model"]]
        or contract["trained"] is not False
        or not isinstance(contract["weights"], str)
        or not contract["weights"]
        or not isinstance(contract["preprocessing"], dict)
        or not contract["preprocessing"]
        or type(contract["l2_normalize"]) is not bool
    ):
        raise ValueError("Incomplete frozen embedding contract.")
    return contract


def load_vectors(directory, sample_ids):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    contract = embedding_contract(meta)
    if meta.get("status") != "complete":
        raise ValueError("Image extraction is not complete.")
    ids = pd.read_csv(directory / "sample_ids.csv", dtype=str, keep_default_na=False)["sample_id"]
    requested = pd.Index(sample_ids)
    if (
        not ids.is_unique
        or ids.str.strip().eq("").any()
        or not requested.is_unique
        or requested.hasnans
        or len(requested) == 0
    ):
        raise ValueError("Embedding and requested IDs must be nonempty and unique.")
    vectors = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
    if vectors.shape != (len(ids), contract["dimension"]):
        raise ValueError("Embedding array and metadata dimensions differ.")
    if "rows" in meta and meta["rows"] != len(ids):
        raise ValueError("Embedding metadata rows differs from the ID count.")
    if "sample_order_sha256" in meta:
        digest = hashlib.sha256()
        for sample_id in ids:
            digest.update(sample_id.encode("utf-8") + b"\0")
        if digest.hexdigest() != meta["sample_order_sha256"]:
            raise ValueError("Embedding ID order differs from the recorded hash.")
    positions = pd.Index(ids).get_indexer(requested)
    if (positions < 0).any():
        raise ValueError("Some input samples do not have embeddings.")
    result = np.array(vectors[positions], dtype=np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Embedding values must be finite.")
    return result, contract


def visual_vectors(
    frame, image_root, config, *, cache=None, expected=None, device="cuda", work_dir=None
):
    if cache is not None:
        vectors, contract = load_vectors(cache, frame["sample_id"])
    else:
        path = str(ROOT / "image_features")
        if path not in sys.path:
            sys.path.append(path)
        from image_features.extract_features import extract

        with tempfile.TemporaryDirectory(prefix="neural_visual_") as folder:
            manifest = Path(folder) / "input.csv"
            frame[["sample_id", "image_path"]].to_csv(manifest, index=False)
            output = Path(work_dir) if work_dir else Path(folder) / "embeddings"
            extract(
                model_name=config["embedding_encoder"],
                csv_path=manifest,
                image_root=Path(image_root),
                output_dir=output,
                batch_size=1,
                workers=0,
                device=device,
                l2_normalize=bool((expected or {}).get("l2_normalize", False)),
            )
            vectors, contract = load_vectors(output, frame["sample_id"])
    if (
        contract["model"] != config["embedding_encoder"]
        or contract["dimension"] != config["embedding_dim"]
    ):
        raise ValueError("Embedding encoder differs from the config.")
    if expected is not None and contract != expected:
        raise ValueError("Embedding weights/preprocessing differ from training.")
    return vectors, contract


class VisualDataset(BatteryDataset):
    def __init__(self, frame, image_root, preprocessor, config, *, vectors=None, labeled=True):
        if (config["encoder"] == "embedding") != (vectors is not None):
            raise ValueError("Visual data does not match the neural encoder.")
        super().__init__(
            frame,
            image_root,
            preprocessor,
            config["image_size"],
            labeled=labeled,
            use_images=vectors is None,
        )
        if vectors is not None and len(vectors) != len(frame):
            raise ValueError("Image vectors and input rows differ.")
        self.vectors = vectors

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["visual"] = (
            torch.from_numpy(self.vectors[index])
            if self.vectors is not None
            else item.pop("images")
        )
        return item
