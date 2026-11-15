"""Audit legacy image-cache precision without fitting or inspecting label values."""

from __future__ import annotations

import importlib.metadata
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from experiments.seungbeom.export_submissions import sha256

POLICIES = ("fp32", "cudnn_tf32")
CACHE_FILES = ("metadata.json", "features.npy", "sample_ids.csv")


@contextmanager
def precision_context(policy):
    """Set extraction math explicitly and restore caller state even on failure."""
    import torch

    if policy not in POLICIES:
        raise ValueError("Unsupported image precision policy.")
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = policy == "cudnn_tf32"
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def runtime_signature(device):
    """Software and execution metadata only; no dataset or model contents."""
    import torch

    target = torch.device(device)
    packages = {}
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "joblib",
        "Pillow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "packages": packages,
        "device": str(target),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else None,
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def probe_positions(length):
    """At most 32 evenly spaced observations, chosen without labels or model scores."""
    if type(length) is not int or length < 1:
        raise ValueError("Precision audit requires nonempty input.")
    return np.linspace(0, length - 1, min(32, length), dtype=np.int64)


def matching_policy(reference, candidates):
    """Require exact vector equality; never choose a policy by validation accuracy."""
    reference = np.asarray(reference)
    if reference.ndim != 2 or min(reference.shape) < 1 or not np.isfinite(reference).all():
        raise ValueError("Invalid reference image vectors.")
    if set(candidates) != set(POLICIES):
        raise ValueError("Audit both fixed image precision policies.")
    exact, differences = {}, {}
    for policy in POLICIES:
        current = np.asarray(candidates[policy])
        if current.shape != reference.shape or not np.isfinite(current).all():
            raise ValueError("Invalid live image vectors.")
        exact[policy] = bool(np.array_equal(reference, current))
        differences[policy] = float(
            np.max(np.abs(reference.astype(np.float64) - current.astype(np.float64)))
        )
    matching = [policy for policy in POLICIES if exact[policy]]
    if not matching:
        raise ValueError(
            "Neither image precision policy exactly reproduces the reference cache. "
            "Check original package, extraction batch/device, and image versions before predicting."
        )
    # A CPU run can produce the same vectors for both policies. Prefer fp32 in a tie.
    return matching[0], {
        "exact_match_by_policy": exact,
        "max_abs_difference_by_policy": differences,
    }


def reference_hashes(reference_dir, recorded):
    directory = Path(reference_dir)
    if not isinstance(recorded, dict) or set(recorded) != set(CACHE_FILES):
        raise ValueError("Neural identity must record the original Validation cache file hashes.")
    hashes = {str((directory / name).resolve()): sha256(directory / name) for name in CACHE_FILES}
    if any(hashes[str((directory / name).resolve())] != recorded[name] for name in CACHE_FILES):
        raise ValueError("Reference cache differs from the neural run's recorded Validation cache.")
    return hashes


def extract_probe(frame, image_root, loaded, policy, device):
    """Use current images with the shared extractor and verify both member contracts."""
    from experiments.seungbeom import multimodal as mm

    bundle = loaded["bundle"]
    with precision_context(policy):
        features, meta = mm.make_features(
            frame,
            None,
            image_root,
            bundle["sensor_spec"],
            encoder=bundle["encoder"],
            embedding_spec=bundle["embedding"],
            device=device,
            batch_size=1,
            workers=0,
        )
    contract = mm.embedding_info(meta)
    if contract != mm.embedding_info(bundle["embedding"]) or contract != mm.embedding_info(
        loaded["checkpoint"]["visual_contract"]
    ):
        raise ValueError("Live image contract differs from the saved members.")
    columns = [f"{mm.EMBED_PREFIX}{index:04d}" for index in range(contract["dimension"])]
    return features.loc[:, columns].to_numpy(dtype=np.float32, copy=True)


def audit_cache(frame, image_root, loaded, reference_dir, device):
    """Find an exact legacy-cache precision match on a label-blind probe, or stop."""
    from experiments.seungbeom import multimodal as mm

    recorded = loaded.get("identity", {}).get("input_provenance", {}).get("validation")
    files = reference_hashes(reference_dir, recorded)
    probe = frame.iloc[probe_positions(len(frame))].copy()
    reference, meta = mm.load_embeddings(reference_dir, probe["sample_id"])
    contract = mm.embedding_info(meta)
    if contract != mm.embedding_info(
        loaded["checkpoint"]["visual_contract"]
    ) or contract != mm.embedding_info(loaded["bundle"]["embedding"]):
        raise ValueError("Reference image contract differs from the saved members.")
    candidates = {
        policy: extract_probe(probe, image_root, loaded, policy, device) for policy in POLICIES
    }
    chosen, diagnostics = matching_policy(reference, candidates)
    if files != reference_hashes(reference_dir, recorded):
        raise ValueError("Reference cache changed during the precision audit.")
    return {
        "image_precision": chosen,
        "reference_files": files,
        "probe": {
            "rows": len(probe),
            "selection": "evenly_spaced_input_positions_without_labels",
            **diagnostics,
        },
        "cache_environment": {
            key: meta.get(key) for key in ("batch_size", "workers", "device", "packages")
        },
    }
