"""Cached-feature OOF head generation adapted from doyu commit 2dae970.

Source: experiments/doyu/prior_correction.py. No correction selection/export here.
The heads are proxies for saved inner_best; the saved epoch is reused, so these
predictions are selection diagnostics, not an independent nested evaluation.
Human-run organizer server only; local tests use synthetic CPU inputs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from experiments.seungbeom import frozen_checkpoint as fc  # noqa: E402
from schema import LABELS, RAW_SENSOR_COLUMNS, read_frame, validate_labels  # noqa: E402

ft = fc.ft
KIND = "doyu_prior_correction"
THIS_FILE = "experiments/seungbeom/prior_oof_reference.py"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- neural OOF
def train_head(features, frame, config, epochs, device):
    """Frozen-arm training with cached backbone features: same seed, init, loader, optimizer."""
    ft.seed_everything(config["seed"])
    prep = ft.NeuralSensorPreprocessor.fit(
        frame, RAW_SENSOR_COLUMNS, imputation="median", scaling="standard"
    )
    sensors = np.asarray(prep.transform(frame), dtype=np.float32)
    targets = validate_labels(frame["label"]) - 1
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)),
        torch.from_numpy(sensors),
        torch.from_numpy(targets.astype(np.int64)),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    # Build the full model so random head initialization matches the image pipeline.
    model = ft.FineTuneModel("frozen", **ft.model_kwargs(config), pretrained=True).to(device)
    optimizer = ft.make_optimizer(
        model, config["head_lr"], config["backbone_lr"], config["weight_decay"]
    )
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for visual, sensor, target in loader:
            visual, sensor, target = visual.to(device), sensor.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.classifier(torch.cat((visual, model.sensor_encoder(sensor)), dim=1))
            loss = torch.nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(target)
        print(f"  epoch {epoch}/{epochs} loss={total / len(dataset):.6f}", flush=True)
    return model, prep


def head_probabilities(model, prep, features, frame, device):
    sensors = torch.from_numpy(np.asarray(prep.transform(frame), dtype=np.float32)).to(device)
    visual = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(device)
    model.eval()
    with torch.inference_mode():
        logits = model.classifier(torch.cat((visual, model.sensor_encoder(sensors)), dim=1))
        return torch.softmax(logits.float(), dim=1).cpu().numpy().astype(np.float64)


def load_tree_oof(tree_run):
    z = np.load(Path(tree_run) / "train_oof.npz", allow_pickle=False)
    names = z["model_names"].tolist()
    if "extra_trees" not in names:
        raise ValueError("Tree OOF has no extra_trees member.")
    return {
        "sample_id": z["sample_ids"].astype(str),
        "label": z["labels"].astype(np.int64),
        "fold": z["fold"].astype(np.int64),
        "extra_trees": z["probabilities"][:, names.index("extra_trees"), :].astype(np.float64),
    }


def load_embeddings(tree_run, split):
    directory = Path(tree_run) / "embeddings" / split
    meta = read_json(directory / "metadata.json")
    if (
        meta.get("status") != "complete"
        or meta.get("model") != "resnet18"
        or meta.get("weights") != ft.WEIGHTS_SOURCE
        or meta.get("l2_normalize") is not False
        or meta.get("dimension") != 512
    ):
        raise ValueError("Embedding cache must be complete raw ImageNet ResNet18 512-d features.")
    features = np.load(directory / "features.npy")
    ids = pd.read_csv(directory / "sample_ids.csv", dtype=str)["sample_id"].to_numpy()
    if (
        features.shape != (len(ids), 512)
        or not np.isfinite(features).all()
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Embedding cache shape/values invalid.")
    return pd.DataFrame(features, index=ids)


def oof(args):
    out = args.output_dir.resolve()
    if out.exists():
        raise ValueError("output_dir exists; choose a new one.")
    device = ft.select_device(args.device)
    frame = read_frame(args.data_dir / "train/train.csv", labeled=True)
    manifest, result = fc.read_result(args.checkpoint_run)
    config = ft.normalize_config(manifest["config"])
    epochs = int(result["selected_epoch"])
    if manifest["inputs"]["train"]["csv_sha256"] != sha256(args.data_dir / "train/train.csv"):
        raise ValueError("Train CSV differs from the checkpoint run.")
    tree = load_tree_oof(args.tree_run)
    frame = frame.set_index("sample_id", drop=False)
    if set(tree["sample_id"]) != set(frame.index) or len(tree["sample_id"]) != len(frame):
        raise ValueError("Tree OOF rows must be exactly the Train rows.")
    frame = frame.loc[tree["sample_id"]].reset_index(drop=True)
    if not np.array_equal(validate_labels(frame["label"]), tree["label"]):
        raise ValueError("Tree OOF labels differ from Train labels.")
    cache = load_embeddings(args.tree_run, "train")
    if set(cache.index) != set(tree["sample_id"]):
        raise ValueError("Embedding cache rows must be exactly the Train rows.")
    features = cache.loc[tree["sample_id"]].to_numpy(dtype=np.float32)
    folds = sorted(set(tree["fold"].tolist()))
    for f in folds:
        fit, held = tree["fold"] != f, tree["fold"] == f
        if set(frame.loc[fit, "experiment_id"]) & set(frame.loc[held, "experiment_id"]):
            raise ValueError("Tree folds are not experiment-disjoint.")
    out.mkdir(parents=True)
    write_json(out / "status.json", {"status": "running"})
    neural = np.full((len(frame), len(LABELS)), np.nan)
    try:
        for f in folds:
            fit, held = tree["fold"] != f, tree["fold"] == f
            print(f"fold {f}: training head on {fit.sum()} rows for {epochs} epochs", flush=True)
            model, prep = train_head(features[fit], frame[fit], config, epochs, device)
            neural[held] = head_probabilities(model, prep, features[held], frame[held], device)
            del model
        if not np.isfinite(neural).all():
            raise ValueError("Some OOF rows were not predicted.")
        np.savez(
            out / "oof.npz",
            sample_ids=tree["sample_id"],
            labels=tree["label"],
            fold=tree["fold"],
            neural=neural,
            extra_trees=tree["extra_trees"],
        )
        info = {
            "kind": KIND + "_oof",
            "config": config,
            "epochs": epochs,
            "neural_member": "frozen head retrained per tree fold on cached embeddings "
            "(proxy for inner_best; same config and epoch count)",
            "tree_run": str(Path(args.tree_run).resolve()),
            "checkpoint_run": str(Path(args.checkpoint_run).resolve()),
            "train_csv_sha256": sha256(args.data_dir / "train/train.csv"),
            "tree_oof_sha256": sha256(Path(args.tree_run) / "train_oof.npz"),
            "embedding_sha256": sha256(Path(args.tree_run) / "embeddings/train/features.npy"),
            "oof_sha256": sha256(out / "oof.npz"),
            "this_file_sha256": sha256(ROOT / THIS_FILE),
            "validation_or_test_used": False,
        }
        write_json(out / "oof.json", info)
        write_json(out / "status.json", {"status": "complete"})
        print("OOF complete; Validation/Test were not read.", flush=True)
    except BaseException as error:
        write_json(out / "status.json", {"status": "failed", "error_type": type(error).__name__})
        raise
