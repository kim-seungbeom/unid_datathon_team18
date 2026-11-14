"""Original row-uniform frozen training with an immediate best-checkpoint callback.

Kept separate from historical modules so saved source fingerprints remain valid.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate import classification_metrics
from experiments.seungbeom import resnet_finetune as ft
from schema import RAW_SENSOR_COLUMNS, validate_split


def probability_digest(values):
    array = np.ascontiguousarray(values, dtype=np.float32)
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def save_checkpoint(path, model, prep, identity):
    """Snapshot tensors, not references; atomically replace only this run's checkpoint."""
    checkpoint = {
        **identity,
        "preprocessor": prep.to_dict(),
        "state_dict": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train_phase(
    frame,
    image_root,
    config,
    device,
    *,
    epochs,
    stopping=None,
    progress=None,
    on_best=None,
    expected_initial=None,
    expected_backbone=None,
):
    """Same CE, seed, batches, AdamW and epoch rule as ft.train_phase(frozen).

    on_best runs synchronously on STRICT inner-holdout improvement, before the
    next optimizer update. No official Validation is accepted for selection.
    The caller supplies the predefined Train-only split and its provenance.
    """
    config = ft.normalize_config(config)
    if type(epochs) is not int or not 1 <= epochs <= config["epochs"]:
        raise ValueError("Invalid epoch count.")
    if (stopping is None) != (on_best is None):
        raise ValueError("Inner selection requires on_best; refit must not select checkpoints.")
    ft.seed_everything(config["seed"])
    prep = ft.NeuralSensorPreprocessor.fit(
        frame, RAW_SENSOR_COLUMNS, imputation="median", scaling="standard"
    )
    dataset = ft.FineTuneDataset(frame, image_root, prep)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    model = ft.FineTuneModel("frozen", **ft.model_kwargs(config), pretrained=True).to(device)
    initial, backbone = ft.state_digest(model), ft.state_digest(model.backbone)
    if (expected_initial is not None and initial != expected_initial) or (
        expected_backbone is not None and backbone != expected_backbone
    ):
        raise ValueError("Initialization/backbone differs from the original frozen control.")
    optimizer = ft.make_optimizer(
        model,
        head_lr=config["head_lr"],
        backbone_lr=config["backbone_lr"],
        weight_decay=config["weight_decay"],
    )
    stop_loader = None
    stopper = ft.neural.EarlyStopping(config["patience"])
    if stopping is not None:
        stop_frame, stop_root = stopping
        validate_split(frame, stop_frame)
        stop_loader = ft.evaluation_loader(stop_frame, stop_root, prep, config)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        started, total, seen = time.monotonic(), 0.0, 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["visual"].to(device), batch["sensors"].to(device))
            loss = torch.nn.functional.cross_entropy(logits, batch["targets"].to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
            ):
                raise ValueError("Nonfinite training gradients.")
            optimizer.step()
            total += float(loss.detach()) * len(batch["targets"])
            seen += len(batch["targets"])
            if seen // 4096 != (seen - len(batch["targets"])) // 4096:
                print(f"frozen epoch={epoch}/{epochs} rows={seen}/{len(dataset)}", flush=True)
        row = {"epoch": epoch, "train_loss": total / len(dataset)}
        should_stop = False
        if stop_loader is not None:
            print("Scoring Train-only inner holdout...", flush=True)
            prob = ft.probabilities(model, stop_loader, device, progress_every=2048)
            row["inner_macro_f1"] = classification_metrics(stop_frame["label"], prob.argmax(1) + 1)[
                "macro_f1"
            ]
            improved, should_stop = stopper.update(row["inner_macro_f1"], epoch)
            if improved:
                on_best(model, prep, epoch, row["inner_macro_f1"], probability_digest(prob))
        row["elapsed_seconds"] = time.monotonic() - started
        history.append(row)
        if progress is not None:
            progress(row)
        if should_stop:
            break
    if ft.state_digest(model.backbone) != backbone:
        raise ValueError("Frozen backbone or BatchNorm state changed.")
    return model, prep, stopper.best_epoch if stopping is not None else epochs, history, initial
