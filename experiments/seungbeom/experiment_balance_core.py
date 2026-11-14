"""Train-only experiment weights and a matched frozen ResNet18 training phase."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from evaluate import classification_metrics
from experiments.seungbeom import resnet_finetune as ft
from schema import RAW_SENSOR_COLUMNS, validate_labels, validate_split

OBJECTIVE = "mean_cross_entropy_times_N_over_G_over_group_rows"
PAIRS = ((3, 2), (5, 4), (5, 3), (2, 3), (4, 5))


def row_weights(frame):
    """Mean-one weights; each experiment has total weight N/G within this fit split."""
    if frame.empty or "experiment_id" not in frame:
        raise ValueError("Training requires nonempty experiment groups.")
    groups = frame["experiment_id"]
    if (
        groups.isna().any()
        or not groups.map(lambda value: isinstance(value, str) and bool(value.strip())).all()
    ):
        raise ValueError("Training experiment IDs must be nonempty strings.")
    counts = groups.map(groups.value_counts()).to_numpy(dtype=np.float64)
    weights = len(frame) / (groups.nunique() * counts)
    return weights.astype(np.float32)


def objective(logits, targets, weights):
    # Global mean-one weights, NOT normalization by the minibatch's weight sum.
    # Even for batch_size=1 the observation must retain its experiment weight.
    if weights.shape != targets.shape or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Each training row needs a finite positive weight.")
    return (F.cross_entropy(logits, targets, reduction="none") * weights).mean()


class WeightedDataset(ft.FineTuneDataset):
    def __init__(self, frame, image_root, prep):
        super().__init__(frame, image_root, prep)
        self.weights = torch.from_numpy(row_weights(frame))

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["weight"] = self.weights[index]
        return item


def train_phase(
    frame,
    image_root,
    config,
    device,
    *,
    epochs,
    stopping=None,
    progress=None,
    expected_initial=None,
    expected_backbone=None,
    balanced=True,
):
    """Match ft.train_phase: same initialization, shuffle, AdamW, preprocessing and stopping.

    balanced=False exists for synthetic equivalence tests. Server trials always
    use balanced=True; the unchanged saved row-uniform model is the control.
    """
    config = ft.normalize_config(config)
    if type(epochs) is not int or not 1 <= epochs <= config["epochs"]:
        raise ValueError("Invalid epoch count.")
    ft.seed_everything(config["seed"])
    prep = ft.NeuralSensorPreprocessor.fit(
        frame, RAW_SENSOR_COLUMNS, imputation="median", scaling="standard"
    )
    dataset = WeightedDataset(frame, image_root, prep)
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
        stop_loader = ft.evaluation_loader(stop_frame, stop_root, prep, config, labeled=False)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        started, total, seen = time.monotonic(), 0.0, 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["visual"].to(device), batch["sensors"].to(device))
            targets = batch["targets"].to(device)
            loss = (
                objective(logits, targets, batch["weight"].to(device))
                if balanced
                else (F.cross_entropy(logits, targets))
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
            ):
                raise ValueError("Nonfinite training gradients.")
            optimizer.step()
            total += float(loss.detach()) * len(targets)
            seen += len(targets)
            if seen // 4096 != (seen - len(targets)) // 4096:
                print(f"balanced epoch={epoch}/{epochs} rows={seen}/{len(dataset)}", flush=True)
        row = {"epoch": epoch, "train_loss": total / len(dataset)}
        should_stop = False
        if stop_loader is not None:
            print("Scoring unweighted inner Train holdout...", flush=True)
            prob = ft.probabilities(model, stop_loader, device, progress_every=2048)
            row["inner_macro_f1"] = classification_metrics(stop_frame["label"], prob.argmax(1) + 1)[
                "macro_f1"
            ]
            _, should_stop = stopper.update(row["inner_macro_f1"], epoch)
        row["elapsed_seconds"] = time.monotonic() - started
        history.append(row)
        if progress is not None:
            progress(row)
        if should_stop:
            break
    if ft.state_digest(model.backbone) != backbone:
        raise ValueError("Frozen backbone or BatchNorm state changed.")
    return model, prep, stopper.best_epoch if stopping is not None else epochs, history, initial


def pair_errors(frame, predicted):
    true, pred = validate_labels(frame["label"]), validate_labels(predicted)
    if len(true) != len(pred):
        raise ValueError("Prediction lengths differ.")
    rows = []
    for group, positions in frame.reset_index(drop=True).groupby("experiment_id").indices.items():
        for source, target in PAIRS:
            support = int(np.count_nonzero(true[positions] == source))
            count = int(np.count_nonzero((true[positions] == source) & (pred[positions] == target)))
            rows.append(
                {
                    "experiment_id": group,
                    "pair": f"{source}->{target}",
                    "support": support,
                    "errors": count,
                    "rate": count / support if support else None,
                }
            )
    grouped = pd.DataFrame(rows)
    summary = {}
    for source, target in PAIRS:
        name = f"{source}->{target}"
        subset = grouped[(grouped["pair"] == name) & (grouped["support"] > 0)]
        support, count = int(subset["support"].sum()), int(subset["errors"].sum())
        summary[name] = {
            "support": support,
            "errors": count,
            "rate": count / support if support else None,
            "groups_with_support": len(subset),
            "groups_with_errors": int((subset["errors"] > 0).sum()),
            "mean_group_rate": float(subset["rate"].mean()) if len(subset) else None,
        }
    return summary, grouped


def pair_changes(reference, candidate):
    keys = ["experiment_id", "pair"]
    joined = reference.merge(
        candidate,
        on=keys,
        suffixes=("_old", "_new"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all() or not joined.support_old.eq(joined.support_new).all():
        raise ValueError("Pair comparison requires identical experiments and true-class supports.")
    result = {}
    for source, target in PAIRS:
        name = f"{source}->{target}"
        subset = joined[(joined["pair"] == name) & (joined.support_old > 0)]
        delta = subset.errors_new - subset.errors_old
        result[name] = {
            "groups": len(subset),
            "improved": int((delta < 0).sum()),
            "tied": int((delta == 0).sum()),
            "worse": int((delta > 0).sum()),
            "error_count_delta": int(delta.sum()),
        }
    return result
