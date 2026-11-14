"""Train-only grouped evaluation of the existing frozen-vector 50:50 architecture.

Outer labels are used only for scoring. An inner group split chooses the neural
epoch count, then a new neural model and a cloned tree pipeline fit outer Train.
"""

from pathlib import Path

import numpy as np

from evaluate import classification_metrics
from experiments.seungbeom import midpoint_blend as midpoint
from experiments.seungbeom import multimodal as mm
from experiments.seungbeom import neural_experiment as neural
from experiments.seungbeom.neural_config import model_config
from experiments.seungbeom.neural_data import VisualDataset
from experiments.seungbeom.neural_model import NeuralFusionClassifier
from experiments.seungbeom.neural_sensors import NeuralSensorPreprocessor
from schema import LABELS, validate_labels, validate_split

METHODS = midpoint.METHODS


def split_plan(frame, seed=42):
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    if (
        "experiment_id" not in frame
        or frame["experiment_id"].isna().any()
        or frame["experiment_id"].astype(str).str.strip().eq("").any()
        or frame["sample_id"].isna().any()
        or frame["sample_id"].duplicated().any()
    ):
        raise ValueError("Group CV requires unique sample IDs and nonempty experiment IDs.")
    labels = validate_labels(frame["label"])
    if set(labels) != set(LABELS):
        raise ValueError("The complete reference Train must contain all six classes.")
    groups = frame["experiment_id"].to_numpy()
    if len(set(groups)) < 6:
        raise ValueError("Need at least six experiments for outer and inner group splits.")
    plan = []
    for number, (fit, holdout) in enumerate(GroupKFold(3).split(frame, labels, groups), 1):
        inner = StratifiedGroupKFold(
            n_splits=min(5, len(set(groups[fit]))), shuffle=True, random_state=seed
        )
        inner_fit, inner_stop = next(inner.split(fit, labels[fit], groups[fit]))
        inner_fit, inner_stop = fit[inner_fit], fit[inner_stop]
        validate_split(frame.iloc[fit], frame.iloc[holdout])
        validate_split(frame.iloc[inner_fit], frame.iloc[inner_stop])
        plan.append(
            {
                "fold": number,
                "fit": fit.tolist(),
                "holdout": holdout.tolist(),
                "inner_fit": inner_fit.tolist(),
                "inner_stop": inner_stop.tolist(),
                "counts": {
                    name: {
                        "rows": len(idx),
                        "experiments": len(set(groups[idx])),
                        "support": {str(c): int((labels[idx] == c).sum()) for c in LABELS},
                    }
                    for name, idx in (
                        ("fit", fit),
                        ("holdout", holdout),
                        ("inner_fit", inner_fit),
                        ("inner_stop", inner_stop),
                    )
                },
            }
        )
    coverage = np.concatenate([part["holdout"] for part in plan])
    if sorted(coverage.tolist()) != list(range(len(frame))):
        raise ValueError("Each Train row must be held out exactly once.")
    return plan


def train_neural(frame, vectors, config, device, *, epochs, stopping=None, progress=None):
    """No outer holdout accepted here. stopping is exclusively the inner split."""
    import torch
    from torch.utils.data import DataLoader

    from runtime import seed_everything
    from training_objectives import build_training_components

    seed_everything(config["seed"])
    prep = NeuralSensorPreprocessor.fit(
        frame,
        config["sensor_features"],
        imputation=config["sensor_imputation"],
        scaling=config["sensor_scaling"],
    )
    dataset = VisualDataset(frame, Path("."), prep, config, vectors=vectors)
    options = {"batch_size": config["batch_size"], "num_workers": config["num_workers"]}
    loader = DataLoader(
        dataset, shuffle=True, generator=torch.Generator().manual_seed(config["seed"]), **options
    )
    model = NeuralFusionClassifier(**model_config(config)).to(device)
    criterion, sampler, _ = build_training_components(frame["label"], config, device)
    if sampler is not None:
        raise ValueError("The reference requires shuffled training.")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    stop_loader = None
    stopper = neural.EarlyStopping(config["early_stopping_patience"], config["min_delta"])
    if stopping is not None:
        stop_frame, stop_vectors = stopping
        validate_split(frame, stop_frame)
        stop_loader = DataLoader(
            VisualDataset(stop_frame, Path("."), prep, config, vectors=stop_vectors),
            shuffle=False,
            **options,
        )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model(batch["visual"].to(device), batch["sensors"].to(device)),
                batch["targets"].to(device),
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
            ):
                raise ValueError("Nonfinite gradients.")
            optimizer.step()
            total += float(loss.item()) * len(batch["targets"])
        row = {"epoch": epoch, "train_loss": total / len(dataset)}
        stop = False
        if stop_loader is not None:
            metrics = neural.evaluate(model, stop_loader, device, stable=True)
            row["inner_macro_f1"] = metrics["macro_f1"]
            _, stop = stopper.update(metrics["macro_f1"], epoch)
        history.append(row)
        if progress:
            progress(row)
        if stop:
            break
    selected = stopper.best_epoch if stop_loader is not None else epochs
    # Inner weights are deliberately discarded; only the epoch count is transferred.
    return model, prep, selected, history


def neural_probabilities(model, prep, frame, vectors, device):
    import torch

    sensors = np.array(prep.transform(frame), dtype=np.float32, order="C", copy=True)
    vectors = np.array(vectors, dtype=np.float32, order="C", copy=True)
    model.eval()
    values = []
    with torch.inference_mode():
        for row in range(len(frame)):
            logits = model(
                torch.from_numpy(vectors[row : row + 1]).to(device),
                torch.from_numpy(sensors[row : row + 1]).to(device),
            )
            values.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(values)


def fit_fold(frame, vectors, features, tree_template, config, part, directory, device, contract):
    import joblib
    import torch
    from sklearn.base import clone

    def progress(stage):
        def emit(row):
            inner = f" inner_f1={row['inner_macro_f1']:.6f}" if "inner_macro_f1" in row else ""
            print(
                f"fold={part['fold']} {stage} epoch={row['epoch']} "
                f"loss={row['train_loss']:.6f}{inner}",
                flush=True,
            )

        return emit

    i, s, f, h = (part[key] for key in ("inner_fit", "inner_stop", "fit", "holdout"))
    inner_model, _, epochs, inner_history = train_neural(
        frame.iloc[i],
        vectors[i],
        config,
        device,
        epochs=config["epochs"],
        stopping=(frame.iloc[s], vectors[s]),
        progress=progress("inner_selection"),
    )
    del inner_model
    neural.write_json(directory / "inner_history.json", inner_history)
    model, prep, _, history = train_neural(
        frame.iloc[f],
        vectors[f],
        config,
        device,
        epochs=epochs,
        progress=progress("outer_refit"),
    )
    neural.write_json(directory / "refit_history.json", history)
    torch.save(
        {
            "kind": "seungbeom_group_cv_neural",
            "config": config,
            "model_config": model.get_model_config(),
            "epoch": epochs,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "preprocessor": prep.to_dict(),
            "visual_contract": contract,
        },
        directory / "neural.pt",
    )
    # clone discards all fitted imputers/scalers/PCA and the original forest.
    tree = clone(tree_template)
    print(f"fold={part['fold']} fitting fresh ExtraTrees + preprocessing...", flush=True)
    tree.fit(features.iloc[f], frame.iloc[f]["label"])
    joblib.dump(tree, directory / "extra_trees.joblib")
    print(f"fold={part['fold']} scoring untouched outer holdout...", flush=True)
    probabilities = {
        "neural": neural_probabilities(model, prep, frame.iloc[h], vectors[h], device),
        "extra_trees": mm.predict_proba_six_stable(tree, features.iloc[h]),
    }
    probabilities["blend_50_50"] = midpoint.blend_probabilities(
        probabilities["neural"], probabilities["extra_trees"]
    )
    np.savez_compressed(directory / "oof.npz", positions=np.asarray(h), **probabilities)
    result = {
        "fold": part["fold"],
        "selected_epoch": epochs,
        "inner_epoch_limit_reached": epochs == config["epochs"],
        "counts": part["counts"],
        "metrics": {
            name: classification_metrics(frame.iloc[h]["label"], values.argmax(axis=1) + 1)
            for name, values in probabilities.items()
        },
    }
    neural.write_json(directory / "result.json", result)
    return result


def summarize(labels, results, probabilities):
    return {
        "folds": results,
        "methods": {
            name: {
                "fold_macro_f1_mean": float(
                    np.mean([r["metrics"][name]["macro_f1"] for r in results])
                ),
                "fold_macro_f1_std": float(
                    np.std([r["metrics"][name]["macro_f1"] for r in results], ddof=1)
                ),
                "pooled_oof": classification_metrics(
                    labels, probabilities[name].argmax(axis=1) + 1
                ),
            }
            for name in METHODS
        },
    }
