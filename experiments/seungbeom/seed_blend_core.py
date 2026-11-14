"""Load completed seed repeats and share one live image extraction across fixed mixtures."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from experiments.seungbeom import blend_ratios, midpoint_blend, neural_experiment
from experiments.seungbeom import neural_seed_repeat as repeat
from experiments.seungbeom.ensemble_core import validate_probabilities
from experiments.seungbeom.export_submissions import checked_score, read_json, sha256
from experiments.seungbeom.neural_config import load_config, model_config
from experiments.seungbeom.neural_data import embedding_contract
from experiments.seungbeom.neural_sensors import NeuralSensorPreprocessor
from experiments.seungbeom.replay_precision import precision_context
from schema import LABELS, RAW_SENSOR_COLUMNS

SEEDS = (42, 43, 44)
METRIC_KEYS = (
    "macro_f1",
    "class_f1",
    "class_precision",
    "class_recall",
    "support",
    "confusion_matrix",
)
METHODS = (
    "original_neural",
    "extra_trees",
    "neural_seed42",
    "neural_seed43",
    "neural_seed44",
    "neural_mean",
    "baseline_50_50",
    "seed44_50_50",
    "seed_mean_50_50",
)
RUN_FILES = (
    "status.json",
    "config.json",
    "identity.json",
    "best_metrics.json",
    "best.pt",
    "environment.json",
)


def metric_projection(metrics):
    """Remove epoch/CE loss, retaining every root classification metric for exact replay."""
    if not isinstance(metrics, dict) or not set(METRIC_KEYS).issubset(metrics):
        raise ValueError("Saved member metrics must contain the full six-class evaluation.")
    checked_score(metrics)
    labels = {str(label) for label in LABELS}
    for key in ("class_f1", "class_precision", "class_recall", "support"):
        if not isinstance(metrics[key], dict) or set(metrics[key]) != labels:
            raise ValueError("Saved class metrics must have the aligned labels 1..6.")
    if np.asarray(metrics["confusion_matrix"]).shape != (6, 6):
        raise ValueError("Saved confusion matrix must be 6 by 6.")
    return {key: copy.deepcopy(metrics[key]) for key in METRIC_KEYS}


def _load_neural(run, config, identity, baseline):
    """Load a trusted neural checkpoint only; the large tree bundle is already loaded."""
    import torch

    from experiments.seungbeom.neural_model import NeuralFusionClassifier

    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if (
        checkpoint.get("kind") != "seungbeom_neural_fusion"
        or checkpoint.get("format_version") != 1
        or checkpoint.get("label_values") != list(LABELS)
        or checkpoint.get("config") != config
        or checkpoint.get("model_config") != model_config(config)
    ):
        raise ValueError("Seed checkpoint/config/label contract differs from its completed run.")
    preprocessor = NeuralSensorPreprocessor.from_dict(checkpoint["preprocessor"])
    if (
        preprocessor.feature_names != RAW_SENSOR_COLUMNS
        or preprocessor.imputation != "median"
        or preprocessor.scaling != "standard"
        or preprocessor.uses_history
    ):
        raise ValueError("All seed members require saved raw5/median/standard preprocessing.")
    if any(
        not np.array_equal(getattr(preprocessor, name), getattr(baseline["preprocessor"], name))
        for name in ("medians", "means", "scales")
    ):
        raise ValueError("Seed preprocessing statistics differ from the original training split.")
    if embedding_contract(checkpoint["visual_contract"]) != embedding_contract(
        baseline["checkpoint"]["visual_contract"]
    ):
        raise ValueError("Seed image weights or preprocessing differ from the original member.")
    model = NeuralFusionClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return {
        "config": config,
        "checkpoint": checkpoint,
        "preprocessor": preprocessor,
        "model": model,
        "identity": identity,
    }


def load_members(base_recipe, experiment_dir, device):
    """Validate metadata and load models without opening any dataset CSV or image cache."""
    directory = Path(experiment_dir).resolve()
    files = dict(base_recipe["provenance"]["files"])
    files.update(
        {
            str(directory / name): sha256(directory / name)
            for name in ("experiment.json", "plan.json")
        }
    )
    baseline = blend_ratios.load_base(base_recipe, device)
    manifest = repeat.read_experiment(directory)
    original = Path(base_recipe["neural_run"]).resolve()
    if Path(manifest["reference_run"]).resolve() != original:
        raise ValueError("Seed repeats must reference the original midpoint neural run.")
    if (
        manifest["reference_config"] != baseline["config"]
        or manifest["reference_identity"] != baseline["identity"]
    ):
        raise ValueError("Seed repeat reference differs from the original neural identity.")
    if manifest["helper_sha256"] != repeat.helper_hashes():
        raise ValueError("Seed repeat helper source changed after preparation.")
    expected = manifest["expected_identity"]
    current_sources = {
        name: sha256(neural_experiment.ROOT / name) for name in neural_experiment.SOURCE_FILES
    }
    if expected.get("source_sha256") != current_sources:
        raise ValueError("Seed training source differs from the current model implementation.")
    if expected.get("config") != baseline["config"]:
        raise ValueError("Seed experiment changed settings beyond the random seed.")
    for key in ("train_csv_sha256", "validation_csv_sha256", "input_provenance"):
        if expected.get(key) != baseline["identity"].get(key):
            raise ValueError("Seed repeats must use the original CSVs and image cache bytes.")
    environment = repeat.environment_signature(repeat.current_environment(device))
    if environment is None or environment != manifest["environment"]:
        raise ValueError("Runtime differs from the seed training environment.")
    runs = {}
    if manifest["reuse_reference"]:
        runs[42] = original
    for trial in manifest["plan"]["trials"]:
        seed = trial["config"]["seed"]
        if seed in runs:
            raise ValueError("Each seed must have exactly one completed source run.")
        runs[seed] = directory / "runs" / trial["id"]
    if set(runs) != set(SEEDS):
        raise ValueError("The mean requires exactly seeds 42, 43, and 44.")
    files.update(manifest["reference_files"])
    members = {}
    metrics = {
        "original_neural": metric_projection(base_recipe["metrics"]["neural"]),
        "extra_trees": metric_projection(base_recipe["metrics"]["extra_trees"]),
        "baseline_50_50": metric_projection(base_recipe["metrics"]["blend_50_50"]),
    }
    for seed in SEEDS:
        run = runs[seed]
        files.update({str(run / name): sha256(run / name) for name in RUN_FILES})
        if read_json(run / "status.json").get("status") != "complete":
            raise ValueError("All seed source runs must be complete.")
        config = load_config(run / "config.json")
        if config != {**baseline["config"], "seed": seed}:
            raise ValueError("Seed member config differs beyond its random seed.")
        identity = read_json(run / "identity.json")
        if identity != {**expected, "config": config}:
            raise ValueError("Seed members must share input, cache, and source identity.")
        if repeat.environment_signature(read_json(run / "environment.json")) != environment:
            raise ValueError("Seed members have different recorded training environments.")
        members[seed] = (
            baseline if run == original else _load_neural(run, config, identity, baseline)
        )
        metrics[f"neural_seed{seed}"] = metric_projection(read_json(run / "best_metrics.json"))
    midpoint_blend.check_files(base_recipe["provenance"]["files"])
    repeat.check_files(manifest["reference_files"])
    midpoint_blend.check_files(files)
    return {"baseline": baseline, "seeds": members}, {
        "files": files,
        "seed_runs": {str(seed): str(runs[seed]) for seed in SEEDS},
        "expected_metrics": metrics,
        "train_csv_sha256": expected["train_csv_sha256"],
        "validation_csv_sha256": expected["validation_csv_sha256"],
        "experiment_dir": str(directory),
    }


def combine_probabilities(original_prob, tree_prob, seed_probs):
    """Average softmax probabilities, then allocate half of the vote to ExtraTrees."""
    if set(seed_probs) != set(SEEDS):
        raise ValueError("Provide exactly the probabilities for seeds 42, 43, and 44.")
    arrays = [np.asarray(original_prob), np.asarray(tree_prob)]
    arrays += [np.asarray(seed_probs[seed]) for seed in SEEDS]
    if any(value.ndim != 2 or value.shape != arrays[0].shape for value in arrays):
        raise ValueError("All members must have matching [rows, 6] probability shapes.")
    values = validate_probabilities(np.stack(arrays, axis=1))
    mean = values[:, 2:].mean(axis=1)
    result = {"original_neural": values[:, 0], "extra_trees": values[:, 1]}
    result.update({f"neural_seed{seed}": values[:, index + 2] for index, seed in enumerate(SEEDS)})
    result.update(
        {
            "neural_mean": mean,
            "baseline_50_50": midpoint_blend.blend_probabilities(original_prob, tree_prob),
            "seed44_50_50": midpoint_blend.blend_probabilities(values[:, 4], values[:, 1]),
            "seed_mean_50_50": midpoint_blend.blend_probabilities(mean, values[:, 1]),
        }
    )
    return result


def _neural_probabilities(member, frame, vectors, target):
    import torch

    sensors = np.array(
        member["preprocessor"].transform(frame), dtype=np.float32, order="C", copy=True
    )
    if sensors.shape != (len(frame), len(RAW_SENSOR_COLUMNS)) or not np.isfinite(sensors).all():
        raise ValueError("Invalid transformed sensor values for a saved member.")
    model = member["model"].to(target).eval()
    probabilities = []
    with torch.inference_mode():
        for row in range(len(frame)):
            logits = model(
                torch.from_numpy(vectors[row : row + 1]).to(target),
                torch.from_numpy(sensors[row : row + 1]).to(target),
            )
            if tuple(logits.shape) != (1, len(LABELS)) or not torch.isfinite(logits).all():
                raise ValueError("Neural logits must be finite with six aligned classes.")
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def method_probabilities(frame, image_root, loaded, device, image_precision="fp32"):
    """Use current rows only; share one live extraction and one ExtraTrees prediction."""
    from experiments.seungbeom import multimodal as mm
    from runtime import seed_everything, select_device

    if len(frame) < 1:
        raise ValueError("Inference requires at least one row.")
    baseline, members = loaded["baseline"], loaded["seeds"]
    if set(members) != set(SEEDS):
        raise ValueError("Expected exactly three loaded seed members.")
    seed_everything(baseline["config"]["seed"])
    target = select_device(device)
    bundle = baseline["bundle"]
    current = frame.drop(columns="label", errors="ignore")
    with precision_context(image_precision):
        features, meta = mm.make_features(
            current,
            None,
            image_root,
            bundle["sensor_spec"],
            encoder=bundle["encoder"],
            embedding_spec=bundle["embedding"],
            device=device,
            batch_size=1,
            workers=0,
        )
    if list(features.columns) != bundle["columns"] or len(features) != len(current):
        raise ValueError("Live feature rows/order differ from the saved tree contract.")
    contract = embedding_contract(meta)
    for member in (baseline, *members.values()):
        if contract != embedding_contract(member["checkpoint"]["visual_contract"]):
            raise ValueError("Live image contract differs from a saved neural member.")
    columns = [
        f"{mm.EMBED_PREFIX}{index:04d}" for index in range(baseline["config"]["embedding_dim"])
    ]
    vectors = np.array(features.loc[:, columns], dtype=np.float32, order="C", copy=True)
    if not np.isfinite(vectors).all():
        raise ValueError("Live image vectors must be finite.")
    print("Image features ready; predicting original, seed members, and ExtraTrees...", flush=True)
    original = _neural_probabilities(baseline, current, vectors, target)
    seed_probs = {
        seed: original
        if member is baseline
        else _neural_probabilities(member, current, vectors, target)
        for seed, member in members.items()
    }
    pipeline = bundle["pipelines"][bundle["models"].index("extra_trees")]
    tree_prob = mm.predict_proba_six_stable(pipeline, features)
    return combine_probabilities(original, tree_prob, seed_probs)
