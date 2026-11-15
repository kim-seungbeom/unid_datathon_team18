"""Train ExtraTrees seeds43/44 with original fitted preprocessing/PCA fixed.

Existing seed42 and inner_best are immutable. Compare on official Validation;
never use Validation to fit trees, preprocessing, weights or select an epoch.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.seungbeom import boosted_trees as inputs  # noqa: E402
from experiments.seungbeom import classwise_ensemble as legacy  # noqa: E402
from experiments.seungbeom import extratrees_seed_core as core  # noqa: E402
from experiments.seungbeom import multimodal as mm  # noqa: E402
from experiments.seungbeom.export_submissions import read_json, sha256, write_json  # noqa: E402
from experiments.seungbeom.neural_data import embedding_contract  # noqa: E402
from schema import RAW_SENSOR_COLUMNS, read_frame, validate_split  # noqa: E402

previous, prior30, cache, original = inputs.previous, inputs.prior30, inputs.cache, inputs.original
KIND = "seungbeom_extratrees_seed_mean"


def sources():
    paths = (
        "experiments/seungbeom/extratrees_seed_mean.py",
        "experiments/seungbeom/extratrees_seed_core.py",
    )
    return {**inputs.sources(), **{p: sha256(ROOT / p) for p in paths}}


def versions():
    return inputs.verified.versions()  # No CatBoost/LightGBM dependency for this experiment.


def unchanged(base, manifest, files, code, env, device):
    previous.unchanged(base, manifest, files, previous.sources(), device)
    if sources() != code or versions() != env:
        raise ValueError("Seed experiment source/runtime changed.")


def load_original(manifest, files):
    path = Path(manifest["reference"]["base_recipe"]["ensemble_run"]) / "ensemble.joblib"
    if files.get(str(path.resolve())) != sha256(path):
        raise ValueError("Original tree bundle is not authenticated by the baseline.")
    bundle = legacy.load_bundle(path)
    if (
        bundle["sensor_spec"] != mm.sensor_spec()
        or bundle["encoder"] != "resnet18"
        or bundle.get("refit_train_val") is not False
    ):
        raise ValueError("Use original Train-only ResNet18/raw5 tree bundle.")
    template = bundle["pipelines"][bundle["models"].index("extra_trees")]
    core.validate_original(template)
    return template, bundle


def features(frame, vectors, spec, columns):
    if spec != mm.sensor_spec() or spec["sensor_features"] != list(RAW_SENSOR_COLUMNS):
        raise ValueError("Only original raw-five current-row sensors allowed.")
    sensors = mm.build_sensor_features(
        frame.drop(columns="label", errors="ignore"),
        spec["sensor_features"],
        imputation=spec["sensor_imputation"],
    ).reset_index(drop=True)
    result = pd.concat(
        [
            sensors.add_prefix("sensor_"),
            pd.DataFrame(vectors, columns=[f"img_{i:04d}" for i in range(vectors.shape[1])]),
        ],
        axis=1,
    )
    if list(result.columns) != columns:
        raise ValueError("Feature columns differ from original pipeline.")
    return result


def display(r):
    print("Original fitted preprocessing/PCA and seed42 preserved; only seeds43/44 trained.")
    print("Neural:mean-tree=30:70 fixed, no tau/Temperature. Validation comparison only.")
    print(
        f"Tree fitting seconds={r['fit_seconds']:.1f}; "
        "original seed42 exact probability replay=True"
    )
    prior30.display(r["metrics"])
    delta = r["metrics"][core.CANDIDATE]["macro_f1"] - r["metrics"][core.BASELINE]["macro_f1"]
    print(f"Seed-mean blend delta vs original 30:70={delta:+.6f}")
    print("Validation is reused; no new independent CV score or Public score prediction.")


def train(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new one.")
    code, env = sources(), versions()
    old, reference, base, manifest, files = previous.load_reference(
        args.baseline_recipe, args.device
    )
    for split, path in (("train", args.train_csv), ("validation", args.input)):
        if sha256(path) != manifest["inputs"][split]["csv_sha256"]:
            raise ValueError(f"Use the original {split} CSV, never Test or altered rows.")
        files.update(original.file_hashes([path]))
    template, bundle = load_original(manifest, files)
    identity_path = Path(manifest["reference"]["base_recipe"]["neural_run"]) / "identity.json"
    if files.get(str(identity_path.resolve())) != sha256(identity_path):
        raise ValueError("Original embedding identity not authenticated.")
    identity = read_json(identity_path)
    train_frame, val_frame = (
        read_frame(args.train_csv, labeled=True),
        read_frame(args.input, labeled=True),
    )
    validate_split(train_frame, val_frame)
    train_vectors, contract = inputs.training_vectors(
        args.train_embeddings, train_frame, identity["input_provenance"]["train"], files
    )
    val_vectors, val_contract = inputs.training_vectors(
        args.validation_embeddings, val_frame, identity["input_provenance"]["validation"], files
    )
    if contract != val_contract or contract != embedding_contract(bundle["embedding"]):
        raise ValueError("Embedding contracts differ from original ExtraTrees.")
    prep = copy.deepcopy(template[:-1])
    prep_hash, seed42_hash = (
        joblib.hash(template[:-1]),
        core.fitted_classifier_hash(template.named_steps["classifier"]),
    )
    if joblib.hash(prep) != prep_hash:
        raise ValueError("Copied preprocessing differs from original fitted state.")
    train_features = features(train_frame, train_vectors, bundle["sensor_spec"], bundle["columns"])
    val_features = features(val_frame, val_vectors, bundle["sensor_spec"], bundle["columns"])
    print("Applying original fitted preprocessing once; no fit/fit_transform calls...", flush=True)
    x, vx = core.stable_transform(prep, train_features), core.stable_transform(prep, val_features)
    members, images = cache.cached_members(
        args, val_frame, base, manifest, files, cache.source_hashes()
    )
    original_p = core.predict_classifier(template.named_steps["classifier"], vx)
    if not np.array_equal(original_p, members[1]):
        raise ValueError(
            "Original seed42 probabilities differ from saved Validation cache. Stop before fitting."
        )
    existing = cache.previous.candidates(*members)
    if any(
        prior30.classification_metrics(val_frame.label, existing[n].argmax(1) + 1)
        != reference["metrics"][n]
        for n in cache.REPLAY
    ):
        raise ValueError("Original member/ratio Validation metrics failed to replay.")
    unchanged(base, manifest, files, code, env, args.device)
    args.output_dir.mkdir(parents=True)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "training"})
    try:
        started = time.perf_counter()
        classifiers = core.fit_members(template, x, train_frame.label)
        seconds = time.perf_counter() - started
        model = {
            "kind": KIND,
            "contract": core.CONTRACT,
            "preprocessing": prep,
            "classifiers": classifiers,
            "columns": bundle["columns"],
            "sensor_spec": bundle["sensor_spec"],
            "embedding": contract,
        }
        probabilities = {s: core.predict_classifier(c, vx) for s, c in classifiers.items()}
        if (
            joblib.hash(prep) != prep_hash
            or core.fitted_classifier_hash(classifiers[42]) != seed42_hash
        ):
            raise ValueError("Original preprocessing or classifier state changed.")
        model_path = args.output_dir / "members.joblib"
        joblib.dump(model, model_path)
        restored = joblib.load(model_path)
        if (
            joblib.hash(restored["preprocessing"]) != prep_hash
            or core.fitted_classifier_hash(restored["classifiers"][42]) != seed42_hash
        ):
            raise ValueError("Saved original states failed to replay.")
        for s in core.SEEDS:
            if not np.array_equal(
                core.predict_classifier(restored["classifiers"][s], vx), probabilities[s]
            ):
                raise ValueError("Saved classifier probabilities failed to replay.")
        if not np.array_equal(probabilities[42], members[1]):
            raise ValueError("Original seed42 was changed.")
        values = core.candidates(members[0], probabilities)
        metrics = {
            n: prior30.classification_metrics(val_frame.label, p.argmax(1) + 1)
            for n, p in values.items()
        }
        if metrics[core.BASELINE] != old["metrics"][core.BASELINE]:
            raise ValueError("Original 30:70 score failed to replay.")
        unchanged(base, manifest, files, code, env, args.device)
        record = {
            "kind": KIND,
            "contract": core.CONTRACT,
            "source_sha256": code,
            "versions": env,
            "baseline_path": str(args.baseline_recipe.resolve()),
            "baseline": old,
            "model_file": model_path.name,
            "model_sha256": sha256(model_path),
            "preprocessing_sha": prep_hash,
            "seed42_sha": seed42_hash,
            "parameter_hash": core.classifier_hash(classifiers[42]),
            "metrics": metrics,
            "fit_seconds": seconds,
            "provenance": files,
            "validation_images_sha256": images,
            "image_precision": manifest["reference"]["base_recipe"]["image_precision"],
            "replay_matches": {
                "seed42_probabilities": True,
                "saved_members": True,
                "fitted_preprocessing": True,
            },
            "note": "New classifiers fit on rowwise transform of saved PCA, never refitted PCA. "
            "Original seed42 is reused, not claimed to be reproducible by fresh fitting.",
        }
        path = args.output_dir / "recipe.json"
        write_json(path, record)
        write_json(status, {"status": "complete", "sha256": sha256(path)})
        display(record)
        return record
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def read_recipe(path):
    r, status = read_json(path), read_json(path.parent / "status.json")
    if (
        status.get("status") != "complete"
        or status.get("sha256") != sha256(path)
        or r.get("kind") != KIND
        or r.get("contract") != core.CONTRACT
        or r.get("source_sha256") != sources()
        or r.get("versions") != versions()
        or r.get("model_file") != "members.joblib"
        or r.get("model_sha256") != sha256(path.parent / "members.joblib")
        or r.get("replay_matches")
        != dict.fromkeys(("seed42_probabilities", "saved_members", "fitted_preprocessing"), True)
    ):
        raise ValueError("Incomplete/changed seed mean experiment or source/runtime.")
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("train")
    for name in (
        "baseline_recipe",
        "train_csv",
        "input",
        "train_embeddings",
        "validation_embeddings",
        "image_root",
        "cache_dir",
        "output_dir",
    ):
        fitting.add_argument("--" + name, type=Path, required=True)
    fitting.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    fitting.set_defaults(handler=train)
    report = commands.add_parser("report")
    report.add_argument("--recipe", type=Path, required=True)
    report.set_defaults(handler=lambda a: display(read_recipe(a.recipe)))
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
