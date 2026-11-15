"""User-run CatBoost + LightGBM, cached ResNet18/raw5, bounded CPU training.

Train fits gradients and sensor medians. Official Validation selects best iteration.
Existing checkpoints/probability caches are immutable. No Test use during training.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.seungbeom import boosted_trees_core as core  # noqa: E402
from experiments.seungbeom import multimodal as mm  # noqa: E402
from experiments.seungbeom import probability_fusion as verified  # noqa: E402
from experiments.seungbeom.export_submissions import check_csv, read_json, sha256, write_json  # noqa: E402
from experiments.seungbeom.neural_data import embedding_contract  # noqa: E402
from experiments.seungbeom.replay_precision import precision_context  # noqa: E402
from schema import read_frame, validate_split  # noqa: E402

KIND = "seungbeom_boosted_trees"
FILES = ("features.npy", "metadata.json", "sample_ids.csv")
previous, prior30, cache, original = (
    verified.previous,
    verified.prior30,
    verified.cache,
    verified.original,
)


def sources():
    names = (
        "experiments/seungbeom/boosted_trees.py",
        "experiments/seungbeom/boosted_trees_core.py",
        "experiments/seungbeom/multimodal.py",
        "experiments/seungbeom/neural_data.py",
    )
    return {**verified.sources(), **{n: sha256(ROOT / n) for n in names}}


def versions():
    return {**verified.versions(), **{n: importlib.metadata.version(n) for n in core.MODELS}}


def unchanged(base, manifest, files, recorded, env, device):
    previous.unchanged(base, manifest, files, previous.sources(), device)
    if sources() != recorded or versions() != env:
        raise ValueError("Boosting code or libraries changed during this run.")


def training_vectors(directory, frame, expected_hashes, files):
    hashes = {n: sha256(directory / n) for n in FILES}
    if hashes != expected_hashes:
        raise ValueError("Use the exact original embedding cache recorded in model identity.")
    files.update({str((directory / n).resolve()): v for n, v in hashes.items()})
    vectors, meta = mm.load_embeddings(directory, frame.sample_id)
    contract = embedding_contract(meta)
    if contract["model"] != "resnet18" or contract["dimension"] != 512:
        raise ValueError("Expected frozen ResNet18 image vectors.")
    return vectors, contract


def display(r):
    print("Train-only gradient fitting; official Validation selected best iterations.")
    print("Validation is a reused selection score, not an independent final evaluation.")
    for name, model in r["models"].items():
        print(
            f"{name}: best_iteration={model['best_iteration']} "
            f"fit_seconds={model['fit_seconds']:.1f} budget_reached={model['budget_reached']}"
        )
    prior30.display(r["metrics"])


def train(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new one.")
    if (
        not 1 <= args.iterations <= 2000
        or not 1 <= args.threads <= 8
        or not 0 < args.minutes_per_model <= 30
    ):
        raise ValueError("Use iterations 1..2000, threads 1..8, minutes_per_model (0,30].")
    # Import both before fitting either, so missing packages fail immediately.
    import catboost  # noqa: F401
    import lightgbm  # noqa: F401

    env, code = versions(), sources()
    old, reference, base, manifest, files = previous.load_reference(
        args.baseline_recipe, args.device
    )
    for split, path in (("train", args.train_csv), ("validation", args.input)):
        if sha256(path) != manifest["inputs"][split]["csv_sha256"]:
            raise ValueError(f"Use the original {split} CSV; not Test or altered rows.")
        files.update(original.file_hashes([path]))
    identity_path = Path(manifest["reference"]["base_recipe"]["neural_run"]) / "identity.json"
    if str(identity_path.resolve()) not in files or files[str(identity_path.resolve())] != sha256(
        identity_path
    ):
        raise ValueError("Original embedding identity is not authenticated by the baseline.")
    identity = read_json(identity_path)
    fit_frame, val_frame = (
        read_frame(args.train_csv, labeled=True),
        read_frame(args.input, labeled=True),
    )
    validate_split(fit_frame, val_frame)
    vectors, contract = training_vectors(
        args.train_embeddings, fit_frame, identity["input_provenance"]["train"], files
    )
    val_vectors, val_contract = training_vectors(
        args.validation_embeddings, val_frame, identity["input_provenance"]["validation"], files
    )
    if contract != val_contract:
        raise ValueError("Train and Validation embedding contracts differ.")
    prep = core.fit_preprocessor(fit_frame)
    x, val_x = (
        core.transform(fit_frame, vectors, prep),
        core.transform(val_frame, val_vectors, prep),
    )
    members, images = cache.cached_members(
        args, val_frame, base, manifest, files, cache.source_hashes()
    )
    existing = cache.previous.candidates(*members)
    if any(
        prior30.classification_metrics(val_frame.label, existing[n].argmax(1) + 1)
        != reference["metrics"][n]
        for n in cache.REPLAY
    ):
        raise ValueError("Original Validation members do not replay.")
    unchanged(base, manifest, files, code, env, args.device)
    args.output_dir.mkdir(parents=True)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "training"})
    try:
        records, probabilities = {}, {}
        for name in core.MODELS:
            print(
                f"Fitting {name}: CPU threads={args.threads}, up to {args.iterations} iterations, "
                f"soft budget={args.minutes_per_model} minutes...",
                flush=True,
            )
            record, p = core.fit_model(
                name,
                x,
                fit_frame.label,
                val_x,
                val_frame.label,
                args.output_dir,
                iterations=args.iterations,
                threads=args.threads,
                seconds=60 * args.minutes_per_model,
            )
            record["sha256"] = sha256(args.output_dir / record["file"])
            records[name], probabilities[name] = record, p
            write_json(args.output_dir / f"{name}_metrics.json", record)
            print(
                f"Saved {name} best checkpoint: iteration={record['best_iteration']} "
                f"validation_macro_f1={record['validation_macro_f1']:.6f}",
                flush=True,
            )
        values = core.candidates(*members, probabilities["catboost"], probabilities["lightgbm"])
        metrics = {
            n: prior30.classification_metrics(val_frame.label, p.argmax(1) + 1)
            for n, p in values.items()
        }
        if metrics[core.BASELINE] != old["metrics"][core.BASELINE]:
            raise ValueError("Original 30:70 baseline changed.")
        unchanged(base, manifest, files, code, env, args.device)
        recipe = {
            "kind": KIND,
            "contract": core.CONTRACT,
            "source_sha256": code,
            "versions": env,
            "baseline_path": str(args.baseline_recipe.resolve()),
            "baseline": old,
            "preprocessor": prep,
            "embedding": contract,
            "models": records,
            "metrics": metrics,
            "image_precision": manifest["reference"]["base_recipe"]["image_precision"],
            "fit_split": "train",
            "selection_split": "official_validation",
            "provenance": files,
            "validation_images_sha256": images,
            "note": "Raw 512 image vectors + raw five sensors; no PCA/scaling. Not a pure "
            "estimator-only ablation against original PCA ExtraTrees. No new OOF score.",
        }
        path = args.output_dir / "recipe.json"
        write_json(path, recipe)
        write_json(status, {"status": "complete", "sha256": sha256(path)})
        display(recipe)
        return recipe
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def read_recipe(path):
    status, r = read_json(path.parent / "status.json"), read_json(path)
    if status.get("status") != "complete" or status.get("sha256") != sha256(path):
        raise ValueError("Incomplete/modified tree experiment.")
    if (
        r.get("kind") != KIND
        or r.get("source_sha256") != sources()
        or r.get("versions") != versions()
        or r.get("contract") != core.CONTRACT
        or r.get("fit_split") != "train"
        or r.get("selection_split") != "official_validation"
        or set(r.get("models", {})) != set(core.MODELS)
    ):
        raise ValueError("Boosting recipe/source/library contract mismatch.")
    for name, suffix in (("catboost", ".cbm"), ("lightgbm", ".txt")):
        m = r["models"][name]
        if m["file"] != name + suffix or sha256(path.parent / m["file"]) != m["sha256"]:
            raise ValueError("Saved tree model changed.")
    return r


def predict(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new one.")
    r = read_recipe(args.recipe)
    old, _, base, manifest, files = previous.load_reference(Path(r["baseline_path"]), args.device)
    if old != r["baseline"]:
        raise ValueError("Saved baseline changed.")
    paths = [args.recipe, args.recipe.parent / "status.json", args.input, args.sample_submission]
    paths += [args.recipe.parent / m["file"] for m in r["models"].values()]
    files.update(original.file_hashes(paths))
    frame = read_frame(args.input)
    ids = frame.sample_id.tolist()
    order = prior30.template_ids(args.sample_submission, ids)
    members, images = cache.cached_members(
        args, frame, base, manifest, files, cache.source_hashes()
    )
    print("Extracting current input image vectors once for both new trees...", flush=True)
    with precision_context(r["image_precision"]):
        features, meta = mm.make_features(
            frame.drop(columns="label", errors="ignore"),
            None,
            args.image_root,
            mm.sensor_spec(),
            encoder="resnet18",
            embedding_spec=r["embedding"],
            device=args.device,
            batch_size=1,
            workers=0,
        )
    if embedding_contract(meta) != r["embedding"]:
        raise ValueError("Live image embedding contract changed.")
    vectors = features[[f"img_{i:04d}" for i in range(512)]].to_numpy()
    x = core.transform(frame, vectors, r["preprocessor"])
    p = {
        n: core.predict_model(n, args.recipe.parent / m["file"], x, threads=m["threads"])
        for n, m in r["models"].items()
    }
    values = core.candidates(*members, p["catboost"], p["lightgbm"])
    labels = {n: v.argmax(1) + 1 for n, v in values.items()}
    if original.image_digest(frame, args.image_root) != images:
        raise ValueError("Images changed during tree inference.")
    unchanged(base, manifest, files, r["source_sha256"], r["versions"], args.device)
    args.output_dir.mkdir(parents=True)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    outputs, seen = {}, {core.BASELINE: labels[core.BASELINE], "extra_trees": labels["extra_trees"]}
    try:
        for name in args.methods:
            duplicate = next((n for n, v in seen.items() if np.array_equal(v, labels[name])), None)
            if duplicate:
                print(f"Skipping {name}: identical labels to {duplicate}")
                continue
            path = args.output_dir / f"submission_{name}.csv"
            with path.open("x", encoding="utf-8", newline="") as f:
                pd.DataFrame(
                    {
                        "sample_id": order,
                        "label": pd.Series(labels[name], index=ids)
                        .loc[order]
                        .to_numpy(dtype=np.int64),
                    }
                ).to_csv(f, index=False)
            outputs[name] = check_csv(path, order)
            seen[name] = labels[name]
            print("Saved", path)
        unchanged(base, manifest, files, r["source_sha256"], r["versions"], args.device)
        write_json(
            args.output_dir / "manifest.json",
            {"recipe": r, "input_files": files, "outputs": outputs},
        )
        write_json(
            status,
            {
                "status": "complete",
                "outputs": outputs,
                "manifest_sha256": sha256(args.output_dir / "manifest.json"),
            },
        )
        return outputs
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("train")
    for n in ("baseline_recipe", "train_csv", "train_embeddings", "validation_embeddings"):
        fitting.add_argument("--" + n, type=Path, required=True)
    fitting.add_argument("--iterations", type=int, default=400)
    fitting.add_argument("--threads", type=int, default=4)
    fitting.add_argument("--minutes_per_model", type=float, default=12)
    fitting.set_defaults(handler=train)
    prediction = commands.add_parser("predict")
    for n in ("recipe", "sample_submission"):
        prediction.add_argument("--" + n, type=Path, required=True)
    prediction.add_argument(
        "--methods", nargs="+", choices=core.METHODS, default=["inner_best_cat_lgbm_30_35_35"]
    )
    prediction.set_defaults(handler=predict)
    for cmd in (fitting, prediction):
        for n in ("input", "image_root", "cache_dir", "output_dir"):
            cmd.add_argument("--" + n, type=Path, required=True)
        cmd.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    report = commands.add_parser("report")
    report.add_argument("--recipe", type=Path, required=True)
    report.set_defaults(handler=lambda a: display(read_recipe(a.recipe)))
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
