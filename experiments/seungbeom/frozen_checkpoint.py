"""Save the best inner-holdout frozen weights and compare against fresh all-Train refit.

Keep the original ExtraTrees and 50:50 rule. Select epochs inside Train, then
freshly refit on Train only. Official Validation is comparison, never fitting.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import classification_metrics  # noqa: E402
from experiments.seungbeom import experiment_balance_core as diagnostics  # noqa: E402
from experiments.seungbeom import frozen_checkpoint_core as core  # noqa: E402
from experiments.seungbeom import frozen_tree_blend as blend  # noqa: E402
from experiments.seungbeom import resnet_finetune as ft  # noqa: E402
from experiments.seungbeom.export_submissions import (  # noqa: E402
    check_csv,
    read_json,
    sha256,
    write_json,
)
from experiments.seungbeom.replay_precision import precision_context, runtime_signature  # noqa: E402
from experiments.seungbeom.tree_candidates import template_ids  # noqa: E402
from schema import LABELS, RAW_SENSOR_COLUMNS, read_frame, validate_split  # noqa: E402

KIND = "seungbeom_frozen_best_checkpoint"
FILES = (
    "inner_best.pt",
    "refit.pt",
    "manifest.json",
    "result.json",
    "selection_history.json",
    "refit_history.json",
    "group_pair_errors.csv",
)
CANDIDATES = {
    "inner_best_neural": "inner_best",
    "inner_best_50_50": "inner_best",
    "refit_neural": "refit",
    "refit_50_50": "refit",
}
METHODS = ("original_frozen", "extra_trees", "original_50_50", *CANDIDATES)
PAIRED = {
    name: "original_50_50" if name.endswith("50_50") else "original_frozen" for name in CANDIDATES
}


def source_hashes():
    return {
        **blend.source_hashes(),
        **{
            f"experiments/seungbeom/{name}.py": sha256(ROOT / f"experiments/seungbeom/{name}.py")
            for name in ("frozen_checkpoint", "frozen_checkpoint_core", "experiment_balance_core")
        },
    }


def print_progress(directory, phase):
    rows = []

    def update(row):
        rows.append(row)
        write_json(directory / f"{phase}_history.json", {"epochs": rows})
        score = f" inner_macro_f1={row['inner_macro_f1']:.6f}" if "inner_macro_f1" in row else ""
        print(
            f"frozen {phase} epoch={row['epoch']} loss={row['train_loss']:.6f}{score}", flush=True
        )

    return update


def checkpoint_identity(manifest, selected, stage, inner_score, inner_digest):
    if stage not in ("inner_best", "refit"):
        raise ValueError("Invalid checkpoint stage.")
    return {
        "kind": KIND,
        "format_version": 1,
        "objective": "row_uniform_cross_entropy",
        "config": manifest["config"],
        "selected_epoch": selected,
        "stage": stage,
        "fit_partition": "train_inner_fit" if stage == "inner_best" else "all_train",
        "selection_metric": "train_inner_macro_f1",
        "selection_score": inner_score,
        "selection_probability_sha256": inner_digest,
        "inner_split": manifest["inner_split"],
        "model_config": {"mode": "frozen", **ft.model_kwargs(manifest["config"])},
        "label_values": list(LABELS),
        "image_contract": ft.IMAGE_CONTRACT,
        "source_sha256": manifest["source_sha256"],
        "runtime": manifest["runtime"],
        "train_csv_sha256": manifest["inputs"]["train"]["csv_sha256"],
        "validation_csv_sha256": manifest["inputs"]["validation"]["csv_sha256"],
    }


def restore(path, identity, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if any(saved.get(key) != value for key, value in identity.items()):
        raise ValueError("Frozen checkpoint differs from its recorded identity.")
    config = ft.normalize_config(identity["config"])
    prep = ft.NeuralSensorPreprocessor.from_dict(saved["preprocessor"])
    if (
        prep.feature_names != RAW_SENSOR_COLUMNS
        or prep.imputation != "median"
        or prep.scaling != "standard"
        or prep.uses_history
    ):
        raise ValueError("Frozen inference requires the saved raw5/median/standard preprocessing.")
    ft.seed_everything(config["seed"])
    model = ft.FineTuneModel("frozen", **ft.model_kwargs(config), pretrained=False).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model, prep


def probability(model, prep, frame, image_root, config, device):
    loader = ft.evaluation_loader(
        frame.drop(columns="label", errors="ignore"), image_root, prep, config, labeled=False
    )
    with precision_context("fp32"):
        return ft.probabilities(model, loader, device, progress_every=2048)


def check_unchanged(manifest, reference, files, device):
    blend.check_unchanged(
        reference["base_recipe"], reference["frozen"], reference["source_sha256"], files, device
    )
    if source_hashes() != manifest["source_sha256"]:
        raise ValueError("Source changed during the run; do not pull while training.")


def run(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    if args.output_dir.resolve().is_relative_to(args.data_dir.resolve()):
        raise ValueError("Write outputs outside the provided data directory.")
    reference = blend.read_comparison(args.recipe)
    if reference["source_sha256"] != blend.source_hashes():
        raise ValueError("Original inference source changed.")
    config = ft.normalize_config(reference["frozen"]["config"])
    ft.seed_everything(config["seed"])
    device = ft.select_device(args.device)
    paths = {split: args.data_dir / split / f"{split}.csv" for split in ("train", "validation")}
    files = {
        str(path.resolve()): sha256(path)
        for path in (args.recipe, args.recipe.parent / "status.json", *paths.values())
    }
    for split, path in paths.items():
        if (
            files[str(path.resolve())]
            != reference["base_recipe"]["provenance"][split + "_csv_sha256"]
        ):
            raise ValueError("Use the original Train/Validation CSVs; no Test fitting.")
    frames = {split: read_frame(path, labeled=True) for split, path in paths.items()}
    validate_split(frames["train"], frames["validation"])
    part = ft.inner_split(frames["train"], config["seed"])
    original_trial = ft.check_trial(Path(reference["frozen"]["run_dir"]))
    initial = original_trial.get("initial_state_sha256")
    if not isinstance(initial, str) or len(initial) != 64:
        raise ValueError("Original frozen trial must record its initial state hash.")
    print("Verifying original models and fingerprinting Train/Validation images...", flush=True)
    legacy, frozen, record = blend.load_members(
        reference["base_recipe"], Path(reference["frozen"]["run_dir"]), args.device
    )
    if record != reference["frozen"]:
        raise ValueError("Original frozen model changed.")
    backbone = ft.state_digest(frozen["model"].backbone)
    manifest = {
        "kind": KIND,
        "format_version": 1,
        "objective": "row_uniform_cross_entropy",
        "config": config,
        "source_sha256": source_hashes(),
        "runtime": runtime_signature(device),
        "reference": reference,
        "inner_split": part,
        "initial_state_sha256": initial,
        "backbone_sha256": backbone,
        "selection": "save_strict_inner_best_then_fresh_all_train_refit",
        "inputs": {
            split: {"csv_sha256": sha256(path), "images_sha256": ft.neural.image_digest(path)}
            for split, path in paths.items()
        },
        "rules": {
            "weights": "uniform_rows_no_group_or_class_weights",
            "checkpoint": "strict_inner_macro_f1_improvement_ties_keep_earlier_epoch",
            "sampler": "original_seeded_shuffle_no_resampling",
            "preprocessing": "original_unweighted_train_median_standard",
            "blend": "fixed_probability_50_50_original_extratrees",
            "validation": "unweighted_comparison_only_not_epoch_or_weight_selection",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    write_json(args.output_dir / "manifest.json", manifest)
    files[str((args.output_dir / "manifest.json").resolve())] = sha256(
        args.output_dir / "manifest.json"
    )
    try:
        print("Replaying original Validation control before training...", flush=True)
        old = blend.infer_members(
            frames["validation"].drop(columns="label"),
            paths["validation"].parent,
            legacy,
            frozen,
            reference["base_recipe"],
            args.device,
        )
        replay = {
            name: classification_metrics(frames["validation"]["label"], prob.argmax(1) + 1)
            == reference["metrics"][name]
            for name, prob in old.items()
        }
        if replay != dict.fromkeys(blend.METHODS, True):
            write_json(args.output_dir / "replay.json", replay)
            raise ValueError(
                "Original Validation metrics did not replay; no candidate was trained."
            )
        del legacy, frozen
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        fit, stop = (frames["train"].iloc[part[key]].copy() for key in ("fit", "stop"))
        print(
            f"Stage 1/2: inner Train epoch selection (at most {config['epochs']} epochs).",
            flush=True,
        )
        best = {}

        def save_best(model, prep, epoch, score, digest):
            identity = checkpoint_identity(manifest, epoch, "inner_best", score, digest)
            core.save_checkpoint(args.output_dir / "inner_best.pt", model, prep, identity)
            best.update(identity=identity, score=score, digest=digest)
            print(f"Saved inner_best.pt: epoch={epoch} inner_macro_f1={score:.6f}", flush=True)

        last_model, _, selected, _, _ = core.train_phase(
            fit,
            paths["train"].parent,
            config,
            device,
            epochs=config["epochs"],
            stopping=(stop, paths["train"].parent),
            progress=print_progress(args.output_dir, "selection"),
            on_best=save_best,
            expected_initial=initial,
            expected_backbone=backbone,
        )
        del last_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if not best or best["identity"]["selected_epoch"] != selected:
            raise ValueError("Best checkpoint does not match the selected inner epoch.")
        best_file_hash = sha256(args.output_dir / "inner_best.pt")
        inner_model, inner_prep = restore(
            args.output_dir / "inner_best.pt", best["identity"], device
        )
        replayed_inner = probability(
            inner_model, inner_prep, stop, paths["train"].parent, config, device
        )
        if (
            core.probability_digest(replayed_inner) != best["digest"]
            or classification_metrics(stop["label"], replayed_inner.argmax(1) + 1)["macro_f1"]
            != best["score"]
        ):
            raise ValueError("Saved best checkpoint does not exactly replay the inner holdout.")
        del inner_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Stage 2/2: fresh all-Train refit for exactly {selected} epochs.", flush=True)
        model, prep, _, _, _ = core.train_phase(
            frames["train"],
            paths["train"].parent,
            config,
            device,
            epochs=selected,
            progress=print_progress(args.output_dir, "refit"),
            expected_initial=initial,
            expected_backbone=backbone,
        )
        identities = {
            "inner_best": best["identity"],
            "refit": checkpoint_identity(
                manifest, selected, "refit", best["score"], best["digest"]
            ),
        }
        core.save_checkpoint(args.output_dir / "refit.pt", model, prep, identities["refit"])
        print("Scoring both models on official Validation; epochs are already fixed...", flush=True)
        refit_prob = probability(
            model, prep, frames["validation"], paths["validation"].parent, config, device
        )
        del model
        restored, saved_prep = restore(args.output_dir / "refit.pt", identities["refit"], device)
        reproduced = probability(
            restored, saved_prep, frames["validation"], paths["validation"].parent, config, device
        )
        if not np.array_equal(refit_prob, reproduced):
            raise ValueError("Refit checkpoint does not exactly replay Validation probabilities.")
        del restored
        inner_model, inner_prep = restore(
            args.output_dir / "inner_best.pt", identities["inner_best"], device
        )
        inner_prob = probability(
            inner_model,
            inner_prep,
            frames["validation"],
            paths["validation"].parent,
            config,
            device,
        )
        del inner_model
        if sha256(args.output_dir / "inner_best.pt") != best_file_hash:
            raise ValueError("Inner best checkpoint changed during refit/evaluation.")
        outputs = {
            "original_frozen": old["frozen"],
            "extra_trees": old["extra_trees"],
            "original_50_50": old["frozen_tree_50_50"],
            "inner_best_neural": inner_prob,
            "inner_best_50_50": blend.midpoint.blend_probabilities(inner_prob, old["extra_trees"]),
            "refit_neural": refit_prob,
            "refit_50_50": blend.midpoint.blend_probabilities(refit_prob, old["extra_trees"]),
        }
        metrics, pairs, grouped = {}, {}, {}
        for name, values in outputs.items():
            predicted = values.argmax(1) + 1
            metrics[name] = classification_metrics(frames["validation"]["label"], predicted)
            pairs[name], grouped[name] = diagnostics.pair_errors(frames["validation"], predicted)
        changes = {
            name: diagnostics.pair_changes(grouped[base], grouped[name])
            for name, base in PAIRED.items()
        }
        check_unchanged(manifest, reference, files, args.device)
        for split, path in paths.items():
            if ft.neural.image_digest(path) != manifest["inputs"][split]["images_sha256"]:
                raise ValueError("Train/Validation images changed during the run.")
        pd.concat([table.assign(method=name) for name, table in grouped.items()]).to_csv(
            args.output_dir / "group_pair_errors.csv", index=False
        )
        result = {
            "kind": KIND,
            "format_version": 1,
            "identities": identities,
            "inner_best_macro_f1": best["score"],
            "inner_probability_sha256": best["digest"],
            "selected_epoch": selected,
            "original_selected_epoch": reference["frozen"]["selected_epoch"],
            "metrics": metrics,
            "pairs": pairs,
            "pair_changes": changes,
            "baseline_replay_matches": replay,
            "checkpoint_replay_matches": {"inner_best": True, "refit": True},
            "notes": [
                "One Train group holdout/seed; not 3-fold CV.",
                "Validation has informed previous choices; comparison is not a blind test.",
                "Inner best uses fewer training groups than refit; they are distinct models.",
                "The inner score is for epoch selection, not official Validation performance.",
                "Original frozen model/ExtraTrees/blend/submissions were not replaced.",
            ],
        }
        write_json(args.output_dir / "result.json", result)
        write_json(
            status,
            {
                "status": "complete",
                "files": {name: sha256(args.output_dir / name) for name in FILES},
            },
        )
        display(result)
        print(
            "Completed. Original models unchanged; Test was not read. No submission CSV generated."
        )
        return result
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def read_result(directory):
    status = read_json(directory / "status.json")
    if status.get("status") != "complete" or set(status.get("files", {})) != set(FILES):
        raise ValueError("Experiment incomplete; inspect the training log.")
    for name, digest in status["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError("Saved experiment artifacts changed.")
    manifest, result = (read_json(directory / name) for name in ("manifest.json", "result.json"))
    if (
        result.get("kind") != KIND
        or manifest.get("kind") != KIND
        or result.get("format_version") != 1
        or manifest.get("format_version") != 1
        or result.get("identities")
        != {
            stage: checkpoint_identity(
                manifest,
                result["selected_epoch"],
                stage,
                result["inner_best_macro_f1"],
                result["inner_probability_sha256"],
            )
            for stage in ("inner_best", "refit")
        }
        or result.get("checkpoint_replay_matches") != {"inner_best": True, "refit": True}
        or result.get("baseline_replay_matches") != dict.fromkeys(blend.METHODS, True)
    ):
        raise ValueError("Invalid comparison identity or replay records.")
    return manifest, result


def display(result, class_details=False):
    print(
        f"Best inner Train Macro F1: {result['inner_best_macro_f1']:.6f}; "
        "selection diagnostic, NOT the official Validation score."
    )
    print(
        "method\tepoch\tmacro_f1\tdelta_vs_original_50_50\t" + "\t".join(f"f1_{c}" for c in LABELS)
    )
    baseline = result["metrics"]["original_50_50"]["macro_f1"]
    for name in METHODS:
        m = result["metrics"][name]
        epoch = (
            result["selected_epoch"]
            if name in CANDIDATES
            else result["original_selected_epoch"]
            if name != "extra_trees"
            else "NA"
        )
        print(
            f"{name}\t{epoch}\t{m['macro_f1']:.6f}\t{m['macro_f1'] - baseline:+.6f}\t"
            + "\t".join(f"{m['class_f1'][str(c)]:.6f}" for c in LABELS)
        )
    print("\nTarget and reverse confusions (rate = errors / true-class support):")
    print("method\tpair\terrors\tsupport\trate\tgroups_with_error\tgroups_with_support")
    for name in METHODS:
        for pair, item in result["pairs"][name].items():
            rate = f"{item['rate']:.4f}" if item["rate"] is not None else "NA"
            print(
                f"{name}\t{pair}\t{item['errors']}\t{item['support']}\t{rate}\t"
                f"{item['groups_with_errors']}\t{item['groups_with_support']}"
            )
    print("\nPaired experiment counts vs the matching original member/blend:")
    print("candidate\tpair\timproved\ttied\tworse\terror_count_delta")
    for name, pairs in result["pair_changes"].items():
        for pair, item in pairs.items():
            print(
                f"{name}\t{pair}\t{item['improved']}\t{item['tied']}\t{item['worse']}\t"
                f"{item['error_count_delta']:+d}"
            )
    if class_details:
        print("\nmethod\tclass\tprecision\trecall\tf1\tsupport")
        for name in METHODS:
            m = result["metrics"][name]
            for label in LABELS:
                key = str(label)
                print(
                    f"{name}\t{label}\t{m['class_precision'][key]:.6f}\t"
                    f"{m['class_recall'][key]:.6f}\t{m['class_f1'][key]:.6f}\t{m['support'][key]}"
                )
    print("Validation is unweighted and reused for comparison, not a new blind test or 3-fold CV.")
    print("No prediction-stage experiment IDs, group counts or label-specific overrides are used.")


def report(args):
    _, result = read_result(args.output_dir)
    display(result, args.class_details)


def predict(args):
    sidecar = args.output.with_suffix(".manifest.json")
    if args.output.exists() or sidecar.exists():
        raise ValueError("Output exists; choose a new filename.")
    manifest, result = read_result(args.run_dir)
    if manifest["source_sha256"] != source_hashes():
        raise ValueError("Candidate source changed; use the recorded revision.")
    device = ft.select_device(args.device)
    ft.seed_everything(manifest["config"]["seed"])
    if runtime_signature(device) != manifest["runtime"]:
        raise ValueError("Use the recorded runtime environment.")
    stage = CANDIDATES[args.method]
    model, prep = restore(args.run_dir / f"{stage}.pt", result["identities"][stage], device)
    frame = read_frame(args.input)
    ids = frame["sample_id"].tolist()
    order = template_ids(args.sample_submission, ids) if args.sample_submission else ids
    files = {
        str(path.resolve()): sha256(path)
        for path in [
            args.input,
            args.run_dir / "status.json",
            *(args.run_dir / name for name in FILES),
        ]
    }
    if args.sample_submission:
        files[str(args.sample_submission.resolve())] = sha256(args.sample_submission)
    neural = probability(model, prep, frame, args.image_root, manifest["config"], device)
    if args.method.endswith("50_50"):
        base = manifest["reference"]["base_recipe"]
        legacy = blend.ratios.load_base(base, args.device)
        old = blend.midpoint.method_probabilities(
            frame.drop(columns="label", errors="ignore"),
            args.image_root,
            legacy,
            args.device,
            image_precision=base["image_precision"],
        )
        neural = blend.midpoint.blend_probabilities(neural, old["extra_trees"])
        blend.midpoint.check_files(base["provenance"]["files"])
    values = pd.Series(neural.argmax(1) + 1, index=ids).loc[order].to_numpy(dtype=np.int64)
    blend.midpoint.check_files(files)
    if (
        source_hashes() != manifest["source_sha256"]
        or runtime_signature(device) != manifest["runtime"]
    ):
        raise ValueError("Sources/runtime changed during inference.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        pd.DataFrame({"sample_id": order, "label": values}).to_csv(handle, index=False)
    write_json(
        sidecar,
        {
            "kind": KIND,
            "method": args.method,
            "identity": result["identities"][stage],
            "source_run": str(args.run_dir.resolve()),
            "input_sha256": files,
            "output": check_csv(args.output, order),
        },
    )
    print("Saved candidate CSV:", args.output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    for name in ("recipe", "data_dir", "output_dir"):
        training.add_argument("--" + name, type=Path, required=True)
    training.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    training.set_defaults(handler=run)
    reporting = commands.add_parser("report")
    reporting.add_argument("--output_dir", type=Path, required=True)
    reporting.add_argument("--class_details", action="store_true")
    reporting.set_defaults(handler=report)
    prediction = commands.add_parser("predict")
    for name in ("run_dir", "input", "image_root", "output"):
        prediction.add_argument("--" + name, type=Path, required=True)
    prediction.add_argument("--method", choices=tuple(CANDIDATES), default="inner_best_50_50")
    prediction.add_argument("--sample_submission", type=Path)
    prediction.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    prediction.set_defaults(handler=predict)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
