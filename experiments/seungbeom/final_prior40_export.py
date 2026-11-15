"""Export saved 40:60+prior models without the original submission or OOF data."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.seungbeom import final_prior40_verify as audit  # noqa: E402
from schema import read_frame  # noqa: E402
from scripts import locate_final_submission as locator  # noqa: E402

pc = audit.pc
KIND = "prior40_without_original_csv"


def correction_at(path):
    correction = pc.read_correction(path)
    prior = correction["prior"]
    if (
        correction.get("this_file_sha256") != locator.SOURCE_SHA
        or correction.get("rule") != "argmax(p_blend / prior**tau)"
        or not all(math.isfinite(p) and p > 0 for p in prior)
        or not math.isclose(sum(prior), 1, rel_tol=0, abs_tol=1e-8)
    ):
        raise ValueError("Invalid saved prior40 correction or source version.")
    return correction


def discover(args):
    if args.output.exists():
        raise ValueError("Discovery output exists; choose a new filename.")
    roots = locator.roots_for(args.runs_root or [ROOT / "runs"], args.team_home)
    files, errors = locator.inventory(roots)
    result = {"kind": KIND, "corrections": [], "recipes": [], "scan_errors": errors}
    seen = {"corrections": set(), "recipes": set()}
    for path in files:
        # Never inspect any original submission CSV or its manifest.
        if path.name not in {"correction.json", "recipe.json"}:
            continue
        try:
            if path.name == "correction.json":
                config = correction_at(path)
                group = "corrections"
                detail = {"tau": config["tau"]}
            else:
                pc.ibr.read_comparison(path)
                group, detail = "recipes", {}
            digest = pc.sha256(path)
            if digest not in seen[group]:
                seen[group].add(digest)
                result[group].append({"path": str(path.resolve()), "sha256": digest, **detail})
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    locator.write_new(args.output, result)
    for group in ("corrections", "recipes"):
        print(f"{group}: {len(result[group])}")
        for item in result[group]:
            print(" ", item["path"], f"tau={item['tau']}" if "tau" in item else "")
    print("Only one of each permits automatic selection; otherwise specify both paths.")
    print("No original CSV, weights, OOF rows, or Test read. Report:", args.output)
    return result


def choose_paths(args):
    if args.correction is not None and args.recipe is not None:
        return args.correction, args.recipe
    if args.correction is not None or args.recipe is not None or args.discovery is None:
        raise ValueError("Supply --discovery OR both --correction and --recipe.")
    report = locator.read_json(args.discovery)
    if report.get("kind") != KIND:
        raise ValueError("Invalid discovery report.")
    selected = []
    for group in ("corrections", "recipes"):
        choices = report[group]
        if len(choices) != 1:
            raise ValueError("Missing or ambiguous saved artifacts; specify both paths explicitly.")
        item = choices[0]
        path = Path(item["path"])
        if pc.sha256(path) != item["sha256"]:
            raise ValueError("Artifact changed since discovery; discover again.")
        selected.append(path)
    return tuple(selected)


def predict(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new output_dir.")
    correction_path, recipe_path = choose_paths(args)
    if pc.sha256(ROOT / pc.THIS_FILE) != locator.SOURCE_SHA:
        raise ValueError("Prior40 inference source changed.")
    correction = correction_at(correction_path)
    watched = {
        str(p.resolve()): pc.sha256(p)
        for p in (correction_path, recipe_path, args.input, args.sample_submission, Path(__file__))
    }
    recipe, manifest, files, loaded = pc.load_saved_members(recipe_path, args.device)
    frame = read_frame(args.input)
    ordered = pc.ibr.template_ids(args.sample_submission, frame.sample_id.tolist())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pc.write_json(args.output_dir / "status.json", {"status": "running"})
    try:
        print(
            f"Saved neural:ExtraTrees=40:60, saved tau={correction['tau']}; no fitting.", flush=True
        )
        images = pc.ibr.image_digest(frame, args.image_root)
        labels = audit.infer(frame, args.image_root, correction, recipe, manifest, loaded)
        positions = np.unique(np.linspace(0, len(frame) - 1, min(32, len(frame)), dtype=int))[::-1]
        partial = audit.infer(
            frame.iloc[positions].reset_index(drop=True),
            args.image_root,
            correction,
            recipe,
            manifest,
            loaded,
        )
        changed = int(np.count_nonzero(partial != labels[positions]))
        if changed:
            raise ValueError("Reordered subset predictions differ; no submission exported.")
        pc.ibr.check_unchanged(manifest, recipe["source_sha256"], files, loaded[-1])
        pc.ibr.blend.midpoint.check_files(watched)
        if pc.ibr.image_digest(frame, args.image_root) != images:
            raise ValueError("Images changed during inference.")
        output = args.output_dir / "submission_inner_best_40_60_prior_tau_regenerated.csv"
        values = pd.Series(labels, index=frame.sample_id).loc[ordered].to_numpy(dtype=np.int64)
        with output.open("x", encoding="utf-8", newline="") as handle:
            pd.DataFrame({"sample_id": ordered, "label": values}).to_csv(handle, index=False)
        record = {
            "kind": KIND,
            "status": "complete",
            "neural_weight": 0.4,
            "tree_weight": 0.6,
            "tau": correction["tau"],
            "prior": correction["prior"],
            "correction": str(correction_path.resolve()),
            "recipe": str(recipe_path.resolve()),
            "input_and_config_sha256": watched,
            "model_files_sha256": files,
            "legacy_model_files_sha256": manifest["reference"]["base_recipe"]["provenance"][
                "files"
            ],
            "runtime": manifest["runtime"],
            "images_sha256": images,
            "output_sha256": pc.sha256(output),
            "subset_rows": len(positions),
            "subset_changed": changed,
            "original_csv_used": False,
            "original_submission_match_verified": False,
            "public_score": None,
            "correction_model_pair_proven_by_original_receipt": False,
            "portable_submission_package": False,
            "notice": "Saved settings used without retuning. Original CSV unavailable: "
            "neither the 0.9093891677 score nor the original model/config pairing is proven. "
            "Subset invariance is a spot check only.",
        }
        pc.write_json(output.with_suffix(".manifest.json"), record)
        pc.write_json(args.output_dir / "status.json", {"status": "complete"})
        print("Saved:", output)
        print("Subset changed=0. New export; original submission/score match is unverified.")
        return record
    except BaseException as error:
        pc.write_json(
            args.output_dir / "status.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    find = commands.add_parser("discover")
    find.add_argument("--runs_root", action="append", type=Path, default=[])
    find.add_argument("--team_home", type=Path)
    find.add_argument("--output", type=Path, required=True)
    find.set_defaults(handler=discover)
    export = commands.add_parser("predict")
    for name in ("discovery", "correction", "recipe"):
        export.add_argument("--" + name, type=Path)
    for name in ("input", "image_root", "sample_submission", "output_dir"):
        export.add_argument("--" + name, type=Path, required=True)
    export.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    export.set_defaults(handler=predict)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    main()
