"""Compare one bounded OOF-only residual correction on the existing prior30 model.

No base training, image extraction, Test fitting or leaderboard optimization.
Export is enabled only by a predefined Train-OOF diagnostic gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.seungbeom import prior_residual_core as core  # noqa: E402
from experiments.seungbeom import probability_fusion as verified  # noqa: E402
from experiments.seungbeom.export_submissions import check_csv, read_json, sha256, write_json  # noqa: E402
from schema import read_frame  # noqa: E402

previous, prior30, cache, original = (
    verified.previous,
    verified.prior30,
    verified.cache,
    verified.original,
)
KIND = "seungbeom_prior_residual"


def sources():
    names = (
        "experiments/seungbeom/prior_residual.py",
        "experiments/seungbeom/prior_residual_core.py",
    )
    return {**verified.sources(), **{n: sha256(ROOT / n) for n in names}}


def unchanged(base, manifest, files, code, env, device):
    previous.unchanged(base, manifest, files, previous.sources(), device)
    if sources() != code or verified.versions() != env:
        raise ValueError("Residual correction code/runtime changed.")


def display(recipe, folds=False):
    d, state = recipe["diagnostics"], recipe["correction"]
    print(f"Original neural:ExtraTrees=30:70; anchor tau={state['tau']}; no base-model fitting.")
    print("Bounded log-score offsets (labels1..6):", state["offsets"])
    print("method\tfold_excluded_macro_f1\tfit_diagnostic_macro_f1")
    for n in (core.BASELINE, core.METHOD):
        print(
            f"{n}\t{d['fold_excluded_diagnostic'][n]['macro_f1']:.6f}\t"
            f"{d['full_fit_diagnostic'][n]['macro_f1']:.6f}"
        )
    if folds:
        print("fold\ttau\tdelta_vs_fold_prior_anchor")
        for f in d["per_fold"]:
            print(f"{f['fold']}\t{f['state']['tau']}\t{f['delta']:+.6f}")
    print(
        f"OOF diagnostic gate: adopt={d['adopt']} delta={d['delta']:+.6f} "
        f"wins={d['wins']}/{len(d['per_fold'])}"
    )
    prior30.display(recipe["validation_metrics"])
    print("Changed Validation rows vs prior30:", recipe["changed_rows"])
    print(d["limitations"])
    print("Validation comparison only; existing highest submission remains unchanged.")
    if not d["adopt"]:
        print("Gate failed: predict will not read Test or export this candidate.")


def compare(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new one.")
    code, env = sources(), verified.versions()
    old, reference, base, manifest, files = previous.load_reference(
        args.baseline_recipe, args.device
    )
    if sha256(args.input) != base["validation"]["csv_sha256"]:
        raise ValueError("Use original Validation input.")
    # Fit/select before opening Validation values or member probabilities.
    neural, tree, y, folds = verified.load_oof(args, old, manifest, files)
    state, diagnosis = core.crossfit(neural, tree, y, folds)
    if state["prior"] != old["correction"]["prior"] or state["tau"] != old["correction"]["tau"]:
        raise ValueError("Full-OOF prior/tau does not replay original prior30 anchor.")
    files.update(original.file_hashes([args.input]))
    frame = read_frame(args.input, labeled=True)
    members, images = cache.cached_members(
        args, frame, base, manifest, files, cache.source_hashes()
    )
    if images != base["validation"]["images_sha256"]:
        raise ValueError("Validation images changed.")
    existing = cache.previous.candidates(*members)
    if any(
        prior30.classification_metrics(frame.label, existing[n].argmax(1) + 1)
        != reference["metrics"][n]
        for n in cache.REPLAY
    ):
        raise ValueError("Original members/ratios failed to replay.")
    p = prior30.mixture(*members)
    predictions = {prior30.BASELINE: p.argmax(1) + 1, **core.apply(p, state)}
    metrics = {n: prior30.classification_metrics(frame.label, v) for n, v in predictions.items()}
    if any(metrics[n] != old["metrics"][n] for n in (prior30.BASELINE, core.BASELINE)):
        raise ValueError("Original raw30/prior30 Validation failed to replay.")
    unchanged(base, manifest, files, code, env, args.device)
    recipe = {
        "kind": KIND,
        "contract": core.CONTRACT,
        "source_sha256": code,
        "versions": env,
        "baseline_path": str(args.baseline_recipe.resolve()),
        "baseline": old,
        "correction": state,
        "diagnostics": diagnosis,
        "validation_metrics": metrics,
        "changed_rows": int(
            np.count_nonzero(predictions[core.METHOD] != predictions[core.BASELINE])
        ),
        "provenance": files,
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "recipe.json"
    write_json(path, recipe)
    write_json(args.output_dir / "status.json", {"status": "complete", "sha256": sha256(path)})
    display(recipe, folds=True)
    return recipe


def read_recipe(path):
    r, status = read_json(path), read_json(path.parent / "status.json")
    if (
        status.get("status") != "complete"
        or status.get("sha256") != sha256(path)
        or r.get("kind") != KIND
        or r.get("contract") != core.CONTRACT
        or r.get("source_sha256") != sources()
        or r.get("versions") != verified.versions()
    ):
        raise ValueError("Residual recipe/source/runtime changed or incomplete.")
    core.check_state(r["correction"])
    return r


def predict(args):
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose a new one.")
    r = read_recipe(args.recipe)
    if not r["diagnostics"]["adopt"]:
        print("OOF diagnostic gate failed; no Test access or CSV export. Keep original best.")
        return {}
    old, _, base, manifest, files = previous.load_reference(Path(r["baseline_path"]), args.device)
    if old != r["baseline"]:
        raise ValueError("Original baseline changed.")
    files.update(
        original.file_hashes(
            [args.recipe, args.recipe.parent / "status.json", args.input, args.sample_submission]
        )
    )
    frame = read_frame(args.input)
    ids = frame.sample_id.tolist()
    order = prior30.template_ids(args.sample_submission, ids)
    members, images = cache.cached_members(
        args, frame, base, manifest, files, cache.source_hashes()
    )
    predictions = core.apply(prior30.mixture(*members), r["correction"])
    unchanged(base, manifest, files, r["source_sha256"], r["versions"], args.device)
    if np.array_equal(predictions[core.METHOD], predictions[core.BASELINE]):
        print("Candidate equals existing prior30 labels; no duplicate CSV exported.")
        return {}
    args.output_dir.mkdir(parents=True)
    status = args.output_dir / "status.json"
    write_json(status, {"status": "running"})
    try:
        values = pd.Series(predictions[core.METHOD], index=ids).loc[order].to_numpy(dtype=np.int64)
        output = args.output_dir / f"submission_{core.METHOD}.csv"
        with output.open("x", encoding="utf-8", newline="") as handle:
            pd.DataFrame({"sample_id": order, "label": values}).to_csv(handle, index=False)
        info = check_csv(output, order)
        write_json(
            output.with_suffix(".manifest.json"),
            {
                "recipe": r,
                "method": core.METHOD,
                "input_files": files,
                "images_sha256": images,
                "output": info,
            },
        )
        unchanged(base, manifest, files, r["source_sha256"], r["versions"], args.device)
        write_json(status, {"status": "complete", "output": info})
        print(f"Saved {output}; fixed correction only. No Train/Validation/OOF data read.")
        print("OOF diagnostic eligibility does not guarantee a higher Public/Private score.")
        return {core.METHOD: info}
    except BaseException as error:
        write_json(status, {"status": "failed", "error_type": type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    comp = commands.add_parser("compare")
    for n in ("baseline_recipe", "oof_dir", "train_csv"):
        comp.add_argument("--" + n, type=Path, required=True)
    comp.set_defaults(handler=compare)
    export = commands.add_parser("predict")
    for n in ("recipe", "sample_submission"):
        export.add_argument("--" + n, type=Path, required=True)
    export.set_defaults(handler=predict)
    for command in (comp, export):
        for n in ("input", "image_root", "cache_dir", "output_dir"):
            command.add_argument("--" + n, type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    report = commands.add_parser("report")
    report.add_argument("--recipe", type=Path, required=True)
    report.add_argument("--folds", action="store_true")
    report.set_defaults(handler=lambda a: display(read_recipe(a.recipe), a.folds))
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
