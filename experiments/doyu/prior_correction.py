"""inner_best 신경망 : ExtraTrees 40:60 위의 클래스 사전확률 보정 p_c / pi_c**tau.

모델·비율·전처리는 그대로 두고, 블렌드 확률에 Train 클래스 비율 pi의 tau제곱을 나눈 뒤
argmax만 바꾼다. tau는 Train group OOF로만 고르고 Validation·Test는 고르는 데 쓰지 않는다.

명령
  oof       ExtraTrees의 5-fold group OOF(train_oof.npz)와 같은 fold로, 고정 ResNet18
            임베딩 캐시 위에서 frozen 신경망 head를 fold마다 새로 학습해 신경망 OOF를 만든다.
            (backbone이 고정이라 head만 학습하면 된다. inner_best와 같은 설정·epoch 수.)
  select    40:60 OOF에서 tau를 격자로 고른다. nested OOF(다른 4 fold로 tau·pi를 고르고
            남은 fold에 적용)가 tau=0보다 엄격히 높을 때만 adopt=true. 최종 tau는 전체 OOF로.
  validate  저장된 inner_best + ExtraTrees를 Validation에 한 번 추론해 40:60과 보정본 비교.
            40:60 집계 지표가 승범 recipe와 정확히 같아야 진행한다.
  predict   Test 추론 -> 보정 -> 제출 CSV. adopt=false면 중단한다.
"""

from __future__ import annotations

import argparse
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

from evaluate import classification_metrics  # noqa: E402
from experiments.seungbeom import frozen_checkpoint as fc  # noqa: E402
from experiments.seungbeom import inner_best_ratios as ibr  # noqa: E402
from schema import LABELS, RAW_SENSOR_COLUMNS, read_frame, validate_labels  # noqa: E402

ft = fc.ft
KIND = "doyu_prior_correction"
BASE_METHOD = "inner_best_40_60"
NEURAL_WEIGHT = ibr.RATIOS[BASE_METHOD]
TAU_GRID = tuple(round(0.05 * i, 2) for i in range(41))  # 0.00 ... 2.00
THIS_FILE = "experiments/doyu/prior_correction.py"


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


# ---------------------------------------------------------------- correction rule
def class_prior(labels):
    labels = validate_labels(labels)
    counts = np.array([(labels == label).sum() for label in LABELS], dtype=np.float64)
    if (counts == 0).any():
        raise ValueError("Every class must appear in the rows used for the prior.")
    return counts / counts.sum()


def corrected_labels(probabilities, prior, tau):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != len(LABELS) or not np.isfinite(p).all():
        raise ValueError("Probabilities must be finite [rows, 6].")
    return (p / np.asarray(prior, dtype=np.float64) ** float(tau)).argmax(1) + 1


def macro_f1(labels, predictions):
    return classification_metrics(labels, predictions)["macro_f1"]


def best_tau(probabilities, labels, prior):
    """Highest macro F1 on these rows; ties keep the smaller tau (closer to no change)."""
    scores = [macro_f1(labels, corrected_labels(probabilities, prior, t)) for t in TAU_GRID]
    index = int(np.argmax(scores))  # argmax returns the first maximum
    return TAU_GRID[index], scores


def blend(neural, tree):
    return NEURAL_WEIGHT * np.asarray(neural) + (1 - NEURAL_WEIGHT) * np.asarray(tree)


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
    if features.shape != (len(ids), 512) or not np.isfinite(features).all():
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


# ---------------------------------------------------------------- tau selection
def select(args):
    out = args.output_dir.resolve()
    if out.exists():
        raise ValueError("output_dir exists; choose a new one.")
    if read_json(args.oof_dir / "status.json").get("status") != "complete":
        raise ValueError("OOF run is not complete.")
    info = read_json(args.oof_dir / "oof.json")
    if sha256(args.oof_dir / "oof.npz") != info["oof_sha256"]:
        raise ValueError("oof.npz changed.")
    z = np.load(args.oof_dir / "oof.npz", allow_pickle=False)
    labels, fold = z["labels"], z["fold"]
    p = blend(z["neural"], z["extra_trees"])
    base_pred = p.argmax(1) + 1
    nested_pred = np.zeros_like(base_pred)
    per_fold = []
    for f in sorted(set(fold.tolist())):
        other, held = fold != f, fold == f
        prior = class_prior(labels[other])
        tau, _ = best_tau(p[other], labels[other], prior)
        nested_pred[held] = corrected_labels(p[held], prior, tau)
        per_fold.append(
            {
                "fold": int(f),
                "tau": tau,
                "baseline_macro_f1": macro_f1(labels[held], base_pred[held]),
                "corrected_macro_f1": macro_f1(labels[held], nested_pred[held]),
            }
        )
    prior = class_prior(labels)
    tau, grid = best_tau(p, labels, prior)
    baseline = classification_metrics(labels, base_pred)
    nested = classification_metrics(labels, nested_pred)
    full = classification_metrics(labels, corrected_labels(p, prior, tau))
    adopt = bool(nested["macro_f1"] > baseline["macro_f1"] and tau > 0)
    correction = {
        "kind": KIND,
        "format_version": 1,
        "rule": "argmax(p_blend / prior**tau)",
        "base_method": BASE_METHOD,
        "neural_weight": NEURAL_WEIGHT,
        "tau": tau,
        "prior": prior.tolist(),
        "prior_source": "all Train labels",
        "label_values": list(LABELS),
        "adopt": adopt,
        "at_grid_edge": tau == TAU_GRID[-1],
        "selection": "Train group OOF only; nested OOF decides adoption",
        "oof_dir": str(args.oof_dir.resolve()),
        "oof_sha256": info["oof_sha256"],
        "this_file_sha256": sha256(ROOT / THIS_FILE),
        "oof_metrics": {"baseline": baseline, "nested": nested, "full_fit": full},
        "per_fold": per_fold,
        "grid": dict(zip(map(str, TAU_GRID), grid)),
    }
    out.mkdir(parents=True)
    write_json(out / "correction.json", correction)
    print_selection(correction)
    return correction


def print_selection(c):
    m = c["oof_metrics"]
    print(f"Train OOF, base = {c['base_method']} (neural weight {c['neural_weight']})")
    print("row\tmacro_f1\t" + "\t".join(f"f1_{label}" for label in LABELS))
    for name in ("baseline", "nested", "full_fit"):
        print(
            f"{name}\t{m[name]['macro_f1']:.6f}\t"
            + "\t".join(f"{m[name]['class_f1'][str(label)]:.4f}" for label in LABELS)
        )
    print("\nfold\ttau\tbaseline\tcorrected\tdelta")
    for row in c["per_fold"]:
        delta = row["corrected_macro_f1"] - row["baseline_macro_f1"]
        print(
            f"{row['fold']}\t{row['tau']:.2f}\t{row['baseline_macro_f1']:.6f}\t"
            f"{row['corrected_macro_f1']:.6f}\t{delta:+.6f}"
        )
    print(f"\nfinal tau={c['tau']:.2f}  adopt={c['adopt']}  at_grid_edge={c['at_grid_edge']}")
    print("prior=" + ", ".join(f"{label}:{v:.4f}" for label, v in zip(LABELS, c["prior"])))
    print("nested = tau chosen on the other folds; only nested > baseline adopts.")


# ---------------------------------------------------------------- saved-model inference
def read_correction(path):
    c = read_json(path)
    if (
        c.get("kind") != KIND
        or c.get("format_version") != 1
        or c.get("base_method") != BASE_METHOD
        or c.get("neural_weight") != NEURAL_WEIGHT
        or c.get("label_values") != list(LABELS)
        or len(c.get("prior", [])) != len(LABELS)
        or c.get("tau") not in TAU_GRID
    ):
        raise ValueError("Invalid correction file.")
    return c


def load_saved_members(recipe_path, device):
    """Same provenance checks as inner_best_ratios.predict before any inference."""
    recipe = ibr.read_comparison(recipe_path)
    if recipe["source_sha256"] != ibr.source_hashes():
        raise ValueError("Ratio comparison source changed; use the recorded revision.")
    ibr.blend.midpoint.check_files(recipe["run_files"])
    manifest, result, files, compatibility, loaded = ibr.load_members(
        Path(recipe["run_dir"]), device
    )
    if (
        files != recipe["run_files"]
        or compatibility != recipe["source_compatibility"]
        or manifest["runtime"] != recipe["runtime"]
        or any(recipe["metrics"][name] != result["metrics"][name] for name in ibr.REPLAY)
    ):
        raise ValueError("Models or replay provenance differ from the comparison.")
    return recipe, manifest, files, loaded


def validate(args):
    correction = read_correction(args.correction)
    recipe, manifest, files, loaded = load_saved_members(args.recipe, args.device)
    path = args.data_dir / "validation/validation.csv"
    if sha256(path) != manifest["inputs"]["validation"]["csv_sha256"]:
        raise ValueError("Validation CSV differs from the checkpoint run.")
    frame = read_frame(path, labeled=True)
    neural, tree = ibr.infer_members(frame, path.parent, manifest, loaded)
    ibr.check_unchanged(manifest, recipe["source_sha256"], files, loaded[-1])
    p = ibr.weighted_probabilities(neural, tree, BASE_METHOD)
    base = classification_metrics(frame["label"], p.argmax(1) + 1)
    if base != recipe["metrics"][BASE_METHOD]:
        raise ValueError("40:60 Validation replay differs from the saved comparison.")
    fixed = classification_metrics(
        frame["label"], corrected_labels(p, correction["prior"], correction["tau"])
    )
    print(f"\nValidation (tau={correction['tau']:.2f}, adopt={correction['adopt']}; "
          "tau was chosen on Train OOF only)")
    print("method\tmacro_f1\t" + "\t".join(f"f1_{label}" for label in LABELS))
    for name, m in ((BASE_METHOD, base), ("prior_corrected", fixed)):
        print(f"{name}\t{m['macro_f1']:.6f}\t"
              + "\t".join(f"{m['class_f1'][str(label)]:.4f}" for label in LABELS))
    print(f"delta={fixed['macro_f1'] - base['macro_f1']:+.6f}")
    print("\nmethod\tclass\tprecision\trecall\tf1")
    for name, m in ((BASE_METHOD, base), ("prior_corrected", fixed)):
        for label in LABELS:
            k = str(label)
            print(f"{name}\t{label}\t{m['class_precision'][k]:.4f}\t"
                  f"{m['class_recall'][k]:.4f}\t{m['class_f1'][k]:.4f}")
    changed = int((corrected_labels(p, correction["prior"], correction["tau"])
                   != p.argmax(1) + 1).sum())
    print(f"rows whose label changed: {changed}/{len(frame)}")


def predict(args):
    correction = read_correction(args.correction)
    if not correction["adopt"] and not args.allow_not_adopted:
        raise ValueError("Correction was not adopted on nested Train OOF; keep the 40:60 CSV.")
    if args.output.exists():
        raise ValueError("Output exists; choose a new filename.")
    recipe, manifest, files, loaded = load_saved_members(args.recipe, args.device)
    frame = read_frame(args.input)
    ids = frame["sample_id"].tolist()
    ordered = ibr.template_ids(args.sample_submission, ids)
    neural, tree = ibr.infer_members(frame, args.image_root, manifest, loaded)
    ibr.check_unchanged(manifest, recipe["source_sha256"], files, loaded[-1])
    p = ibr.weighted_probabilities(neural, tree, BASE_METHOD)
    labels = corrected_labels(p, correction["prior"], correction["tau"])
    values = pd.Series(labels, index=ids).loc[ordered].to_numpy(dtype=np.int64)
    validate_labels(values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        pd.DataFrame({"sample_id": ordered, "label": values}).to_csv(handle, index=False)
    write_json(
        args.output.with_suffix(".manifest.json"),
        {
            "kind": KIND,
            "correction": correction["tau"],
            "correction_sha256": sha256(args.correction),
            "recipe_sha256": sha256(args.recipe),
            "input_sha256": sha256(args.input),
            "output_sha256": sha256(args.output),
            "this_file_sha256": sha256(ROOT / THIS_FILE),
        },
    )
    print("Submission CSV saved:", args.output)
    print("Saved models only; no fitting on Test.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("oof")
    for name in ("tree_run", "checkpoint_run", "data_dir", "output_dir"):
        a.add_argument("--" + name, type=Path, required=True)
    a.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a.set_defaults(handler=oof)
    b = sub.add_parser("select")
    for name in ("oof_dir", "output_dir"):
        b.add_argument("--" + name, type=Path, required=True)
    b.set_defaults(handler=select)
    c = sub.add_parser("validate")
    for name in ("correction", "recipe", "data_dir"):
        c.add_argument("--" + name, type=Path, required=True)
    c.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    c.set_defaults(handler=validate)
    d = sub.add_parser("predict")
    for name in ("correction", "recipe", "input", "image_root", "sample_submission", "output"):
        d.add_argument("--" + name, type=Path, required=True)
    d.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    d.add_argument("--allow_not_adopted", action="store_true")
    d.set_defaults(handler=predict)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    main()
