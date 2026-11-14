"""Validation 한 번 추론으로 멤버·블렌드·tau 보정·OOF 프록시 신경망을 나란히 비교한다.

- 저장 모델(inner_best, ExtraTrees, refit 신경망)은 추론만 한다.
- 프록시 신경망 = prior_correction oof와 같은 방식(고정 ResNet18 임베딩 캐시 위 head 학습)을
  전체 Train으로 학습한 것. refit 신경망(이미지 경로, 전체 Train, 같은 epoch·초기값)과의
  예측 일치율로 OOF 프록시의 충실도를 확인한다.
- Validation은 학습·선택에 쓰지 않는다. Test는 읽지 않는다. 출력은 사람이 읽는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from evaluate import classification_metrics  # noqa: E402
from experiments.doyu import prior_correction as pc  # noqa: E402
from schema import LABELS, read_frame  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("correction", "recipe", "data_dir", "tree_run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Output exists; choose a new filename.")

    correction = pc.read_correction(args.correction)
    recipe, manifest, files, loaded = pc.load_saved_members(args.recipe, args.device)
    device = loaded[-1]
    val_path = args.data_dir / "validation/validation.csv"
    train_path = args.data_dir / "train/train.csv"
    if pc.sha256(val_path) != manifest["inputs"]["validation"]["csv_sha256"]:
        raise ValueError("Validation CSV differs from the checkpoint run.")
    if pc.sha256(train_path) != manifest["inputs"]["train"]["csv_sha256"]:
        raise ValueError("Train CSV differs from the checkpoint run.")
    val = read_frame(val_path, labeled=True)
    train = read_frame(train_path, labeled=True)
    y = val["label"].to_numpy()

    # Saved members (inference only).
    inner, tree = pc.ibr.infer_members(val, val_path.parent, manifest, loaded)
    pc.ibr.check_unchanged(manifest, recipe["source_sha256"], files, device)
    blend = pc.ibr.weighted_probabilities(inner, tree, pc.BASE_METHOD)
    if classification_metrics(y, blend.argmax(1) + 1) != recipe["metrics"][pc.BASE_METHOD]:
        raise ValueError("40:60 Validation replay differs from the saved comparison.")

    run_dir = Path(recipe["run_dir"])
    _, result = pc.fc.read_result(run_dir)
    print("Predicting saved refit neural (full Train, image path; no training)...", flush=True)
    refit_model, refit_prep = pc.fc.restore(
        run_dir / "refit.pt", result["identities"]["refit"], device
    )
    refit = pc.fc.probability(
        refit_model, refit_prep, val, val_path.parent, manifest["config"], device
    )
    del refit_model

    # Proxy: same cached-feature head training as the OOF, on all Train rows.
    print("Training proxy head on cached Train embeddings (Validation not used)...", flush=True)
    config = pc.ft.normalize_config(manifest["config"])
    epochs = int(result["selected_epoch"])
    train_cache = pc.load_embeddings(args.tree_run, "train")
    val_cache = pc.load_embeddings(args.tree_run, "validation")
    ids_train = train["sample_id"].astype(str).to_numpy()
    ids_val = val["sample_id"].astype(str).to_numpy()
    if set(train_cache.index) != set(ids_train) or set(val_cache.index) != set(ids_val):
        raise ValueError("Embedding caches must match Train/Validation rows exactly.")
    proxy_model, proxy_prep = pc.train_head(
        train_cache.loc[ids_train].to_numpy(np.float32), train, config, epochs, device
    )
    proxy = pc.head_probabilities(
        proxy_model, proxy_prep, val_cache.loc[ids_val].to_numpy(np.float32), val, device
    )

    prior, tau = np.asarray(correction["prior"]), correction["tau"]
    w = pc.NEURAL_WEIGHT
    preds = {
        "inner_best_neural": inner.argmax(1) + 1,
        "extra_trees": tree.argmax(1) + 1,
        "blend_40_60": blend.argmax(1) + 1,
        f"blend_40_60_tau{tau:.2f}": pc.corrected_labels(blend, prior, tau),
        "refit_neural": refit.argmax(1) + 1,
        "proxy_neural": proxy.argmax(1) + 1,
        "proxy_blend_40_60": (w * proxy + (1 - w) * tree).argmax(1) + 1,
        f"proxy_blend_40_60_tau{tau:.2f}": pc.corrected_labels(
            w * proxy + (1 - w) * tree, prior, tau
        ),
    }
    metrics = {name: classification_metrics(y, p) for name, p in preds.items()}

    lines = ["Validation comparison (inference only; tau chosen on Train OOF)",
             "method\tmacro_f1\t" + "\t".join(f"f1_{c}" for c in LABELS)]
    for name, m in metrics.items():
        lines.append(f"{name}\t{m['macro_f1']:.6f}\t"
                     + "\t".join(f"{m['class_f1'][str(c)]:.4f}" for c in LABELS))
    lines.append("\nmethod\tclass\tprecision\trecall")
    for name, m in metrics.items():
        for c in LABELS:
            lines.append(f"{name}\t{c}\t{m['class_precision'][str(c)]:.4f}\t"
                         f"{m['class_recall'][str(c)]:.4f}")
    agree = float((preds["proxy_neural"] == preds["refit_neural"]).mean())
    gap = float(np.abs(proxy - refit).max())
    lines.append(f"\nproxy vs refit neural: label agreement={agree:.4f}  max|prob diff|={gap:.4f}")
    lines.append("agreement near 1 -> OOF proxy is faithful; low -> OOF neural is not representative.")
    text = "\n".join(lines)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")
    with args.output.with_suffix(".json").open("x", encoding="utf-8") as handle:
        json.dump({"proxy_refit_agreement": agree, "proxy_refit_max_abs": gap}, handle, indent=2)


if __name__ == "__main__":
    main()
