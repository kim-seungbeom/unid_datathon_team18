"""최종 제출 모델: inner_best 신경망 : ExtraTrees = 40:60 확률 평균 + 클래스 사전확률(tau) 보정.

`submission_inner_best_40_60_prior_tau.csv`를 만든 `prior_correction.py predict`와 같은 함수로
추론하되, 서버 절대경로·실행 폴더 대신 레포의 `weights/tau_final/` 묶음만 읽는다.

명령
  export   (서버에서 한 번) recipe·correction이 가리키는 저장 모델을 원래 검사 그대로 불러와
           weights/tau_final/ 에 신경망 가중치·ExtraTrees·설정을 묶는다. 학습하지 않는다.
  predict  묶음으로 입력 CSV의 행만 추론해 sample_id,label CSV를 입력 순서로 저장한다.
  compare  두 제출 CSV의 라벨이 sample_id별로 완전히 같은지 확인한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from schema import LABELS, read_frame, validate_labels  # noqa: E402

KIND = "doyu_tau_final"
FORMAT_VERSION = 1
NEURAL_FILE = "inner_best.pt"
TREE_FILE = "extra_trees.joblib"
CONFIG_FILE = "final.json"
DEFAULT_DIR = ROOT / "weights" / "tau_final"


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_bundle(path) -> bool:
    path = Path(path)
    if not (path.is_dir() and (path / CONFIG_FILE).is_file()):
        return False
    try:
        return json.loads((path / CONFIG_FILE).read_text(encoding="utf-8")).get("kind") == KIND
    except (OSError, ValueError):
        return False


# ------------------------------------------------------------------ export (server, once)
def export(args):
    import joblib

    from experiments.doyu import prior_correction as pc
    from experiments.seungbeom import frozen_checkpoint as checkpoint

    out = Path(args.out)
    if out.exists():
        raise ValueError(f"{out} already exists; choose a new folder.")
    correction = pc.read_correction(args.correction)
    if not correction["adopt"]:
        raise ValueError("Correction was not adopted on nested Train OOF.")
    # Same provenance/source/runtime checks as the original predict command.
    recipe, manifest, files, loaded = pc.load_saved_members(args.recipe, args.device)
    _model, _prep, legacy, target = loaded
    run_dir = Path(recipe["run_dir"])
    _manifest, result = checkpoint.read_result(run_dir)
    bundle = legacy["bundle"]
    tree = {
        "sensor_spec": bundle["sensor_spec"],
        "encoder": bundle["encoder"],
        "embedding": bundle["embedding"],
        "columns": list(bundle["columns"]),
        "pipeline": bundle["pipelines"][bundle["models"].index("extra_trees")],
    }
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        raise ValueError(f"{tmp} exists from an interrupted export; remove it first.")
    tmp.mkdir(parents=True)
    shutil.copyfile(run_dir / NEURAL_FILE, tmp / NEURAL_FILE)
    joblib.dump(tree, tmp / TREE_FILE, compress=3)
    config = {
        "kind": KIND,
        "format_version": FORMAT_VERSION,
        "label_values": list(LABELS),
        "rule": "argmax(p / prior**tau), p = 0.4*inner_best_neural + 0.6*extra_trees",
        "base_method": pc.BASE_METHOD,
        "neural_weight": pc.NEURAL_WEIGHT,
        "prior": correction["prior"],
        "tau": correction["tau"],
        "image_precision": manifest["reference"]["base_recipe"]["image_precision"],
        "neural": {
            "file": NEURAL_FILE,
            "sha256": sha256(tmp / NEURAL_FILE),
            "identity": result["identities"]["inner_best"],
            "config": manifest["config"],
        },
        "tree": {"file": TREE_FILE, "sha256": sha256(tmp / TREE_FILE),
                 "seed": legacy["config"]["seed"]},
        "uses_history": False,
        "source": {
            "recipe_sha256": sha256(args.recipe),
            "correction_sha256": sha256(args.correction),
            "neural_source_sha256": sha256(run_dir / NEURAL_FILE),
            "runtime": checkpoint.runtime_signature(target),
        },
    }
    (tmp / CONFIG_FILE).write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
    tmp.rename(out)
    for name in (NEURAL_FILE, TREE_FILE, CONFIG_FILE):
        size = (out / name).stat().st_size / 2**20
        flag = "  <- 100MB 초과: GitHub 일반 업로드 불가" if size > 100 else ""
        print(f"{out / name}: {size:.1f} MB{flag}")
    print(f"tau={config['tau']}  image_precision={config['image_precision']}")
    return config


# ------------------------------------------------------------------ inference
def load(model_dir=DEFAULT_DIR, device="auto"):
    import joblib

    from experiments.seungbeom import frozen_checkpoint as checkpoint

    model_dir = Path(model_dir)
    config = json.loads((model_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    if (
        config.get("kind") != KIND
        or config.get("format_version") != FORMAT_VERSION
        or config.get("label_values") != list(LABELS)
        or len(config.get("prior", [])) != len(LABELS)
    ):
        raise ValueError("Unsupported tau_final bundle.")
    for part in ("neural", "tree"):
        path = model_dir / config[part]["file"]
        if sha256(path) != config[part]["sha256"]:
            raise ValueError(f"{path} differs from final.json; re-download the weights.")
    # Same order as inner_best_ratios.load_members: seed, select device, restore.
    checkpoint.ft.seed_everything(config["neural"]["config"]["seed"])
    target = checkpoint.ft.select_device(device)
    model, prep = checkpoint.restore(
        model_dir / config["neural"]["file"], config["neural"]["identity"], target
    )
    tree = joblib.load(model_dir / config["tree"]["file"])
    return config, (model, prep, tree, target)


def probabilities(frame, image_root, config, loaded):
    """Member probabilities with the same calls as inner_best_ratios.infer_members."""
    from experiments.seungbeom import frozen_checkpoint as checkpoint
    from experiments.seungbeom import multimodal as mm
    from experiments.seungbeom.replay_precision import precision_context
    from runtime import seed_everything

    model, prep, tree, target = loaded
    current = frame.drop(columns="label", errors="ignore")
    print("Predicting inner_best neural (saved weights, no training)...", flush=True)
    neural = checkpoint.probability(model, prep, current, image_root, config["neural"]["config"],
                                    target)
    print("Predicting ExtraTrees (saved pipeline, no training)...", flush=True)
    seed_everything(config["tree"]["seed"])
    with precision_context(config["image_precision"]):
        features, _meta = mm.make_features(
            current, None, image_root, tree["sensor_spec"], encoder=tree["encoder"],
            embedding_spec=tree["embedding"], device=str(target), batch_size=1, workers=0,
        )
    if list(features.columns) != tree["columns"]:
        raise ValueError("Tree feature order differs from training.")
    return neural, mm.predict_proba_six_stable(tree["pipeline"], features)


def predict_labels(frame, image_root, config, loaded):
    from experiments.doyu import prior_correction as pc
    from experiments.seungbeom import inner_best_ratios as ibr

    neural, tree = probabilities(frame, image_root, config, loaded)
    p = ibr.weighted_probabilities(neural, tree, config["base_method"])
    return validate_labels(pc.corrected_labels(p, config["prior"], config["tau"]))


def run_test(input_path, image_root, model_dir, output, device="auto"):
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists. Choose a new path.")
    config, loaded = load(model_dir, device)
    frame = read_frame(input_path)
    labels = predict_labels(frame, image_root, config, loaded)
    if len(labels) != len(frame):
        raise ValueError("Prediction count does not match input row count.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        pd.DataFrame({"sample_id": frame["sample_id"], "label": labels}).to_csv(handle, index=False)
    print("Inference complete: sample_id,label CSV saved in input row order.")


def compare(args):
    a, b = (pd.read_csv(p, dtype={"sample_id": str}, keep_default_na=False) for p in (args.a, args.b))
    if sorted(a.sample_id) != sorted(b.sample_id) or a.sample_id.duplicated().any():
        raise ValueError("The two CSVs have different sample_id sets.")
    merged = a.merge(b, on="sample_id", suffixes=("_a", "_b"))
    diff = int((merged["label_a"] != merged["label_b"]).sum())
    same_order = a.sample_id.tolist() == b.sample_id.tolist()
    print(f"rows={len(a)}  label_differences={diff}  same_row_order={same_order}")
    print("IDENTICAL" if diff == 0 else "DIFFERENT")
    if diff:
        raise SystemExit(1)
    return diff


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("--recipe", type=Path, required=True)
    e.add_argument("--correction", type=Path, required=True)
    e.add_argument("--out", type=Path, default=DEFAULT_DIR)
    e.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    e.set_defaults(handler=export)
    p = sub.add_parser("predict")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--image_root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model_dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.set_defaults(handler=lambda a: run_test(a.input, a.image_root, a.model_dir, a.output, a.device))
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.set_defaults(handler=compare)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    main()
