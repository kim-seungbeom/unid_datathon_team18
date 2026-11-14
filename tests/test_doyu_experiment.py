"""experiments/doyu end-to-end on synthetic data only. Never read competition data or outputs."""

import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from runtime import ROOT

EXP = ROOT / "experiments" / "doyu"
sys.path.insert(0, str(EXP))  # best_*.joblib 언피클(models.ShiftLabels)과 실험 모듈 import
SENSORS = ["meas_v", "meas_a", "meas_t1", "meas_t2", "meas_p"]


def write_split(root, split, experiments, labeled=True, rows=24):
    rng = np.random.default_rng(len(split))
    folder = root / split
    (folder / "images").mkdir(parents=True)
    records = []
    for exp in experiments:
        for k in range(rows):
            label = k * 6 // rows + 1
            sample_id = f"{split}_{exp}_{k:03}"
            pixels = (rng.random((16, 16)) * 40 * label).clip(0, 255).astype("uint8")
            Image.fromarray(pixels).save(folder / "images" / f"{sample_id}.png")
            records.append(
                {
                    "sample_id": sample_id,
                    "experiment_id": f"E{exp}",
                    "relative_time_s": k * 2.0,
                    "image_path": f"images/{sample_id}.png",
                    **{s: label * (i + 1) + rng.normal(0, 0.1) for i, s in enumerate(SENSORS)},
                    "label": label,
                }
            )
    frame = pd.DataFrame(records)
    if not labeled:
        frame = frame.drop(columns="label")
    frame.to_csv(folder / f"{split}.csv", index=False)
    return folder


def run(script, *args):
    result = subprocess.run(
        [sys.executable, str(EXP / script), *map(str, args)],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    data = tmp_path_factory.mktemp("synthetic_data")
    write_split(data, "train", range(4))
    write_split(data, "validation", range(4, 6))
    write_split(data, "test", range(6, 7))
    out = tmp_path_factory.mktemp("tabular")
    run("train_compare.py", "--data_dir", data, "--out_dir", out, "--folds", 2)
    return data, out


def test_train_compare_saves_one_bundle_per_model(trained):
    _, out = trained
    for name in ("xgboost", "lightgbm", "mlp"):
        bundle = joblib.load(out / f"best_{name}.joblib")
        assert bundle["model_name"] == name
        assert len(bundle["features"]) == 33
    assert not (out / "best_model.joblib").exists()


def test_predict_writes_labels_and_metrics(trained, tmp_path):
    data, out = trained
    output = tmp_path / "submission.csv"
    run("predict.py", "--input", data / "test" / "test.csv", "--image_root", data / "test",
        "--model", out / "best_lightgbm.joblib", "--output", output)
    submission = pd.read_csv(output)
    assert list(submission.columns) == ["sample_id", "label"]
    assert submission.label.between(1, 6).all()
    assert (tmp_path / "submission_classification_report.csv").exists()


def test_dataana_runs_per_model(trained, tmp_path):
    data, out = trained
    run("dataana.py", "--data_dir", data, "--model", out / "best_mlp.joblib",
        "--out_dir", tmp_path / "mlp")
    assert (tmp_path / "mlp" / "delays.csv").exists()
    assert (tmp_path / "mlp" / "confusion_matrix.png").exists()


def test_cross_attention_training_runs(trained, tmp_path):
    data, _ = trained
    run("train_cross_attention.py", "--data_dir", data, "--out_dir", tmp_path, "--folds", 2,
        "--epochs", 1, "--batch_size", 16, "--device", "cpu")
    assert (tmp_path / "attention_model.pt").exists()
    assert (tmp_path / "scores.json").exists()


def test_cross_attention_forward_and_causal_sequences():
    import torch

    from models.cross_attention_model import CrossAttentionNet
    from train_cross_attention import make_sequences

    logits = CrossAttentionNet()(torch.randn(2, 1, 64, 64), torch.randn(2, 16, 5))
    assert logits.shape == (2, 6)
    frame = pd.DataFrame({"experiment_id": ["a"] * 3, "relative_time_s": [2.0, 0.0, 1.0],
                          **{s: [30.0, 10.0, 20.0] for s in SENSORS}})
    seq = make_sequences(frame, 4)[:, :, 0]
    np.testing.assert_array_equal(seq[1], [10, 10, 10, 10])  # earliest row: no future values
    np.testing.assert_array_equal(seq[0], [10, 10, 20, 30])


# ---------------------------------------------------------------------------
# fusion: 센서(sensor_features) + 이미지 인코더(image_features) + 최종 분류기
# ---------------------------------------------------------------------------
HISTORY_FEATURES = [*SENSORS, "meas_v_mean30s", "meas_v_delta30s"]


def split_csv(data, split):
    return data / split / f"{split}.csv"


def random_encoder():
    sys.path.append(str(ROOT / "image_features"))
    from encoders import build_encoder

    return build_encoder("resnet18", pretrained=False)  # synthetic test only


@pytest.fixture(scope="module")
def fused(trained, tmp_path_factory):
    import fusion as F

    data, _ = trained
    encoder = random_encoder()
    emb = tmp_path_factory.mktemp("embeddings")
    for split in ("train", "validation", "test"):
        F.extract_embeddings(split_csv(data, split), data / split, "resnet18",
                             emb / split, device="cpu", workers=0, _encoder=encoder)
    out = tmp_path_factory.mktemp("fusion") / "run"
    run("train_fusion.py", "--data_dir", data, "--out_dir", out, "--encoder", "resnet18",
        "--train_embeddings", emb / "train", "--val_embeddings", emb / "validation",
        "--sensor_features", *HISTORY_FEATURES, "--pca_dim", 8, "--folds", 2)
    return data, emb, out, encoder


def test_fusion_saves_bundles_with_sensor_and_image_columns(fused):
    _, _, out, _ = fused
    for name in ("xgboost", "lightgbm", "mlp"):
        bundle = joblib.load(out / f"best_{name}.joblib")
        assert bundle["kind"] == "fusion" and bundle["encoder"] == "resnet18"
        assert bundle["sensor_features"] == HISTORY_FEATURES
        assert sum(c.startswith("img_") for c in bundle["columns"]) == 512
        assert sum(c.startswith("sensor_") for c in bundle["columns"]) == len(HISTORY_FEATURES)
        pca = bundle["pipeline"].named_steps["fusion"].named_transformers_["image"]["pca"]
        assert pca.n_components_ == 8


def test_fusion_onthefly_embeddings_match_precomputed(fused):
    import fusion as F

    data, emb, out, encoder = fused
    bundle = joblib.load(out / "best_lightgbm.joblib")
    frame = F.read_frame(split_csv(data, "test"))
    pre, _ = F.features_for(frame, split_csv(data, "test"), data / "test", bundle,
                            embeddings_dir=emb / "test")
    live, _ = F.features_for(frame, split_csv(data, "test"), data / "test", bundle,
                             device="cpu", _encoder=encoder)
    np.testing.assert_allclose(live.to_numpy(), pre.to_numpy(), rtol=1e-5, atol=1e-5)


def test_fusion_predict_and_dataana(fused, tmp_path):
    data, emb, out, _ = fused
    output = tmp_path / "submission.csv"
    run("predict_fusion.py", "--input", split_csv(data, "test"), "--image_root", data / "test",
        "--model", out / "best_xgboost.joblib", "--output", output, "--embeddings", emb / "test")
    submission = pd.read_csv(output)
    assert list(submission.columns) == ["sample_id", "label"]
    assert submission.label.between(1, 6).all()
    run("dataana.py", "--data_dir", data, "--model", out / "best_mlp.joblib",
        "--out_dir", tmp_path / "ana", "--embeddings", emb / "validation")
    assert (tmp_path / "ana" / "delays.csv").exists()


def test_fusion_rejects_mismatched_embeddings(fused):
    import fusion as F

    data, emb, out, _ = fused
    bundle = joblib.load(out / "best_mlp.joblib")
    frame = F.read_frame(split_csv(data, "validation"))
    with pytest.raises(ValueError, match="no embedding"):
        F.features_for(frame, None, None, bundle, embeddings_dir=emb / "test")
    bundle = {**bundle, "embedding": {**bundle["embedding"], "l2_normalize": True}}
    with pytest.raises(ValueError, match="l2_normalize"):
        F.features_for(frame, None, None, bundle, embeddings_dir=emb / "validation")


def test_fusion_current_row_sensor_features_are_partial_input_invariant(trained):
    import fusion as F

    data, _ = trained
    frame = F.read_frame(split_csv(data, "test"))
    spec = {**F.sensor_spec(None, SENSORS), "encoder": None}
    full, _ = F.features_for(frame, None, None, spec)
    part, _ = F.features_for(frame.iloc[::3].reset_index(drop=True), None, None, spec)
    np.testing.assert_array_equal(part.to_numpy(), full.iloc[::3].to_numpy())
