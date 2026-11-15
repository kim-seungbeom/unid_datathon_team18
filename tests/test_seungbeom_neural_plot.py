"""Learning curves use synthetic report files, never competition outputs."""

import json

import pytest

from experiments.seungbeom.neural_plot import plot_history, read_history


def reports(root, *, status="complete"):
    root.mkdir()
    values = {
        "status.json": {"status": status},
        "config.json": {"encoder": "embedding", "fusion": "gated", "seed": 42, "loss": "focal"},
        "best_metrics.json": {"epoch": 3, "macro_f1": 0.75},
        "history.json": {
            "epochs": [
                {
                    "epoch": epoch,
                    "train_loss": 1.0 / epoch,
                    "validation_ce_loss": 1.2 / epoch,
                    "macro_f1": min(0.75, 0.25 * epoch),
                    "class_f1": {
                        str(label): min(0.95, 0.2 * epoch + 0.03 * label) for label in range(1, 7)
                    },
                }
                for epoch in range(1, 5)
            ]
        },
    }
    for name, content in values.items():
        (root / name).write_text(json.dumps(content), encoding="utf-8")
    return root


def test_render_synthetic_learning_curves_and_preserve_existing_output(tmp_path):
    pytest.importorskip("matplotlib")
    from PIL import Image

    run = reports(tmp_path / "synthetic_report")
    output = tmp_path / "curves.png"
    assert plot_history(run, output) == output
    with Image.open(output) as image:
        assert image.size == (1920, 1280)
    original = output.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        plot_history(run, output)
    assert output.read_bytes() == original


def test_reject_failed_or_corrupt_history(tmp_path):
    with pytest.raises(ValueError, match="complete"):
        read_history(reports(tmp_path / "failed", status="failed"))
    run = reports(tmp_path / "corrupt")
    history = json.loads((run / "history.json").read_text())
    history["epochs"][1]["macro_f1"] = float("nan")
    (run / "history.json").write_text(json.dumps(history))
    with pytest.raises(ValueError, match="finite"):
        read_history(run)


def test_reject_best_epoch_not_in_history(tmp_path):
    run = reports(tmp_path / "missing_best")
    (run / "best_metrics.json").write_text(json.dumps({"epoch": 9}))
    with pytest.raises(ValueError, match="absent"):
        read_history(run)
