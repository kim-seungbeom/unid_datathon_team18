"""Generated images only; random initialization, no downloads or competition data."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from encoders import DIMENSIONS, FILL, build_encoder, preprocess
from extract_all import run_all
from extract_features import ExtractionError, extract, read_manifest
from PIL import Image


def make_inputs(root):
    (root / "images").mkdir()
    rows = []
    for i in range(4):
        Image.new("RGB", (31 + i, 75 - i), (20 * i, 40, 190)).save(
            root / "images" / f"synthetic_{i}.png"
        )
        rows.append({"sample_id": f"synthetic_{i}", "image_path": f"images/synthetic_{i}.png"})
    write_manifest(root / "input.csv", rows)
    return rows


def write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "image_path"])
        writer.writeheader()
        writer.writerows(rows)


def offline_job(**kwargs):
    """Picklable CPU worker to exercise the actual spawn-based parallel runner."""
    torch.set_num_threads(1)
    model = build_encoder(kwargs["model_name"], pretrained=False)
    return extract(**kwargs, _encoder=model)


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_preprocessing(self):
        transparent = preprocess(Image.new("RGBA", (31, 81), (255, 0, 0, 0)))
        filled = preprocess(Image.new("RGB", (224, 224), FILL))
        torch.testing.assert_close(transparent, filled)
        self.assertEqual(tuple(filled.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(filled).all())

    def test_all_three_encoder_dimensions_and_freezing(self):
        tensor = preprocess(Image.new("RGB", (60, 90))).unsqueeze(0)
        for name, dimension in DIMENSIONS.items():
            with self.subTest(encoder=name):
                model = build_encoder(name, pretrained=False)
                before = {k: v.clone() for k, v in model.state_dict().items()}
                model.train()  # forward must restore eval mode and prevent BN updates.
                output = model(tensor)
                self.assertEqual(tuple(output.shape), (1, dimension))
                self.assertFalse(output.requires_grad)
                self.assertFalse(model.training)
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                self.assertTrue(
                    all(torch.equal(before[k], v) for k, v in model.state_dict().items())
                )

    def test_storage_mapping_batch_and_order_invariance(self):
        with tempfile.TemporaryDirectory(prefix="unid_features_test_") as temp:
            root = Path(temp)
            rows = make_inputs(root)
            model = build_encoder("resnet18", pretrained=False)
            settings = dict(
                model_name="resnet18", image_root=root, device="cpu", workers=0, _encoder=model
            )
            result = extract(
                csv_path=root / "input.csv", output_dir=root / "first", batch_size=3, **settings
            )
            baseline = np.load(root / "first/features.npy", allow_pickle=False)
            self.assertEqual(baseline.shape, (4, 512))
            self.assertEqual(baseline.dtype, np.float32)
            self.assertEqual(result["status"], "complete")
            self.assertFalse(result["trained"])
            self.assertEqual(result["weights"], "SYNTHETIC_TEST_RANDOM_WEIGHTS")
            with (root / "first/sample_ids.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(
                    [r["sample_id"] for r in csv.DictReader(handle)], [r["sample_id"] for r in rows]
                )
            write_manifest(root / "reordered.csv", list(reversed(rows)))
            extract(
                csv_path=root / "reordered.csv",
                output_dir=root / "second",
                batch_size=1,
                **settings,
            )
            changed = np.load(root / "second/features.npy", allow_pickle=False)
            np.testing.assert_allclose(baseline, changed[::-1], rtol=1e-4, atol=2e-5)
            with self.assertRaises(FileExistsError):
                extract(csv_path=root / "input.csv", output_dir=root / "first", **settings)
            extract(
                csv_path=root / "input.csv",
                output_dir=root / "normalized",
                l2_normalize=True,
                **settings,
            )
            normalized = np.load(root / "normalized/features.npy", allow_pickle=False)
            np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1, atol=1e-5)

    def test_manifest_failures_and_image_folder_root(self):
        with tempfile.TemporaryDirectory(prefix="unid_features_test_") as temp:
            root = Path(temp)
            rows = make_inputs(root)
            self.assertEqual(len(read_manifest(root / "input.csv", root / "images")), 4)
            self.assertNotIn("synthetic_0", repr(read_manifest(root / "input.csv", root)[0]))
            for bad in (
                [rows[0], rows[0]],
                [{"sample_id": "s", "image_path": "../escape.png"}],
                [{"sample_id": "s", "image_path": "absent.png"}],
            ):
                write_manifest(root / "bad.csv", bad)
                with self.assertRaises(ExtractionError):
                    read_manifest(root / "bad.csv", root)

    def test_decode_failure_marks_output_incomplete(self):
        with tempfile.TemporaryDirectory(prefix="unid_features_test_") as temp:
            root = Path(temp)
            make_inputs(root)
            (root / "images/synthetic_0.png").write_bytes(b"corrupt synthetic image")
            with self.assertRaises(ExtractionError):
                extract(
                    model_name="resnet18",
                    csv_path=root / "input.csv",
                    image_root=root,
                    output_dir=root / "failed",
                    workers=0,
                    device="cpu",
                    _encoder=build_encoder("resnet18", pretrained=False),
                )
            metadata = json.loads((root / "failed/metadata.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertFalse((root / "failed/features.npy").exists())

    def test_sequential_runner_and_existing_output_guard(self):
        with tempfile.TemporaryDirectory(prefix="unid_features_test_") as temp:
            root = Path(temp)
            with patch(
                "extract_all.extract", side_effect=lambda **kwargs: kwargs["model_name"]
            ) as mocked:
                result = run_all(
                    csv_path=root / "input.csv", image_root=root, output_dir=root / "results"
                )
                self.assertEqual(result, list(DIMENSIONS))
                self.assertEqual(mocked.call_count, 3)
            (root / "results/resnet18").mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                run_all(csv_path=root / "input.csv", image_root=root, output_dir=root / "results")

    def test_spawn_parallel_runner_with_offline_encoders(self):
        with tempfile.TemporaryDirectory(prefix="unid_features_parallel_test_") as temp:
            root = Path(temp)
            make_inputs(root)
            with patch("extract_all.extract", offline_job):
                results = run_all(
                    csv_path=root / "input.csv",
                    image_root=root,
                    output_dir=root / "parallel",
                    device="cpu",
                    workers=0,
                    batch_size=2,
                    parallel=2,
                )
            self.assertEqual({r["model"] for r in results}, set(DIMENSIONS))
            for name, dimension in DIMENSIONS.items():
                self.assertEqual(
                    np.load(root / "parallel" / name / "features.npy", allow_pickle=False).shape,
                    (4, dimension),
                )

    def test_notebook_cell_rerun_preserves_existing_output(self):
        notebook = json.loads(
            (Path(__file__).parent / "01_extract_features.ipynb").read_text(encoding="utf-8")
        )
        source = next(
            "".join(c["source"])
            for c in notebook["cells"]
            if c["cell_type"] == "code" and "def extract_model_splits(" in "".join(c["source"])
        )
        with tempfile.TemporaryDirectory(prefix="unid_notebook_retry_test_") as temp:
            root = Path(temp)
            original = root / "dinov2/train"
            original.mkdir(parents=True)
            (original / "preserve.txt").write_text("original synthetic result")
            outputs = []

            def fake_extract(**kwargs):
                output = kwargs["output_dir"]
                output.mkdir(parents=True, exist_ok=False)
                outputs.append(output)
                return dict(model="dinov2", rows=0, dimension=384, seconds=0, status="complete")

            context = dict(
                OUTPUT_ROOT=root,
                DATA_DIR=root / "data",
                SPLITS=("train", "validation"),
                BATCH_SIZE=32,
                WORKERS=0,
                DEVICE="cpu",
                extract=fake_extract,
            )
            exec(compile(source, "<notebook retry function>", "exec"), context)
            context["extract_model_splits"]("dinov2")
            context["extract_model_splits"]("dinov2")
            self.assertEqual((original / "preserve.txt").read_text(), "original synthetic result")
            self.assertEqual(len(set(outputs)), 4)
            self.assertTrue(outputs[0].name.startswith("train_retry_"))
            self.assertEqual(outputs[1].name, "validation")


if __name__ == "__main__":
    unittest.main()
