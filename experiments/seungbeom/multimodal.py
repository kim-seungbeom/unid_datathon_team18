"""Private experiment adapters: frozen image vectors + causal sensor features.

No preprocessing is learned while constructing the feature matrix. Every learned
transform lives inside the estimator pipeline and is fitted by the caller on a
Train fold only. The neural sensor config contributes feature names and imputation
semantics only; its neural loss, scaling option, and network settings are not reused.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from schema import LABELS, RAW_SENSOR_COLUMNS, validate_labels  # noqa: E402
from sensor_features import (  # noqa: E402
    SENSOR_FEATURE_VERSION,
    build_sensor_features,
    uses_history,
    validate_feature_names,
    validate_imputation,
)
from temporal import TEMPORAL_VERSION  # noqa: E402

MODEL_NAMES = ("xgboost", "lightgbm", "extra_trees", "logistic", "mlp")
ENCODERS = ("resnet18", "dinov2", "convnext_tiny")
DIMENSIONS = {"resnet18": 512, "dinov2": 384, "convnext_tiny": 768}
EMBED_PREFIX = "img_"


def sensor_spec(config_path=None, sensor_features=None, imputation=None) -> dict:
    """Read only feature selection and causal imputation from an existing config."""
    config = {}
    if config_path is not None:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Sensor config must be a JSON object.")
    names = config.get("sensor_features", list(RAW_SENSOR_COLUMNS))
    method = config.get("sensor_imputation", "median")
    if sensor_features is not None:
        names = sensor_features
    if imputation is not None:
        method = imputation
    names = list(validate_feature_names(names))
    method = validate_imputation(method)
    return {
        "sensor_features": names,
        "sensor_imputation": method,
        "sensor_feature_version": SENSOR_FEATURE_VERSION,
        "temporal_version": TEMPORAL_VERSION if uses_history(names, method) else None,
    }


def check_versions(spec: dict) -> None:
    names = validate_feature_names(spec["sensor_features"])
    method = validate_imputation(spec["sensor_imputation"])
    if spec.get("sensor_feature_version") != SENSOR_FEATURE_VERSION:
        raise ValueError("Saved sensor feature version differs from the current code.")
    expected = TEMPORAL_VERSION if uses_history(names, method) else None
    if spec.get("temporal_version") != expected:
        raise ValueError("Saved temporal version differs from the current code.")


def embedding_info(meta: dict) -> dict:
    """Portable embedding contract, excluding split-specific IDs and row counts."""
    keys = ("model", "weights", "dimension", "l2_normalize", "preprocessing", "trained")
    result = {key: meta.get(key) for key in keys}
    if result["model"] not in ENCODERS:
        raise ValueError("Unsupported embedding encoder.")
    if result["dimension"] != DIMENSIONS[result["model"]]:
        raise ValueError("Embedding dimension differs from the encoder contract.")
    if not isinstance(result["weights"], str) or not result["weights"].strip():
        raise ValueError("Embedding weights source is missing.")
    if type(result["l2_normalize"]) is not bool:
        raise ValueError("Embedding l2_normalize metadata must be boolean.")
    if not isinstance(result["preprocessing"], dict) or not result["preprocessing"]:
        raise ValueError("Embedding preprocessing metadata is missing.")
    if meta.get("trained") is not False:
        raise ValueError("Only frozen, unfitted image encoders are supported.")
    return result


def check_embedding_compatible(saved: dict, actual: dict) -> None:
    left, right = embedding_info(saved), embedding_info(actual)
    for key in left:
        if left[key] != right[key]:
            raise ValueError(f"Embedding {key} differs from the training contract.")


def _checked_ids(values, description: str) -> pd.Index:
    ids = pd.Index(values)
    if ids.empty or ids.hasnans or not ids.is_unique:
        raise ValueError(f"{description} IDs must be nonempty and unique.")
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError(f"{description} IDs must be nonempty strings.")
    return ids


def load_embeddings(directory, sample_ids) -> tuple[np.ndarray, dict]:
    """Join cached vectors by exact sample_id; never rely on input row position."""
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("status") != "complete":
        raise ValueError("Embedding extraction is not complete.")
    contract = embedding_info(meta)
    frame = pd.read_csv(directory / "sample_ids.csv", dtype=str, keep_default_na=False)
    if "sample_id" not in frame:
        raise ValueError("Embedding sample_ids.csv is missing sample_id.")
    ids = _checked_ids(frame["sample_id"], "Embedding")
    requested = _checked_ids(sample_ids, "Requested")
    values = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
    if values.ndim != 2 or values.shape != (len(ids), contract["dimension"]):
        raise ValueError("Embedding array shape differs from IDs or metadata dimension.")
    if "rows" in meta and meta["rows"] != len(ids):
        raise ValueError("Embedding metadata rows differs from the ID count.")
    if "sample_order_sha256" in meta:
        digest = hashlib.sha256()
        for sid in ids:
            digest.update(sid.encode("utf-8") + b"\0")
        if digest.hexdigest() != meta["sample_order_sha256"]:
            raise ValueError("Embedding ID order differs from the recorded hash.")
    positions = ids.get_indexer(requested)
    if (positions < 0).any():
        raise ValueError("Some requested sample IDs have no embedding.")
    selected = np.asarray(values[positions], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("Selected embeddings contain nonfinite values.")
    return selected, meta


def make_features(
    frame,
    csv_path,
    image_root,
    spec,
    *,
    embeddings_dir=None,
    encoder="resnet18",
    embedding_spec=None,
    device="cuda",
    batch_size=1,
    workers=0,
    work_dir=None,
    _encoder=None,
) -> tuple[pd.DataFrame, dict]:
    """Build multimodal features without fitting; input frame determines all rows.

    csv_path is accepted for the CLI interface, but online extraction deliberately
    writes a frame-only manifest so a subset cannot read extra CSV observations.
    Cached vectors have encoder/ID provenance, not an image-content hash; use live
    extraction to evaluate changed/blank images and for final input-only inference.
    """
    del csv_path
    if encoder not in ENCODERS:
        raise ValueError("A supported image encoder is required for multimodal training.")
    check_versions(spec)
    _checked_ids(frame["sample_id"], "Input")
    if embeddings_dir is None:
        if image_root is None or "image_path" not in frame:
            raise ValueError("Live extraction requires image_root and image_path.")
        encoder_directory = str(REPO_ROOT / "image_features")
        if encoder_directory not in sys.path:
            sys.path.append(encoder_directory)
        from image_features.extract_features import extract

        with tempfile.TemporaryDirectory(prefix="seungbeom_embed_") as temporary:
            manifest = Path(temporary) / "manifest.csv"
            frame[["sample_id", "image_path"]].to_csv(manifest, index=False)
            destination = Path(work_dir) if work_dir is not None else Path(temporary) / "vectors"
            extract(
                model_name=encoder,
                csv_path=manifest,
                image_root=Path(image_root),
                output_dir=destination,
                batch_size=batch_size,
                workers=workers,
                device=device,
                l2_normalize=bool((embedding_spec or {}).get("l2_normalize", False)),
                _encoder=_encoder,
            )
            vectors, meta = load_embeddings(destination, frame["sample_id"])
    else:
        vectors, meta = load_embeddings(embeddings_dir, frame["sample_id"])
    if meta["model"] != encoder:
        raise ValueError("Embedding encoder differs from the requested encoder.")
    if embedding_spec is not None:
        check_embedding_compatible(embedding_spec, meta)
    sensor = build_sensor_features(
        frame, spec["sensor_features"], imputation=spec["sensor_imputation"]
    ).reset_index(drop=True)
    sensor.columns = [f"sensor_{name}" for name in sensor.columns]
    images = pd.DataFrame(
        vectors, columns=[f"{EMBED_PREFIX}{i:04d}" for i in range(vectors.shape[1])]
    )
    return pd.concat([sensor, images], axis=1), meta


class EncodedClassifier(ClassifierMixin, BaseEstimator):
    """Map fold-local labels to consecutive indices and preserve their true classes."""

    def __init__(self, estimator, multiclass_num_class=False):
        self.estimator = estimator
        self.multiclass_num_class = multiclass_num_class

    def fit(self, X, y):
        labels = validate_labels(y)
        self.classes_, encoded = np.unique(labels, return_inverse=True)
        if len(self.classes_) == 1:
            self.estimator_ = DummyClassifier(strategy="constant", constant=0)
        else:
            self.estimator_ = clone(self.estimator)
            if self.multiclass_num_class:
                self.estimator_.set_params(num_class=len(self.classes_))
        self.estimator_.fit(X, encoded)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.classes_[np.asarray(self.estimator_.predict(X), dtype=np.int64)]

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        values = np.asarray(self.estimator_.predict_proba(X), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.classes_):
            raise ValueError("Estimator probabilities do not match its fitted classes.")
        return values


def _classifier(name: str, seed: int):
    if name == "xgboost":
        from xgboost import XGBClassifier

        return EncodedClassifier(
            XGBClassifier(
                objective="multi:softprob",
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.9,
                reg_lambda=2.0,
                tree_method="hist",
                random_state=seed,
                n_jobs=4,
                eval_metric="mlogloss",
            ),
            multiclass_num_class=True,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return EncodedClassifier(
            LGBMClassifier(
                objective="multiclass",
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=30,
                reg_lambda=2.0,
                random_state=seed,
                n_jobs=4,
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
            ),
            multiclass_num_class=True,
        )
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        estimator = ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=seed,
            n_jobs=4,
        )
    elif name == "logistic":
        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    elif name == "mlp":
        from sklearn.neural_network import MLPClassifier

        # No row-level internal validation: outer experiment folds provide validation.
        estimator = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=0.01,
            learning_rate_init=0.001,
            batch_size=128,
            max_iter=250,
            early_stopping=False,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported model: {name}")
    return EncodedClassifier(estimator)


def effective_pca_dim(pca_dim: int, n_rows: int, n_emb: int) -> int:
    if any(type(value) is not int for value in (pca_dim, n_rows, n_emb)):
        raise ValueError("PCA dimensions and row counts must be integers.")
    if pca_dim < 0 or n_rows < 1 or n_emb < 1:
        raise ValueError("PCA requires nonnegative requested size and positive matrix sizes.")
    return min(pca_dim, n_emb, max(0, n_rows - 1))


def build_pipeline(model_name: str, columns, pca_dim: int, seed: int = 42) -> Pipeline:
    columns = list(columns)
    sensors = [name for name in columns if name.startswith("sensor_")]
    images = [name for name in columns if name.startswith(EMBED_PREFIX)]
    if not sensors or not images or len(sensors) + len(images) != len(columns):
        raise ValueError("Pipeline requires only sensor_ and img_ features, with both present.")
    if len(set(columns)) != len(columns):
        raise ValueError("Pipeline feature columns must be unique.")
    if type(pca_dim) is not int or pca_dim < 0:
        raise ValueError("pca_dim must be a nonnegative integer.")
    image_steps = [("scale", StandardScaler())]
    if pca_dim:
        image_steps.append(("pca", PCA(n_components=pca_dim, random_state=seed)))
    transforms = ColumnTransformer(
        [
            ("sensor", SimpleImputer(strategy="median", keep_empty_features=True), sensors),
            ("image", Pipeline(image_steps), images),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    steps = [("features", transforms)]
    if model_name in ("logistic", "mlp"):
        steps.append(("scale", StandardScaler()))
    steps.append(("classifier", _classifier(model_name, seed)))
    return Pipeline(steps)


def _align_probabilities(probabilities, classes, n_rows) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    classes = validate_labels(classes)
    if probabilities.shape != (n_rows, len(classes)) or len(np.unique(classes)) != len(classes):
        raise ValueError("Probability shape differs from classifier classes.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Classifier probabilities must be finite and nonnegative.")
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("Classifier probability rows must sum to one.")
    aligned = np.zeros((n_rows, len(LABELS)), dtype=np.float64)
    aligned[:, classes - 1] = probabilities / sums[:, None]
    return aligned


def predict_proba_six(pipeline, X) -> np.ndarray:
    """Return aligned columns [1,2,3,4,5,6], including zero for absent fold classes."""
    return _align_probabilities(pipeline.predict_proba(X), pipeline.classes_, len(X))


def predict_proba_six_stable(pipeline, X) -> np.ndarray:
    """Fix numeric preprocessing paths while avoiding one forest job per input row.

    PCA and dense linear algebra are transformed one row at a time under one BLAS
    thread. Tree predictions themselves are independent per row, so their final
    classifier is run in one batch with a single thread and a fixed tree reduction
    order. Linear/MLP predictions retain single-row evaluation. Fitted statistics
    and permanent estimator settings are unchanged. Call this outside concurrent
    threads sharing the same estimator because n_jobs is temporarily overridden.
    """
    from threadpoolctl import threadpool_limits

    if len(X) == 0:
        return np.empty((0, len(LABELS)), dtype=np.float64)
    classifier = pipeline.steps[-1][1]
    check_is_fitted(classifier, "estimator_")
    estimator = classifier.estimator_
    preprocessing = pipeline[:-1]
    tree_types = {"XGBClassifier", "LGBMClassifier", "ExtraTreesClassifier", "DummyClassifier"}
    with threadpool_limits(limits=1):
        transformed = np.concatenate(
            [preprocessing.transform(X.iloc[index : index + 1]) for index in range(len(X))],
            axis=0,
        )
        if type(estimator).__name__ in tree_types:
            parameters = estimator.get_params(deep=False)
            old_jobs = parameters.get("n_jobs")
            try:
                if "n_jobs" in parameters:
                    estimator.set_params(n_jobs=1)
                probabilities = classifier.predict_proba(transformed)
            finally:
                if "n_jobs" in parameters:
                    estimator.set_params(n_jobs=old_jobs)
        else:
            probabilities = np.concatenate(
                [
                    classifier.predict_proba(transformed[index : index + 1])
                    for index in range(len(X))
                ],
                axis=0,
            )
    return _align_probabilities(probabilities, classifier.classes_, len(X))
