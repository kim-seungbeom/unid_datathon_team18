"""Six-class metrics; rows of the confusion matrix are true classes 1..6."""

import numpy as np

from schema import LABELS, validate_labels


def classification_metrics(y_true, y_pred) -> dict:
    true = validate_labels(y_true)
    pred = validate_labels(y_pred)
    if true.shape != pred.shape or not len(true):
        raise ValueError("Evaluation requires nonempty, equally sized label arrays.")
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(matrix, (true - 1, pred - 1), 1)
    tp = np.diag(matrix)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2.0 * tp, denominator, out=np.zeros(6), where=denominator != 0)
    return {
        "macro_f1": float(f1.mean()),
        "class_f1": {str(label): float(f1[label - 1]) for label in LABELS},
        "support": {str(label): int(matrix[label - 1].sum()) for label in LABELS},
        "confusion_matrix": matrix.tolist(),
    }
