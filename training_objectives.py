"""Train-only class imbalance objectives and sampling, independent of inference."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler

from schema import LABELS, validate_labels


def validate_training_options(config: dict) -> None:
    """Validate optional training settings without changing the supplied config."""
    if config.get("loss", "cross_entropy") not in (
        "cross_entropy",
        "weighted_cross_entropy",
        "focal",
    ):
        raise ValueError("loss must be cross_entropy, weighted_cross_entropy, or focal.")
    if config.get("sampling", "shuffle") not in ("shuffle", "balanced"):
        raise ValueError("sampling must be shuffle or balanced.")
    power = config.get("class_weight_power", 0.5)
    gamma = config.get("focal_gamma", 2.0)
    for name, value in (("class_weight_power", power), ("focal_gamma", gamma)):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
    if not 0 <= power <= 1:
        raise ValueError("class_weight_power must be in [0, 1].")
    if gamma < 0:
        raise ValueError("focal_gamma must be nonnegative.")


class FocalLoss(nn.Module):
    """Unweighted multiclass focal loss with a mean over examples."""

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        validate_training_options({"focal_gamma": gamma})
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probabilities = F.log_softmax(logits, dim=1)
        if self.gamma == 0:
            return F.nll_loss(log_probabilities, targets)
        log_pt = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        # expm1 avoids cancellation near p_t=1. The floor also keeps fractional-gamma
        # gradients finite when log_softmax rounds a confident correct example to zero.
        one_minus_pt = (-torch.expm1(log_pt)).clamp_min(torch.finfo(log_pt.dtype).tiny)
        return (-one_minus_pt.pow(self.gamma) * log_pt).mean()


def build_training_components(
    train_labels, config: dict, device: torch.device
) -> tuple[nn.Module, WeightedRandomSampler | None, dict]:
    """Build loss, optional sampler, and aggregate metadata from Train labels only.

    Labels use the external 1..6 convention; loss targets retain the model's 0..5
    convention. Validation and Test inputs are deliberately absent from this API.
    """
    validate_training_options(config)
    labels = validate_labels(train_labels)
    if labels.size == 0:
        raise ValueError("Train labels must contain at least one row.")
    counts = np.bincount(labels - 1, minlength=len(LABELS))
    loss = config.get("loss", "cross_entropy")
    sampling = config.get("sampling", "shuffle")
    power = float(config.get("class_weight_power", 0.5))
    gamma = float(config.get("focal_gamma", 2.0))
    class_weights = None
    if loss == "weighted_cross_entropy":
        present = counts > 0
        weights = np.zeros(len(LABELS), dtype=np.float64)
        weights[present] = counts[present].astype(np.float64) ** (-power)
        weights[present] /= weights[present].mean()
        criterion = nn.CrossEntropyLoss(
            weight=torch.as_tensor(weights, dtype=torch.float32, device=device)
        )
        class_weights = weights.tolist()
    elif loss == "focal":
        criterion = FocalLoss(gamma)
    else:
        # Preserve the previous default loss and avoid consuming any random numbers.
        criterion = nn.CrossEntropyLoss()

    sampler = None
    sampler_metadata = {
        "method": "shuffle_without_replacement",
        "replacement": False,
        "num_samples": int(labels.size),
        "description": "Each Train row is visited once per epoch in shuffled order.",
    }
    if sampling == "balanced":
        sample_weights = 1.0 / counts[labels - 1]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=int(labels.size),
            replacement=True,
            generator=torch.Generator().manual_seed(config["seed"]),
        )
        sampler_metadata = {
            "method": "inverse_train_class_frequency",
            "replacement": True,
            "num_samples": int(labels.size),
            "seed": config["seed"],
            "description": "Present Train classes have equal expected sampling mass.",
        }
    metadata = {
        "loss": loss,
        "sampling": sampling,
        "class_weight_power": power,
        "focal_gamma": gamma,
        "train_class_counts": {str(label): int(counts[label - 1]) for label in LABELS},
        "class_weights": class_weights,
        "sampler": sampler_metadata,
    }
    return criterion, sampler, metadata
