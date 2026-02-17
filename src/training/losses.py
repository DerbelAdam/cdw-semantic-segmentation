from __future__ import annotations

from typing import Iterable, Optional
import torch
import torch.nn as nn


def make_weighted_ce_loss(
    num_classes: int,
    class_weights: Optional[Iterable[float]] = None,
    ignore_index: int = 0,
) -> nn.Module:
    """
    Weighted Cross-Entropy for semantic segmentation.

    Args:
        num_classes: number of classes (11 for labels 0..10)
        class_weights: iterable of length num_classes (w0..w10). If None => unweighted.
        ignore_index: label to ignore in target mask (0 in your dataset).

    Returns:
        nn.CrossEntropyLoss configured for segmentation.
    """
    weight_tensor = None
    if class_weights is not None:
        w = list(class_weights)
        if len(w) != num_classes:
            raise ValueError(f"class_weights must have length {num_classes}, got {len(w)}")
        weight_tensor = torch.tensor(w, dtype=torch.float32)

    return nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=ignore_index)