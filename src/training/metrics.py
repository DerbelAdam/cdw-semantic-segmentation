from __future__ import annotations

import torch


@torch.no_grad()
def update_confusion_matrix(
    confmat: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 0,
) -> torch.Tensor:
    """
    confmat: [C, C] where rows=true, cols=pred
    preds, targets: [B, H, W] (int64)
    """
    preds = preds.view(-1).to(torch.int64)
    targets = targets.view(-1).to(torch.int64)

    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    # filter out-of-range just in case
    in_range = (targets >= 0) & (targets < num_classes) & (preds >= 0) & (preds < num_classes)
    preds = preds[in_range]
    targets = targets[in_range]

    idx = targets * num_classes + preds
    binc = torch.bincount(idx, minlength=num_classes * num_classes)
    confmat += binc.reshape(num_classes, num_classes)
    return confmat


def compute_iou_from_confmat(confmat: torch.Tensor, ignore_index: int = 0) -> dict:
    """
    Returns per-class IoU, mean IoU, and pixel accuracy excluding ignore_index.
    """
    confmat = confmat.to(torch.float64)

    tp = torch.diag(confmat)
    fp = confmat.sum(dim=0) - tp
    fn = confmat.sum(dim=1) - tp

    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(denom))

    # pixel accuracy
    total = confmat.sum()
    correct = tp.sum()
    pixel_acc = correct / total if total > 0 else torch.tensor(0.0, dtype=torch.float64)

    # exclude ignore_index from mean IoU
    valid_classes = [c for c in range(confmat.shape[0]) if c != ignore_index]
    miou = iou[valid_classes].mean() if len(valid_classes) > 0 else torch.tensor(0.0, dtype=torch.float64)

    return {
        "per_class_iou": iou,
        "miou": miou,
        "pixel_acc": pixel_acc,
    }