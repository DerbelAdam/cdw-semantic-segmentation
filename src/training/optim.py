from __future__ import annotations
import torch
import torch.nn as nn


def make_optimizer_two_lrs(
    model: nn.Module,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_student_backbone = name.startswith("backbone.body.")
        is_prof_backbone = name.startswith("backbone.") and (not name.startswith("backbone.fpn")) and (not name.startswith("backbone.head"))
        # For prof model, backbone params are "backbone.conv1", "backbone.layer1", etc.
        # For prof model, fpn/head are separate attributes, so they won't start with "backbone." anyway.
        if is_student_backbone or is_prof_backbone:
            backbone_params.append(p)
        else:
            head_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )