from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FPNSegmenter(nn.Module):
    """
    Semantic segmentation model:
      image -> ResNet50 backbone -> FPN features -> fuse -> segmentation logits

    Notes:
    - We fuse FPN levels by upsampling them to the highest resolution (usually P2)
      and summing them, then predict per-pixel class logits.
    """
    def __init__(self, num_classes: int = 11, backbone_name: str = "resnet50", pretrained: bool = True):
        super().__init__()

        # ResNet50 + FPN. Returned features are typically keys: "0","1","2","3" (P2..P5)
        self.backbone = resnet_fpn_backbone(backbone_name=backbone_name, weights="DEFAULT" if pretrained else None)

        # For ResNet50-FPN, out_channels is 256 for each pyramid level
        fpn_out = 256

        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_out, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]  # (H, W)

        feats: Dict[str, torch.Tensor] = self.backbone(x)
        # feats keys: "0","1","2","3" with decreasing resolutions

        # pick the highest-res level as reference (usually "0" = P2)
        p2 = feats["0"]
        target_hw = p2.shape[-2:]

        # upsample all pyramid levels to p2 size and sum (simple fusion)
        fused = torch.zeros_like(p2)
        for k, f in feats.items():
            if f.shape[-2:] != target_hw:
                f = F.interpolate(f, size=target_hw, mode="nearest")
            fused = fused + f

        fused = self.fuse(fused)
        logits = self.classifier(fused)  # [B, num_classes, H_p2, W_p2]

        # upscale to input size so it matches mask size
        if logits.shape[-2:] != input_hw:
            logits = F.interpolate(logits, size=input_hw, mode="bilinear", align_corners=False)

        return logits