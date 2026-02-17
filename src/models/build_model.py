from __future__ import annotations

from src.models.fpn_segmenter import FPNSegmenter
from src.models.prof_fpn_segmenter import ResNet50_FPN_Segmenter


def build_model(arch: str, num_classes: int = 11):
    arch = arch.lower().strip()
    if arch in {"student", "ours", "fpn"}:
        return FPNSegmenter(num_classes=num_classes, backbone_name="resnet50", pretrained=True)
    if arch in {"prof", "teacher"}:
        return ResNet50_FPN_Segmenter(num_classes=num_classes, pretrained=True)
    raise ValueError(f"Unknown arch={arch!r}. Use 'student' or 'prof'.")