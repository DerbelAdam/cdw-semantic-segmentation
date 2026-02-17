from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.cdw_dataset import CDWSegmentationDataset
from src.data.transforms import SegmentationTransformConfig, SegmentationBasicTransform
from src.models.fpn_segmenter import FPNSegmenter
from src.training.losses import make_weighted_ce_loss
from src.training.pixel_stats import load_pixel_stats_txt
from src.training.class_weights import median_frequency_balancing_from_counts


def main():
    DATASET_ROOT = Path(r"C:\Users\lenovo\Desktop\P2M\Data\Ground_Truths_VOC_Format")
    SPLITS_DIR = Path(r"C:\Users\lenovo\Desktop\project\cdw-semantic-segmentation\splits")
    train_split = SPLITS_DIR / "train.txt"
    pixel_stats_path = SPLITS_DIR / "pixel_stats.txt"

    num_classes = 11
    ignore_index = 0

    stats = load_pixel_stats_txt(pixel_stats_path, num_classes=num_classes, ignore_index=ignore_index)
    weights = median_frequency_balancing_from_counts(stats.class_pixels, ignore_index=ignore_index, clip_max=10.0)

    print("Loaded pixel stats from:", str(pixel_stats_path))
    print("class_pixels (0..10):", stats.class_pixels.tolist())
    print("class_weights(0..10):", [round(float(x), 4) for x in weights.tolist()])

    tfm = SegmentationBasicTransform(SegmentationTransformConfig(size=(512, 512)))
    ds = CDWSegmentationDataset(dataset_root=DATASET_ROOT, split_file=train_split, transform=tfm)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    images, masks, ids = next(iter(dl))
    model = FPNSegmenter(num_classes=num_classes, backbone_name="resnet50", pretrained=True)
    model.train()
    logits = model(images)

    criterion = make_weighted_ce_loss(
        num_classes=num_classes,
        class_weights=weights.tolist(),
        ignore_index=ignore_index,
    )
    loss = criterion(logits, masks)

    print("logits:", logits.shape, logits.dtype)
    print("masks :", masks.shape, masks.dtype)
    print("loss  :", float(loss.item()))
    assert torch.isfinite(loss).item(), "Loss is NaN or Inf"
    print("OK: Step 4 (weighted loss from pixel_stats.txt) works.")


if __name__ == "__main__":
    main()