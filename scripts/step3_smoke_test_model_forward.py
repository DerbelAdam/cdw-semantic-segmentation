from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.cdw_dataset import CDWSegmentationDataset
from src.data.transforms import SegmentationTransformConfig, SegmentationBasicTransform
from src.models.fpn_segmenter import FPNSegmenter


def main():
    DATASET_ROOT = Path(r"C:\Users\lenovo\Desktop\P2M\Data\Ground_Truths_VOC_Format")
    SPLITS_DIR = Path(r"C:\Users\lenovo\Desktop\project\cdw-semantic-segmentation\splits")
    train_split = SPLITS_DIR / "train.txt"

    tfm = SegmentationBasicTransform(SegmentationTransformConfig(size=(512, 512)))
    ds = CDWSegmentationDataset(dataset_root=DATASET_ROOT, split_file=train_split, transform=tfm)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    images, masks, ids = next(iter(dl))
    print("images:", images.shape, images.dtype)
    print("masks :", masks.shape, masks.dtype)

    model = FPNSegmenter(num_classes=11, backbone_name="resnet50", pretrained=True)
    model.eval()

    with torch.no_grad():
        logits = model(images)

    print("logits:", logits.shape, logits.dtype)

    # Expected: [B, 11, 512, 512]
    assert logits.shape[0] == images.shape[0]
    assert logits.shape[1] == 11
    assert logits.shape[-2:] == images.shape[-2:]
    print("OK: Step 3 forward pass works.")


if __name__ == "__main__":
    main()