from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.cdw_dataset import CDWSegmentationDataset
from src.data.transforms import SegmentationTransformConfig, SegmentationBasicTransform


def main():
    
    DATASET_ROOT = Path(r"C:\Users\lenovo\Desktop\P2M\Data\Ground_Truths_VOC_Format")
    SPLITS_DIR = Path(r"C:\Users\lenovo\Desktop\project\cdw-semantic-segmentation\splits")

    train_split = SPLITS_DIR / "train.txt"

    tfm = SegmentationBasicTransform(SegmentationTransformConfig(size=(512, 512)))
    ds = CDWSegmentationDataset(dataset_root=DATASET_ROOT, split_file=train_split, transform=tfm)

    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    images, masks, ids = next(iter(dl))

    print("images:", images.shape, images.dtype)  # [B,3,H,W] float32
    print("masks :", masks.shape, masks.dtype)    # [B,H,W] int64
    print("ids   :", list(ids))

    u = torch.unique(masks)
    print("unique(mask) count:", u.numel())
    print("unique(mask) min/max:", int(u.min()), int(u.max()))
    print("unique(mask) values:", u.tolist())

    # Sanity checks pour ta config (0..10 + ignore_index=0)
    assert masks.dtype == torch.int64, "Mask must be int64/long for CrossEntropyLoss."
    assert int(u.min()) >= 0, "Found negative labels in mask."
    assert int(u.max()) <= 10, "Found label > 10 in mask (expected 0..10)."

    print("OK: Step 2 dataloader pipeline is consistent.")


if __name__ == "__main__":
    main()