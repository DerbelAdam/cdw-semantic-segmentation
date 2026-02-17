from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.cdw_dataset import CDWSegmentationDataset
from src.data.transforms import SegmentationTransformConfig, SegmentationBasicTransform
from src.models.fpn_segmenter import FPNSegmenter
from src.training.metrics import update_confusion_matrix, compute_iou_from_confmat


@torch.no_grad()
def main():
    DATASET_ROOT = Path(r"C:\Users\lenovo\Desktop\P2M\Data\Ground_Truths_VOC_Format")
    SPLITS_DIR = Path(r"C:\Users\lenovo\Desktop\project\cdw-semantic-segmentation\splits")
    val_split = SPLITS_DIR / "val.txt"   # change to test.txt if needed
    ckpt_path = Path(r"runs\step5\model_step5.pt")

    device = torch.device("cpu")
    num_classes = 11
    ignore_index = 0
    image_size = (512, 512)

    tfm = SegmentationBasicTransform(SegmentationTransformConfig(size=image_size))
    ds = CDWSegmentationDataset(dataset_root=DATASET_ROOT, split_file=val_split, transform=tfm)
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    # Load model
    model = FPNSegmenter(num_classes=num_classes, backbone_name="resnet50", pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for images, masks, ids in dl:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)  # [B,H,W]

        confmat = update_confusion_matrix(
            confmat=confmat,
            preds=preds,
            targets=masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    metrics = compute_iou_from_confmat(confmat, ignore_index=ignore_index)
    per_class_iou = metrics["per_class_iou"]

    print("Checkpoint:", ckpt_path)
    print("Split     :", val_split)
    print(f"Pixel Acc (excl 0): {metrics['pixel_acc'].item():.4f}")
    print(f"mIoU      (1..10):  {metrics['miou'].item():.4f}")

    print("\nIoU per class:")
    for c in range(num_classes):
        if c == ignore_index:
            continue
        print(f"  class {c}: {per_class_iou[c].item():.4f}")


if __name__ == "__main__":
    main()