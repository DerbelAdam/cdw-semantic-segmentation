from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader

from src.data.cdw_dataset import CDWSegmentationDataset
from src.data.transforms import SegmentationTransformConfig, SegmentationBasicTransform
from src.models.fpn_segmenter import FPNSegmenter
from src.training.losses import make_weighted_ce_loss
from src.training.pixel_stats import load_pixel_stats_txt
from src.training.class_weights import median_frequency_balancing_from_counts


def main():
    # Paths
    DATASET_ROOT = Path(r"C:\Users\lenovo\Desktop\P2M\Data\Ground_Truths_VOC_Format")
    SPLITS_DIR = Path(r"C:\Users\lenovo\Desktop\project\cdw-semantic-segmentation\splits")
    train_split = SPLITS_DIR / "train.txt"
    pixel_stats_path = SPLITS_DIR / "pixel_stats.txt"
    out_dir = Path("runs") / "step5"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Settings (CPU)
    device = torch.device("cpu")
    num_classes = 11
    ignore_index = 0
    image_size = (512, 512)
    batch_size = 2  # if slow -> 1
    num_workers = 0
    lr = 1e-4
    weight_decay = 1e-4
    grad_accum_steps = 1  # set 2 or 4 if batch_size=1 and you want larger effective batch
    max_batches = 50  # for quick validation; set None for full epoch

    # Data
    tfm = SegmentationBasicTransform(SegmentationTransformConfig(size=image_size))
    ds = CDWSegmentationDataset(dataset_root=DATASET_ROOT, split_file=train_split, transform=tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Model
    model = FPNSegmenter(num_classes=num_classes, backbone_name="resnet50", pretrained=True).to(device)
    model.train()

    # Loss weights from pixel_stats
    stats = load_pixel_stats_txt(pixel_stats_path, num_classes=num_classes, ignore_index=ignore_index)
    weights = median_frequency_balancing_from_counts(stats.class_pixels, ignore_index=ignore_index, clip_max=10.0)

    criterion = make_weighted_ce_loss(
        num_classes=num_classes,
        class_weights=weights.tolist(),
        ignore_index=ignore_index,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train (one epoch)
    t0 = time.time()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, (images, masks, ids) in enumerate(dl, start=1):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks) / grad_accum_steps
        loss.backward()

        if step % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running += float(loss.item()) * grad_accum_steps

        if step % 10 == 0:
            avg = running / step
            print(f"step={step:04d}  loss={float(loss.item())*grad_accum_steps:.4f}  avg_loss={avg:.4f}")

        if max_batches is not None and step >= max_batches:
            break

    dt = time.time() - t0
    avg_loss = running / step
    print(f"Done. steps={step} avg_loss={avg_loss:.4f} time_sec={dt:.1f}")

    # Save checkpoint
    ckpt_path = out_dir / "model_step5.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "num_classes": num_classes,
            "ignore_index": ignore_index,
            "image_size": image_size,
            "class_weights": weights.tolist(),
        },
        ckpt_path,
    )
    print("Saved:", ckpt_path)


if __name__ == "__main__":
    main()