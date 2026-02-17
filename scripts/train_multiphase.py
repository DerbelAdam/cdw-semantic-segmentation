from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
import argparse

from src.data.cdw_dataset import CDWSegmentationDataset
from src.data.transforms import SegmentationTransformConfig, SegmentationBasicTransform
from src.models.build_model import build_model
from src.training.losses import make_weighted_ce_loss
from src.training.pixel_stats import load_pixel_stats_txt
from src.training.class_weights import median_frequency_balancing_from_counts
from src.training.freeze import freeze_backbone_all, unfreeze_backbone_c4_c5, unfreeze_backbone_all
from src.training.optim import make_optimizer_two_lrs


def train_one_epoch(model, dl, criterion, optimizer, device, grad_accum_steps=1, max_batches=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running = 0.0

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

        if step % 20 == 0:
            print(f"  step={step:04d} loss={float(loss.item())*grad_accum_steps:.4f} avg={running/step:.4f}")

        if max_batches is not None and step >= max_batches:
            break

    return running / step


@torch.no_grad()
def evaluate_pixel_acc_miou(model, dl, device, num_classes=11, ignore_index=0):
    from src.training.metrics import update_confusion_matrix, compute_iou_from_confmat

    model.eval()
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for images, masks, ids in dl:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        confmat = update_confusion_matrix(confmat, preds, masks, num_classes, ignore_index)

    m = compute_iou_from_confmat(confmat, ignore_index)
    return float(m["pixel_acc"].item()), float(m["miou"].item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="student", choices=["student", "prof"])
    args = parser.parse_args()

    DATASET_ROOT = Path(r"C:\Users\lenovo\Desktop\P2M\Data\Ground_Truths_VOC_Format")
    SPLITS_DIR = Path(r"C:\Users\lenovo\Desktop\project\cdw-semantic-segmentation\splits")
    train_split = SPLITS_DIR / "train.txt"
    val_split = SPLITS_DIR / "val.txt"
    pixel_stats_path = SPLITS_DIR / "pixel_stats.txt"

    device = torch.device("cpu")
    num_classes = 11
    ignore_index = 0
    image_size = (512, 512)
    batch_size = 2
    num_workers = 0
    grad_accum_steps = 1

    # epochs per phase (adjust)
    epochs_p1 = 3
    epochs_p2 = 10
    epochs_p3 = 10

    # LRs (head always higher)
    lr_head_p1 = 1e-4
    lr_backbone_p1 = 0.0  # frozen anyway

    lr_head_p2 = 1e-4
    lr_backbone_p2 = 1e-5

    lr_head_p3 = 5e-5
    lr_backbone_p3 = 5e-6

    weight_decay = 1e-4

    print("Architecture:", args.arch)

    # Data
    tfm = SegmentationBasicTransform(SegmentationTransformConfig(size=image_size))
    train_ds = CDWSegmentationDataset(dataset_root=DATASET_ROOT, split_file=train_split, transform=tfm)
    val_ds = CDWSegmentationDataset(dataset_root=DATASET_ROOT, split_file=val_split, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model (student or prof)
    model = build_model(arch=args.arch, num_classes=num_classes).to(device)

    # Loss weights
    stats = load_pixel_stats_txt(pixel_stats_path, num_classes=num_classes, ignore_index=ignore_index)
    weights = median_frequency_balancing_from_counts(stats.class_pixels, ignore_index=ignore_index, clip_max=10.0)
    criterion = make_weighted_ce_loss(num_classes, weights.tolist(), ignore_index).to(device)

    # Separate output dirs per architecture (avoid mixing incompatible checkpoints)
    out_dir = Path("runs") / f"multiphase_{args.arch}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_ckpt(tag: str):
        path = out_dir / f"model_{tag}.pt"
        torch.save({"model_state": model.state_dict(), "tag": tag, "arch": args.arch}, path)
        print("Saved:", path)

    # -------- Phase 1: backbone frozen --------
    print("\nPHASE 1: Freeze backbone (stabilization)")
    freeze_backbone_all(model)
    optimizer = make_optimizer_two_lrs(model, lr_backbone=lr_backbone_p1, lr_head=lr_head_p1, weight_decay=weight_decay)

    for ep in range(1, epochs_p1 + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_dl, criterion, optimizer, device, grad_accum_steps)
        acc, miou = evaluate_pixel_acc_miou(model, val_dl, device, num_classes, ignore_index)
        print(f"[P1][ep {ep}/{epochs_p1}] train_loss={tr_loss:.4f} val_acc={acc:.4f} val_miou={miou:.4f} time={time.time()-t0:.1f}s")
    save_ckpt("phase1_last")

    # -------- Phase 2: unfreeze C4-C5 --------
    print("\nPHASE 2: Unfreeze C4-C5 (layer3-layer4)")
    unfreeze_backbone_c4_c5(model)
    optimizer = make_optimizer_two_lrs(model, lr_backbone=lr_backbone_p2, lr_head=lr_head_p2, weight_decay=weight_decay)

    best_miou_p2 = -1.0
    for ep in range(1, epochs_p2 + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_dl, criterion, optimizer, device, grad_accum_steps)
        acc, miou = evaluate_pixel_acc_miou(model, val_dl, device, num_classes, ignore_index)
        print(f"[P2][ep {ep}/{epochs_p2}] train_loss={tr_loss:.4f} val_acc={acc:.4f} val_miou={miou:.4f} time={time.time()-t0:.1f}s")
        if miou > best_miou_p2 + 1e-4:
            best_miou_p2 = miou
            save_ckpt(f"phase2_best_ep{ep}")
    save_ckpt("phase2_last")

    # -------- Phase 3: unfreeze all --------
    print("\nPHASE 3: Unfreeze all backbone (C1-C5)")
    unfreeze_backbone_all(model)
    optimizer = make_optimizer_two_lrs(model, lr_backbone=lr_backbone_p3, lr_head=lr_head_p3, weight_decay=weight_decay)

    best_miou = -1.0
    patience = 4
    bad = 0

    for ep in range(1, epochs_p3 + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_dl, criterion, optimizer, device, grad_accum_steps)
        acc, miou = evaluate_pixel_acc_miou(model, val_dl, device, num_classes, ignore_index)
        print(f"[P3][ep {ep}/{epochs_p3}] train_loss={tr_loss:.4f} val_acc={acc:.4f} val_miou={miou:.4f} time={time.time()-t0:.1f}s")

        if miou > best_miou + 1e-4:
            best_miou = miou
            bad = 0
            save_ckpt(f"phase3_best_ep{ep}")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping: no mIoU improvement for {patience} epochs.")
                break

    save_ckpt("phase3_last")
    print("\nDone multiphase training.")


if __name__ == "__main__":
    main()