import os
import random
from pathlib import Path

import numpy as np
from PIL import Image



IMAGES_DIR = Path("C:/Users/lenovo/Desktop/P2M/Data/Ground_Truths_VOC_Format/JPEGImages")            
MASKS_DIR  = Path("C:/Users/lenovo/Desktop/P2M/Data/Ground_Truths_VOC_Format/SegmentationClassPNG")  
OUTPUT_DIR = Path("C:/Users/lenovo/Desktop/project/cdw-semantic-segmentation/splits")      


NUM_CLASSES = 10          # classes 1..10
IGNORE_LABEL = 0          # pixels non annotés
VAL_RATIO = 0.2    # 20% des données pour validation, 80% pour entraînement
SEED = 42
CHECK_FIRST_N = 30        # nombre de masques inspectés pour vérifier min/max/valeurs


def list_images(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sorted([p for p in images_dir.iterdir() if p.suffix in exts])


def corresponding_mask_path(img_path: Path, masks_dir: Path):
    # même "stem" (nom sans extension), masque en .png
    return masks_dir / f"{img_path.stem}.png"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    images = list_images(IMAGES_DIR)
    if not images:
        raise RuntimeError(f"Aucune image trouvée dans {IMAGES_DIR}")

    # 1) Vérifier présence des masques + dimensions
    pairs = []
    missing_masks = []
    size_mismatch = []

    for img_path in images:
        mask_path = corresponding_mask_path(img_path, MASKS_DIR)
        if not mask_path.exists():
            missing_masks.append((img_path.name, str(mask_path)))
            continue

        # vérifier dimensions
        with Image.open(img_path) as im:
            w_img, h_img = im.size
        with Image.open(mask_path) as mm:
            w_m, h_m = mm.size

        if (w_img, h_img) != (w_m, h_m):
            size_mismatch.append((img_path.name, (w_img, h_img), (w_m, h_m)))
            continue

        pairs.append((img_path, mask_path))

    print(f"Images trouvées: {len(images)}")
    print(f"Paires valides (image+mask, même taille): {len(pairs)}")
    print(f"Masques manquants: {len(missing_masks)}")
    print(f"Tailles non correspondantes: {len(size_mismatch)}")

    if missing_masks[:5]:
        print("\nExemples masques manquants (max 5):")
        for x in missing_masks[:5]:
            print(" -", x)

    if size_mismatch[:5]:
        print("\nExemples mismatch taille (max 5):")
        for x in size_mismatch[:5]:
            print(" -", x)

    if len(pairs) == 0:
        raise RuntimeError("Aucune paire valide. Corrige chemins/nommage des fichiers.")

    # 2) Vérifier les valeurs des labels sur un échantillon
    sample = pairs[: min(CHECK_FIRST_N, len(pairs))]
    all_vals = set()
    for _, mask_path in sample:
        m = np.array(Image.open(mask_path))
        if m.ndim != 2:
            raise RuntimeError(
                f"Le masque {mask_path} n'est pas 2D (shape={m.shape}). "
                "Pour une segmentation sémantique, on attend un label par pixel (H,W)."
            )
        vals = np.unique(m)
        all_vals.update(vals.tolist())

    print("\nSanity check labels (sur un échantillon):")
    print(" - valeurs uniques observées (triées, max 50):", sorted(all_vals)[:50])
    print(" - contient IGNORE_LABEL=0 ?", (IGNORE_LABEL in all_vals))
    print(" - max valeur observée:", max(all_vals))

    # 3) Stats: distribution de pixels par classe (ignore exclu)
    # Ici on calcule le nombre de pixels pour chaque label 1..10 sur tout le dataset.
    pixel_counts = np.zeros(NUM_CLASSES + 1, dtype=np.int64)  # index 0..10
    ignore_pixels = 0
    total_pixels = 0

    for _, mask_path in pairs:
        m = np.array(Image.open(mask_path))
        total_pixels += m.size
        ignore_pixels += int(np.sum(m == IGNORE_LABEL))
        # compter labels 1..10
        for c in range(1, NUM_CLASSES + 1):
            pixel_counts[c] += int(np.sum(m == c))

    labeled_pixels = total_pixels - ignore_pixels

    print("\nPixel stats:")
    print(f" - total pixels: {total_pixels}")
    print(f" - ignore pixels (label=0): {ignore_pixels} ({ignore_pixels/total_pixels:.3f})")
    print(f" - labeled pixels: {labeled_pixels} ({labeled_pixels/total_pixels:.3f})")

    print("\nPixels par classe (1..10):")
    for c in range(1, NUM_CLASSES + 1):
        ratio = (pixel_counts[c] / labeled_pixels) if labeled_pixels > 0 else 0.0
        print(f" - class {c:2d}: {pixel_counts[c]:12d}  ratio={ratio:.4f}")

    # Sauver stats
    stats_path = OUTPUT_DIR / "pixel_stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"total_pixels={total_pixels}\n")
        f.write(f"ignore_pixels={ignore_pixels}\n")
        f.write(f"labeled_pixels={labeled_pixels}\n")
        for c in range(1, NUM_CLASSES + 1):
            f.write(f"class_{c}_pixels={pixel_counts[c]}\n")
    print(f"\nStats sauvegardées dans: {stats_path}")

    # 4) Split train/val
    random.shuffle(pairs)
    n_val = int(len(pairs) * VAL_RATIO)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_list = OUTPUT_DIR / "train.txt"
    val_list = OUTPUT_DIR / "val.txt"

    # On sauvegarde juste les "stems" (nom sans extension), c'est pratique.
    with open(train_list, "w", encoding="utf-8") as f:
        for img_path, _ in train_pairs:
            f.write(img_path.stem + "\n")

    with open(val_list, "w", encoding="utf-8") as f:
        for img_path, _ in val_pairs:
            f.write(img_path.stem + "\n")

    print("\nSplit:")
    print(f" - train: {len(train_pairs)} échantillons -> {train_list}")
    print(f" - val  : {len(val_pairs)} échantillons -> {val_list}")

    # 5) Indication utile pour la suite
    print("\nProchaine étape (Step 2): créer un PyTorch Dataset/Dataloader")
    print(" - qui lit image + masque à partir de ces stems")
    print(" - applique transforms (resize/augmentation)")
    print(" - renvoie (image_tensor, mask_tensor) avec ignore=0")


if __name__ == "__main__":
    main()