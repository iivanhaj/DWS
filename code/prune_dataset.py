import os
import random
from pathlib import Path

# Paths
base_dir = "../dataset/watermarked"

# Config: how many to keep
train_limit = 2000
val_limit = 500

methods = ["DCT", "DWT", "SVD", "DWT_DCT", "DWT_SVD"]

for method in methods:
    for split, limit in [("train", train_limit), ("val", val_limit)]:
        folder = Path(base_dir) / method / split
        if not folder.exists():
            print(f"Skipping missing folder: {folder}")
            continue

        images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(images)

        # Keep only 'limit' images
        keep = set(images[:limit])
        delete = [f for f in images if f not in keep]

        for img in delete:
            os.remove(folder / img)

        print(f"{method} - {split}: kept {len(keep)}, deleted {len(delete)}")
