import os
import cv2
import random
from glob import glob
from tqdm import tqdm

# Paths
RAW_DIR = "../raw_images"
DATASET_DIR = "dataset/original"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

# Create folders if they don't exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Parameters
PATCH_SIZE = 256
VAL_SPLIT = 0.2  # 20% for validation

# Get all image paths
image_paths = glob(os.path.join(RAW_DIR, "*.*"))
random.shuffle(image_paths)

# Split into train/val
split_idx = int(len(image_paths) * (1 - VAL_SPLIT))
train_paths = image_paths[:split_idx]
val_paths = image_paths[split_idx:]

def save_patches(image_paths, save_dir):
    for img_path in tqdm(image_paths, desc=f"Processing {save_dir}"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w, _ = img.shape
        
        # Sliding window to create patches
        for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
            for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
                patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                filename = f"{os.path.splitext(os.path.basename(img_path))[0]}_{y}_{x}.png"
                cv2.imwrite(os.path.join(save_dir, filename), patch)

# Save patches for train and val
save_patches(train_paths, TRAIN_DIR)
save_patches(val_paths, VAL_DIR)

print("Preprocessing & splitting done!")
