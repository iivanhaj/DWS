import cv2
import numpy as np
import os
from glob import glob
from pathlib import Path

train_dir = "../../dataset/original/train"
val_dir = "../../dataset/original/val"
watermark_path = "../../watermark.png"
output_base = "../../dataset/watermarked/SVD"

ALPHA = 0.5  # Adjust to control watermark visibility (lower = more invisible)

for split in ["train", "val"]:
    Path(f"{output_base}/{split}").mkdir(parents=True, exist_ok=True)

watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
if watermark is None:
    raise FileNotFoundError(f"Watermark not found at {watermark_path}")

def apply_svd_watermark(cover_img, watermark_img, alpha=0.05):
    # Convert cover image to YCrCb
    ycrcb = cv2.cvtColor(cover_img, cv2.COLOR_BGR2YCrCb)
    y_channel = np.float32(ycrcb[:, :, 0])

    # Resize watermark to match Y channel
    wm_resized = cv2.resize(watermark_img, (y_channel.shape[1], y_channel.shape[0]))
    wm_resized = np.float32(wm_resized)

    # Apply SVD to Y channel
    U, S, Vt = np.linalg.svd(y_channel, full_matrices=False)
    # Apply SVD to watermark
    Uw, Sw, Vtw = np.linalg.svd(wm_resized, full_matrices=False)

    # Modify singular values
    S_watermarked = S + alpha * Sw

    # Reconstruct watermarked Y channel
    y_watermarked = np.dot(U, np.dot(np.diag(S_watermarked), Vt))

    # Clip values and replace Y channel
    ycrcb[:, :, 0] = np.uint8(np.clip(y_watermarked, 0, 255))

    # Convert back to BGR
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def process_folder(input_dir, output_dir):
    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        img_paths.extend(glob(os.path.join(input_dir, ext)))

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping unreadable file: {img_path}")
            continue

        watermarked_img = apply_svd_watermark(img, watermark, alpha=ALPHA)

        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, watermarked_img)

    print(f"Processed {len(img_paths)} images from {input_dir}")

process_folder(train_dir, f"{output_base}/train")
process_folder(val_dir, f"{output_base}/val")

print("SVD watermarking complete.")
