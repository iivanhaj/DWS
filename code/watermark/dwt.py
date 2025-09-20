import cv2
import numpy as np
import pywt
import os
from glob import glob
from pathlib import Path

train_dir = "../../dataset/original/train"
val_dir = "../../dataset/original/val"
watermark_path = "../../watermark.png"
output_base = "../../dataset/watermarked/DWT"

ALPHA = 0.5 # Lower for more invisible
WM_SIZE = 64  # Watermark size

for split in ["train", "val"]:
    Path(f"{output_base}/{split}").mkdir(parents=True, exist_ok=True)

watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
if watermark is None:
    raise FileNotFoundError(f"Watermark not found at {watermark_path}")
watermark = cv2.resize(watermark, (WM_SIZE, WM_SIZE), interpolation=cv2.INTER_AREA)

def apply_dwt_watermark(cover_img, watermark_img, alpha=0.01):
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(cover_img, cv2.COLOR_BGR2YCrCb)
    y_channel = np.float32(ycrcb[:, :, 0])

    # Apply single-level DWT
    coeffs2 = pywt.dwt2(y_channel, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # Embed watermark into LL sub-band
    wm_resized = cv2.resize(watermark_img, (LL.shape[1], LL.shape[0]))
    LL += alpha * wm_resized

    # Inverse DWT
    y_channel_watermarked = pywt.idwt2((LL, (LH, HL, HH)), 'haar')

    # Clip and replace Y channel
    ycrcb[:, :, 0] = np.uint8(np.clip(y_channel_watermarked, 0, 255))

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

        watermarked_img = apply_dwt_watermark(img, watermark, alpha=ALPHA)

        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, watermarked_img)

    print(f"Processed {len(img_paths)} images from {input_dir}")

process_folder(train_dir, f"{output_base}/train")
process_folder(val_dir, f"{output_base}/val")

print("DWT watermarking complete.")
