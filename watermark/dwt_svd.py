import cv2
import numpy as np
import pywt
import os
from glob import glob
from pathlib import Path

train_dir = "../../dataset/original/train"
val_dir = "../../dataset/original/val"
watermark_path = "../../watermark.png"
output_base = "../../dataset/watermarked/DWT_SVD"

ALPHA = 0.5       # Controls strength of watermark embedding
WM_SIZE = 64       # Size of watermark embedding block

for split in ["train", "val"]:
    Path(f"{output_base}/{split}").mkdir(parents=True, exist_ok=True)

watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
if watermark is None:
    raise FileNotFoundError(f"Watermark not found at {watermark_path}")
watermark = cv2.resize(watermark, (WM_SIZE, WM_SIZE), interpolation=cv2.INTER_AREA)

def apply_dwt_svd_watermark(cover_img, watermark_img, alpha=0.05):
    # Convert to YCrCb and take Y channel
    ycrcb = cv2.cvtColor(cover_img, cv2.COLOR_BGR2YCrCb)
    y_channel = np.float32(ycrcb[:, :, 0])

    # Step 1: Apply single-level DWT
    LL, (LH, HL, HH) = pywt.dwt2(y_channel, 'haar')

    # Step 2: SVD on LL band
    U, S, Vt = np.linalg.svd(LL, full_matrices=False)

    # Step 3: Resize watermark to match size of S
    wm_resized = cv2.resize(watermark_img, (S.shape[0], 1))  # Shape (1, n) after flatten
    wm_resized = wm_resized.flatten()

    # Step 4: Embed watermark in singular values
    S_watermarked = S + alpha * wm_resized

    # Step 5: Reconstruct LL band with modified singular values
    LL_modified = np.dot(U, np.dot(np.diag(S_watermarked), Vt))

    # Step 6: Inverse DWT
    y_channel_watermarked = pywt.idwt2((LL_modified, (LH, HL, HH)), 'haar')

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

        watermarked_img = apply_dwt_svd_watermark(img, watermark, alpha=ALPHA)

        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, watermarked_img)

    print(f"Processed {len(img_paths)} images from {input_dir}")

process_folder(train_dir, f"{output_base}/train")
process_folder(val_dir, f"{output_base}/val")

print("DWT-SVD watermarking complete.")
