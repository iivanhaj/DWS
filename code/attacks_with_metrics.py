import cv2
import numpy as np
import os
import csv
from glob import glob
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# === CONFIG ===
input_base = "../dataset/watermarked"   # where your watermarked folders are
output_base = "../dataset/attacks"      # where attacked images will be saved
metrics_file = "../dataset/attack_metrics.csv"

methods = ["DCT", "DWT", "SVD", "DWT_DCT", "DWT_SVD"]
splits = ["train", "val"]

# === ATTACKS ===
def jpeg_compression(img, quality=50):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)

def add_gaussian_noise(img, mean=0, var=0.01):
    h, w, c = img.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (h, w, c)) * 255
    noisy = np.clip(img + gauss, 0, 255).astype(np.uint8)
    return noisy

def gaussian_blur(img, ksize=5):
    # ksize must be odd
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def crop_resize(img, crop_ratio=0.8):
    h, w = img.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    top = (h - ch) // 2
    left = (w - cw) // 2
    cropped = img[top:top+ch, left:left+cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)

def rotate(img, angle=15):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# attack dictionary - add/remove attacks as needed
attacks = {
    "JPEG50": lambda img: jpeg_compression(img, 50),
    "JPEG70": lambda img: jpeg_compression(img, 70),
    "Noise": lambda img: add_gaussian_noise(img, var=0.005),
    "Blur": lambda img: gaussian_blur(img, ksize=5),
    "Crop": lambda img: crop_resize(img, crop_ratio=0.8),
    "Rotate": lambda img: rotate(img, angle=15)
}

# === PROCESSING FUNCTION ===
def process_attacks(method, split, writer):
    input_dir = os.path.join(input_base, method, split)
    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        img_paths.extend(glob(os.path.join(input_dir, ext)))

    if not img_paths:
        print(f"No images found in {input_dir}")
        return

    for attack_name, attack_fn in attacks.items():
        output_dir = os.path.join(output_base, method, attack_name, split)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping unreadable file: {img_path}")
                continue

            attacked_img = attack_fn(img)

            # Save attacked image
            filename = os.path.basename(img_path)
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, attacked_img)

            # Compute metrics (PSNR & SSIM) comparing attacked image to the watermarked original
            try:
                psnr_val = psnr(img, attacked_img, data_range=255)
            except Exception:
                psnr_val = None

            try:
                # compute SSIM on grayscale for speed and stability
                ssim_val = ssim(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(attacked_img, cv2.COLOR_BGR2GRAY),
                    data_range=255
                )
            except Exception:
                ssim_val = None

            # Placeholder for BER - uncomment and implement when extraction function is available
            # Example: extracted_wm = extract_watermark_method(attacked_img, method)
            #          ber_val = compute_ber(original_wm, extracted_wm)
            ber_val = ""  # keep empty for now

            # Write row: image, method, split, attack, psnr, ssim, ber
            writer.writerow([filename, method, split, attack_name, psnr_val, ssim_val, ber_val])

        print(f"[{method}-{split}] {attack_name} → {len(img_paths)} images processed.")

# === MAIN ===
if __name__ == "__main__":
    # Ensure output base exists
    Path(output_base).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(metrics_file)).mkdir(parents=True, exist_ok=True)

    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "method", "split", "attack", "PSNR", "SSIM", "BER"])

        for method in methods:
            for split in splits:
                process_attacks(method, split, writer)

    print(f"All attacks applied successfully. Metrics logged to {metrics_file}")
