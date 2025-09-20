import os
import cv2
import numpy as np
import csv
import pywt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# CONFIG
WATERMARKED_DIR = "../dataset/watermarked"
ATTACKS_DIR = "../dataset/attacks"
METRICS_FILE = "../dataset/metrics_with_ber.csv"

methods = ["DCT", "DWT", "SVD", "DWT_DCT", "DWT_SVD"]
splits = ["train", "val"]
attacks = ["Blur", "Crop", "JPEG50", "JPEG70", "Noise", "Rotate"]

# === Load original watermark ===
wm_path = "../watermark.png"
original_watermark = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
if original_watermark is None:
    raise FileNotFoundError(f"Watermark not found at {wm_path}")
original_watermark = cv2.resize(original_watermark, (32, 32))
_, original_watermark = cv2.threshold(original_watermark, 128, 1, cv2.THRESH_BINARY)
original_bits = original_watermark.flatten()

# === BER ===
def compute_ber(orig_bits, ext_bits):
    min_len = min(orig_bits.size, ext_bits.size)
    if min_len == 0:
        return None
    return np.sum(orig_bits[:min_len] != ext_bits[:min_len]) / min_len

WM_SIZE = 32  # watermark is resized to 32x32

# === Extractors (Fixed: use original image too) ===
def extract_dct(orig_img, attacked_img, alpha=0.5):
    y_orig = np.float32(cv2.cvtColor(orig_img, cv2.COLOR_BGR2YCrCb)[:, :, 0])
    y_att = np.float32(cv2.cvtColor(attacked_img, cv2.COLOR_BGR2YCrCb)[:, :, 0])
    dct_orig = cv2.dct(y_orig)
    dct_att = cv2.dct(y_att)
    wm_est = (dct_att[:WM_SIZE, :WM_SIZE] - dct_orig[:WM_SIZE, :WM_SIZE]) / alpha
    wm_norm = cv2.normalize(wm_est, None, 0, 1, cv2.NORM_MINMAX)
    return (wm_norm > 0.5).astype(np.uint8)

def extract_dwt(orig_img, attacked_img, alpha=0.5):
    y_orig = np.float32(cv2.cvtColor(orig_img, cv2.COLOR_BGR2YCrCb)[:, :, 0])
    y_att = np.float32(cv2.cvtColor(attacked_img, cv2.COLOR_BGR2YCrCb)[:, :, 0])
    LL_o, _ = pywt.dwt2(y_orig, 'haar')
    LL_a, _ = pywt.dwt2(y_att, 'haar')
    wm_est = (LL_a - LL_o) / alpha
    wm_resized = cv2.resize(wm_est, (WM_SIZE, WM_SIZE))
    wm_norm = cv2.normalize(wm_resized, None, 0, 1, cv2.NORM_MINMAX)
    return (wm_norm > 0.5).astype(np.uint8)

def extract_svd(orig_img, attacked_img, alpha=0.5):
    y_orig = np.float32(cv2.cvtColor(orig_img, cv2.COLOR_BGR2YCrCb)[:, :, 0])
    y_att = np.float32(cv2.cvtColor(attacked_img, cv2.COLOR_BGR2YCrCb)[:, :, 0])
    _, S_orig, _ = np.linalg.svd(y_orig, full_matrices=False)
    _, S_att, _ = np.linalg.svd(y_att, full_matrices=False)
    wm_est = (S_att - S_orig) / alpha
    wm_resized = cv2.resize(wm_est.reshape(-1, 1), (WM_SIZE, WM_SIZE))
    wm_norm = cv2.normalize(wm_resized, None, 0, 1, cv2.NORM_MINMAX)
    return (wm_norm > 0.5).astype(np.uint8)

def extract_dwt_dct(orig_img, attacked_img, alpha=0.5):
    y_orig = np.float32(cv2.cvtColor(orig_img, cv2.COLOR_BGR2YCrCb)[:, :, 0])
    y_att = np.float32(cv2.cvtColor(attacked_img, cv2.COLOR_BGR2YCrCb)[:, :, 0])
    LL_o, _ = pywt.dwt2(y_orig, 'haar')
    LL_a, _ = pywt.dwt2(y_att, 'haar')
    dct_LL_o = cv2.dct(LL_o)
    dct_LL_a = cv2.dct(LL_a)
    wm_est = (dct_LL_a[:WM_SIZE, :WM_SIZE] - dct_LL_o[:WM_SIZE, :WM_SIZE]) / alpha
    wm_norm = cv2.normalize(wm_est, None, 0, 1, cv2.NORM_MINMAX)
    return (wm_norm > 0.5).astype(np.uint8)

def extract_dwt_svd(orig_img, attacked_img, alpha=0.5):
    y_orig = np.float32(cv2.cvtColor(orig_img, cv2.COLOR_BGR2YCrCb)[:, :, 0])
    y_att = np.float32(cv2.cvtColor(attacked_img, cv2.COLOR_BGR2YCrCb)[:, :, 0])
    LL_o, _ = pywt.dwt2(y_orig, 'haar')
    LL_a, _ = pywt.dwt2(y_att, 'haar')
    _, S_orig, _ = np.linalg.svd(LL_o, full_matrices=False)
    _, S_att, _ = np.linalg.svd(LL_a, full_matrices=False)
    wm_est = (S_att - S_orig) / alpha
    wm_resized = cv2.resize(wm_est.reshape(-1, 1), (WM_SIZE, WM_SIZE))
    wm_norm = cv2.normalize(wm_resized, None, 0, 1, cv2.NORM_MINMAX)
    return (wm_norm > 0.5).astype(np.uint8)

extractors = {
    "DCT": extract_dct,
    "DWT": extract_dwt,
    "SVD": extract_svd,
    "DWT_DCT": extract_dwt_dct,
    "DWT_SVD": extract_dwt_svd
}

# === MAIN PROCESS ===
def process_metrics():
    with open(METRICS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "method", "split", "attack", "PSNR", "SSIM", "BER"])

        for method in methods:
            for split in splits:
                for attack_name in attacks:
                    input_dir = Path(ATTACKS_DIR) / method / attack_name / split
                    orig_dir = Path(WATERMARKED_DIR) / method / split
                    img_paths = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

                    if not img_paths:
                        print(f"[{method}-{split}] No images for attack={attack_name}")
                        continue

                    total_imgs = len(img_paths)
                    print(f"[{method}-{split}] Starting attack={attack_name} → {total_imgs} images")

                    for idx, attacked_path in enumerate(img_paths, 1):
                        img_file = attacked_path.name
                        orig_path = orig_dir / img_file

                        orig_img = cv2.imread(str(orig_path))
                        attacked_img = cv2.imread(str(attacked_path))

                        if orig_img is None or attacked_img is None:
                            print(f"  Skipping {img_file}, unreadable.")
                            continue

                        # Compute metrics
                        try:
                            psnr_val = psnr(orig_img, attacked_img, data_range=255)
                        except Exception:
                            psnr_val = None
                        try:
                            ssim_val = ssim(
                                cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY),
                                cv2.cvtColor(attacked_img, cv2.COLOR_BGR2GRAY),
                                data_range=255
                            )
                        except Exception:
                            ssim_val = None

                        # Extract watermark & compute BER
                        try:
                            # Extract watermark & compute BER
                            wm_extractor = extractors[method]
                            extracted_wm = wm_extractor(orig_img, attacked_img)
                            ber_val = compute_ber(original_bits, extracted_wm.flatten())

                        except Exception as e:
                            print(f"    Extractor failed for {img_file}: {e}")
                            ber_val = None

                        # Safe formatting
                        psnr_str = f"{psnr_val:.2f}" if psnr_val is not None else "None"
                        ssim_str = f"{ssim_val:.4f}" if ssim_val is not None else "None"
                        ber_str  = f"{ber_val:.4f}" if ber_val is not None else "None"

                        print(f"  [{idx}/{total_imgs}] {img_file} → PSNR={psnr_str}, SSIM={ssim_str}, BER={ber_str}")

                        # Save to CSV
                        writer.writerow([img_file, method, split, attack_name, psnr_val, ssim_val, ber_val])

# Entry
if __name__ == "__main__":
    process_metrics()
