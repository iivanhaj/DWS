# Digital Watermarking System (DWS) with XAI for Watermark Prediction

## Overview
The **Digital Watermarking System (DWS)** is a comprehensive framework for embedding, attacking, and evaluating watermarks in digital images. It supports multiple watermarking techniques, simulates real-world image attacks, and provides robust evaluation metrics. Additionally, this project incorporates **Explainable AI (XAI)** techniques for predicting watermark robustness and analyzing how attacks affect watermark integrity.

This project is designed for **research, benchmarking, and educational purposes** in image security and digital watermarking.

---

## Features
- **Image Preprocessing:** Extracts patches from raw images and organizes them into `train` and `val` sets.
- **Watermark Embedding:** Implements watermarking algorithms including:
  - DCT (Discrete Cosine Transform)
  - DWT (Discrete Wavelet Transform)
  - SVD (Singular Value Decomposition)
  - Hybrid methods: DWT+DCT, DWT+SVD
- **Attack Simulation:** Applies common distortions to watermarked images:
  - Blur, Crop, JPEG Compression (50%, 70%), Noise, Rotation
- **Metrics & Evaluation:**
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - BER (Bit Error Rate) for watermarks
- **Explainable AI for Watermark Prediction:** 
  - Predicts watermark resilience using interpretable ML models
  - Provides visual insights into which parts of images affect watermark survival
- **Visualization & Reporting:** Generates plots, heatmaps, and summary tables for benchmarking results.

---

## Dataset Structure

The dataset is organized to facilitate **watermark embedding, attack simulations, and metric computation**. Folder structure:

```
dataset/
├── original/
│   ├── train/            # Original image patches for training
│   └── val/              # Original image patches for validation
├── watermarked/
│   ├── DCT/               # Watermarked images using DCT
│   │   ├── train/
│   │   └── val/
│   ├── DWT/               # Watermarked images using DWT
│   │   ├── train/
│   │   └── val/
│   ├── DWT_DCT/           # Watermarked images using DWT+DCT
│   │   ├── train/
│   │   └── val/
│   ├── DWT_SVD/           # Watermarked images using DWT+SVD
│   │   ├── train/
│   │   └── val/
│   └── SVD/               # Watermarked images using SVD
│       ├── train/
│       └── val/
├── attacks/               # Attacked watermarked images
│   ├── DCT/               # Each watermark type has attack subfolders
│   │   ├── Blur/train/
│   │   ├── Blur/val/
│   │   ├── Crop/train/
│   │   └── ... etc.
│   ├── DWT/
│   └── ... (other watermark types)
├── attack_metrics.csv      # Metrics for attacked images
└── metrics_with_ber.csv    # Watermark BER metrics
```

**Notes:**
- Each watermark type (`DCT`, `DWT`, etc.) has the same **train/val split**.
- Under `attacks/`, images are grouped by **attack type** (Blur, Crop, JPEG50, JPEG70, Noise, Rotate), with corresponding `train` and `val` folders.
- Folder structure is preserved on GitHub using `.gitkeep` files.

---

## Getting Started

### 1. Preprocess Images
```bash
python code/preprocess.py
```
* Converts raw images into patch-based datasets.
* Creates `train/` and `val/` splits for each watermarking experiment.

### 2. Embed Watermarks
```bash
# Example: Embed DCT watermark
python code/watermark/dct.py
```
* Supports DCT, DWT, SVD, DWT+DCT, and DWT+SVD embedding.
* Generates watermarked images in corresponding `dataset/watermarked/<method>/` folders.

### 3. Apply Attacks
```bash
python code/attacks_with_metrics.py
```
* Applies all attack types on watermarked images.
* Saves attacked images in `dataset/attacks/<method>/<attack_type>/`.

### 4. Compute Metrics
```bash
python code/metrics_with_ber.py
```
* Computes PSNR, SSIM, and Bit Error Rate (BER) for attacked watermarked images.
* Outputs `attack_metrics.csv` and `metrics_with_ber.csv`.

### 5. Visualize Results
```bash
python code/plot_attack_metrics.py
```
* Generates plots, boxplots, and heatmaps summarizing metrics for benchmarking.
* Saves outputs to `results/` folder.

---

## Explainable AI for Watermark Prediction

* Uses ML models to **predict watermark survival** under attacks.
* Provides **visual interpretations** showing which regions in an image are most sensitive.
* Helps researchers understand the **robustness of each watermarking method**.

---

## Requirements

* Python 3.7+
* Libraries:
  * `opencv-python`
  * `numpy`
  * `scikit-image`
  * `pandas`
  * `matplotlib`
  * `seaborn`
  * `tqdm`

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Example Workflow

```bash
# Preprocess images
python code/preprocess.py

# Embed watermark (DWT example)
python code/watermark/dwt.py

# Apply attacks and compute metrics
python code/attacks_with_metrics.py

# Compute BER for watermarks
python code/metrics_with_ber.py

# Generate visualization
python code/plot_attack_metrics.py
```

---

## Directory Structure

```
project/
├── code/                    # All scripts for watermarking, attacks, metrics, plotting
├── dataset/                 # Original, watermarked, attacked images
├── results/                 # Visualizations and summary tables
├── raw_images/              # Source images
├── watermark.png            # Watermark image
├── architecture_diagram.pdf # High-level system overview
└── README.md                # (You are here)
```
