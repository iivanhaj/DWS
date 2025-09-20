# plot_attack_metrics.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from skimage import io

# CONFIG - update paths if different
METRICS_CSV = "../dataset/attack_metrics.csv"
ATTACKS_DIR = "../dataset/attacks"
OUTPUT_DIR = "../results"
SAMPLE_IMAGE = None  # set like "img001.jpg" to show an example; None = auto pick first

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load metrics
df = pd.read_csv(METRICS_CSV)

# Basic cleaning: drop rows where PSNR/SSIM are NaN
df_clean = df.copy()
df_clean = df_clean.dropna(subset=["PSNR", "SSIM"])

# Convert PSNR/SSIM to numeric (if strings)
df_clean["PSNR"] = pd.to_numeric(df_clean["PSNR"], errors="coerce")
df_clean["SSIM"] = pd.to_numeric(df_clean["SSIM"], errors="coerce")
df_clean = df_clean.dropna(subset=["PSNR", "SSIM"])

# 1) Summary table (mean, std)
summary = df_clean.groupby(["method", "attack"]).agg(
    PSNR_mean=("PSNR","mean"),
    PSNR_std=("PSNR","std"),
    SSIM_mean=("SSIM","mean"),
    SSIM_std=("SSIM","std"),
    count=("image","count")
).reset_index()
summary.to_csv(os.path.join(OUTPUT_DIR, "attack_summary.csv"), index=False)
print("Saved summary to", os.path.join(OUTPUT_DIR, "attack_summary.csv"))

# 2) Boxplots: PSNR by method for each attack (one figure per attack)
attacks = df_clean["attack"].unique()
for atk in attacks:
    sub = df_clean[df_clean["attack"]==atk]
    plt.figure(figsize=(10,6))
    sns.boxplot(x="method", y="PSNR", data=sub, order=sorted(sub["method"].unique()))
    plt.title(f"PSNR distribution by method — Attack: {atk}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    outp = os.path.join(OUTPUT_DIR, f"boxplot_PSNR_{atk}.png")
    plt.savefig(outp, dpi=150)
    plt.close()
    print("Saved", outp)

    plt.figure(figsize=(10,6))
    sns.boxplot(x="method", y="SSIM", data=sub, order=sorted(sub["method"].unique()))
    plt.title(f"SSIM distribution by method — Attack: {atk}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    outp = os.path.join(OUTPUT_DIR, f"boxplot_SSIM_{atk}.png")
    plt.savefig(outp, dpi=150)
    plt.close()
    print("Saved", outp)

# 3) Aggregate plots: mean PSNR per method across attacks (heatmap)
pivot_psnr = summary.pivot(index="method", columns="attack", values="PSNR_mean")
plt.figure(figsize=(8,6))
sns.heatmap(pivot_psnr, annot=True, fmt=".2f", cmap="viridis")
plt.title("Mean PSNR (method × attack)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_psnr_mean.png"), dpi=150)
plt.close()

pivot_ssim = summary.pivot(index="method", columns="attack", values="SSIM_mean")
plt.figure(figsize=(8,6))
sns.heatmap(pivot_ssim, annot=True, fmt=".3f", cmap="magma")
plt.title("Mean SSIM (method × attack)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_ssim_mean.png"), dpi=150)
plt.close()

print("Saved heatmaps to", OUTPUT_DIR)

# 4) Show a sample visual comparison (original watermarked attacked)
# Attempt to pick a sample image that exists across methods
if SAMPLE_IMAGE is None:
    SAMPLE_IMAGE = df_clean["image"].iloc[0]

sample = SAMPLE_IMAGE
method = df_clean["method"].iloc[0]
attack = df_clean["attack"].iloc[0]
split = df_clean["split"].iloc[0]

# find actual file paths in attacks dir
def find_path(base_dir, method, attack, split, image_name):
    p = Path(base_dir) / method / attack / split / image_name
    return p if p.exists() else None

found = None
for m in df_clean["method"].unique():
    for a in df_clean["attack"].unique():
        for s in df_clean["split"].unique():
            p = find_path(ATTACKS_DIR, m, a, s, sample)
            if p:
                found = (m,a,s,p)
                break
        if found:
            break
    if found:
        break

if found:
    m,a,s,p = found
    # load files: original watermarked (input), attacked
    orig_path = Path("../dataset/watermarked") / m / s / sample
    attacked_path = p
    if orig_path.exists():
        try:
            orig = io.imread(str(orig_path))
            attacked = io.imread(str(attacked_path))
            fig, axes = plt.subplots(1,2, figsize=(10,5))
            axes[0].imshow(orig)
            axes[0].set_title(f"{m} - watermarked ({s})")
            axes[0].axis("off")
            axes[1].imshow(attacked)
            axes[1].set_title(f"Attacked: {a}")
            axes[1].axis("off")
            plt.tight_layout()
            outp = os.path.join(OUTPUT_DIR, f"sample_{sample}_{m}_{a}.png")
            plt.savefig(outp, dpi=150)
            plt.close()
            print("Saved sample visual to", outp)
        except Exception as e:
            print("Could not render sample image:", e)
else:
    print("Sample image not found on disk; skipping sample visual step.")

print("All plots and summaries saved under", OUTPUT_DIR)
