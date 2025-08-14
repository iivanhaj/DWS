import cv2
import matplotlib.pyplot as plt
import os

original_dir = "../dataset/original/train"
methods = ["DCT", "DWT", "SVD", "DWT_DCT", "DWT_SVD"]
watermarked_base = "../dataset/watermarked"
sample_image_name = "0003_0_0.png"  

orig_path = os.path.join(original_dir, sample_image_name)
original = cv2.imread(orig_path)
if original is None:
    raise FileNotFoundError(f"Original image not found: {orig_path}")
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

watermarked_images = []
for method in methods:
    wm_path = os.path.join(watermarked_base, method, "train", sample_image_name)
    img = cv2.imread(wm_path)
    if img is None:
        raise FileNotFoundError(f"Watermarked image not found for {method}: {wm_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    watermarked_images.append((method, img))

fig, axes = plt.subplots(1, len(methods) + 1, figsize=(18, 5))
axes[0].imshow(original)
axes[0].set_title("Original")
axes[0].axis("off")

for ax, (method, img) in zip(axes[1:], watermarked_images):
    ax.imshow(img)
    ax.set_title(method)
    ax.axis("off")

plt.suptitle("Phase 1: Dataset Creation – Original & Watermarked Images", fontsize=16)
plt.tight_layout()
plt.show()
