"""
visualize.py — Look at the auxiliary dataset images.

Saves PNG grids so you can see what the 1000 candidate images look like.
No GPU needed, just matplotlib.

Outputs (saved in task1/):
  - sample_grid.png        : 10x10 grid of random images
  - class_distribution.png : bar chart of how many images per class
  - class_samples.png      : 10 rows (one per class), 10 sample images each
"""

import torch
import numpy as np
import os

import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Load the dataset ----
print("Loading auxiliary_dataset.pt ...")
dataset = torch.load(os.path.join(WORK_DIR, "auxiliary_dataset.pt"), weights_only=False)
images = dataset["images"]  # (1000, 3, 32, 32) float32 in [0, 1]
labels = dataset["labels"]  # (1000,)

print(f"  {images.shape[0]} images, {images.shape[2]}x{images.shape[3]} pixels, {images.shape[1]} channels")
print(f"  Labels: {sorted(set(labels.tolist()))}")
print()

# ---- 1. Sample grid: 10x10 random images ----
print("Saving sample_grid.png ...")
fig, axes = plt.subplots(10, 10, figsize=(12, 12))
fig.suptitle("100 Random Images from Auxiliary Dataset", fontsize=14)
indices = np.random.default_rng(42).choice(len(images), 100, replace=False)
for i, idx in enumerate(indices):
    ax = axes[i // 10][i % 10]
    # images are (C, H, W) — transpose to (H, W, C) for display
    img = images[idx].permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(f"#{idx}\nl={labels[idx].item()}", fontsize=6)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "sample_grid.png"), dpi=150)
plt.close()

# ---- 2. Class distribution bar chart ----
print("Saving class_distribution.png ...")
label_list = labels.tolist()
unique_labels = sorted(set(label_list))
counts = [label_list.count(l) for l in unique_labels]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(unique_labels, counts, color="steelblue")
ax.set_xlabel("Class Label")
ax.set_ylabel("Count")
ax.set_title("Images per Class in Auxiliary Dataset")
ax.set_xticks(unique_labels)
for l, c in zip(unique_labels, counts):
    ax.text(l, c + 1, str(c), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "class_distribution.png"), dpi=150)
plt.close()

# ---- 3. Per-class sample grid: 10 rows x 10 columns ----
print("Saving class_samples.png ...")
fig, axes = plt.subplots(10, 10, figsize=(12, 14))
fig.suptitle("10 Sample Images per Class (rows = classes 0–9)", fontsize=14, y=1.0)

for cls in range(10):
    cls_indices = [i for i, l in enumerate(label_list) if l == cls]
    # pick first 10 (or fewer if not enough)
    shown = cls_indices[:10]
    for col, idx in enumerate(shown):
        ax = axes[cls][col]
        img = images[idx].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(f"Class {cls}", fontsize=10, rotation=0, labelpad=40)
    # blank out unused columns
    for col in range(len(shown), 10):
        axes[cls][col].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "class_samples.png"), dpi=150)
plt.close()

print()
print("Done! Open these files to see your images:")
print(f"  {os.path.join(WORK_DIR, 'sample_grid.png')}")
print(f"  {os.path.join(WORK_DIR, 'class_distribution.png')}")
print(f"  {os.path.join(WORK_DIR, 'class_samples.png')}")
