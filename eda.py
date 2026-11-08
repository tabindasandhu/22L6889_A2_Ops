"""
Task 1.3 - Visual EDA
Builds a sample-image grid per class and histograms of width, height and file size.
Reads the table produced by audit.py.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

AUDIT_CSV = Path("reports/audit.csv")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42            # fixed seed -> the same sample images every run
N_PER_CLASS = 8
CLASSES = ["ai", "real"]
COLOURS = {"ai": "tab:red", "real": "tab:blue"}

df = pd.read_csv(AUDIT_CSV)
df = df[df["readable"] & ~df["is_duplicate"]]       # clean images only

# ---------- Figure 1: sample grid ----------
fig, axes = plt.subplots(len(CLASSES), N_PER_CLASS,
                         figsize=(2.3 * N_PER_CLASS, 2.8 * len(CLASSES)))
for r, cls in enumerate(CLASSES):
    sample = df[df["class"] == cls].sample(N_PER_CLASS, random_state=SEED)
    for c, (_, row) in enumerate(sample.iterrows()):
        ax = axes[r, c]
        with Image.open(row["path"]) as img:
            img = img.convert("RGB")          # L / P / RGBA -> RGB for display
            img.thumbnail((256, 256))         # shrink for a light figure
            ax.imshow(img)
        ax.set_title(f"{cls.upper()} | {int(row['width'])}x{int(row['height'])}",
                     fontsize=8)
        ax.axis("off")
fig.suptitle("Random sample images per class (seed = 42)", fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "sample_grid.png", dpi=150)
plt.close(fig)

# ---------- Figure 2: histograms ----------
metrics = [("width", "Width (pixels)"),
           ("height", "Height (pixels)"),
           ("file_size_kb", "File size (KB, log scale)")]

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
for ax, (col, label) in zip(axes, metrics):
    if col == "file_size_kb":
        # file sizes are very skewed -> log-spaced bins show the shape better
        bins = np.logspace(np.log10(df[col].min()), np.log10(df[col].max()), 40)
        ax.set_xscale("log")
    else:
        bins = 40
    for cls in CLASSES:
        vals = df.loc[df["class"] == cls, col]
        ax.hist(vals, bins=bins, alpha=0.55, color=COLOURS[cls],
                label=f"{cls} (n={len(vals)}, median={vals.median():.0f})")
        ax.axvline(vals.median(), color=COLOURS[cls], linestyle="--", linewidth=1.2)
    ax.set_xlabel(label)
    ax.set_ylabel("Number of images")
    ax.legend(fontsize=8)
fig.suptitle("Distribution of image width, height and file size per class "
             "(dashed line = median)", fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "histograms.png", dpi=150)
plt.close(fig)

print("Saved:", FIG_DIR / "sample_grid.png")
print("Saved:", FIG_DIR / "histograms.png")