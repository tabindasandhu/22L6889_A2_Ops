"""
Task 1.2 - Dataset audit
Walks the dataset, records one row per file, finds corrupt files and exact duplicates.
"""
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

# ---------- settings ----------
DATA_DIR = Path("data/raw/archive")
CLASS_MAP = {"AiArtData": "ai", "RealArt": "real"}   # folder name -> class label
OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)


def sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 fingerprint of a file, reading 1 MB at a time."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


# ---------- 1. walk every file and build one row per file ----------
rows = []
for folder, label in CLASS_MAP.items():
    for path in sorted((DATA_DIR / folder).rglob("*")):
        if not path.is_file():
            continue

        row = {
            "path": path.as_posix(),
            "class": label,
            "file_size_kb": round(path.stat().st_size / 1024, 2),
            "width": None,
            "height": None,
            "mode": None,
            "format": None,
            "readable": False,
            "error": "",
            "sha256": sha256_of(path),
        }

        try:
            # Pass 1: verify() checks the file structure is valid (fast, no decoding)
            with Image.open(path) as img:
                img.verify()
            # Pass 2: load() actually decodes all pixels (catches truncated files)
            with Image.open(path) as img:
                img.load()
                row.update(
                    width=img.width,
                    height=img.height,
                    mode=img.mode,
                    format=img.format,
                    readable=True,
                )
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"

        rows.append(row)

df = pd.DataFrame(rows)

# ---------- 2. duplicates ----------
# keep="first": the first copy is kept as the original; later copies are flagged
df["is_duplicate"] = df.duplicated(subset="sha256", keep="first")

# the same image appearing in BOTH classes = a contradictory label
classes_per_hash = df.groupby("sha256")["class"].nunique()
cross_class_hashes = classes_per_hash[classes_per_hash > 1].index
df["cross_class_duplicate"] = df["sha256"].isin(cross_class_hashes)

df.to_csv(OUT_DIR / "audit.csv", index=False)

# ---------- 3. summaries for the report ----------
print("\n=== Counts per class ===")
counts = df.groupby("class").agg(
    total_files=("path", "count"),
    readable=("readable", "sum"),
    corrupt=("readable", lambda s: (~s).sum()),
    duplicates=("is_duplicate", "sum"),
)
print(counts.to_string())

ok = df[df["readable"]]
print("\n=== Width / height / file size (readable images only) ===")
stats = ok.groupby("class")[["width", "height", "file_size_kb"]].agg(["min", "median", "max"])
print(stats.to_string())

print("\n=== Formats per class ===")
print(ok.groupby(["class", "format"]).size().unstack(fill_value=0).to_string())

print("\n=== Colour modes per class ===")
print(ok.groupby(["class", "mode"]).size().unstack(fill_value=0).to_string())

print(f"\nTotal files           : {len(df)}")
print(f"Corrupt / unreadable  : {(~df['readable']).sum()}")
print(f"Exact duplicates      : {df['is_duplicate'].sum()}")
print(f"Cross-class duplicates: {df.loc[df['cross_class_duplicate'], 'sha256'].nunique()} images")

if (~df["readable"]).any():
    print("\n=== Corrupt files ===")
    print(df.loc[~df["readable"], ["path", "error"]].to_string(index=False))

stats.to_csv(OUT_DIR / "audit_stats.csv")
counts.to_csv(OUT_DIR / "audit_counts.csv")
print(f"\nSaved: {OUT_DIR / 'audit.csv'}")