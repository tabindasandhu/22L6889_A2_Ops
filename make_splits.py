"""
Task 2.1 - Create reproducible, stratified train/val/test split lists.
Uses the audit table, so corrupt files and duplicates are excluded.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
OUT_DIR = Path("data/splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("reports/audit.csv")
df = df[df["readable"] & ~df["is_duplicate"]].copy()     # clean images only

# label: 1 = AI-generated (the thing we want to detect), 0 = real
df["label"] = (df["class"] == "ai").astype(int)

# 70% train, 30% temporary
train, temp = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=SEED)
# split the 30% in half -> 15% val, 15% test
val, test = train_test_split(
    temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

for name, part in [("train", train), ("val", val), ("test", test)]:
    part[["path", "label", "class"]].to_csv(OUT_DIR / f"{name}.csv", index=False)
    print(f"{name:5s}: {len(part):4d} images  {part['class'].value_counts().to_dict()}")

# safety check: no image appears in two splits
assert not (set(train.path) & set(val.path) | set(train.path) & set(test.path) | set(val.path) & set(test.path))
print("No overlap between splits.")