import pathlib, os
import os, pathlib
from pathlib import Path
import sys
import tensorflow as tf
from tensorflow import keras
from ai_shared_utilities import get_asset_path

DATA_ROOT = get_asset_path("aclImdb")

def count_files(root):
    root = pathlib.Path(root)
    n = sum(1 for _ in root.rglob("*.txt"))
    print(root, "txt files:", n)
    for sub in ["pos", "neg"]:
        p = root / sub
        if p.exists():
            print("  ", p, ":", sum(1 for _ in p.glob("*.txt")))
    print()

for split in ["train", "val", "test"]:
    count_files(DATA_ROOT / split)

import hashlib, pathlib, re

def norm_text(p: pathlib.Path) -> str:
    s = p.read_text(errors="ignore")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def file_hash(p: pathlib.Path) -> str:
    h = hashlib.sha1(norm_text(p).encode("utf-8")).hexdigest()
    return h

def collect_hashes(dir_path: pathlib.Path):
    hashes = {}
    for p in dir_path.rglob("*.txt"):
        # skip known non-review files if any sneak in
        if p.name.startswith("urls_") or p.name in {"imdb.vocab", "imdbEr.txt"}:
            continue
        hashes[file_hash(p)] = p
    return hashes

train_dir = get_asset_path("aclImdb") / "train"
val_dir   = get_asset_path("aclImdb") / "val"

train_hashes = collect_hashes(train_dir)
val_hashes   = collect_hashes(val_dir)

overlap = set(train_hashes.keys()) & set(val_hashes.keys())
print("Train unique:", len(train_hashes))
print("Val unique:", len(val_hashes))
print("Exact text overlaps train∩val:", len(overlap))

# show a few examples
for i, h in enumerate(list(overlap)[:5]):
    print("\nOverlap example:")
    print(" train:", train_hashes[h])
    # find one val file with same hash (need reverse map)
    # easiest: rescan val to find it
    for p in val_dir.rglob("*.txt"):
        if p.suffix == ".txt" and file_hash(p) == h:
            print("   val:", p)
            break