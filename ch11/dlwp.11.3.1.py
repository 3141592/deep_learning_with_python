# Suppress warnings
import os, pathlib
import shutil
from ai_shared_data import ensure_asset, get_asset_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 11.3.1 Preparing the IMDB movie reviews data
print("11.3.1 Preparing the IMDB movie reviews data")

print("Prepare a validation set by setting apart 20% of the training text files in a new directory")
import os, pathlib, shutil, random

ensure_asset("aclImdb")
base_dir = get_asset_path("aclImdb")
val_dir = base_dir / "val"
train_dir = base_dir / "train"

if val_dir.exists():
    shutil.rmtree(val_dir)

for category in ("neg", "pos"):
    os.makedirs(val_dir / category, exist_ok = True)
    files = os.listdir(train_dir / category)
    # Shuffle the list of training files using a seed,
    # to ensure we get the same validationset every time we run the code.
    random.Random(1337).shuffle(files)
    # Take 20% of the training files to use for validation.
    num_val_samples = int(0.2 * len(files))
    val_files = files[-num_val_samples:]
    # Move the files to aclImdb/val/neg and aclImdb/val/pos
    for fname in val_files:
        shutil.copy(train_dir / category / fname,
                    val_dir / category / fname)


