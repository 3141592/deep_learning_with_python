# Suppress warnings
import os, pathlib
from ai_surgery.data_paths import get_data_root

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_ROOT = get_data_root()

# 11.3.1 Preparing the IMDB movie reviews data
print("11.3.1 Preparing the IMDB movie reviews data")

print("Prepare a validation set by setting apart 20% of the training text files in a new directory")
import os, pathlib, shutil, random

base_dir = DATA_ROOT  / "aclImdb"
val_dir = base_dir / "val"
train_dir = base_dir / "train"
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
        shutil.move(train_dir / category / fname,
                    val_dir / category / fname)


