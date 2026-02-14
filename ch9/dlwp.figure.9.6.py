# 9.2 An image segmentation example
# Suppress warnings
import os, pathlib
from pathlib import Path
from tensorflow.keras.utils import load_img, img_to_array 
from deep_learning_with_python.data_paths import get_data_root

def get_data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", Path.home() / "src" / "data"))

MODEL_PATH = (
    get_data_root()
    / "models"
)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

input_dir = get_data_root() / "images"
target_dir = get_data_root() / "annotations" / "trimaps"

input_img_paths = sorted(
        [os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")])
target_paths = sorted(
        [os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")])

#
# Load inputs and targets into two NumPy arrays.
import numpy as np
import random

# We resize everything to 200 x 200
img_size = (200, 200)
# Total number of samples in the data
num_imgs = len(input_img_paths)

# Shuffle the file paths (they were originally sorted by breed).
# We use the same seed (1337) in both statements to ensure
# that the input paths and target paths stay in the same order.
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_paths)

def path_to_input_image(path):
    return img_to_array(load_img(path, target_size=img_size))

def path_to_target(path):
    img = img_to_array(load_img(path, target_size=img_size, color_mode="grayscale"))
    # Subtract 1 so that our labels become 0, 1, and 2.
    img = img.astype("uint8") - 1
    return img

# Load all images in the input_imgs float32 array and their masks in the 
# targets uint array (same order). The inputs have three channels (RBG values)
# and the targets have a single channel (which contains integer labels).
input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32")
targets = np.zeros((num_imgs,) + img_size + (1,), dtype="uint8")
for i in range(num_imgs):
    input_imgs[i] = path_to_input_image(input_img_paths[i])
    targets[i] = path_to_target(target_paths[i])

# Reserve 1,000 samples for validation.
num_val_samples = 1000
# Split the data into a training and a validation set.
train_input_imgs = input_imgs[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_input_imgs = input_imgs[-num_val_samples:]
val_targets = targets[-num_val_samples:]

#
# Figure 9.6 A test image and its predicted segmentation mask
from tensorflow import keras
from tensorflow.keras.utils import array_to_img
import matplotlib.pyplot as plt

model = keras.models.load_model(MODEL_PATH / "oxford.segmentation.keras")

i = 5
test_image = val_input_imgs[i]
plt.axis("off")
plt.imshow(array_to_img(test_image))
plt.show()

mask = model.predict(np.expand_dims(test_image, 0))[0]

# Utility to display a model's prediction
def display_mask(pred):
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.axis("off")
    plt.imshow(mask)
    plt.show()

display_mask(mask)

