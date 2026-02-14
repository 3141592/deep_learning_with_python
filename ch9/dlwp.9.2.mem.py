# 9.2 An image segmentation example
# Suppress warnings
import os, pathlib
from pathlib import Path
from deep_learning_with_python.data_paths import get_data_root
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", Path.home() / "src" / "data"))

MODEL_PATH = (
    get_data_root()
    / "models"
)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

input_dir = get_data_root() / "images"
target_dir = get_data_root() / "annotations/trimaps"

input_img_paths = sorted(
        [os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")])
target_paths = sorted(
        [os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")])

#
# Figure 9.3 An example image
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array

plt.axis("off")
# Display input image number 9.
plt.imshow(load_img(input_img_paths[9]))
plt.show()

def display_target(target_array):
    # The original labels are 1, 2, and 3. We subtract 1 so that the
    # labels range from 0 to 2, and then we multiply by 127 so that
    # the labels become 0(black), 127(gray), 254(near white).
    normalized_array = (target_array.astype("uint8") - 1) * 127
    plt.axis("off")
    plt.imshow(normalized_array[:, :, 0])
    plt.show()

# We use color_mode = "grayscale"so that the image we load is treated as having a single color channel.
img = img_to_array(load_img(target_paths[9], color_mode="grayscale"))
display_target(img)

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
# Training on Large Datasets That Don’t Fit In Memory in Keras
# Rajat Garg
# Mar 19, 2019
# https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
#
#from skimage.io import imread
#from skimage.transform import resize
from tensorflow import keras

class My_Custom_Generator(keras.utils.Sequence) :

  def __init__(self, image_filenames, labels, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size


  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)


  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    return np.array(batch_x), np.array(batch_y)
            #resize(imread('/content/all_images/' + str(file_name)), (80, 80, 3))
            #   for file_name in batch_x])/255.0, np.array(batch_y)

# Create instances of class
batch_size = 32

my_training_batch_generator = My_Custom_Generator(train_input_imgs, train_targets, batch_size)
my_validation_batch_generator = My_Custom_Generator(val_input_imgs, val_targets, batch_size)

#
# Define the model
print("tensorflow imports")
from tensorflow import keras
from tensorflow.keras import layers

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    # Don't forget to rescale input images to the [0-1] range.
    x = layers.Rescaling(1./255)(inputs)

    print("Creating layers")
    # Note how we use padding="same" everywhere to avoid
    # the influence of border padding on feature map size.
    print("Creating Conv2D layers")
    x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(256, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)

    print("Creating Conv2DTranspose layers")
    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)

    # We end the model with a per-pixel three-way softmax to 
    # classify each output pixel into one of our three categories.
    print("Creating output layer")
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    
    print("Instantiating keras.Model")
    model = keras.Model(inputs, outputs)
    return model

print("Creating model")
model = get_model(img_size=img_size, num_classes=3)
model.summary()

# Compile and fit model
print("Compile and fit model")
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

callbacks = [
        keras.callbacks.ModelCheckpoint(MODEL_PATH / "oxford.segmentation.keras",
            save_best_only=True)
]

history = model.fit(
        my_training_batch_generator,
        steps_per_epoch = int(3800 // batch_size),
        epochs=30,
        callbacks=callbacks,
        validation_data = my_validation_batch_generator,
        validation_steps = int(950 // batch_size))

# Figure 9.5
print("Figure 9.5 Displaying training and validation loss curves")
epochs = range(1, len(history.history["loss"]) + 1)
loss = history.history["loss"]
val_loss = history.history["val_loss"]
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()


