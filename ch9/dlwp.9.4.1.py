# 9.4.1 Visualizing intermediate activations
# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path
import os

def get_data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", Path.home() / "src" / "data"))

MODEL_PATH = (
    get_data_root()
    / "models"
    / "convnet_from_scratch_with_augmentation.keras"
)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

from tensorflow import keras
model = keras.models.load_model(MODEL_PATH)
model.summary()

#
# Listing 9.6 Preprocessing a single image
print("Listing 9.6 Preprocessing a single image")
import numpy as np

# Download a test image.
img_path = keras.utils.get_file(fname="cat.jpg", origin="https://img-datasets.s3.amazonaws.com/cat.jpg")

def get_img_array(img_path, target_size):
    # Open the image file and resize it.
    img = keras.utils.load_img(img_path, target_size=target_size)

    # Turn the image into a float32 NumPy array of shape (180, 180, 3).
    array = keras.utils.img_to_array(img)
    # Add a dimension to transform the array into a "batch" of a single smaple. 
    # It's shape is now (1, 180, 180, 3).
    array = np.expand_dims(array, axis=0)
    return array

img_tensor = get_img_array(img_path, target_size=(180, 180))

#
# Listing 9.7 Displaying the test picture
print("Listing 9.7 Displaying the test picture")
import matplotlib.pyplot as plt
plt.axis("off")
plt.imshow(img_tensor[0].astype("uint8"))
plt.show()

#
# Listing 9.8 Instantiating a model that returns layer activations
print("Listing 9.8 Instantiating a model that returns layer activations")
from tensorflow.keras import layers

layer_outputs = []
layer_names = []

# Extract the outputs of all Conv2D and MaxPooling2D layers and put them in a list.
for layer in model.layers:
    if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
        layer_outputs.append(layer.output)
        layer_names.append(layer.name)

# Create a model that will return these outputs, given the model input.
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

#
# Listing 9.9 Using the model to compute layer activations
print("Listing 9.9 Using the model to compute layer activations")
activations = activation_model.predict(img_tensor)

print("This is the activation of the first convolution for the cat image input:")
first_layer_activation = activations[0]
print(f"first_layer_activation.shape: {first_layer_activation.shape}")

#
# Listing 9.10 Visualizing the fifth channel
print("Listing 9.10 Visualizing the fifth channel")
import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0, :, :, 5], cmap="viridis")
plt.show()

#
# Listing 9.11 Visualizing every channel in every intermediate activation
print("Listing 9.11 Visualizing every channel in every intermediate activation")
images_per_row = 16

# Iterate over the activations (and the names of the corresponding layers).
for layer_name, layer_activation in zip(layer_names, activations):
    # The layer activation has shape (1, size, size, n_features).
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    # Prepare an empty grid for displaying all the channels in this activation.
    display_grid = np.zeros(((size + 1) * n_cols - 1,
        images_per_row * (size + 1) - 1))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_index = col * images_per_row + row
            # This is a single channel (or feature).
            channel_image = layer_activation[0, :, :, channel_index].copy()
            
            # Normalize channel values within the [0, 255] range.
            # All-zero channels are kept at 0.
            if channel_image.sum() != 0:
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype("uint8")
            display_grid[
                    col * (size + 1): (col + 1) * size + col,
                    row * (size + 1) : (row + 1) * size + row] = channel_image

    # Display the grid for the layer
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.axis("off")
    plt.imshow(display_grid, aspect="auto", cmap="viridis")
    plt.show()


