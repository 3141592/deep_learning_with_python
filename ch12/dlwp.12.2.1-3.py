# Suppress warnings
import os, pathlib
from deep_learning_with_python.data_paths import get_data_root
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_ROOT = get_data_root()

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("12.2.1 Implementing DeepDream in Keras")

print("Listing 12.9 Fetching the test image")
from tensorflow import keras
import matplotlib.pyplot as plt

base_image_path = keras.utils.get_file(
        "coast.jpg",
        origin="https://img-datasets.s3.amazonaws.com/coast.jpg")


plt.axis("off")
plt.imshow(keras.utils.load_img(base_image_path))
plt.show()

print("Listing 12.10 Instantiating a pretrained InceptionV3 model")
from tensorflow.keras.applications import inception_v3
model = inception_v3.InceptionV3(weights="imagenet", include_top=False)
model.summary()

print("Listing 12.11 Configuring the contribution of each layer to the DeepDream loss")
# Layers for which we try to maximize activation, as
# well as their weight in the total loss. You can tweak
# these setting to obtain new visual effects.
layer_settings = {
        "mixed0": 1.0,
        "mixed1": 1.5,
        "mixed2": 2.0,
        "mixed3": 2.5,
        "mixed4": 1.0,
        "mixed5": 1.5,
        "mixed6": 2.0,
        "mixed7": 2.5,
}
outputs_dict = dict(
        [
            (layer.name, layer.output)
            for layer in [model.get_layer(name)
                for name in layer_settings.keys()]
        ]
)
# Model that returns the activation values for every target layer (as a dict)
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

print("Listing 12.12 The DeepDream loss")
def compute_loss(input_image):
    # Extract activations.
    features = feature_extractor(input_image)
    # Initialize the loss to 0.
    loss = tf.zeros(shape=())
    for name in features.keys():
        coeff = layer_settings[name]
        activation = features[name]
        # We avoid border artifacts by only involving non-border pixels in the loss.
        loss += coeff * tf.reduce_mean(tf.square(activation[:, 2:-2, 2:-2, :]))
    return loss

print("Listing 12.13 The DeepDream gradient ascent process")
import tensorflow as tf

# We make the training step fast by compiling it as a tf.function.
@tf.function
def gradient_ascent_step(image, learning_rate):
    # Compute the gradients of DeepDream loss with respect to the current image.
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = compute_loss(image)
    grads = tape.gradient(loss, image)
    # Normalize gradients (the same trick we used in chapter 9).
    grads = tf.math.l2_normalize(grads)
    image += learning_rate * grads
    return loss, image

# This runs gradient ascent for a given image scale (octave).
def gradient_ascent_loop(image, iterations, learning_rate, max_loss=None):
    # Repeatedly update the image in a way that increases the DeepDream loss.
    for i in range(iterations):
        loss, image = gradient_ascent_step(image, learning_rate)
        # Break out if the loss crosses a certain
        # threshold (over-optimizing would create unwanted image artifacts).
        if max_loss is not None and loss > max_loss:
            break
        print(f"... Loss value at step {i}: {loss:.2f}")
    return image

print("Parameters of process")
# Gradient ascent step size
step = 20.
# Number of scales at which to run gradient ascent
num_octave = 3
# Size ratio betwen successive scales
octave_scale = 1.4
# Number of gradient ascent stpes per scale
iterations = 30
# We'll stop the gradient ascent process for a scale if the loss gets higher than this.
max_loss = 15.

print("Listing 12.14 Image processing utilities")
import numpy as np

# Util function to open, resize, and format pictures into appropriate arrays")
def preprocess_image(image_path):
    img = keras.utils.load_img(image_path)
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.inception_v3.preprocess_input(img)
    return img

# Util function to convert a NumPy array into a valid image
def deprocess_image(img):
    img = img.reshape((img.shape[1], img.shape[2], 3))
    # Undo inception v3 processing
    img /= 2.0
    img += 0.5
    img *= 255
    # Convert to uint8 and clip to valid range [0, 255].
    img = np.clip(img, 0, 255).astype("uint8")
    return img

print("Listing 12.15 Running gradient ascent over multiple successive \"octaves\"")
# Load the test image.
original_img = preprocess_image(base_image_path)
original_shape = original_img.shape[1:3]

# Compute the target shape of the image at different octaves.
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]

shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])

# Make a copy of the image
img = tf.identity(original_img)
# Iterate over the different octaves
for i, shape in enumerate(successive_shapes):
    print(f"Processing octave {i} with shape {shape}")
    # Scale up the dream image.
    img = tf.image.resize(img, shape)
    # Run gradient ascent, altering the dream
    img = gradient_ascent_loop(
            img,
            iterations=iterations,
            learning_rate=step,
            max_loss=max_loss
    )
    # Scale up the smaller version of the original image:
    # it wiill be pixelated.
    upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
    # Compute the high-quality version of the original image at this size.
    same_size_original = tf.image.resize(original_img, shape)
    # The difference between the two is the detail that was lost when scaling up.
    lost_detail = same_size_original - upscaled_shrunk_original_img
    img += lost_detail
    shrunk_original_img = tf.image.resize(original_img, shape)

# Save the final result
keras.utils.save_img("dream.png", deprocess_image(img.numpy()))

