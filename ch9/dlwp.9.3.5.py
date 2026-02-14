# 9.3.5 Putting it together: A mini Xception-like model
# Suppress warnings
import os, pathlib
from deep_learning_with_python.data_paths import get_data_root
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 8.2.2
new_base_dir = get_data_root() / "cats_vs_dogs_small"

#
# Listing 8.9 Using image_dataset_from_directory to read images
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

train_dataset = image_dataset_from_directory(
        new_base_dir / "train",
        image_size=(180, 180),
        batch_size=32)
validation_dataset = image_dataset_from_directory(
        new_base_dir / "validation",
        image_size=(180, 180),
        batch_size=32)
test_dataset = image_dataset_from_directory(
        new_base_dir / "test",
        image_size=(180, 180),
        batch_size=32)

#
# Listing 8.14 Define a data augtmentation stage to add to an image model
data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
        ]
)

#
# Listing 8.15 Displaying some randomly augmented training images
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
# We can use take(N) to only sample N batches from the dataset.
# This is equivalent to inserting a break in the loop after the Nth batch.
for images, _ in train_dataset.take(1):
    for i in range(9):
        # Apply the augmentationstage to the batch of images.
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        # Display the first image in the output batch.
        # For each of the nine iterations, this is a different augmentation of the same image.
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
plt.figure()        
#plt.show()

# Replace model definition with the following convnet
inputs = keras.Input(shape=(180, 180, 3))

# We use the same data augmentation configuration as before.
x = data_augmentation(inputs)

# Don't forget the inpupt rescaling!
x = layers.Rescaling(1./255)(x)

# Note the assumption that underlies separable convolution, "feature channels are largely independent,"
# does not hold for RGB images! Red, green and blue color channels are actually highly correlated in natural images.
# As such, the first layer in our model is a regular Conv2D layer. We'll start using SeparableConv2D afterwards.
x = layers.Conv2D(filters=32, kernel_size=5, use_bias=False)(x)

# We apply a series of convolutional blocks with increasing feature depth.
# Each block consists of two batch-normalized depthwise separable conviolution layers
# and a max pooling layer, with a residual connection around the entire block.
for size in [32, 64, 128, 256, 512]:
    residual = x

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    residual = layers.Conv2D(size, 1, strides=2, padding="same", use_bias=False)(residual)
    x = layers.add([x, residual])

# In the original model, we used a Flatten layer before the Dense layer. Here we go with a GlobalAveragePooling2D layer.
x = layers.GlobalAveragePooling2D()(x)
# Like the original model, we add a dropout layer for regularization.
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(loss="binary_crossentropy",
            optimizer="rmsprop",
            metrics=["accuracy"])

#
# Listing 8.17 Training the regularized convnet
callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="convnet_with_xception_architecture.keras",
            monitor="val_loss",
            save_best_only=True)
]
history = model.fit(
        train_dataset,
        epochs=100,
        validation_data=validation_dataset,
        callbacks=callbacks)

#
# Figure 8.11 Training and validation metrics with data augmentation
import matplotlib.pyplot as plt
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Train Acc")
plt.plot(epochs, val_accuracy, "b", label="Val Acc")
plt.title("Training and validation accuracy/loss")
plt.plot(epochs, loss, "ro", label="Train Loss")
plt.plot(epochs, val_loss, "r", label="Val Loss")
plt.legend()
plt.show()

#
# Listing 8.18 Evaluating the model on the test set
print("Listing 8.18 Evaluating the model on the test set.")
test_model = keras.models.load_model("convnet_with_xception_architecture.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


