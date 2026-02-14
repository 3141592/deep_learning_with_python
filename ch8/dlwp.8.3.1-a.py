# 8.3.1 Feature extraction with a pretrained model
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
# Listing 8.19 Instantiating the VGG16 convolutional base
print("Listing 8.19 Instantiating the VGG16 convolutional base")
conv_base = keras.applications.vgg16.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(180, 180, 3))

conv_base.summary()

#
# Listing 8.20 Extracting the VGG16 features and corresponding labels
print("Listing 8.20 Extracting the VGG16 features and corresponding labels")
import numpy as np

def get_features_and_labels(dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = keras.applications.vgg16.preprocess_input(images)
        features = conv_base.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)

train_features, train_labels = get_features_and_labels(train_dataset)
val_features, val_labels = get_features_and_labels(validation_dataset)
test_features, test_labels = get_features_and_labels(test_dataset)

print(f"train_features.shape: {train_features.shape}")
print(f"train_labels.shape: {train_labels.shape}")

#
# Listing 8.21 Defining and training the densely connected classifier
print("Listing 8.21 Defining and training the densely connected classifier")
inputs = keras.Input(shape=(5, 5, 512))
# Note use of Flatten layer before passing the features to a Dense layer.
x = layers.Flatten()(inputs)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()

model.compile(loss="binary_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"])

callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="feature_extraction.keras",
            save_best_only=True,
            monitor="val_loss")
]
history = model.fit(
        train_features,
        train_labels,
        epochs=20,
        validation_data=(val_features, val_labels),
        callbacks=callbacks)

#
# Listing 8.22 Plotting the results
print("Listing 8.22 Plotting the results")
import matplotlib.pyplot as plt
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation less")
plt.title("Training and validation loss")
plt.legend()
plt.show()


