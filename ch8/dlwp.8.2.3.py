# 8.2.3 Building the model
# Suppress warnings
import os, pathlib
from deep_learning_with_python.data_paths import get_data_root
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#
# Listing 8.7 Instantiating a small convnet for dogs vs cats classification
from tensorflow import keras
from tensorflow.keras import layers

DATA_ROOT = get_data_root() / "cats_vs_dogs_small"

 # 8.2.2
new_base_dir = DATA_ROOT

# The model expects RGB images of size 180 x 180
inputs = keras.Input(shape=(180, 180, 3))
# Rescale inputs to the [0, 1] range by dividing them by 255.
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=512, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

#
# Listing 8.8 Configuring the model for training
model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

#
# Listing 8.9 Using image_dataset_from_directory to read images
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
# Listing 8.10 Displaying the shapes of the data and labels yeilded by the Dataset
for data_batch, labels_batch in train_dataset:
    print("data batch shape: ", data_batch.shape)
    print("labels batch shape: ", labels_batch.shape)
    break

#
# Listing 8.11 Fitting the model using a Dataset
callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="convnet_from_scratch.keras",
            save_best_only=True,
            monitor="val_loss")
]
history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks)

#
# Listing 8.12 Displaying curves of loss and accuracy dur9ing training
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



