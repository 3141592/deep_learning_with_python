# 8.3.1 Feature extraction with a pretrained model
# Suppress warnings
import os, pathlib
from deep_learning_with_python.data_paths import get_data_root
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 8.2.2
new_base_dir = pathlib.Path(get_data_root()) / "cats_vs_dogs_small"

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
# Listing 8.23 Instantiating and freezing the VGG16 convolutional base
print("Listing 8.23 Instantiating and freezing the VGG16 convolutional base")
conv_base = keras.applications.vgg16.VGG16(
        weights="imagenet",
        include_top=False)
conv_base.trainable = True
print(f"conv_base.trainable: {conv_base.trainable}")
print("This is the number of trainable weights before freezing the conv base: ", len(conv_base.trainable_weights))
conv_base.trainable = False
print("This is the number of trainable weights after freezing the conv base: ", len(conv_base.trainable_weights))

#
# Listing 8.25 Adding a data augmentation stage and a classifier to the convolutional base
print("Listing 8.25 Adding a data augmentation stage and a classifier to the convolutional base")
data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2)
        ]
)

inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = keras.applications.vgg16.preprocess_input(x)
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="binary_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"])
model.summary()

callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="feature_extraction_with_data_augmentation.keras",
            save_best_only=True,
            monitor="val_loss")
]
history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=validation_dataset,
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

# 
# Listing 8.26 Evaluating the model on the test set
print("Listing 8.26 Evaluating the model on the test set")
test_model = keras.models.load_model("feature_extraction_with_data_augmentation.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")
print(f"Test loss: {test_loss:.3f}")

#
# 8.3.2 Fine-tuning a pretrained model
#
# Listing 8.27 Freezing all layers until the fourth from the last
print("Listing 8.27 Freezing all layers until the fourth from the last")
conv_base.trainable = True
for layer in conv_base.layers[:-4]:
    layer.trainable = False

#
# Listing 8.28 Fine-tuning the model
print("Listing 8.28 Fine-tuning the model")
model.compile(loss="binary_crossentropy",
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
        metrics=["accuracy"])

callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="fine_tuning.keras",
            save_best_only=True,
            monitor="val_loss")
]
history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=validation_dataset,
        callbacks=callbacks)

model = keras.models.load_model("fine_tuning.keras")

test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")
print(f"Test loss: {test_loss:.3f}")





