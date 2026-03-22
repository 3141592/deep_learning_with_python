# 7.3.2 Using callbacks

#
# Listing 7.17 The standard workflow: compile(), fit(), evaluate(), predict()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import keras.backend as K
from ai_shared_data import ensure_asset, get_asset_path, get_data_home

MODEL_PATH = get_data_home() / "models" / "checkpoint_path.keras"

# Create a model (we factor this into a separate function so as to reuse it later).
def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model

# Load your data, reserving some for validation
(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

#
# Listing 7.19 Using the callbacks argument in the fit() method

# Callbacks are passed to the model via the callbacks argument in fit(),
# which takes a list of callbacks. You can pass any number of callbacks.
callbacks_list = [
    # Interrupts training when improvement stops
    keras.callbacks.EarlyStopping(
        # Monitors the model's validation accuracy
        monitor="val_accuracy",
        # Interrupts training when accuracy has stopped improving for two epochs
        patience=2,
    ),
    # Saves the current weights after every epoch
    keras.callbacks.ModelCheckpoint(
        # Path to the destination model file
        filepath=MODEL_PATH,
        # These two arguments mean you won't overwrite the model file unless val_loss
        # has improved, which allows you to keep the best model seen during training.
        monitor="val_loss",
        save_best_only=True,
    )
]

# Compile the model by specifying its optimizer, 
# the loss function to minimize,
# and the metrics to monitor.
model = get_mnist_model()
model.compile(optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
print("About to model.fit()")
model.summary()

# Use fit() to train the model, optionally providing
# validation data to monitor performance on unseen data.
model.fit(train_images,
        train_labels,
        epochs=10,
        callbacks=callbacks_list,
        validation_data=(val_images, val_labels))


