# Suppress warnings
import os, pathlib
from ai_surgery.data_paths import get_data_root

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_ROOT = get_data_root() / "aclImdb"

print("Listing 11.2 Displaying the shapes and dtypes of the first batch")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
batch_size = 32

train_ds = keras.utils.text_dataset_from_directory(
        DATA_ROOT / "train/",
        batch_size=batch_size)

val_ds = keras.utils.text_dataset_from_directory(
        DATA_ROOT / "val/",
        batch_size=batch_size)

test_ds = keras.utils.text_dataset_from_directory(
        DATA_ROOT / "test/",
        batch_size=batch_size)

for inputs, targets in train_ds:
    print("inputs.shape: ", inputs.shape)
    print("inputs.dtype: ", inputs.dtype)
    print("targets.shape: ", targets.shape)
    print("targets.dtype: ", targets.dtype)
    print("inputs[0]: ", inputs[0])
    print("targets[0]: ", targets[0])
    break

print("11.3.2 Processing words as a set: The bag-of-words approach")
print("Listing 11.3 Preprocessing our datasets with a TextVectorization layer")
text_vectorization = TextVectorization(
        # Limit vocabulary to the 20,000 most frequent words.
        max_tokens=20000,
        # Encode the output tokens as multi-hot binary vectors.
        output_mode="multi_hot")
print("Prepare a dataset that only yeilds raw text inputs (no labels).")
text_only_train_ds = train_ds.map(lambda x, y: x)
# Use that dataset to index the dataset vocabulary via the adapt() method.
text_vectorization.adapt(text_only_train_ds)

print("Prepare processed versions of our training, validation, and test dataset.")
binary_1gram_train_ds = train_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE)
binary_1gram_val_ds = val_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE)
binary_1gram_test_ds = test_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE)

print("Listing 11.4 Inspecting the output of our binary unigram dataset")
for inputs, targets in binary_1gram_train_ds:
    print("inputs.shape: ", inputs.shape)
    print("inputs.dtype: ", inputs.dtype)
    print("targets.shape: ", targets.shape)
    print("targets.dtype: ", targets.dtype)
    print("inputs[0]: ", inputs[0])
    print("targets[0]: ", targets[0])
    break

print("Listing 11.5 Our model-building utility")
from tensorflow import keras
from tensorflow.keras import layers

def get_model(max_tokens=20000, hidden_dim =16):
    inputs = keras.Input(shape=(max_tokens,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop",
            loss="binary_crossentropy",
            metrics=["accuracy"])
    return model

print("Listing 11.6 Training and testing the binary unigram model")
model = get_model()
model.summary()
callbacks = [
        keras.callbacks.ModelCheckpoint("binary_1gram.keras",
                                        save_best_only=True)
]
# We call cache() on the datasets to cache them in memory: this way we will only do the preprocessing once,
# during the first epoch, and we'll use the preprocessed texts for the following epochs.
# This can only be done if the data is small enough to fit in memory.
model.fit(binary_1gram_train_ds.cache(),
        validation_data=binary_1gram_val_ds.cache(),
        epochs=10,
        callbacks=callbacks)
model = keras.models.load_model("binary_1gram.keras")
print(f"Test acc: {model.evaluate(binary_1gram_test_ds)[1]:.3f}")

