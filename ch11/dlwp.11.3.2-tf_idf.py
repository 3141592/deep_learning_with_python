# Suppress warnings
import os, pathlib
from ai_shared_data import ensure_asset, get_asset_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

basedir = get_asset_path("aclImdb")
MODEL_PATH = get_asset_path("tfidf_2gram")

print("Listing 11.2 Displaying the shapes and dtypes of the first batch")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization

batch_size = 32
seed = 1337
val_split = 0.2  # 20% of train -> val

train_ds = keras.utils.text_dataset_from_directory(
    basedir / "train",
    batch_size=batch_size,
    validation_split=val_split,
    subset="training",
    seed=seed,
)

val_ds = keras.utils.text_dataset_from_directory(
    basedir / "train",
    batch_size=batch_size,
    validation_split=val_split,
    subset="validation",
    seed=seed,
)

test_ds = keras.utils.text_dataset_from_directory(
    basedir / "test",
    batch_size=batch_size,
    shuffle=False,
)

print("11.3.2 Processing words as a set: The bag-of-words approach")
print("Listing 11.10 Configuring TextVectorization to return TF-IDF-weighted outputs")
text_vectorization = TextVectorization(
        ngrams=2,
        max_tokens=20000,
        output_mode="tf_idf")

print("Prepare a dataset that only yields raw text inputs (no labels).")
text_only_train_ds = train_ds.map(lambda x, y: x)

# The adapt() call will learn the TF-IDF weights in addition to the vocabulary. 
text_vectorization.adapt(text_only_train_ds)

print("Listing 11.11 Training and testing the TF-IDF bigram model")
tfidf_2gram_train_ds = train_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE)
tfidf_2gram_val_ds = val_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE)
tfidf_2gram_test_ds = test_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE)

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

model = get_model()
model.summary()
callbacks = [
        keras.callbacks.ModelCheckpoint(MODEL_PATH,
                                        save_best_only=True)
]
# We call cache() on the datasets to cache them in memory: this way we will only do the preprocessing once,
# during the first epoch, and we'll use the preprocessed texts for the following epochs.
# This can only be done if the data is small enough to fit in memory.
model.fit(tfidf_2gram_train_ds.cache(),
        validation_data=tfidf_2gram_val_ds.cache(),
        epochs=10,
        callbacks=callbacks)
model = keras.models.load_model(MODEL_PATH)
print(f"Test acc: {model.evaluate(tfidf_2gram_test_ds)[1]:.3f}")

print("Exporting a model the processes raw strings")
# Our sample would be one string
inputs = keras.Input(shape=(1,), dtype="string")
# Apply text preprocessing
processed_inputs = text_vectorization(inputs)
# Apply the previously trained model
outputs = model(processed_inputs)
# Instantiate the end-to-end model
inference_model = keras.Model(inputs, outputs)
inference_model.summary()
raw_text_data = tf.convert_to_tensor([
    ["That was the best movie I've ever seen."],
])
predictions = inference_model(raw_text_data)
print(f"{float(predictions[0] * 100):.2f} percent positive")


