# Suppress warnings
import os, pathlib
from ai_surgery.data_paths import get_data_root

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

DATA_ROOT = get_data_root() / "aclImdb"

print("11.3.3 Processing words as a sequence: The sequence model approach")
import tensorflow as tf
from tensorflow import keras
batch_size = 16

train_ds = keras.utils.text_dataset_from_directory(
                DATA_ROOT / "train",
                batch_size=batch_size)

val_ds = keras.utils.text_dataset_from_directory(
                DATA_ROOT / "val",
                batch_size=batch_size)

test_ds = keras.utils.text_dataset_from_directory(
                DATA_ROOT / "test",
                batch_size=batch_size)

text_only_train_ds = train_ds.map(lambda x, y: x)

print("Listing 11.12 Preparing integer sequence datasets")
from tensorflow.keras import layers

max_length = 600
max_tokens = 20000
text_vectorization = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        # In order to keep a manageable input size, we'll truncate the inputs after the first 600 words.
        output_sequence_length=max_length,
)
text_vectorization.adapt(text_only_train_ds)

int_train_ds = train_ds.map(
                lambda x, y: (text_vectorization(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
int_val_ds = val_ds.map(
                lambda x, y: (text_vectorization(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
int_test_ds = test_ds.map(
                lambda x, y: (text_vectorization(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)

print("11.17 Using an Embedding layer with masking enabled")
inputs = keras.Input(shape=(None,), dtype="int64")
embedded = layers.Embedding(input_dim=max_tokens, output_dim=256, mask_zero=True)(inputs)
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"])
model.summary()

callbacks = [
        keras.callbacks.ModelCheckpoint("one_hot_bidir_gru_with_masking.keras",
            save_best_only=True)
]
model.fit(int_train_ds,
        validation_data=int_val_ds,
        epochs=10,
        callbacks=callbacks)
model = keras.models.load_model("one_hot_bidir_gru_with_masking.keras")
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

