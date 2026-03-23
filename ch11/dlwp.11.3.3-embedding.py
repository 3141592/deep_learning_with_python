# Suppress warnings
import os, pathlib
from ai_shared_utilities import get_asset_path, get_asset, save_registered_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

DATA_ROOT = get_asset_path("aclImdb")

MODEL_ARTIFACT = get_asset("one_hot_bidir_gru")

print("11.3.3 Processing words as a sequence: The sequence model approach")
import tensorflow as tf
from tensorflow import keras

batch_size = 16
seed = 1337
val_split = 0.2  # 20% of train -> val

train_ds = keras.utils.text_dataset_from_directory(
    DATA_ROOT / "train",
    batch_size=batch_size,
    validation_split=val_split,
    subset="training",
    seed=seed,
)

val_ds = keras.utils.text_dataset_from_directory(
    DATA_ROOT / "train",
    batch_size=batch_size,
    validation_split=val_split,
    subset="validation",
    seed=seed,
)

test_ds = keras.utils.text_dataset_from_directory(
    DATA_ROOT / "test",
    batch_size=batch_size,
    shuffle=False,
)

print(train_ds.class_names)  # should show ['neg', 'pos']

for texts, labels in train_ds.take(1):
    print(labels[:10].numpy())
    print(texts[0].numpy()[:200])

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

print("11.16 Model that uses an Embedding layer trained from scratch")
inputs = keras.Input(shape=(None,), dtype="int64")
embedded = layers.Embedding(input_dim=max_tokens, output_dim=256)(inputs)
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(MODEL_ARTIFACT.path, save_best_only=True)
]
model.fit(int_train_ds,
        validation_data=int_val_ds,
        epochs=1,
        callbacks=callbacks)
model = keras.models.load_model(MODEL_ARTIFACT.path)
print("Evaluating best checkpoint on test set")
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

save_registered_model(model, MODEL_ARTIFACT)