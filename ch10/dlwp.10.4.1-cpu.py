import os, pathlib
from deep_learning_with_python.data_paths import get_data_root
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_ROOT = get_data_root()

# 10.4 Advanced use of recurrent neural networks
print("10.4 Advanced use of recurrent neural networks")

#
# Listing 10.1 Inspecting the data of the Jena weather dataset
print("Listing 10.1 Inspecting the data of the Jena weather dataset")
fname = os.path.join(DATA_ROOT / "jena_climate_2009_2016.csv")

with open(fname) as f:
    data = f.read()

lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]

#
# Listing 10.2 Parsing the data
print("Listing 10.2 Parsing the data")
import numpy as np
temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]]
    temperature[i] = values[1]
    raw_data[i, :] = values[:]

#
# Listing 10.5 Computing the number of samples we'll use for each data split
print("Listing 10.5 Computing the number of samples we'll use for each data split")
num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples
print("num_train_samples: ", num_train_samples)
print("num_val_samples: ", num_val_samples)
print("num_test_samples: ", num_test_samples)

#
# Listing 10.6 Normalizing the data
print("Listing 10.6 Normalizing the data")
#print("raw_data[0]: ", raw_data[0])
mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std
#print("raw_data[0]: ", raw_data[0])

#
# Listing 10.7 Instantiating datasets for training, validation, and testing
print("Listing 10.7 Instantiating datasets for training, validation, and testing")

# Force CPU use for keras.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow import keras

sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 -1)
batch_size = 256

train_dataset = keras.utils.timeseries_dataset_from_array(
        data=raw_data[:-delay],
        targets=temperature[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=True,
        batch_size=batch_size,
        start_index=0,
        end_index=num_train_samples)

val_dataset = keras.utils.timeseries_dataset_from_array(
        raw_data[:-delay],
        targets=temperature[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=True,
        batch_size=batch_size,
        start_index=num_train_samples,
        end_index=num_train_samples + num_val_samples)

test_dataset = keras.utils.timeseries_dataset_from_array(
        raw_data[:-delay],
        targets=temperature[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=True,
        batch_size=batch_size,
        start_index=num_train_samples + num_val_samples)

#
# Listing 10.8 Inspecting the output of one of our datasets
print("Listing 10.8 Inspecting the output of one of our datasets")
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
for samples, targets in train_dataset:
    print(f"samples.__class__.__name__: {samples.__class__.__name__}")
    print("sample size: ", samples.size)
    print("samples.shape: ", samples.shape)
    print("targets.shape: ", targets.shape)
    print("samples[0][0]: ", samples[0][0])
    print("targets[0]: ", targets[0])
    break

# 10.4.1 Using recurrent dropout to fight overfitting
print("10.4.1 Using recurrent dropout to fight overfitting")

#
# Listing 10.22 Training and evaluating a dropout-regularized LSTM
print("Listing 10.22 Training and evaluating a dropout-regularized LSTM")
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.LSTM(32, recurrent_dropout=0.2, unroll=True)(inputs)
# To regularize the Dense layer, we also add a Dropout layer after the LSTM.
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()

callbacks = [
        # We use a callback to save the best-performing model
        keras.callbacks.ModelCheckpoint("jena_lstm_dropout_cpu.keras",
            save_best_only=True)
]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
        epochs=50,
        validation_data=val_dataset,
        callbacks=callbacks)

# Reload the best model and evaluate it on test data.
print("Reload the best model and evaluate it on test data.")
model = keras.models.load_model("jena_lstm_dropout_cpu.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")

#
# Listing 10.11 Plotting results
print("Listing 10.11 Plotting results")
import matplotlib.pyplot as plt
loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs[1:], loss[1:], "bo", label="Training MAE")
plt.plot(epochs[1:], val_loss[1:], "b", label="Validation MAE")
plt.title("Training and validation MAE")
plt.legend()
plt.show()

