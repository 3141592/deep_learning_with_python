# 10.2 A temperature forecasting example
import os, pathlib
from deep_learning_with_python.data_paths import get_data_root
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_ROOT = get_data_root()

#
# Listing 10.1 Inspecting the data of the Jena weather dataset
print("Listing 10.1 Inspecting the data of the Jena weather dataset")
import os
fname = os.path.join(DATA_ROOT / "jena_climate_2009_2016.csv")

with open(fname) as f:
    data = f.read()

lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]
print(header)
print(len(lines))

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
# Listing 10.3 Plotting the temperature timeseries
print("Listing 10.3 Plotting the temperature time series")
from matplotlib import pyplot as plt
plt.plot(range(len(temperature)), temperature)
plt.show()

#
# Listing 10.4 Plotting the first 10 days of the temperature timeseries
print("Listing 10.4 Plotting the first 10 days of the temperature timeseries")
plt.plot(range(1440), temperature[:1440])
plt.show()

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

#
# Listing 10.9 Computing the common-sense baseline MAE
print("Listing 10.9 Computing the common-sense baseline MAE")
def evaluate_naive_method(dataset):
    total_abs_err = 0
    samples_seen = 0
    for samples, targets in dataset:
        # The temperature feature is at column 1, so samples[:, -1, 1] is the last temperature measurement
        # in the inputs sequence. Recall that we normalized our features, so to retrieve a temperature
        # in degrees Celcius, we need to un-normalize it by multiplying it by the standard deviation,
        # and adding back the mean.
        preds = samples[:, -1, 1] * std[1] + mean[1]
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += samples.shape[0]
    return total_abs_err / samples_seen

print(f"Validation MAE: {evaluate_naive_method(val_dataset):.2f}")
print(f"Test MAE: {evaluate_naive_method(test_dataset):.2f}")



