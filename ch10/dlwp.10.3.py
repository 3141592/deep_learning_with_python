import os, pathlib
from deep_learning_with_python.data_paths import get_data_root
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_ROOT = get_data_root()

# 10.3 Understanding recurrent neural networks
print("10.3 Understanding recurrent neural networks")

#
# Listing 10.15 NumPy implementation of a simple RNN
print("Listing 10.15 NumPy implementation of a simple RNN")
import numpy as np
timesteps = 100
input_features = 32
output_features = 64
inputs = np.random.random((timesteps, input_features))
state_t = np.zeros((output_features,))
# Create random weight matrices
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))
successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b) 
    #print("output_t: ", output_t)
    successive_outputs.append(output_t)
    state_t = output_t
final_output_sequence = np.stack(successive_outputs, axis=0)

for i in final_output_sequence:
    print(i)
