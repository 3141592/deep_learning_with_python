# 9.3.2 Residual connections
# Suppress warnings
import os, pathlib
from deep_learning_with_python.data_paths import get_data_root
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#
# Listing 9.2 Residual block where the number of filters changes
print("Listing 9.2 Residual block where the number of filters changes")
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(32, 32, 3))
print(f"inputs.name: {inputs.name}")
print(f"inputs.shape: {inputs.shape}")

x = layers.Conv2D(32, 3, activation="relu")(inputs)
print(f"x.name: {x.name}")
print(f"x.shape: {x.shape}")
# Set aside the residual.
residual = x

# This is the lyaer around which we crate a residual connection: it increases 
# the number of output filters from 32 to 64. Note that we use padding="smae"
# to avoid downsampling due to padding.
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
print(f"x.name: {x.name}")
print(f"x.shape: {x.shape}")

# The residual only had 32 filters, so we use a 1 x 1 Conv2D to 
# project it to the correct shape.
residual = layers.Conv2D(64, 1)(residual)
print(f"residual.name: {residual.name}")
print(f"residual.shape: {residual.shape}")

# Now the block output and the residual have the same shape and can be added.
x = layers.add([x, residual])
print(f"x.name: {x.name}")
print(f"x.shape: {x.shape}")

#
# Listing 9.3 Case where the target block includes a max pooling layer
print("\nListing 9.3 Case where the target block includes a max pooling layer")
inputs = keras.Input(shape=(32, 32, 3))
print(f"inputs.name: {inputs.name}")
print(f"inputs.shape: {inputs.shape}")

x = layers.Conv2D(32, 3, activation="relu")(inputs)
print(f"x.name: {x.name}")
print(f"x.shape: {x.shape}")

# Set aside the residual.
residual = x

# This is the block of two layers around which we create a residual connection: 
# it includes a 2 x 2 max pooling layer. Note that we use padding="same"
# in both the convolution layer and the max pooling layer to
# avoid downsmapling due to padding.
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
print(f"x.name: {x.name}")
print(f"x.shape: {x.shape}")

x = layers.MaxPooling2D(2, padding="same")(x)
print(f"x.name: {x.name}")
print(f"x.shape: {x.shape}")

# We use strides=2 in the residual projection to match the downsampling
# created by the max pooling layer.
residual = layers.Conv2D(64, 1, strides=2)(residual)
print(f"residual.name: {residual.name}")
print(f"residual.shape: {residual.shape}")

# Now the block output and the residual have the same shape and can be added.
x = layers.add([x, residual])
print(f"x.name: {x.name}")
print(f"x.shape: {x.shape}")

# Example of a simple convnet structured into a series of block
print("\nExample of a simple convnet structured into a series of block")

# Destroys the current TF graph and creates a new one. Useful to avoid clutter from old models / layers.
keras.backend.clear_session()

inputs = keras.Input(shape=(32, 32, 3))
print(f"inputs.name: {inputs.name}")
print(f"inputs.shape: {inputs.shape}")

x = layers.Rescaling(1./255)(inputs)
print(f"x.name: {x.name}")
print(f"x.shape: {x.shape}")

# Utility function to apply a convolutional block with a
# residual connection, with an option to add max pooling
def residual_block(x, filters, pooling=False):
    residual = x
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    
    if pooling:
        x = layers.MaxPooling2D(2, padding="same")(x)
        # If we use max pooling, we add a strided convolution
        # to project the residual to the expected shape.
        residual = layers.Conv2D(filters, 1, strides=2)(residual)
    elif filters != residual.shape[-1]:
        # If we don't use max pooling, we only project the residual
        # if the number of channels has changed.
        residual = layers.Conv2D(filters, 1)(residual)
    x = layers.add([x, residual])
    return x

# First block
x = residual_block(x, filters=32, pooling=True)
print(f"x.name: {x.name}")
print(f"x.shape: {x.shape}")

# Second block; note the increasing filter count in each block.
x = residual_block(x, filters=64, pooling=True)
print(f"x.name: {x.name}")
print(f"x.shape: {x.shape}")

# The last block doesn't need a max pooling layer, since we will 
# apply global average pooling right after it.
x = residual_block(x, filters=128, pooling=False)
print(f"x.name: {x.name}")
print(f"x.shape: {x.shape}")


# The last block doesn't need a max pooling layer, since we will 
# apply global average pooling right after it.

x = layers.GlobalAveragePooling2D()(x)
print(f"x.name: {x.name}")
print(f"x.shape: {x.shape}")

outputs = layers.Dense(1, activation="sigmoid")(x)
print(f"outputs.name: {outputs.name}")
print(f"outputs.shape: {outputs.shape}")

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()


