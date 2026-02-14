# Suppress warnings
import os, pathlib
from pathlib import Path
from deep_learning_with_python.data_paths import get_data_root
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", Path.home() / "src" / "data"))

MODEL_PATH = (
    get_data_root()
    / "models"
    / "encoder.keras"
)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("12.4.4 Implementing a VAE with Keras")

print("Listing 12.24 VAE encoder network")
from tensorflow import keras
from tensorflow.keras import layers

# Dimensionality of the latent space: a 2D plane.
latent_dim = 2

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
# The input image ends up being encoded into these two parameters.
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
encoder.summary()

print("Listing 12.25 Latent-space-sampling layer")
import tensorflow as tf

class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        # Draw a batch of random normal vectors.
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        # Apply the VAE sampling formula.
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

print("Listing 12.27 VAE model with custom train_step()")
class VAE(keras.Model):
    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.sampler = Sampler()
        # We use these metrics to keep track of the loss averages over each epoch.
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        #self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss_tracker")
        #self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    # We list the metrics in metrics property to enable the model to reset
    # them after each epoch (or between mutiple calss to fit()/evaluate()).
    def metrics(self):
        return [self.total_loss_tracker]#,
                #self.reconstruction_loss_tracker],
                #self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean, z_log_var)
            #reconstruction = decoder(z)
            # We sum the reconstruction loss over the spatial dimensions
            # (axes 1 and 2) and take its mean over the batch dimension.
            #reconstruction_loss = tf.reduce_mean(
            #        tf.reduce_sum(
            #            keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            #            )
            #        )
            # Add the regularization term (Kullback-Leibler divergence).
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = tf.reduce_mean(kl_loss)
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            #self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            #self.kl_loss_tracker.update_state(kl_loss)
            return {
                    "total_loss": self.total_loss_tracker.result(),
                    #"reconstruction_loss": self.reconstruction_loss_tracker.result()#,
                    #"kl_loss": self.kl_loss_tracker.result(),
                    }

print("Listing 12.28 Training the VAE")
import numpy as np

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
# We train on all MNIST digits, so we concatenate
# the training and test samples.
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

vae = VAE(encoder)
# Note that we don't pass a loss argument in compile(), since the loss is already part of the train_step().
vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
# Note that we don't pass targets in fit(), since train_step doesn't expect any.
vae.fit(mnist_digits, epochs=30, batch_size=128, callbacks=[callbacks])

print("Listing 12.29 Sampling a grid of images from the 2D latent space")
import matplotlib.pyplot as plt

# We'll display a grid of 30 x 30 digits (900 digits total).
n = 30
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# Sample points linearly on a 2D grid.
grid_x = np.linspace(-1, 1, n)
grid_y = np.linspace(-1, 1, n)[::-1]

