# Suppress warnings
import os, pathlib
from deep_learning_with_python.data_paths import get_data_root
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_PATH = get_data_root() / "celeba_gan" / "img_align_celeba"

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("Listing 12.31 Creating a dataset from a directory of images")
from tensorflow import keras
dataset = keras.utils.image_dataset_from_directory(
    DATA_PATH,
    # Only the images will be returned--no labels.
    label_mode=None,
    image_size=(64, 64),
    batch_size=32,
)

print("Listing 12.32 Rescaling the images")
dataset = dataset.map(lambda x: x / 255.)

print("Listing 12.33 Displaying the first image")
import matplotlib.pyplot as plt
for x in dataset:
    plt.axis("off")
    plt.imshow((x.numpy() * 255).astype("int32")[0])
    plt.show()
    break

print("Listing 12.34 The GAN discriminator network")
from tensorflow.keras import layers

discriminator = keras.Sequential(
        [
            keras.Input(shape=(64, 64, 3)),
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            # One dropout layer: an important trick!
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )

discriminator.summary()

print("Listing 12.35 GAN generator network")
# The latent space will be made of 128-dimensional vectors.
latent_dim = 128

generator = keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            # Produce the same number of coefficients we had at 
            # the level of the Flatten layer in the encoder.
            layers.Dense(8 * 8 * 128),
            # Revert the Flatten layer of the encoder.
            layers.Reshape((8, 8, 128)),
            # Revert the Conv2D layers of the encoder.
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            # We use LeakyReLU as our activation.
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
        ],
        name="generator",
    )

generator.summary()

print("Listing 12.36 The GAN Model")
import tensorflow as tf
class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        # Set up metrics to track the two losses over each training epoch
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latest space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
                )
        # Decodes them to fake images
        generated_images = self.generator(random_latent_vectors)
        # Combines them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)
        # Assembles labels, discriminating real from fake images
        labels = tf.concat(
                [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))],
                axis=0
                )
        # Adds random noise to the labels--an importabnt trick

        # Trains the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
                )

        # Samples random points in the latent space
        random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim))

        # Assembles labels that say "these are all real images" (it's a lie!)
        misleading_labels = tf.zeros((batch_size, 1))

        # Trains the generator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(
                    self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
                zip(grads, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"d_loss": self.d_loss_metric.result(),
                "g_loss": self.g_loss_metric.result()}

print("Listing 12.37 A callback that samples generated images during training") 
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(
                shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.utils.array_to_img(generated_images[i])
            img.save(f"{DATA_PATH}/generated_img_{epoch:03d}_{i}.png")

print("Listing 12.38 Compiling and training the GAN")
# You'll start getting interesting results after epoch 20.
epochs = 100

gan = GAN(discriminator=discriminator, generator=generator,
        latent_dim=latent_dim)
gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
        )

gan.fit(
        dataset, epochs=epochs,
        callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)]
        )








