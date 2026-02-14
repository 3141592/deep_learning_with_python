# Suppress warnings
import os, pathlib
from deep_learning_with_python.data_paths import get_data_root
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_PATH = get_data_root()

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("Listing 11.24 Implementing positional embedding as a subclassed layer")
from tensorflow.keras import layers

class PositionalEmbedding(layers.Layer):
    # A downside of position embeddings is that the sequence
    # length needs to be known in advance.
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        # Prepare an embedding layer for the token indices
        self.token_embeddings = layers.Embedding(
                input_dim=input_dim,
                output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
                # And another one for the token positions.
                input_dim=sequence_length,
                output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    # Like the Embedding layer, this layer should be able to generate a mask so
    # we can ignore padding zeros in the inputs. The compute_mask method will be
    # called automatically by the framework, and the mask will get propogated
    # to the next layer.
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    # Implement serialization so we can save the model.
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config

print("Listing 11.33 The TransformerDecoder")
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim= embed_dim)
        self.attention_2 = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim= embed_dim)
        self.dense_proj = keras.Sequential(
                [layers.Dense(dense_dim, activation="relu"),
                 layers.Dense(embed_dim),]
                )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        # This attribute ensures that the layer will propogate its input mask to its outputs.
        # If you pass a mask to a layer that doesn't implement compute_mask() and that doesn't
        # expose this supports_masking attribute, that's an error.
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
            })
        return config

    print("Listing 11.34 TransformerDecoder method that generates a causal mask")
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
                [tf.expand_dims(batch_size, -1),
                 tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

    print("Listing 11.35 The forward pass of the TransformerDecoder")
    def call(self, inputs, encoder_outputs, mask=None):
        # Retrieve the causal mask
        causal_mask = self.get_causal_attention_mask(inputs)
        # Prepare input mask
        if mask is not None:
            padding_mask = tf.cast(
                    mask[:, tf.newaxis, :], dtype="int32")
            # Merge the two masks together
            padding_mask = tf.minimum(padding_mask, causal_mask)
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            # Pass causal mask to first attention layer
            attention_mask=causal_mask)
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            # Pass causal mask to first attention layer
            attention_mask=padding_mask)
        attention_output_2 = self.layernorm_2(attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)

print("12.1.4 Implementing text generation with Keras")
print("Listing 12.3 Creating a database from text files (one file = one sample)")
import tensorflow as tf
from tensorflow import keras
dataset = keras.utils.text_dataset_from_directory(
        directory=DATA_PATH / "aclImdb",
        label_mode=None,
        batch_size=256)

# Strip the <br /> HTML tag that occurs in many of the reviews.
dataset = dataset.map(lambda x: tf.strings.regex_replace(x, "<br />", " "))
print("list(dataset)[0]:")
print(list(dataset)[0])

print("")
print("Listing 12.4 Preparing a TextVectorization layer")
from tensorflow.keras.layers import TextVectorization

sequence_length = 100
# We'll only consider the 15,000 most common words--
# anything else will be treated as out-of-vocabulary token, "[UNK]".
vocab_size = 15000
text_vectorization = TextVectorization(
        max_tokens=vocab_size,
        # We want to return integer word index sequences.
        output_mode="int",
        # We'll work with inputs and targets of length 100
        # (but since we'll offset the targets by 1, the model will actually see sequences of length 99).
        output_sequence_length=sequence_length,
)
text_vectorization.adapt(dataset)

print("Listing 12.5 Setting up a language modeling dataset")
def prepare_lm_dataset(text_batch):
    # Convert a bunch of texts to a batch of integer sequences.
    vectorized_sequences = text_vectorization(text_batch)
    # Create inputs by cutting off the last word of the sequences.
    x = vectorized_sequences[:, :-1]
    # Create targets by offsetting the sequences by 1.
    y = vectorized_sequences[:, 1:]
    return x, y

lm_dataset = dataset.map(prepare_lm_dataset, num_parallel_calls=4)

print("Listing 12.6 A simple Transformer-based language model")
from tensorflow.keras import layers
embed_dim = 256
latent_dim = 2048
num_heads = 2

inputs = keras.Input(shape=(None,), dtype="int64")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, x)
# Softmax over possible vocabulary words, computer for each output sequence timestep.
outputs = layers.Dense(vocab_size, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop")
model.summary()

print("Listing 12.7 The text-generation callback")
import numpy as np

#Dict that maps word indices back to strings, to be used for text decoding
tokens_index = dict(enumerate(text_vectorization.get_vocabulary()))

# Implements variable-temperature sampling, to be used for text decoding
def sample_next(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype("float64")
    #print("== sample_next with temperature: ", temperature)
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

class TextGenerator(keras.callbacks.Callback):
    def __init__(self,
                  prompt,
                  # How many words to generate
                  generate_length,
                  model_input_length,
                  # Range of temperatures to use for sampling
                  temperatures=(1.,),
                  print_freq=1):
        self.prompt = prompt
        self.generate_length = generate_length
        self.model_input_length = model_input_length
        self.temperatures = temperatures
        self.print_freq = print_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_freq != 0:
            return
        for temperature in self.temperatures:
            print("== Generating with temperature", temperature)
            # When generating text, we start fromn our prompt.
            sentence = self.prompt
            for i in range(self.generate_length):
                # Feed the current sequence into our model.
                tokenized_sentence = text_vectorization([sentence])
                predictions = self.model(tokenized_sentence)
                # Retrieve the predictions for the last timestep,
                # and use them to sample a new word.
                next_token = sample_next(predictions[0, i, :], temperature)
                sampled_token = tokens_index[next_token]
                # Append the new word to the current sequence and repeat.
                sentence += " " + sampled_token
            print(sentence)

prompt = "This movie"
text_gen_callback = TextGenerator(
        prompt,
        generate_length=50,
        model_input_length=sequence_length,
        # We'll use a diverse range of temperatures to sample text,
        # to demonstrate the effect of temperature on text generation.
        temperatures=(0.2, 0.5, 0.7, 1., 1.5))

print("Listing 12.8 Fitting the language model")
model.fit(lm_dataset, epochs=20, callbacks=[text_gen_callback])


