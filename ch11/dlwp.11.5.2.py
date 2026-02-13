# Suppress warnings
import os, pathlib
from ai_surgery.data_paths import get_data_root

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("11.5 Beyond text classification: Sequence-to-sequence learning")
print("11.5.1 A machine translation example")
print("Parse the file")
text_file = get_data_root() / "spa-eng/spa.txt"

with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []

# Iterate over the file in lines
for line in lines:
    # Each line contains an English phrase and its Spanish translation, tab-separated.
    english, spanish = line.split("\t")
    # We prepend "[start]" and append "[end]" to the Spanish sentence, to match the template from figure 11.12.
    spanish = "[start] " + spanish + " [end]"
    text_pairs.append((english, spanish))

print("Shuffle and split data into training, validation, and test")
import random
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples]

print(random.choice(text_pairs))

print("Listing 11.6 Vectorizing the English and Spanish text pairs")
import tensorflow as tf
from tensorflow.keras import layers
import string
import re

# Hold ALT+168 from keypad
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
            lowercase, f"[{re.escape(strip_chars)}]", "")

# To keep things simple, we'll only look at the top 15,000 words in each
# language, and we'll restrict sentences to 20 words.
vocab_size = 15000
sequence_length = 20

# The English layer
source_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length,
)
# The Spanish layer
target_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        # Generate Spanish sentences that have one extra token,
        # since we'll need to offset the sentence by one step during training.
        output_sequence_length=sequence_length + 1,
        standardize=custom_standardization
)
train_english_texts = [pair[0] for pair in train_pairs]
train_spanish_texts = [pair[1] for pair in train_pairs]
source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_spanish_texts)

print("Listing 11.27 Preparing datasets for the translation task")
batch_size = 64

def format_dataset(eng, spa):
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)
    return({
        "english": eng,
        # The input Spanish sentence does not include the last token
        # to keep inputs and targets at the same length.
        "spanish": spa[:, :-1],
        # The target Spanish sentence is one step ahead. Both are still the same length.
        }, spa[:, 1:])

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    # Use in-memory caching to speed up preprocessing.
    return dataset.shuffle(2048).prefetch(16).cache()

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

print("Here's what our dataset outputs look like:")
for inputs, targets in train_ds.take(1):
    print(f"inputs['english'].shape: {inputs['english'].shape}")
    print(f"inputs['spanish'].shape: {inputs['spanish'].shape}")
    print(f"targets.shape: {targets.shape}")

print("Listing 11.28 GRU-based encoder")
from tensorflow import keras
from tensorflow.keras import layers

embed_dim = 256
latent_dim = 1024

# The English source sentence goes here. Specifying the name of the input
# enables us to fit() the model with a dict of inputs.
source = keras.Input(shape=(None,), dtype="int64", name="english")
# Don't forget masking: it's critical in this setup.
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(source)
# Our encoded source sentence is the last output of a bidirectional GRU.
encoded_source = layers.Bidirectional(layers.GRU(latent_dim), merge_mode="sum")(x)

print("Listing 11.29 GRU-based decoder and the end-to-end model")

# The Spanish target goes here
past_target = keras.Input(shape=(None,), dtype="int64", name="spanish")
# Don't forget masking
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(past_target)
decoder_gru = layers.GRU(latent_dim, return_sequences=True)
# The encoded source sentence serves as the initial state of the decoder GRU.
x = decoder_gru(x, initial_state=encoded_source)
x = layers.Dropout(0.5)(x)
# Predicts the next token
target_next_step = layers.Dense(vocab_size, activation="softmax")(x)
# End-to-end model: maps the source sentence and the target sentence to the target sentence one step in the future
seq2seq_rnn = keras.Model([source, past_target], target_next_step)
seq2seq_rnn.summary()

print("Listing 11.30 Training our recurrent sequence-to-sequence model")
seq2seq_rnn.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
seq2seq_rnn.fit(train_ds, epochs=15, validation_data=val_ds)

print("Listing 11.31 Translating new sentences with our RNN encoder and decoder")
import numpy as np
# Prepare a dict to convert token index predictions to string tokens.
spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20

def decode_sequence(input_sentence):
    tokenized_input_sequence = source_vectorization([input_sentence])
    # Seed token
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sequence = target_vectorization([decoded_sentence])
        # Sample the next token.
        next_token_predictions = seq2seq_rnn.predict(
                [tokenized_input_sequence, tokenized_target_sequence])
        sampled_token_index = np.argmax(next_token_predictions[0, i, :])
        # Convert the next token prediction to a string and append it to the generated sentence.
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        # Exit condition: either hit max length or sample a stop character
        if sampled_token == "[end]":
            break
    return decoded_sentence

test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(20):
    input_sentence = random.choice(test_eng_texts)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))


