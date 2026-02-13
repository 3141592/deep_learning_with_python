# Suppress warnings
import os, pathlib
from ai_surgery.data_paths import get_data_root

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

DATA_ROOT = get_data_root()

print("11.3.3 Processing words as a sequence: The sequence model approach")
import tensorflow as tf
from tensorflow import keras
batch_size = 16

train_ds = keras.utils.text_dataset_from_directory(
                DATA_ROOT / "aclImdb" / "train", 
                batch_size=batch_size)

val_ds = keras.utils.text_dataset_from_directory(
                DATA_ROOT / "aclImdb" / "val", 
                batch_size=batch_size)

test_ds = keras.utils.text_dataset_from_directory(
                DATA_ROOT / "aclImdb" / "test", 
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

print("Listing 11.18 Parsing the GloVe word-embeddings file")
import numpy as np
path_to_glove_file = "~/src/data/glove.6B/glove.6B.100d.txt"

embeddings_index = {}
with open(DATA_ROOT / "glove.6B" / "glove.6B.100d.txt") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print(f"Found {len(embeddings_index)} word vectors.")

print("Listing 11.19 Preparing the GloVe word-embeddings matrix")
embedding_dim = 100

# Retrieve the vocabulary indexed by our previous TextVectorization layer.
vocabulary = text_vectorization.get_vocabulary()

# Use it to create a mapping from words to their index in the vocabulary.
word_index = dict(zip(vocabulary, range(len(vocabulary))))

# Use it to create a mapping from words to their index in the vocabulary.
word_index = dict(zip(vocabulary, range(len(vocabulary))))

# Prepare a matrix that we'll fill with the GloVe vectors.
train = embeddings_index.get("train")
station = embeddings_index.get("station")
carrot = embeddings_index.get("carrot")
radio = embeddings_index.get("radio")
beet = embeddings_index.get("beet")
attention = embeddings_index.get("attention")
score = embeddings_index.get("score")
space = embeddings_index.get("space")

print("train and station:", train.dot(station))
print("train and carrot: ", train.dot(carrot))
print("radio and station: ", radio.dot(station))
print("carrot and beet: ", carrot.dot(beet))
print("attention and score: ", attention.dot(score))
print("station and train: ", station.dot(train))
print("station and space: ", station.dot(space))
print("dog and wolf: ", embeddings_index.get("dog").dot(embeddings_index.get("wolf")))
print("see and saw: ", embeddings_index.get("see").dot(embeddings_index.get("saw")))
print("station and station: ", embeddings_index.get("station").dot(embeddings_index.get("station")))
print("station and stations: ", embeddings_index.get("station").dot(embeddings_index.get("stations")))

