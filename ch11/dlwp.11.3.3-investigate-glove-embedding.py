# Suppress warnings
import os, pathlib
from ai_shared_utilities import get_asset_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

DATA_ROOT = get_asset_path("aclImdb")

MODEL_PATH = (
    get_asset_path("glove_embeddings_sequence_model")
)

print("11.3.4 Using pretrained word embeddings")
import tensorflow as tf
from tensorflow import keras
batch_size = 16

train_ds = keras.utils.text_dataset_from_directory(
                DATA_ROOT / "train/",
                batch_size=batch_size)

val_ds = keras.utils.text_dataset_from_directory(
                DATA_ROOT / "val/",
                batch_size=batch_size)

test_ds = keras.utils.text_dataset_from_directory(
                DATA_ROOT / "test/",
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
path_to_glove_file = get_asset_path("glove.6B.100d")

embeddings_index = {}
with open(path_to_glove_file) as f:
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

# Prepare a matrix that we'll fill with the GloVe vectors.
embedding_matrix = np.zeros((max_tokens, embedding_dim))
for word, i in word_index.items():
    if i < max_tokens:
        embedding_vector = embeddings_index.get(word)
    # Fill entry i in the matrix with the word vector for index i.
    # Words not foundin the embedding index will be all zeros.
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print("11.4 Independent investigation of the GloVe word embeddings")

def word_similarity(word1, word2):
    vector1 = embedding_matrix[word_index[word1]]
    vector2 = embedding_matrix[word_index[word2]]
    similarity = np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    return similarity

print("Similarity score between 'cat' and 'dog':")
similarity = word_similarity("cat", "dog")
print(similarity)
print("Similarity score between 'queen' and 'king':")
similarity = word_similarity("queen", "king")
print(similarity)
print("Similarity score between 'princess' and 'prince':")
similarity = word_similarity("princess", "prince")
print(similarity)
print("Similarity score between 'king' and 'prince':")
similarity = word_similarity("king", "prince")
print(similarity)
print("Similarity score between 'queen' and 'princess':")
similarity = word_similarity("queen", "princess")
print(similarity)
print("Similarity score between 'king' and 'good':")
similarity = word_similarity("king", "good")
print(similarity)
print("Similarity score between 'queen' and 'good':")
similarity = word_similarity("queen", "good")
print(similarity)
print("Similarity score between 'queen' and 'beautiful':")
similarity = word_similarity("queen", "beautiful")
print(similarity)
print("Similarity score between 'king' and 'beautiful':")
similarity = word_similarity("king", "beautiful")
print(similarity)
print("Similarity score between 'ugly' and 'beautiful':")
similarity = word_similarity("ugly", "beautiful")
print(similarity)
