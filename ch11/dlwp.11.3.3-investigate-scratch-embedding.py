# Suppress warnings
import os, pathlib
from pathlib import Path
import sys
import tensorflow as tf
from tensorflow import keras
from ai_shared_utilities import get_asset_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_ROOT = get_asset_path("aclImdb")

MODEL_PATH = (
    get_asset_path("one_hot_bidir_gru")
)

if not MODEL_PATH.exists():
    print(f"\nError: {MODEL_PATH} does not exist.")
    print("You must run dlwp.11.3.3-embedding.py first to create the model.\n")
    sys.exit(1)

print("11.3.3 Processing words as a sequence: The sequence model approach")
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

model = keras.models.load_model(MODEL_PATH)

print("# --- Embedding exploration: trained-from-scratch embedding ---")

import numpy as np

print("# 1 Get the vocabulary and build word -> index mapping")
vocab = text_vectorization.get_vocabulary()  # list where index = token id
word_to_idx = {w: i for i, w in enumerate(vocab)}

print(f"Vocab size from TextVectorization: {len(vocab):,}")
print("Special tokens (usually):", vocab[:5])  # often: ['', '[UNK]', ...]

print("# 2 Extract embedding matrix W: shape (max_tokens, 256)")
# Find the Embedding layer (safer than assuming layer index)
emb_layer = next((l for l in model.layers if isinstance(l, keras.layers.Embedding)), None)
if emb_layer is None:
    raise RuntimeError("Could not find an Embedding layer in the loaded model.")
W = emb_layer.get_weights()[0]
print("Embedding matrix shape:", W.shape)

def sentiment_score(word, pos_word="great", neg_word="awful"):
    v = get_word_vector(word)
    v_pos = get_word_vector(pos_word)
    v_neg = get_word_vector(neg_word)

    if v is None or v_pos is None or v_neg is None:
        return None

    sentiment_axis = v_pos - v_neg
    return cosine_similarity(v, sentiment_axis)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float | None:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return None
    return float(np.dot(v1, v2) / (n1 * n2))

def get_word_vector(word: str) -> np.ndarray | None:
    idx = word_to_idx.get(word)
    if idx is None:
        return None
    if idx >= W.shape[0]:
        return None
    return W[idx]

def get_closest_words(word: str, top_k: int = 10) -> None:
    print(f"\nGet top {top_k} words similar to '{word}' (scratch embedding):")

    v = get_word_vector(word)
    if v is None:
        print("Word not found in TextVectorization vocabulary.")
        return

    sims: list[tuple[str, float]] = []
    for w, idx in word_to_idx.items():
        # Skip special tokens and the query word
        if w in {"", "[UNK]"} or w == word:
            continue
        if idx >= W.shape[0]:
            continue

        sim = cosine_similarity(v, W[idx])
        if sim is None:
            continue

        sims.append((w, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    for w, sim in sims[:top_k]:
        print(f"{w}: {sim:.4f}")

# Try a few probes (pick words that actually appear in IMDB)
get_closest_words("good", top_k=5)
get_closest_words("bad", top_k=5)
get_closest_words("great", top_k=5)
get_closest_words("terrible", top_k=5)
get_closest_words("no", top_k=5)
get_closest_words("never", top_k=5)
get_closest_words("not", top_k=5)

test_words = [
    "great", "excellent", "wonderful",
    "good",
    "bad", "terrible", "awful", "boring",
    "not", "never", "no", "dont"
]

print("\nSentiment alignment (positive ≈ +, negative ≈ -):")
for w in test_words:
    score = sentiment_score(w)
    if score is not None:
        print(f"{w:12s}: {score:.4f}")

s1 = get_word_vector("great") - get_word_vector("terrible")
s2 = get_word_vector("excellent") - get_word_vector("awful")

axis_cos = cosine_similarity(s1, s2)
print("Cosine similarity between sentiment axes:", axis_cos)

print("The top 10 most aligned with the sentiment axis and the 10 most negatively aligned words.")
def top_aligned_words(pos_word="great", neg_word="terrible", top_k=10, anchor_thresh=0.45):
    v_pos = get_word_vector(pos_word)
    v_neg = get_word_vector(neg_word)
    if v_pos is None or v_neg is None:
        raise ValueError("Missing anchor word(s) in vocabulary.")

    sentiment_axis = v_pos - v_neg

    sims = []  # IMPORTANT: reset each run

    for w, idx in word_to_idx.items():
        # basic hygiene + specials
        if w in {"", "[UNK]"}:
            continue
        if idx >= W.shape[0]:
            continue
        if not w.isalpha() or not w.islower():
            continue
        if len(w) <= 3:
            continue

        # restrict to words in the sentiment neighborhood
        sim_to_pos = cosine_similarity(W[idx], v_pos)
        sim_to_neg = cosine_similarity(W[idx], v_neg)
        if sim_to_pos is None or sim_to_neg is None:
            continue
        if max(sim_to_pos, sim_to_neg) < anchor_thresh:
            continue

        # axis alignment
        sim = cosine_similarity(W[idx], sentiment_axis)
        if sim is None:
            continue

        sims.append((w, sim))

    sims.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_k} aligned with axis ({pos_word} - {neg_word}) [anchor_thresh={anchor_thresh}]")
    for w, s in sims[:top_k]:
        print(f"{w:12s}: {s:.4f}")

    print(f"\nTop {top_k} negatively aligned with axis ({pos_word} - {neg_word}) [anchor_thresh={anchor_thresh}]")
    for w, s in sims[-top_k:][::-1]:
        print(f"{w:12s}: {s:.4f}")


top_aligned_words(top_k=10, anchor_thresh=0.65)