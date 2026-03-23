# Suppress warnings / TF noise (even though we aren't using TF here)
import os
import numpy as np
from ai_shared_utilities import get_asset_path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("Parsing the GloVe word-embeddings file (full vocabulary)")

# Pick the GloVe file you want:
# path_to_glove_file = get_data_root() / "glove.6B/glove.6B.100d.txt"
path_to_glove_file = get_asset_path("glove.6B.300d")

embeddings_index: dict[str, np.ndarray] = {}

with open(path_to_glove_file, "r", encoding="utf-8") as f:
    for line in f:
        values = line.rstrip().split(" ")
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = vector

# Infer dim from the first entry
embedding_dim = next(iter(embeddings_index.values())).shape[0]
print(f"Loaded {len(embeddings_index):,} words with dim={embedding_dim}")

print("11.4 Independent investigation of the GloVe word embeddings")


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float | None:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return None
    return float(np.dot(v1, v2) / (norm1 * norm2))


def get_word_vector(word: str) -> np.ndarray | None:
    return embeddings_index.get(word)


def get_closest_words(word: str, top_k: int = 5) -> None:
    print(f"\nGet top {top_k} words similar to '{word}':")

    word_vector = get_word_vector(word)
    if word_vector is None:
        print("Word not found in GloVe vocabulary.")
        return

    similarities: list[tuple[str, float]] = []

    # Brute-force nearest neighbors over the full GloVe vocab.
    # (This is fine for exploration; it's ~400k comparisons.)
    for w, vec in embeddings_index.items():
        if w == word:
            continue

        sim = cosine_similarity(word_vector, vec)
        if sim is None:
            continue

        similarities.append((w, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    for w, sim in similarities[:top_k]:
        print(f"{w}: {sim:.4f}")


# Try a few probes
get_closest_words("good")
get_closest_words("bad")
get_closest_words("ugly")