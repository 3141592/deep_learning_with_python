# Suppress warnings / TF noise (even though we aren't using TF here)
import os
import numpy as np
from ai_shared_utilities import get_asset_path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("Parsing the fastText word-embeddings file (full vocabulary)")

path_to_vec_file = get_asset_path("wiki-news-300d-1M")
    
embeddings_index: dict[str, np.ndarray] = {}

with open(path_to_vec_file, "r", encoding="utf-8", newline="\n") as f:
    header = f.readline().strip().split()
    if len(header) == 2 and header[0].isdigit() and header[1].isdigit():
        expected_vocab = int(header[0])
        embedding_dim = int(header[1])
    else:
        # No header (unexpected for this file) — treat it as a vector line
        raise ValueError("fastText .vec header missing or malformed")

    for line in f:
        parts = line.rstrip().split(" ", maxsplit=embedding_dim)
        if len(parts) != embedding_dim + 1:
            # Skip malformed lines
            continue
        word = parts[0]
        vector = np.asarray(parts[1:], dtype="float32")
        embeddings_index[word] = vector

print(f"Loaded {len(embeddings_index):,} words (expected ~{expected_vocab:,}) with dim={embedding_dim}")
print("11.4 Independent investigation of the fastText word embeddings")


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
        print("Word not found in fastText vocabulary.")
        return

    similarities: list[tuple[str, float]] = []

    # Brute-force nearest neighbors over the full fastText vocab (~1M vectors).
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

def closest_words_to_vector(target_vec, top_k=5, exclude=None):
    if exclude is None:
        exclude = set()

    exclude_lower = {w.lower() for w in exclude}

    sims = []
    for w, vec in embeddings_index.items():
        if w in exclude:
            continue
        if w.lower() in exclude_lower:   # filters Princess vs princess etc.
            continue

        sim = cosine_similarity(target_vec, vec)
        if sim is None:
            continue
        sims.append((w, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]

def analogy(a: str, b: str, c: str, top_k: int = 5):
    # a - b + c
    va, vb, vc = get_word_vector(a), get_word_vector(b), get_word_vector(c)
    if va is None or vb is None or vc is None:
        print("One of the words is missing from the vocabulary.")
        return

    target = va - vb + vc
    results = closest_words_to_vector(target, top_k=top_k, exclude={a, b, c})
    print(f"\nAnalogy: {a} - {b} + {c} ≈")
    for w, sim in results:
        print(f"{w}: {sim:.4f}")

def analogy_abc(a: str, b: str, c: str, top_k: int = 10):
    """
    a : b :: c : ?
    returns nearest neighbors to vec(b) - vec(a) + vec(c)
    """
    va, vb, vc = get_word_vector(a), get_word_vector(b), get_word_vector(c)
    if va is None or vb is None or vc is None:
        print("One of the words is missing from the vocabulary.")
        return

    target = vb - va + vc

    results = []
    for w, vec in embeddings_index.items():
        # keep only clean lowercase alphabetic tokens
        if not w.isalpha() or not w.islower():
            continue

        # optional: drop simple plurals to reduce 'doctors/physicians/programmers'
        if w.endswith("s") and len(w) > 3:
            continue

        if w in {a, b, c}:
            continue
        sim = cosine_similarity(target, vec)
        if sim is None:
            continue
        results.append((w, sim))

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\nAnalogy: {a} : {b} :: {c} : ?")
    for w, sim in results[:top_k]:
        print(f"{w}: {sim:.4f}")

# Try a few probes
get_closest_words("good")

#analogy("king", "man", "woman")
analogy_abc("man", "king", "woman")
analogy_abc("man", "programmer", "woman")
analogy_abc("man", "doctor", "woman")

