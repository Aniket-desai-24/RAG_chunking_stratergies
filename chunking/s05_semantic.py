"""
Strategy 05: Semantic Chunking
================================
Detect topic shifts using cosine similarity between adjacent sentence
embeddings. Split where similarity drops below a threshold.

Requirements:
    pip install numpy nltk

For embeddings choose ONE of:
  A) OpenAI (cloud, best quality):
       pip install openai
       Set OPENAI_API_KEY in your environment or .env file

  B) sentence-transformers (local, free, no API key):
       pip install sentence-transformers

The script auto-detects which is available and falls back gracefully.
"""

import os
import re
from typing import List, Tuple

import numpy as np


# ── Embedding backends ────────────────────────────────────────────────

def _embed_openai(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed via OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(input=texts, model=model)
    return np.array([d.embedding for d in response.data])


def _embed_local(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed locally using sentence-transformers (no API key needed)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def _embed_mock(texts: List[str]) -> np.ndarray:
    """
    Deterministic mock embeddings for unit-testing without any API.
    NOT useful for real retrieval — similarity values are random.
    """
    rng = np.random.default_rng(seed=42)
    vecs = rng.random((len(texts), 64))
    # Normalise
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Auto-select best available embedding backend:
      1. OpenAI (if OPENAI_API_KEY is set)
      2. sentence-transformers (if installed)
      3. Mock (always available — warns user)
    """
    if os.getenv("OPENAI_API_KEY"):
        try:
            return _embed_openai(texts)
        except Exception as e:
            print(f"[warn] OpenAI embedding failed: {e}. Trying local model...")

    try:
        return _embed_local(texts)
    except ImportError:
        pass

    print("[warn] No embedding backend found. Using mock embeddings (not suitable for production).")
    print("       Install: pip install sentence-transformers  OR  set OPENAI_API_KEY")
    return _embed_mock(texts)


# ── Cosine similarity ─────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Core algorithm ────────────────────────────────────────────────────

def semantic_chunk(
    text: str,
    similarity_threshold: float = 0.75,
    buffer_size: int = 1,
) -> List[str]:
    """
    Split text into semantically coherent chunks by detecting topic
    shifts using cosine similarity between adjacent sentence embeddings.

    Args:
        text:                 Input text
        similarity_threshold: Similarity below this value → new chunk (0–1)
        buffer_size:          Sentences included on each side when computing
                              similarity (smooths noisy short sentences)

    Returns:
        List of text chunks
    """
    # Step 1: Sentence tokenise
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return sentences

    # Step 2: Create buffered windows for embedding
    buffered = []
    for i in range(len(sentences)):
        start = max(0, i - buffer_size)
        end = min(len(sentences), i + buffer_size + 1)
        buffered.append(" ".join(sentences[start:end]))

    # Step 3: Embed all buffered windows
    embeddings = get_embeddings(buffered)

    # Step 4: Compute adjacent similarities
    similarities: List[float] = [
        cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    # Step 5: Find split points
    split_points = [
        i for i, sim in enumerate(similarities)
        if sim < similarity_threshold
    ]

    # Step 6: Build chunks
    chunks: List[str] = []
    start = 0
    for point in split_points:
        chunk = " ".join(sentences[start: point + 1])
        if chunk.strip():
            chunks.append(chunk)
        start = point + 1

    # Remaining sentences
    if start < len(sentences):
        last = " ".join(sentences[start:])
        if last.strip():
            chunks.append(last)

    return chunks if chunks else [text]


def semantic_chunk_with_scores(
    text: str,
    similarity_threshold: float = 0.75,
    buffer_size: int = 1,
) -> Tuple[List[str], List[float]]:
    """
    Same as semantic_chunk but also returns per-boundary similarity scores.
    Useful for tuning the threshold.

    Returns:
        (chunks, similarities)
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return sentences, []

    buffered = [
        " ".join(sentences[max(0, i - buffer_size): min(len(sentences), i + buffer_size + 1)])
        for i in range(len(sentences))
    ]

    embeddings = get_embeddings(buffered)
    similarities = [
        cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    split_points = [i for i, s in enumerate(similarities) if s < similarity_threshold]
    chunks: List[str] = []
    start = 0
    for point in split_points:
        chunk = " ".join(sentences[start: point + 1])
        if chunk.strip():
            chunks.append(chunk)
        start = point + 1
    if start < len(sentences):
        chunks.append(" ".join(sentences[start:]))

    return chunks, similarities


# ── Helpers ───────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    from chunking._sentence_utils import split_sentences
    return split_sentences(text)


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    The solar system consists of the Sun and the celestial objects bound to it.
    There are eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.
    The inner planets are rocky, while the outer planets are gas or ice giants.
    Jupiter is the largest planet and has at least 95 known moons.

    Machine learning is a subfield of artificial intelligence.
    Algorithms learn patterns from data without being explicitly programmed.
    Neural networks are modelled loosely on the human brain.
    Transformers have become the dominant architecture for NLP tasks since 2017.

    The French Revolution began in 1789 and ended in 1799.
    It led to the rise of Napoleon Bonaparte and reshaped European politics.
    The Revolution abolished feudalism and promoted ideals of liberty and equality.
    Its effects are still felt in modern democratic governance.
    """

    print("=" * 60)
    print("Strategy 05: Semantic Chunking")
    print("=" * 60)

    chunks, scores = semantic_chunk_with_scores(sample, similarity_threshold=0.75)
    print(f"\nBoundary similarities: {[round(s, 3) for s in scores]}")
    print(f"Split threshold: 0.75")
    print(f"Total chunks: {len(chunks)}\n")

    for i, c in enumerate(chunks):
        print(f"── Chunk {i+1} ──")
        print(f"  {c[:120].strip()}")
        print()
