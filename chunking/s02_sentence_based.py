"""
Strategy 02: Sentence-Based Chunking
======================================
Split text at sentence boundaries, then group N sentences per chunk.

Requirements: pip install nltk
Run once: python -c "import nltk; nltk.download('punkt_tab')"

Optional (better accuracy): pip install spacy
                             python -m spacy download en_core_web_sm
No API key needed.
"""

from typing import List


def _split_sentences_nltk(text: str) -> List[str]:
    """Split using NLTK → regex fallback."""
    from chunking._sentence_utils import split_sentences
    return split_sentences(text, backend="nltk")


def _split_sentences_spacy(text: str) -> List[str]:
    """Split using spaCy → NLTK → regex fallback."""
    from chunking._sentence_utils import split_sentences
    return split_sentences(text, backend="spacy")


def sentence_chunk(
    text: str,
    sentences_per_chunk: int = 3,
    sentence_overlap: int = 1,
    backend: str = "nltk"
) -> List[str]:
    """
    Group sentences into chunks of N sentences with optional overlap.

    Args:
        text:                Input text
        sentences_per_chunk: How many sentences per chunk
        sentence_overlap:    Sentences shared between consecutive chunks
        backend:             "nltk" (default) or "spacy" (more accurate)

    Returns:
        List of text chunks
    """
    if not text.strip():
        return []

    # Split into sentences
    if backend == "spacy":
        try:
            sentences = _split_sentences_spacy(text)
        except Exception:
            print("spaCy not available, falling back to NLTK")
            sentences = _split_sentences_nltk(text)
    else:
        sentences = _split_sentences_nltk(text)

    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    step = sentences_per_chunk - sentence_overlap
    if step <= 0:
        raise ValueError("sentence_overlap must be less than sentences_per_chunk")

    chunks = []
    for i in range(0, len(sentences), step):
        group = sentences[i: i + sentences_per_chunk]
        if group:
            chunks.append(" ".join(group))

    return chunks


def sentence_chunk_with_metadata(
    text: str,
    sentences_per_chunk: int = 3,
    sentence_overlap: int = 1,
) -> List[dict]:
    """
    Same as sentence_chunk but returns dicts with index metadata.

    Returns:
        List of {"text": str, "start_sentence": int, "end_sentence": int}
    """
    sentences = _split_sentences_nltk(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    step = sentences_per_chunk - sentence_overlap
    results = []
    for i in range(0, len(sentences), step):
        group = sentences[i: i + sentences_per_chunk]
        if group:
            results.append({
                "text": " ".join(group),
                "start_sentence": i,
                "end_sentence": min(i + sentences_per_chunk - 1, len(sentences) - 1),
                "sentence_count": len(group),
            })
    return results


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    The sky is a vast expanse of blue that stretches across the horizon.
    Clouds drift lazily overhead, pushed by a gentle breeze.
    Birds sing their morning songs from the treetops.
    The sun climbs higher as the day begins to warm.
    Children play in the park below, their laughter filling the air.
    A dog chases a ball across the grass with boundless energy.
    Flowers bloom in every colour imaginable along the garden paths.
    The world feels peaceful and full of quiet promise.
    """

    print("=" * 60)
    print("Strategy 02: Sentence-Based Chunking")
    print("=" * 60)

    chunks = sentence_chunk(sample, sentences_per_chunk=2, sentence_overlap=1)
    print(f"\n[NLTK] sentences_per_chunk=2, overlap=1")
    print(f"  Total chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i+1}: {c[:90]}...")

    print()
    meta = sentence_chunk_with_metadata(sample, sentences_per_chunk=2, sentence_overlap=0)
    print("[With metadata]")
    for m in meta[:3]:
        print(f"  Sentences {m['start_sentence']}–{m['end_sentence']}: {m['text'][:60]}...")
