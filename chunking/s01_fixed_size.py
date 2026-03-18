"""
Strategy 01: Fixed-Size Chunking
=================================
Split text into equal-sized blocks (by characters or tokens) with optional overlap.

Requirements: pip install tiktoken  (optional, for token-based chunking)
No API key needed.
"""

from typing import List, Optional


def fixed_char_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[str]:
    """
    Split text into fixed-size character chunks with overlap.

    Args:
        text:       Input text to chunk
        chunk_size: Max characters per chunk
        overlap:    Characters repeated from previous chunk

    Returns:
        List of text chunks
    """
    if not text.strip():
        return []

    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be less than chunk_size")

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += step

    return chunks


def fixed_token_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    Split text into fixed-size TOKEN chunks with overlap.
    Uses tiktoken (same tokenizer as OpenAI models).

    Args:
        text:          Input text
        chunk_size:    Max tokens per chunk
        overlap:       Tokens repeated from previous chunk
        encoding_name: Tiktoken encoding ("cl100k_base" for GPT-4/3.5)

    Returns:
        List of text chunks (decoded back to strings)
    """
    try:
        import tiktoken
    except ImportError:
        raise ImportError("Run: pip install tiktoken")

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)

    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be less than chunk_size")

    for i in range(0, len(tokens), step):
        token_chunk = tokens[i: i + chunk_size]
        decoded = enc.decode(token_chunk)
        if decoded.strip():
            chunks.append(decoded)

    return chunks


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "Artificial intelligence is transforming every industry across the globe. "
        "From healthcare to finance, AI systems are automating complex tasks. "
        "Machine learning models can now process vast amounts of data in seconds. "
        "Natural language processing enables computers to understand human speech. "
        "The future of AI holds enormous promise for solving global challenges. "
    ) * 5

    print("=" * 60)
    print("Strategy 01: Fixed-Size Chunking")
    print("=" * 60)

    # Character-based
    char_chunks = fixed_char_chunk(sample, chunk_size=200, overlap=30)
    print(f"\n[Character-based] chunk_size=200, overlap=30")
    print(f"  Total chunks : {len(char_chunks)}")
    print(f"  Chunk 1      : {char_chunks[0][:80]}...")
    print(f"  Chunk 2 start: {char_chunks[1][:30]}...")  # Should overlap with chunk 1

    # Token-based (requires tiktoken)
    try:
        tok_chunks = fixed_token_chunk(sample, chunk_size=50, overlap=10)
        print(f"\n[Token-based] chunk_size=50 tokens, overlap=10")
        print(f"  Total chunks : {len(tok_chunks)}")
        print(f"  Chunk 1      : {tok_chunks[0][:80]}...")
    except ImportError:
        print("\n[Token-based] Skipped — install tiktoken: pip install tiktoken")
