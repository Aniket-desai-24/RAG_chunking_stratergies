"""
Strategy 04: Recursive Character Text Splitting
================================================
Try separators in priority order (\\n\\n → \\n → . → " ") until
chunks are within the desired size. This is LangChain's default splitter.

No external libraries required for the pure implementation.
Optional: pip install langchain langchain-text-splitters  (for the LangChain version)
No API key needed.
"""

from typing import List, Optional


class RecursiveCharacterSplitter:
    """
    Splits text using a hierarchy of separators, recursing to smaller
    separators when a split piece is still too large.

    This matches LangChain's RecursiveCharacterTextSplitter behaviour.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    # Language-specific separator sets
    PYTHON_SEPARATORS = [
        "\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""
    ]
    MARKDOWN_SEPARATORS = [
        "\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
        keep_separator: bool = False,
    ):
        """
        Args:
            chunk_size:     Target maximum characters per chunk
            chunk_overlap:  Characters to repeat at chunk boundaries
            separators:     Custom separator list (highest to lowest priority)
            keep_separator: Re-attach the separator to the start of the
                            following chunk (useful for code)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.keep_separator = keep_separator

    # ── Public API ────────────────────────────────────────────────────

    def split_text(self, text: str) -> List[str]:
        """Split a single string into chunks."""
        return self._split_recursive(text.strip(), self.separators)

    def split_documents(self, texts: List[str]) -> List[str]:
        """Split a list of documents, returning all chunks together."""
        all_chunks: List[str] = []
        for t in texts:
            all_chunks.extend(self.split_text(t))
        return all_chunks

    # ── Internals ─────────────────────────────────────────────────────

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Core recursive split logic."""
        # Base case: already fits
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        for idx, sep in enumerate(separators):
            # Last separator is "" → character-level split
            if sep == "":
                return self._char_split(text)

            if sep not in text:
                continue  # This separator doesn't appear — try next

            splits = text.split(sep)
            if self.keep_separator:
                splits = [splits[0]] + [sep + s for s in splits[1:]]

            merged = self._merge(splits, sep)
            remaining_seps = separators[idx + 1:]

            result: List[str] = []
            for piece in merged:
                if len(piece) > self.chunk_size:
                    result.extend(self._split_recursive(piece, remaining_seps))
                elif piece.strip():
                    result.append(piece)
            return result

        return [text]

    def _merge(self, splits: List[str], separator: str) -> List[str]:
        """
        Greedily merge small split pieces into chunks up to chunk_size,
        then apply overlap by backtracking.
        """
        merged: List[str] = []
        current_pieces: List[str] = []
        current_len = 0

        for piece in splits:
            piece_len = len(piece)
            join_len = len(separator) if current_pieces else 0

            if current_len + join_len + piece_len > self.chunk_size and current_pieces:
                # Flush current group
                merged.append(separator.join(current_pieces))

                # Apply overlap: keep trailing pieces whose combined length
                # is within chunk_overlap
                while current_pieces and current_len > self.chunk_overlap:
                    removed = current_pieces.pop(0)
                    current_len -= len(removed) + len(separator)

            current_pieces.append(piece)
            current_len += piece_len + (len(separator) if len(current_pieces) > 1 else 0)

        if current_pieces:
            merged.append(separator.join(current_pieces))

        return [m for m in merged if m.strip()]

    def _char_split(self, text: str) -> List[str]:
        """Last-resort: split by raw characters with overlap."""
        step = self.chunk_size - self.chunk_overlap
        return [text[i: i + self.chunk_size] for i in range(0, len(text), step)
                if text[i: i + self.chunk_size].strip()]


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
# Introduction

Artificial intelligence is transforming every industry.
From healthcare to finance, AI systems are automating complex tasks.

## Machine Learning

Machine learning is a subset of AI that enables systems to learn from data.
Neural networks, inspired by the brain, power modern deep learning models.

### Supervised Learning

In supervised learning, models train on labelled data.
They learn to map inputs to correct outputs iteratively.

### Unsupervised Learning

Unsupervised learning discovers hidden patterns without labels.

## Deep Learning

Deep learning uses stacked neural network layers to extract rich features.
It excels at image recognition, natural language processing, and speech.
Models like GPT-4 and BERT have billions of parameters.
Training requires massive datasets and GPU clusters.
"""

    print("=" * 60)
    print("Strategy 04: Recursive Character Text Splitting")
    print("=" * 60)

    splitter = RecursiveCharacterSplitter(chunk_size=300, chunk_overlap=40)
    chunks = splitter.split_text(sample)
    print(f"\n[Default separators] chunk_size=300, overlap=40 → {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"\n  ── Chunk {i+1} ({len(c)} chars) ──")
        print(f"  {c[:120].strip()}...")

    # Markdown-optimised
    md_splitter = RecursiveCharacterSplitter(
        chunk_size=300,
        chunk_overlap=40,
        separators=RecursiveCharacterSplitter.MARKDOWN_SEPARATORS,
    )
    md_chunks = md_splitter.split_text(sample)
    print(f"\n[Markdown separators] → {len(md_chunks)} chunks")
    for i, c in enumerate(md_chunks):
        print(f"  Chunk {i+1}: {c[:80].strip()}...")

    # LangChain one-liner (if installed)
    print("\n[LangChain equivalent]")
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        lc = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
        lc_chunks = lc.split_text(sample)
        print(f"  langchain → {len(lc_chunks)} chunks ✓")
    except ImportError:
        print("  Install with: pip install langchain-text-splitters")
