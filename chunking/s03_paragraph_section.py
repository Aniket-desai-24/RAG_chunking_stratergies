"""
Strategy 03: Paragraph / Section-Based Chunking
================================================
Split at natural document boundaries: blank lines (paragraphs) or
heading markers (Markdown sections). No external libraries required
for the core logic.

Requirements: pip install nltk  (only for the large-paragraph fallback)
No API key needed.
"""

import re
from typing import List, Dict


def paragraph_chunk(
    text: str,
    max_chars: int = 1500,
    min_chars: int = 50,
) -> List[str]:
    """
    Split text on blank lines. Paragraphs exceeding max_chars are
    further split by sentences (NLTK fallback).

    Args:
        text:      Raw input text
        max_chars: Maximum characters per chunk before fallback splitting
        min_chars: Skip paragraphs shorter than this (e.g., single-word headers)

    Returns:
        List of text chunks
    """
    if not text.strip():
        return []

    # Split on one or more blank lines
    raw_paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks: List[str] = []

    for para in raw_paragraphs:
        para = para.strip()
        if len(para) < min_chars:
            continue

        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # Fallback: split large paragraph by sentence
            chunks.extend(_split_by_sentences(para, max_chars))

    return chunks


def _split_by_sentences(text: str, max_chars: int) -> List[str]:
    """
    Break a large block of text into sentence-grouped sub-chunks
    each fitting within max_chars. Uses NLTK → regex fallback.
    """
    from chunking._sentence_utils import split_sentences
    sentences = split_sentences(text)

    sub_chunks: List[str] = []
    current = ""

    for s in sentences:
        if len(current) + len(s) + 1 > max_chars:
            if current:
                sub_chunks.append(current.strip())
            current = s
        else:
            current = (current + " " + s).strip()

    if current:
        sub_chunks.append(current.strip())

    return sub_chunks


# ── Markdown section-aware chunking ──────────────────────────────────

def markdown_section_chunk(
    markdown_text: str,
    max_chars: int = 2000,
    include_heading_in_chunk: bool = True,
) -> List[Dict]:
    """
    Chunk Markdown by heading sections.
    Each chunk includes its heading and full breadcrumb path.

    Args:
        markdown_text:            Raw markdown string
        max_chars:                Max chars before further splitting a section
        include_heading_in_chunk: Prepend the heading to each chunk text

    Returns:
        List of dicts:
            text         – chunk content sent to embedding model
            heading      – section heading
            level        – heading depth (1–6)
            breadcrumb   – full "H1 > H2 > H3" path
    """
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(markdown_text))

    if not matches:
        # No headings found — fall back to paragraph chunking
        return [{"text": c, "heading": "", "level": 0, "breadcrumb": ""}
                for c in paragraph_chunk(markdown_text, max_chars)]

    chunks: List[Dict] = []
    heading_stack: List[Dict] = []  # Track ancestor headings

    for i, match in enumerate(matches):
        raw_hashes = match.group(1)
        heading_text = match.group(2).strip()
        level = len(raw_hashes)

        # Section content runs until the next heading (or end of doc)
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else None
        section_content = markdown_text[content_start:content_end].strip()

        # Update heading stack
        heading_stack = [h for h in heading_stack if h["level"] < level]
        heading_stack.append({"level": level, "text": heading_text})
        breadcrumb = " > ".join(h["text"] for h in heading_stack)

        # Build chunk text
        prefix = f"{heading_text}\n\n" if include_heading_in_chunk else ""
        full_text = prefix + section_content

        if len(full_text) <= max_chars:
            chunks.append({
                "text": full_text,
                "heading": heading_text,
                "level": level,
                "breadcrumb": breadcrumb,
            })
        else:
            # Split oversized section further
            for sub in _split_by_sentences(section_content, max_chars):
                chunks.append({
                    "text": f"{heading_text}\n\n{sub}" if include_heading_in_chunk else sub,
                    "heading": heading_text,
                    "level": level,
                    "breadcrumb": breadcrumb,
                })

    return chunks


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_plain = """
    Artificial intelligence is changing the world rapidly.
    It touches nearly every industry and profession today.

    Machine learning is a subset of AI that learns from data.
    Neural networks are inspired by the structure of the human brain.
    Deep learning uses many layers to extract complex features.

    The history of AI dates back to the 1950s with pioneers like Alan Turing.
    Early AI was rule-based and could not handle uncertainty well.
    Modern AI uses statistical methods and vast datasets.
    """

    sample_markdown = """
# Introduction to AI

Artificial intelligence is the simulation of human intelligence by machines.

## Machine Learning

Machine learning allows systems to learn without being explicitly programmed.
It improves automatically through experience and data.

### Supervised Learning

In supervised learning, models are trained on labelled examples.
The model learns to map inputs to correct outputs.

### Unsupervised Learning

Unsupervised learning finds hidden patterns without labels.
Clustering and dimensionality reduction are common techniques.

## Deep Learning

Deep learning uses many-layered neural networks to learn representations.
It excels at image recognition, NLP, and speech processing.
    """

    print("=" * 60)
    print("Strategy 03: Paragraph / Section-Based Chunking")
    print("=" * 60)

    para_chunks = paragraph_chunk(sample_plain, max_chars=300)
    print(f"\n[Paragraph] max_chars=300 → {len(para_chunks)} chunks")
    for i, c in enumerate(para_chunks):
        print(f"  Chunk {i+1}: {c[:80]}...")

    print()
    md_chunks = markdown_section_chunk(sample_markdown)
    print(f"[Markdown sections] → {len(md_chunks)} chunks")
    for c in md_chunks:
        print(f"  [{c['breadcrumb']}] {c['text'][:60].strip()}...")
