"""
chunking/_sentence_utils.py
============================
Shared sentence-splitting utility used by strategies 02, 05, and 10.

Fallback chain (best → safest):
  1. spaCy           – most accurate; needs: pip install spacy
                                             python -m spacy download en_core_web_sm
  2. NLTK punkt_tab  – good; needs: pip install nltk + punkt_tab data download
  3. Regex           – always works; no install needed; handles .!? boundaries
                       (may trip on abbreviations like "Dr." — acceptable fallback)
"""

import re
from typing import List


def split_sentences(text: str, backend: str = "auto") -> List[str]:
    """
    Split text into sentences using the best available backend.

    Args:
        text:    Input text
        backend: "auto" | "spacy" | "nltk" | "regex"
                 "auto" tries spaCy → NLTK → regex automatically.

    Returns:
        List of sentence strings (stripped, non-empty)
    """
    if not text.strip():
        return []

    if backend == "spacy":
        return _try_spacy(text) or _try_nltk(text) or _regex_split(text)
    if backend == "nltk":
        return _try_nltk(text) or _regex_split(text)
    if backend == "regex":
        return _regex_split(text)

    # auto: try each in order
    return (_try_spacy(text)
            or _try_nltk(text)
            or _regex_split(text))


# ── Backends ──────────────────────────────────────────────────────────

def _try_spacy(text: str) -> List[str]:
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        result = [s.text.strip() for s in doc.sents if s.text.strip()]
        return result if result else []
    except Exception:
        return []


def _try_nltk(text: str) -> List[str]:
    try:
        from nltk.tokenize import sent_tokenize
        result = sent_tokenize(text)
        return [s.strip() for s in result if s.strip()]
    except Exception:
        # Try to download punkt_tab (will fail if network is blocked)
        try:
            import nltk
            nltk.download("punkt_tab", quiet=True)
            from nltk.tokenize import sent_tokenize
            result = sent_tokenize(text)
            return [s.strip() for s in result if s.strip()]
        except Exception:
            return []


def _regex_split(text: str) -> List[str]:
    """
    Regex-based sentence splitter.
    Handles .  !  ?  followed by whitespace.
    Avoids splitting on common abbreviations (Mr. Dr. U.S. etc.).
    Always available — no dependencies.
    """
    # Protect common abbreviations
    protected = re.sub(
        r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|i\.e|e\.g|St|Ave|Blvd|U\.S|U\.K)\.',
        r'\1<DOT>',
        text,
    )
    # Split on sentence-ending punctuation
    parts = re.split(r'(?<=[.!?])\s+', protected)
    # Restore protected dots
    parts = [p.replace('<DOT>', '.').strip() for p in parts]
    return [p for p in parts if p]
