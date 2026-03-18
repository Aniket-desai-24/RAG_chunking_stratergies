"""
Strategy 07: Proposition-Based (Agentic) Chunking
===================================================
Use an LLM to decompose text into atomic, self-contained factual propositions.
Each proposition becomes its own retrievable chunk.

Requirements:
    pip install openai          (for OpenAI / GPT)
  OR
    pip install anthropic       (for Claude)

Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment / .env file.

Reference: "Dense X Retrieval" – Chen et al. 2023
           https://arxiv.org/abs/2312.06648
"""

import json
import os
from typing import List

# ── Prompt template ───────────────────────────────────────────────────

PROPOSITION_SYSTEM_PROMPT = """You are an expert at extracting atomic facts from text.

Given a passage, extract ALL factual propositions. Each proposition must be:
1. A SINGLE atomic fact — one claim only, no conjunctions joining two claims
2. SELF-CONTAINED — no pronouns like "he", "she", "it", "they"; replace with full nouns
3. FAITHFUL — only state what the source explicitly says; no inferences
4. A COMPLETE sentence with a subject, verb, and object

Return ONLY a JSON array of strings. No preamble, no markdown fences, no explanation.

Example input:
  "Marie Curie was born in Warsaw in 1867 and later discovered polonium with her husband Pierre."

Example output:
  [
    "Marie Curie was born in Warsaw.",
    "Marie Curie was born in 1867.",
    "Marie Curie discovered polonium.",
    "Pierre Curie co-discovered polonium with Marie Curie."
  ]"""


# ── LLM backends ─────────────────────────────────────────────────────

def _extract_openai(passage: str) -> List[str]:
    """Extract propositions using OpenAI."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",        # Cheap but capable for extraction
        messages=[
            {"role": "system", "content": PROPOSITION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Passage:\n{passage}"},
        ],
        temperature=0.0,            # Deterministic for fact extraction
        max_tokens=1024,
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


def _extract_anthropic(passage: str) -> List[str]:
    """Extract propositions using Anthropic Claude."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",   # Fast and cheap for extraction
        max_tokens=1024,
        system=PROPOSITION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Passage:\n{passage}"}],
    )
    raw = message.content[0].text.strip()
    return json.loads(raw)


def _extract_mock(passage: str) -> List[str]:
    """
    Deterministic mock for testing without an API key.
    Splits on ". " as a crude proposition approximation.
    NOT suitable for production.
    """
    import re
    sentences = re.split(r"(?<=[.!?])\s+", passage.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _extract_propositions(passage: str) -> List[str]:
    """Auto-select best available LLM backend."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            return _extract_openai(passage)
        except Exception as e:
            print(f"[warn] OpenAI extraction failed: {e}")

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return _extract_anthropic(passage)
        except Exception as e:
            print(f"[warn] Anthropic extraction failed: {e}")

    print("[warn] No API key found. Using mock sentence-split (not for production).")
    print("       Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable real extraction.")
    return _extract_mock(passage)


# ── Core algorithm ────────────────────────────────────────────────────

def proposition_chunk(
    text: str,
    passage_size: int = 500,
    passage_overlap: int = 50,
) -> List[str]:
    """
    Decompose text into atomic propositions using an LLM.
    Processes text in passage-sized windows to stay within token limits.

    Args:
        text:            Input document text
        passage_size:    Characters per passage sent to the LLM
        passage_overlap: Characters of overlap between passages

    Returns:
        List of proposition strings (each is one chunk)
    """
    if not text.strip():
        return []

    # Slice text into manageable passages
    step = passage_size - passage_overlap
    passages = [text[i: i + passage_size] for i in range(0, len(text), step)]

    all_propositions: List[str] = []
    seen: set = set()  # Deduplicate across overlapping passages

    for passage in passages:
        passage = passage.strip()
        if not passage:
            continue

        try:
            props = _extract_propositions(passage)
        except (json.JSONDecodeError, Exception) as e:
            print(f"[warn] Failed to parse propositions for passage: {e}")
            props = [passage]   # Fallback: use the passage as-is

        for p in props:
            p = p.strip()
            if p and p not in seen:
                all_propositions.append(p)
                seen.add(p)

    return all_propositions


def proposition_chunk_batch(
    texts: List[str],
    passage_size: int = 500,
) -> List[List[str]]:
    """
    Run proposition chunking on multiple documents.

    Returns:
        List of proposition lists — one list per input document.
    """
    return [proposition_chunk(t, passage_size) for t in texts]


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    Albert Einstein was born on March 14, 1879, in Ulm, Germany.
    He developed the theory of special relativity in 1905, which introduced
    the famous equation E=mc². In 1915 he published the general theory of
    relativity, fundamentally changing our understanding of gravity and space-time.
    Einstein was awarded the Nobel Prize in Physics in 1921, not for relativity,
    but for his discovery of the photoelectric effect.
    He emigrated to the United States in 1933 to escape Nazi persecution and
    joined the Institute for Advanced Study in Princeton, New Jersey.
    Einstein died on April 18, 1955, in Princeton.
    """

    print("=" * 60)
    print("Strategy 07: Proposition-Based (Agentic) Chunking")
    print("=" * 60)

    props = proposition_chunk(sample, passage_size=600)
    print(f"\nExtracted {len(props)} propositions:\n")
    for i, p in enumerate(props, 1):
        print(f"  {i:>2}. {p}")

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("\n[tip] Set OPENAI_API_KEY or ANTHROPIC_API_KEY for real LLM extraction.")
