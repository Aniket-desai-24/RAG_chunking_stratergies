"""
Strategy 10: Hierarchical Chunking
=====================================
Build a full multi-level chunk tree: Document → Section → Paragraph → Sentence.
Query at the right granularity; traverse up/down the tree for context.

Requirements:
    pip install nltk

Optional (RAPTOR-style LLM summaries at each level):
    pip install openai  OR  pip install anthropic
No API key needed for the core chunking.
"""

import os
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ── Data model ────────────────────────────────────────────────────────

@dataclass
class HierarchicalChunk:
    id: str
    text: str
    level: int            # 0=document, 1=section, 2=paragraph, 3=sentence
    parent_id: Optional[str]
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    summary: Optional[str] = None   # Optional LLM-generated summary

    LEVEL_NAMES = {0: "document", 1: "section", 2: "paragraph", 3: "sentence"}

    @property
    def level_name(self) -> str:
        return self.LEVEL_NAMES.get(self.level, f"level_{self.level}")

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text[:120] + "..." if len(self.text) > 120 else self.text,
            "level": self.level,
            "level_name": self.level_name,
            "parent_id": self.parent_id,
            "children_count": len(self.children_ids),
            "metadata": self.metadata,
        }


# ── Chunker ───────────────────────────────────────────────────────────

class HierarchicalChunker:
    """
    Builds a 4-level chunk tree from Markdown text.

    Level 0: Entire document (one chunk, with optional summary)
    Level 1: Top-level sections (split on # headings)
    Level 2: Paragraphs within each section
    Level 3: Individual sentences within each paragraph

    All chunks are stored in self.store keyed by their UUID.
    """

    def __init__(self, min_text_len: int = 15):
        self.min_text_len = min_text_len
        self.store: Dict[str, HierarchicalChunk] = {}

    # ── Public API ────────────────────────────────────────────────────

    def chunk_markdown(
        self,
        text: str,
        doc_title: str = "Document",
        doc_id: Optional[str] = None,
    ) -> HierarchicalChunk:
        """
        Build the full hierarchy from a Markdown string.

        Returns:
            The root (level-0) document chunk.
            Use self.get_all_at_level(N) to retrieve chunks at any level.
        """
        self.store.clear()  # Fresh tree per call

        # Level 0: Document root
        root = self._make_chunk(
            text=text[:600],    # Snippet for embedding; summary added later
            level=0,
            parent_id=None,
            metadata={"doc_title": doc_title, "doc_id": doc_id or str(uuid.uuid4())},
        )

        # Level 1: Sections (split on ## headings or --- dividers)
        sections = re.split(r"(?=^#{1,2}\s)", text, flags=re.MULTILINE)
        sections = [s.strip() for s in sections if s.strip()]

        for section_text in sections:
            # Extract heading (first line) if present
            first_line = section_text.split("\n")[0].strip()
            heading = re.sub(r"^#+\s*", "", first_line) if first_line.startswith("#") else ""

            sec = self._make_chunk(
                text=section_text[:800],
                level=1,
                parent_id=root.id,
                metadata={"heading": heading},
            )
            root.children_ids.append(sec.id)

            # Level 2: Paragraphs
            paragraphs = re.split(r"\n\s*\n", section_text)
            paragraphs = [p.strip() for p in paragraphs
                          if len(p.strip()) >= self.min_text_len]

            for para_text in paragraphs:
                para = self._make_chunk(
                    text=para_text,
                    level=2,
                    parent_id=sec.id,
                    metadata={"heading_path": heading},
                )
                sec.children_ids.append(para.id)

                # Level 3: Sentences
                for sent_text in self._split_sentences(para_text):
                    if len(sent_text) < self.min_text_len:
                        continue
                    sent = self._make_chunk(
                        text=sent_text,
                        level=3,
                        parent_id=para.id,
                        metadata={"heading_path": heading},
                    )
                    para.children_ids.append(sent.id)

        return root

    def get_all_at_level(self, level: int) -> List[HierarchicalChunk]:
        """Return all chunks at a specific level (0–3)."""
        return [c for c in self.store.values() if c.level == level]

    def get_ancestors(self, chunk_id: str) -> List[HierarchicalChunk]:
        """Walk up the tree and return all ancestor chunks."""
        ancestors: List[HierarchicalChunk] = []
        chunk = self.store.get(chunk_id)
        while chunk and chunk.parent_id:
            parent = self.store.get(chunk.parent_id)
            if parent:
                ancestors.append(parent)
            chunk = parent
        return ancestors

    def get_descendants(self, chunk_id: str) -> List[HierarchicalChunk]:
        """Return all descendant chunks (BFS)."""
        results: List[HierarchicalChunk] = []
        queue = [chunk_id]
        while queue:
            cid = queue.pop(0)
            chunk = self.store.get(cid)
            if chunk:
                for child_id in chunk.children_ids:
                    child = self.store.get(child_id)
                    if child:
                        results.append(child)
                        queue.append(child_id)
        return results

    def add_llm_summaries(self, levels: List[int] = (0, 1, 2)) -> None:
        """
        Generate LLM summaries for chunks at the specified levels.
        Requires OPENAI_API_KEY or ANTHROPIC_API_KEY.
        """
        for chunk in self.store.values():
            if chunk.level in levels:
                chunk.summary = _summarise(chunk.text)

    def print_tree(self, node_id: Optional[str] = None, depth: int = 0) -> None:
        """Pretty-print the chunk tree for debugging."""
        if node_id is None:
            roots = [c for c in self.store.values() if c.parent_id is None]
            for r in roots:
                self.print_tree(r.id, 0)
            return

        chunk = self.store.get(node_id)
        if not chunk:
            return

        indent = "  " * depth
        preview = chunk.text[:60].replace("\n", " ")
        meta = chunk.metadata.get("heading", "") or chunk.metadata.get("doc_title", "")
        print(f"{indent}[L{chunk.level} {chunk.level_name}] "
              f"({meta}) {preview}...")

        for child_id in chunk.children_ids:
            self.print_tree(child_id, depth + 1)

    # ── Internals ─────────────────────────────────────────────────────

    def _make_chunk(self, **kwargs) -> HierarchicalChunk:
        chunk = HierarchicalChunk(id=str(uuid.uuid4()), **kwargs)
        self.store[chunk.id] = chunk
        return chunk

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        from chunking._sentence_utils import split_sentences
        return split_sentences(text)


# ── Optional LLM summary helper ───────────────────────────────────────

def _summarise(text: str, max_words: int = 30) -> str:
    """Generate a short summary of text using an available LLM."""
    prompt = (
        f"Summarise the following text in at most {max_words} words. "
        f"Be concise and factual.\n\n{text[:1000]}"
    )

    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            pass

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception:
            pass

    # Fallback: first sentence of text
    first = re.split(r"(?<=[.!?])\s+", text.strip())[0]
    return first[:120]


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_markdown = """
# Introduction to AI

Artificial intelligence is the simulation of human intelligence by machines.
It encompasses learning, reasoning, and self-correction.
The field was founded as an academic discipline in 1956.

## Machine Learning

Machine learning is a subset of AI that enables systems to learn from data.
It improves automatically through experience without being explicitly programmed.

### Supervised Learning

Supervised learning uses labelled training data.
The model learns to map inputs to correct outputs.
Common algorithms include linear regression, SVMs, and neural networks.

### Unsupervised Learning

Unsupervised learning finds patterns in data without labels.
Clustering and dimensionality reduction are the main tasks.

## Deep Learning

Deep learning uses many-layered neural networks to extract rich features.
It excels at image recognition, NLP, and speech synthesis.
Models like GPT-4 and BERT changed the NLP landscape dramatically.
    """

    print("=" * 60)
    print("Strategy 10: Hierarchical Chunking")
    print("=" * 60)

    chunker = HierarchicalChunker()
    root = chunker.chunk_markdown(sample_markdown, doc_title="AI Overview")

    print(f"\n  Level 0 (documents) : {len(chunker.get_all_at_level(0))}")
    print(f"  Level 1 (sections)  : {len(chunker.get_all_at_level(1))}")
    print(f"  Level 2 (paragraphs): {len(chunker.get_all_at_level(2))}")
    print(f"  Level 3 (sentences) : {len(chunker.get_all_at_level(3))}")
    print(f"  Total chunks stored : {len(chunker.store)}")

    print("\n── Tree preview ──")
    chunker.print_tree()

    # Show ancestor traversal
    first_sentence = chunker.get_all_at_level(3)[0]
    ancestors = chunker.get_ancestors(first_sentence.id)
    print(f"\n── Ancestors of first sentence ──")
    for a in ancestors:
        print(f"  [L{a.level}] {a.text[:60].replace(chr(10), ' ')}...")
