"""
Strategy 09: Small-to-Big (Parent-Child) Chunking
===================================================
Index small child chunks for precise vector search, but return
their large parent chunks to the LLM for rich context.

No external libraries required.
No API key needed.
"""

import uuid
from typing import List, Dict, Optional


class ParentChildChunker:
    """
    Two-level chunk hierarchy:
        Child chunks  — small (e.g. 128 chars), embedded and stored in vector DB
        Parent chunks — large (e.g. 512 chars), returned to the LLM at retrieval

    Usage:
        chunker = ParentChildChunker()
        children = chunker.chunk(document_text)
        # → index children in your vector store
        # → at retrieval, call chunker.get_parent_text(child) to get rich context
    """

    def __init__(
        self,
        parent_chunk_size: int = 512,
        child_chunk_size: int = 128,
        child_overlap: int = 20,
    ):
        self.parent_size = parent_chunk_size
        self.child_size = child_chunk_size
        self.child_overlap = child_overlap
        self._parent_store: Dict[str, Dict] = {}

    def chunk(
        self,
        text: str,
        doc_id: str = "",
        metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Split text into child chunks and register parent chunks internally.

        Args:
            text:     Document text
            doc_id:   Identifier for the source document
            metadata: Extra key-value pairs attached to every chunk

        Returns:
            List of child chunk dicts.  Each dict has:
                id          – unique child ID (use as vector store key)
                text        – small chunk text (this is what you embed)
                parent_id   – key to fetch the parent from get_parent_text()
                doc_id      – the document this chunk came from
                child_index – position of this child within its parent
        """
        if not text.strip():
            return []

        meta = metadata or {}
        children: List[Dict] = []
        parent_step = self.parent_size  # No overlap at the parent level

        for p_start in range(0, len(text), parent_step):
            parent_text = text[p_start: p_start + self.parent_size]
            if not parent_text.strip():
                continue

            parent_id = str(uuid.uuid4())
            self._parent_store[parent_id] = {
                "id": parent_id,
                "text": parent_text,
                "doc_id": doc_id,
                "start_char": p_start,
                **meta,
            }

            # Create child chunks within this parent
            child_step = self.child_size - self.child_overlap
            child_index = 0

            for c_start in range(0, len(parent_text), child_step):
                child_text = parent_text[c_start: c_start + self.child_size]
                if not child_text.strip():
                    continue

                children.append({
                    "id": str(uuid.uuid4()),
                    "text": child_text,          # ← embed this
                    "parent_id": parent_id,      # ← use at retrieval time
                    "doc_id": doc_id,
                    "child_index": child_index,
                    **meta,
                })
                child_index += 1

        return children

    def get_parent(self, child: Dict) -> Optional[Dict]:
        """
        Given a retrieved child chunk dict, return its parent chunk dict.
        """
        return self._parent_store.get(child.get("parent_id", ""))

    def get_parent_text(self, child: Dict) -> str:
        """
        Convenience: return just the parent's text string.
        Returns the child text itself if the parent cannot be found.
        """
        parent = self.get_parent(child)
        return parent["text"] if parent else child["text"]

    def swap_children_for_parents(
        self,
        retrieved_children: List[Dict],
    ) -> List[str]:
        """
        Post-retrieval step: replace each child with its parent, deduplicating.

        Args:
            retrieved_children: Top-k children from vector search (ranked)

        Returns:
            Ordered list of unique parent texts for the LLM context window
        """
        seen_parent_ids: set = set()
        context: List[str] = []

        for child in retrieved_children:
            pid = child.get("parent_id")
            if pid and pid not in seen_parent_ids:
                parent_text = self.get_parent_text(child)
                context.append(parent_text)
                seen_parent_ids.add(pid)

        return context

    @property
    def parent_count(self) -> int:
        return len(self._parent_store)


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    Artificial intelligence (AI) is intelligence demonstrated by machines,
    as opposed to the natural intelligence displayed by animals including humans.
    AI research has been defined as the field of study of intelligent agents,
    which refers to any system that perceives its environment and takes actions
    that maximize its chance of achieving its goals.

    The term artificial intelligence had previously been used to describe machines
    that mimic and display human cognitive skills associated with the human mind,
    such as learning and problem-solving. This definition has since been rejected
    by major AI researchers who now describe AI in terms of rationality and acting
    rationally, which does not limit how intelligence can be articulated.

    Machine learning is a method of data analysis that automates analytical model
    building. It is based on the idea that systems can learn from data, identify
    patterns and make decisions with minimal human intervention.
    """

    print("=" * 60)
    print("Strategy 09: Small-to-Big (Parent-Child) Chunking")
    print("=" * 60)

    chunker = ParentChildChunker(parent_chunk_size=300, child_chunk_size=80, child_overlap=15)
    children = chunker.chunk(sample, doc_id="ai_overview")

    print(f"\nParent chunks : {chunker.parent_count}")
    print(f"Child chunks  : {len(children)}")
    print(f"Avg children per parent: {len(children) / max(chunker.parent_count, 1):.1f}")

    print(f"\n── Child 0 (what vector DB stores and embeds) ──")
    print(f"  {children[0]['text'][:100]}...")

    print(f"\n── Parent of Child 0 (what LLM receives) ──")
    parent_text = chunker.get_parent_text(children[0])
    print(f"  {parent_text[:200]}...")

    # Simulate retrieval: top-k children → swap for parents
    mock_top_k = [children[0], children[2], children[1]]  # Pretend these were retrieved
    context = chunker.swap_children_for_parents(mock_top_k)
    print(f"\n── After swap_children_for_parents: {len(context)} unique parent(s) for LLM ──")
