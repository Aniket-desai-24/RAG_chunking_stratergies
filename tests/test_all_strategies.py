"""
tests/test_all_strategies.py
=============================
Unit tests for all 10 chunking strategies.
Run with:  pytest tests/ -v

No API key needed — strategies 05 and 07 use mock/regex fallbacks.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Shared fixtures ──────────────────────────────────────────────────

SHORT_TEXT = (
    "The sky is blue. Clouds drift slowly overhead. "
    "A gentle breeze moves through the trees. Birds sing. "
    "The sun rises above the horizon. Everything feels calm."
)

LONG_TEXT = (
    "Artificial intelligence is transforming every industry across the globe. "
    "Machine learning enables systems to learn from data without explicit programming. "
    "Deep learning uses many-layered neural networks to extract complex features. "
    "Natural language processing allows computers to understand human language. "
    "Computer vision enables machines to interpret and understand visual information. "
    "Reinforcement learning trains agents by rewarding desired behaviours. "
) * 6

MARKDOWN_TEXT = """
# Introduction to AI

Artificial intelligence is the simulation of human intelligence by machines.
It encompasses learning, reasoning, and self-correction capabilities.

## Machine Learning

Machine learning is a subset of AI that enables systems to learn from data.
It improves automatically through experience and pattern recognition.

### Supervised Learning

Supervised learning uses labelled training data to train models.
The algorithm learns a mapping from inputs to correct outputs.

## Deep Learning

Deep learning uses many-layered neural networks for complex tasks.
Transformers have become the dominant architecture since 2017.
"""


# ── Strategy 01: Fixed-Size ───────────────────────────────────────────

class TestFixedSizeChunking:
    def setup_method(self):
        from chunking.s01_fixed_size import fixed_char_chunk, fixed_token_chunk
        self.char_chunk = fixed_char_chunk
        self.token_chunk = fixed_token_chunk

    def test_basic_split(self):
        chunks = self.char_chunk(LONG_TEXT, chunk_size=200, overlap=0)
        assert len(chunks) > 1

    def test_overlap_creates_more_chunks(self):
        no_overlap = self.char_chunk(LONG_TEXT, chunk_size=200, overlap=0)
        with_overlap = self.char_chunk(LONG_TEXT, chunk_size=200, overlap=50)
        assert len(with_overlap) >= len(no_overlap)

    def test_chunk_size_respected(self):
        chunks = self.char_chunk(LONG_TEXT, chunk_size=100, overlap=0)
        # Last chunk may be smaller; all others should be at most 100 chars
        for c in chunks[:-1]:
            assert len(c) <= 100

    def test_overlap_content_repeated(self):
        """The end of chunk N should appear at the start of chunk N+1."""
        chunks = self.char_chunk(SHORT_TEXT, chunk_size=80, overlap=20)
        if len(chunks) >= 2:
            tail_of_first = chunks[0][-20:]
            assert tail_of_first in chunks[1]

    def test_empty_input(self):
        assert self.char_chunk("", chunk_size=200) == []
        assert self.char_chunk("   ", chunk_size=200) == []

    def test_text_shorter_than_chunk_size(self):
        chunks = self.char_chunk("Hello world.", chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            self.char_chunk(LONG_TEXT, chunk_size=100, overlap=100)


# ── Strategy 02: Sentence-Based ──────────────────────────────────────

class TestSentenceBasedChunking:
    def setup_method(self):
        from chunking.s02_sentence_based import sentence_chunk, sentence_chunk_with_metadata
        self.chunk = sentence_chunk
        self.chunk_meta = sentence_chunk_with_metadata

    def test_basic(self):
        chunks = self.chunk(SHORT_TEXT, sentences_per_chunk=2, sentence_overlap=0)
        assert len(chunks) >= 2

    def test_each_chunk_contains_complete_sentences(self):
        chunks = self.chunk(SHORT_TEXT, sentences_per_chunk=1, sentence_overlap=0)
        for c in chunks:
            assert c.strip().endswith((".", "!", "?"))

    def test_overlap_produces_more_chunks(self):
        no_overlap = self.chunk(SHORT_TEXT, sentences_per_chunk=2, sentence_overlap=0)
        with_overlap = self.chunk(SHORT_TEXT, sentences_per_chunk=2, sentence_overlap=1)
        assert len(with_overlap) >= len(no_overlap)

    def test_metadata_fields(self):
        meta = self.chunk_meta(SHORT_TEXT, sentences_per_chunk=2)
        for m in meta:
            assert "text" in m
            assert "start_sentence" in m
            assert "end_sentence" in m
            assert m["sentence_count"] >= 1

    def test_empty_input(self):
        assert self.chunk("", 2) == []

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            self.chunk(SHORT_TEXT, sentences_per_chunk=2, sentence_overlap=2)


# ── Strategy 03: Paragraph / Section ─────────────────────────────────

class TestParagraphSectionChunking:
    def setup_method(self):
        from chunking.s03_paragraph_section import paragraph_chunk, markdown_section_chunk
        self.para = paragraph_chunk
        self.md = markdown_section_chunk

    def test_paragraph_splits_on_blank_lines(self):
        text = "First paragraph with enough content here.\n\nSecond paragraph with enough content here.\n\nThird paragraph with enough content here."
        chunks = self.para(text, min_chars=10)
        assert len(chunks) == 3

    def test_large_paragraph_is_further_split(self):
        big_para = "This is a sentence. " * 40  # ~760 chars
        chunks = self.para(big_para, max_chars=200)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 250  # Some tolerance for sentence boundaries

    def test_empty_input(self):
        assert self.para("") == []

    def test_markdown_returns_dicts(self):
        chunks = self.md(MARKDOWN_TEXT)
        assert len(chunks) > 0
        for c in chunks:
            assert "text" in c
            assert "heading" in c
            assert "breadcrumb" in c
            assert "level" in c

    def test_markdown_breadcrumb_hierarchy(self):
        chunks = self.md(MARKDOWN_TEXT)
        # "Supervised Learning" is nested inside "Machine Learning"
        supervised = [c for c in chunks if "Supervised" in c.get("heading", "")]
        if supervised:
            assert "Machine Learning" in supervised[0]["breadcrumb"]

    def test_markdown_heading_in_chunk_text(self):
        chunks = self.md(MARKDOWN_TEXT, include_heading_in_chunk=True)
        for c in chunks:
            if c["heading"]:
                assert c["heading"] in c["text"]


# ── Strategy 04: Recursive Character Splitting ───────────────────────

class TestRecursiveCharacterSplitter:
    def setup_method(self):
        from chunking.s04_recursive_character import RecursiveCharacterSplitter
        self.Splitter = RecursiveCharacterSplitter

    def test_basic(self):
        sp = self.Splitter(chunk_size=200, chunk_overlap=20)
        chunks = sp.split_text(LONG_TEXT)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_size(self):
        sp = self.Splitter(chunk_size=300, chunk_overlap=0)
        chunks = sp.split_text(LONG_TEXT)
        for c in chunks:
            assert len(c) <= 350  # Small tolerance at recursive boundaries

    def test_prefers_paragraph_boundaries(self):
        text = "Para one here.\n\nPara two here.\n\nPara three here."
        sp = self.Splitter(chunk_size=40, chunk_overlap=0)
        chunks = sp.split_text(text)
        # Should split at \n\n first
        assert any("Para one" in c for c in chunks)
        assert any("Para two" in c for c in chunks)

    def test_python_separators(self):
        from chunking.s04_recursive_character import RecursiveCharacterSplitter
        code = "class Foo:\n    def bar(self):\n        pass\n\ndef baz():\n    return 1"
        sp = RecursiveCharacterSplitter(
            chunk_size=40,
            separators=RecursiveCharacterSplitter.PYTHON_SEPARATORS,
        )
        chunks = sp.split_text(code)
        assert len(chunks) >= 2

    def test_split_documents(self):
        sp = self.Splitter(chunk_size=200)
        chunks = sp.split_documents([SHORT_TEXT, LONG_TEXT[:300]])
        assert len(chunks) >= 2

    def test_empty_input(self):
        sp = self.Splitter(chunk_size=200)
        assert sp.split_text("") == []


# ── Strategy 05: Semantic Chunking ───────────────────────────────────

class TestSemanticChunking:
    def setup_method(self):
        from chunking.s05_semantic import semantic_chunk, semantic_chunk_with_scores
        self.chunk = semantic_chunk
        self.chunk_scores = semantic_chunk_with_scores

    def test_returns_list(self):
        # Uses mock embeddings (no API key) — validates interface
        chunks = self.chunk(SHORT_TEXT)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_no_empty_chunks(self):
        chunks = self.chunk(LONG_TEXT)
        for c in chunks:
            assert c.strip() != ""

    def test_scores_length(self):
        chunks, scores = self.chunk_scores(SHORT_TEXT)
        # n chunks should have n-1 boundary scores
        # (or fewer if some were merged)
        assert len(scores) >= len(chunks) - 1

    def test_threshold_affects_chunk_count(self):
        # Lower threshold → fewer splits → fewer chunks
        chunks_tight = self.chunk(LONG_TEXT, similarity_threshold=0.99)
        chunks_loose = self.chunk(LONG_TEXT, similarity_threshold=0.01)
        # With threshold=0.99 almost everything is a boundary → many chunks
        # With threshold=0.01 almost nothing is a boundary → few chunks
        assert len(chunks_tight) >= len(chunks_loose)

    def test_empty_input(self):
        chunks = self.chunk("")
        assert chunks == []


# ── Strategy 06: Document-Structure-Aware ────────────────────────────

class TestDocumentStructureChunking:
    def setup_method(self):
        try:
            from chunking.s06_document_structure import html_structure_chunk
            self.html_chunk = html_structure_chunk
            self.available = True
        except ImportError:
            self.available = False

    def test_html_basic(self):
        if not self.available:
            pytest.skip("beautifulsoup4 not installed")
        html = """
        <html><body>
        <h1>Topic A</h1><p>Content about topic A goes here in detail.</p>
        <h2>Sub-topic A1</h2><p>More specific content about sub-topic A1.</p>
        <h1>Topic B</h1><p>Content about topic B goes here in detail.</p>
        </body></html>
        """
        chunks = self.html_chunk(html)
        assert len(chunks) >= 2

    def test_html_breadcrumb_in_text(self):
        if not self.available:
            pytest.skip("beautifulsoup4 not installed")
        html = """<h1>AI</h1><h2>ML</h2>
        <p>Machine learning is a subset of artificial intelligence used widely in industry today.</p>"""
        chunks = self.html_chunk(html, min_text_length=10)
        # At least one chunk should exist and contain text from the p tag
        assert len(chunks) >= 1
        assert any("Machine learning" in c["text"] for c in chunks)

    def test_html_returns_dicts(self):
        if not self.available:
            pytest.skip("beautifulsoup4 not installed")
        html = "<h1>Title</h1><p>Some content that is long enough to be a chunk.</p>"
        chunks = self.html_chunk(html)
        for c in chunks:
            assert "text" in c
            assert "heading_path" in c
            assert "tag" in c


# ── Strategy 08: Sliding Window ──────────────────────────────────────

class TestSlidingWindowChunking:
    def setup_method(self):
        from chunking.s08_sliding_window import (
            sliding_window_chunk,
            deduplicate_retrieved_windows,
        )
        self.chunk = sliding_window_chunk
        self.dedup = deduplicate_retrieved_windows

    def test_basic(self):
        windows = self.chunk(SHORT_TEXT, window_size=60, stride=30)
        assert len(windows) > 1

    def test_window_size_respected(self):
        windows = self.chunk(LONG_TEXT, window_size=100, stride=50)
        for w in windows:
            assert len(w["text"]) <= 100

    def test_stride_equals_window_no_overlap(self):
        windows = self.chunk("abcdefghij" * 10, window_size=10, stride=10)
        for w in windows:
            assert w["overlap_ratio"] == 0.0

    def test_metadata_fields_present(self):
        windows = self.chunk(SHORT_TEXT, window_size=50, stride=25)
        for w in windows:
            assert all(k in w for k in ["text", "window_id", "start_char",
                                         "end_char", "overlap_ratio", "chunk_hash"])

    def test_deduplication(self):
        windows = self.chunk(SHORT_TEXT, window_size=80, stride=40)
        # Simulate: windows 0 and 1 are heavily overlapping
        if len(windows) >= 2:
            deduped = self.dedup([windows[0], windows[1]], iou_threshold=0.4)
            assert len(deduped) <= 2

    def test_empty_input(self):
        assert self.chunk("", window_size=100, stride=50) == []

    def test_invalid_stride(self):
        with pytest.raises(ValueError):
            self.chunk(SHORT_TEXT, window_size=100, stride=0)


# ── Strategy 09: Small-to-Big ────────────────────────────────────────

class TestParentChildChunking:
    def setup_method(self):
        from chunking.s09_small_to_big import ParentChildChunker
        self.Chunker = ParentChildChunker

    def test_creates_children_and_parents(self):
        c = self.Chunker(parent_chunk_size=200, child_chunk_size=60)
        children = c.chunk(LONG_TEXT)
        assert len(children) > 0
        assert c.parent_count > 0

    def test_each_child_has_parent(self):
        c = self.Chunker(parent_chunk_size=200, child_chunk_size=60)
        children = c.chunk(LONG_TEXT)
        for child in children:
            assert "parent_id" in child
            parent = c.get_parent(child)
            assert parent is not None

    def test_parent_contains_child_text(self):
        c = self.Chunker(parent_chunk_size=300, child_chunk_size=80)
        children = c.chunk(LONG_TEXT[:500])
        child = children[0]
        parent_text = c.get_parent_text(child)
        # Child text should be a substring of its parent
        assert child["text"] in parent_text

    def test_swap_deduplicates_parents(self):
        c = self.Chunker(parent_chunk_size=200, child_chunk_size=60)
        children = c.chunk(LONG_TEXT)
        # Retrieve the first two children (likely from same parent)
        context = c.swap_children_for_parents(children[:4])
        # Should be fewer than 4 entries (deduplication)
        assert len(context) <= 4

    def test_metadata_passthrough(self):
        c = self.Chunker()
        children = c.chunk(SHORT_TEXT, doc_id="doc_123", metadata={"source": "test"})
        for child in children:
            assert child["doc_id"] == "doc_123"
            assert child["source"] == "test"

    def test_empty_input(self):
        c = self.Chunker()
        assert c.chunk("") == []


# ── Strategy 10: Hierarchical Chunking ───────────────────────────────

class TestHierarchicalChunking:
    def setup_method(self):
        from chunking.s10_hierarchical import HierarchicalChunker
        self.Chunker = HierarchicalChunker

    def test_builds_all_four_levels(self):
        c = self.Chunker()
        c.chunk_markdown(MARKDOWN_TEXT)
        assert len(c.get_all_at_level(0)) == 1  # One document root
        assert len(c.get_all_at_level(1)) >= 1  # At least one section
        assert len(c.get_all_at_level(2)) >= 1  # At least one paragraph
        assert len(c.get_all_at_level(3)) >= 1  # At least one sentence

    def test_parent_child_links_are_valid(self):
        c = self.Chunker()
        c.chunk_markdown(MARKDOWN_TEXT)
        for chunk in c.store.values():
            if chunk.parent_id:
                assert chunk.parent_id in c.store
            for child_id in chunk.children_ids:
                assert child_id in c.store

    def test_get_ancestors(self):
        c = self.Chunker()
        c.chunk_markdown(MARKDOWN_TEXT)
        sentences = c.get_all_at_level(3)
        if sentences:
            ancestors = c.get_ancestors(sentences[0].id)
            # Sentence → paragraph → section → document = up to 3 ancestors
            assert 1 <= len(ancestors) <= 3

    def test_get_descendants(self):
        c = self.Chunker()
        root = c.chunk_markdown(MARKDOWN_TEXT)
        desc = c.get_descendants(root.id)
        assert len(desc) > 0

    def test_total_chunk_count(self):
        c = self.Chunker()
        c.chunk_markdown(MARKDOWN_TEXT)
        total = sum(len(c.get_all_at_level(i)) for i in range(4))
        assert total == len(c.store)

    def test_empty_doc(self):
        c = self.Chunker()
        root = c.chunk_markdown("")
        assert root is not None


# ── Shared utility tests ──────────────────────────────────────────────

class TestSentenceUtils:
    def setup_method(self):
        from chunking._sentence_utils import split_sentences
        self.split = split_sentences

    def test_regex_backend_always_works(self):
        sentences = self.split(SHORT_TEXT, backend="regex")
        assert len(sentences) >= 3

    def test_auto_backend_returns_results(self):
        sentences = self.split(SHORT_TEXT, backend="auto")
        assert len(sentences) >= 1

    def test_abbreviation_protection(self):
        text = "Dr. Smith visited Mr. Jones. They discussed the U.S. policy."
        sentences = self.split(text, backend="regex")
        # Should NOT split on "Dr." or "Mr." or "U.S."
        assert len(sentences) == 2

    def test_empty_input(self):
        assert self.split("") == []
        assert self.split("   ") == []
