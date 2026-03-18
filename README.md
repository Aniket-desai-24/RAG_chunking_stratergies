# RAG Chunking Strategies 📦

A complete, runnable reference for **10 text chunking strategies** used in Retrieval-Augmented Generation (RAG) pipelines — from the simplest fixed-size split to full hierarchical trees and LLM-powered proposition extraction.

Each strategy comes with:
- ✅ A clean, well-commented Python implementation
- 📚 Inline theory (when to use it, pros/cons, key parameters)
- 🖥️ An interactive HTML masterclass (`docs/index.html`)

---

## 🗂️ Repository Structure

```
rag-chunking-strategies/
├── chunking/                   # All 10 strategy implementations
│   ├── s01_fixed_size.py
│   ├── s02_sentence_based.py
│   ├── s03_paragraph_section.py
│   ├── s04_recursive_character.py
│   ├── s05_semantic.py
│   ├── s06_document_structure.py
│   ├── s07_proposition_based.py
│   ├── s08_sliding_window.py
│   ├── s09_small_to_big.py
│   └── s10_hierarchical.py
├── examples/
│   └── run_all.py              # One-shot demo of all strategies
├── docs/                 
│   └── index.html              # Interactive theory + code masterclass
├── .env.example                # API key template (copy → .env)
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/rag-chunking-strategies.git
cd rag-chunking-strategies

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (needed for sentence tokenisation)
python -c "import nltk; nltk.download('punkt_tab')"

# 5. (Optional) Set up API keys for strategies 05/07
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY or ANTHROPIC_API_KEY

# 6. Run all strategies on a sample document
python examples/run_all.py
```

---

## 🧠 The 10 Strategies at a Glance

| # | Strategy | Difficulty | Cost | Quality | API Key? |
|---|---|---|---|---|---|
| 01 | Fixed-Size Chunking | Easy | Very Low | Low | ❌ No |
| 02 | Sentence-Based | Easy | Low | Medium | ❌ No |
| 03 | Paragraph / Section | Easy | Low | Medium-High | ❌ No |
| 04 | Recursive Character Split | Easy | Low | Medium-High | ❌ No |
| 05 | Semantic Chunking | Medium | Medium | High | Optional* |
| 06 | Document-Structure-Aware | Medium | Low-Med | High | ❌ No |
| 07 | Proposition-Based (Agentic) | Hard | Very High | Very High | ✅ Yes |
| 08 | Sliding Window | Easy | Low | Medium-High | ❌ No |
| 09 | Small-to-Big (Parent-Child) | Medium | Medium | Very High | ❌ No |
| 10 | Hierarchical Chunking | Hard | Medium-High | Very High | ❌ No |

*Strategy 05 works without an API key using `sentence-transformers` (local model).

---

## 📖 Strategy Details

### 01 — Fixed-Size Chunking
The simplest possible strategy. Split text into chunks of N characters (or tokens) with an optional overlap.

```python
from chunking.s01_fixed_size import fixed_char_chunk, fixed_token_chunk

chunks = fixed_char_chunk("Your long text here...", chunk_size=512, overlap=50)
# Token-based (requires tiktoken):
chunks = fixed_token_chunk("Your long text here...", chunk_size=512, overlap=50)
```

**When to use:** Baseline experiments, homogeneous text corpora, prototyping.

---

### 02 — Sentence-Based Chunking
Respects sentence boundaries using NLTK or spaCy. Groups N sentences per chunk.

```python
from chunking.s02_sentence_based import sentence_chunk

chunks = sentence_chunk(text, sentences_per_chunk=3, sentence_overlap=1)
```

**When to use:** Factual Q&A, news retrieval, any content where individual facts matter.

---

### 03 — Paragraph / Section Chunking
Splits on blank lines or Markdown heading markers. Attaches full heading breadcrumbs to each chunk.

```python
from chunking.s03_paragraph_section import paragraph_chunk, markdown_section_chunk

# Plain text
chunks = paragraph_chunk(text, max_chars=1500)

# Markdown (returns dicts with heading metadata)
chunks = markdown_section_chunk(markdown_text)
# → [{"text": "...", "heading": "Machine Learning", "breadcrumb": "AI > ML", ...}]
```

**When to use:** Wikipedia, documentation, books, structured articles.

---

### 04 — Recursive Character Text Splitting
Tries separators in priority order (`\n\n → \n → . → " "`) until chunks fit. LangChain's default splitter, re-implemented from scratch.

```python
from chunking.s04_recursive_character import RecursiveCharacterSplitter

splitter = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_text(text)

# Code-specific separators:
splitter = RecursiveCharacterSplitter(
    chunk_size=1000,
    separators=RecursiveCharacterSplitter.PYTHON_SEPARATORS
)
```

**When to use:** Mixed documents, general RAG baseline — the best safe default.

---

### 05 — Semantic Chunking
Embeds each sentence, measures cosine similarity between adjacent sentences, and splits where similarity drops below a threshold.

```python
from chunking.s05_semantic import semantic_chunk, semantic_chunk_with_scores

# Auto-selects: OpenAI → sentence-transformers → mock
chunks = semantic_chunk(text, similarity_threshold=0.75, buffer_size=1)

# Also get the similarity scores for threshold tuning
chunks, scores = semantic_chunk_with_scores(text, similarity_threshold=0.75)
```

**When to use:** Transcripts, essays, blog posts — content where topics shift without headings.

---

### 06 — Document-Structure-Aware Chunking
Parses format-specific structure: HTML tags, Markdown headings, PDF text blocks, DOCX paragraph styles.

```python
from chunking.s06_document_structure import (
    html_structure_chunk,
    pdf_structure_chunk,
    docx_structure_chunk,
)

html_chunks  = html_structure_chunk(html_string)
pdf_chunks   = pdf_structure_chunk("report.pdf", strategy="page")  # needs pymupdf
docx_chunks  = docx_structure_chunk("document.docx")               # needs python-docx
```

**When to use:** Websites, PDFs, Word documents with clear formatting.

---

### 07 — Proposition-Based (Agentic) Chunking
Uses an LLM to decompose text into atomic, self-contained factual propositions. Each proposition is one chunk.

```python
from chunking.s07_proposition_based import proposition_chunk

# Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env
propositions = proposition_chunk(text, passage_size=500)
# → ["Einstein was born in 1879.", "Einstein developed special relativity.", ...]
```

**When to use:** Maximum retrieval precision — medical knowledge bases, legal research, fact-checking.

Reference: [Dense X Retrieval (Chen et al. 2023)](https://arxiv.org/abs/2312.06648)

---

### 08 — Sliding Window Chunking
Dense overlapping windows — every part of the text appears in multiple chunks, maximising recall.

```python
from chunking.s08_sliding_window import sliding_window_chunk, deduplicate_retrieved_windows

windows = sliding_window_chunk(text, window_size=512, stride=256)
# After retrieval, remove duplicates:
unique = deduplicate_retrieved_windows(retrieved_top_k, iou_threshold=0.5)
```

**When to use:** Legal contracts, medical records, compliance documents where missing a fact is costly.

---

### 09 — Small-to-Big (Parent-Child) Chunking
Indexes small child chunks for precise matching; returns their large parent chunks to the LLM for rich context.

```python
from chunking.s09_small_to_big import ParentChildChunker

chunker = ParentChildChunker(parent_chunk_size=512, child_chunk_size=128)
children = chunker.chunk(text)            # → embed and index these
# At retrieval time:
context = chunker.swap_children_for_parents(retrieved_children)  # → send to LLM
```

**When to use:** Long documents, enterprise search, when you need both precision and context.

---

### 10 — Hierarchical Chunking
Builds a 4-level tree: Document → Section → Paragraph → Sentence. Query at any level; traverse up/down for context.

```python
from chunking.s10_hierarchical import HierarchicalChunker

chunker = HierarchicalChunker()
root = chunker.chunk_markdown(markdown_text)

sentences  = chunker.get_all_at_level(3)   # Fine-grained retrieval
sections   = chunker.get_all_at_level(1)   # Broad retrieval
ancestors  = chunker.get_ancestors(chunk_id)  # Walk up to section/document

chunker.print_tree()  # Debug the full hierarchy
```

**When to use:** Books, technical manuals, legal codebooks — complex hierarchical documents.

---

## 🌐 Interactive Visualizer

Open `docs/index.html` in any browser for a full interactive masterclass with:
- Theory explanation for each strategy
- Colour-coded visual of how text is split
- Copyable code for every strategy
- "When to use / when to avoid" guidance

No server required — it's a self-contained HTML file.

---

## 🛠️ Dependencies

| Package | Used in | Required? |
|---|---|---|
| `nltk` | 02, 05, 10 | ✅ Core |
| `numpy` | 05 | ✅ Core |
| `sentence-transformers` | 05 (local embed) | Recommended |
| `tiktoken` | 01 (token split) | Optional |
| `beautifulsoup4` | 06 (HTML) | Optional |
| `pymupdf` | 06 (PDF) | Optional |
| `python-docx` | 06 (DOCX) | Optional |
| `openai` | 05 (embed), 07 | Optional |
| `anthropic` | 07 | Optional |

---

## 📜 License

MIT — free to use, modify, and distribute.

---

## 🤝 Contributing

PRs welcome! Ideas:
- Add strategy 11: **Schema/Table-Aware Chunking** for structured data
- Add a Jupyter notebook walkthrough
- Add unit tests for edge cases
- Add a benchmarking script comparing retrieval quality across strategies

---

## ⭐ If this helped you, give it a star!
