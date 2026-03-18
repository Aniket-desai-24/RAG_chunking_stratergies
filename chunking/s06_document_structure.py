"""
Strategy 06: Document-Structure-Aware Chunking
================================================
Parse format-specific structure (HTML, Markdown, PDF, DOCX) and extract
chunks at meaningful structural boundaries with full heading metadata.

Requirements:
    pip install beautifulsoup4          (HTML)
    pip install pymupdf                 (PDF)  -- optional
    pip install python-docx             (DOCX) -- optional
No API key needed.
"""

import re
from typing import List, Dict, Optional


# ── HTML chunking ─────────────────────────────────────────────────────

def html_structure_chunk(
    html: str,
    min_text_length: int = 30,
) -> List[Dict]:
    """
    Extract semantically meaningful chunks from HTML.
    Each chunk carries a full heading breadcrumb.

    Args:
        html:             Raw HTML string
        min_text_length:  Skip text nodes shorter than this

    Returns:
        List of {"text": str, "heading_path": str, "tag": str}
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Run: pip install beautifulsoup4")

    soup = BeautifulSoup(html, "html.parser")
    chunks: List[Dict] = []
    heading_stack: List[Dict] = []

    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6",
                               "p", "li", "td", "blockquote", "article"]):
        text = tag.get_text(separator=" ", strip=True)
        if not text or len(text) < min_text_length:
            continue

        level = int(tag.name[1]) if tag.name.startswith("h") else None

        if level:  # Heading — update the ancestor stack
            heading_stack = [h for h in heading_stack if h["level"] < level]
            heading_stack.append({"level": level, "text": text})
        else:      # Content — create a chunk
            breadcrumb = " > ".join(h["text"] for h in heading_stack)
            chunk_text = f"{breadcrumb}: {text}" if breadcrumb else text
            chunks.append({
                "text": chunk_text,
                "heading_path": breadcrumb,
                "tag": tag.name,
            })

    return chunks


# ── PDF chunking ──────────────────────────────────────────────────────

def pdf_structure_chunk(
    pdf_path: str,
    strategy: str = "page",  # "page" | "block"
) -> List[Dict]:
    """
    Chunk a PDF file.

    Strategies:
        "page"  – one chunk per page (fast, good for most PDFs)
        "block" – one chunk per text block (finer granularity)

    Args:
        pdf_path: Path to a .pdf file
        strategy: "page" or "block"

    Returns:
        List of {"text": str, "page": int, "source": str}
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Run: pip install pymupdf")

    doc = fitz.open(pdf_path)
    chunks: List[Dict] = []

    for page_num, page in enumerate(doc):
        if strategy == "page":
            text = page.get_text("text").strip()
            if text:
                chunks.append({
                    "text": text,
                    "page": page_num + 1,
                    "source": pdf_path,
                })
        elif strategy == "block":
            for block in page.get_text("blocks"):
                block_text = block[4].strip()  # index 4 is the text content
                if block_text:
                    chunks.append({
                        "text": block_text,
                        "page": page_num + 1,
                        "source": pdf_path,
                        "block_bbox": block[:4],
                    })

    doc.close()
    return chunks


# ── DOCX chunking ─────────────────────────────────────────────────────

def docx_structure_chunk(docx_path: str) -> List[Dict]:
    """
    Chunk a Word document using paragraph styles.
    Heading paragraphs update the heading breadcrumb;
    body paragraphs become chunks.

    Args:
        docx_path: Path to a .docx file

    Returns:
        List of {"text": str, "heading_path": str, "style": str}
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Run: pip install python-docx")

    doc = Document(docx_path)
    chunks: List[Dict] = []
    heading_stack: List[Dict] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style = para.style.name  # e.g. "Heading 1", "Normal"
        heading_match = re.match(r"Heading (\d+)", style)

        if heading_match:
            level = int(heading_match.group(1))
            heading_stack = [h for h in heading_stack if h["level"] < level]
            heading_stack.append({"level": level, "text": text})
        else:
            breadcrumb = " > ".join(h["text"] for h in heading_stack)
            chunks.append({
                "text": f"{breadcrumb}: {text}" if breadcrumb else text,
                "heading_path": breadcrumb,
                "style": style,
            })

    return chunks


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_html = """
    <html><body>
    <h1>Artificial Intelligence</h1>
    <p>AI is the simulation of human intelligence by machines.</p>
    <h2>Machine Learning</h2>
    <p>ML allows systems to learn from data without explicit programming.</p>
    <h3>Supervised Learning</h3>
    <p>Models train on labelled examples to learn input-output mappings.</p>
    <h3>Unsupervised Learning</h3>
    <p>Finds hidden patterns in data without labels or guidance.</p>
    <h2>Deep Learning</h2>
    <p>Uses many-layered neural networks for complex feature extraction.</p>
    </body></html>
    """

    print("=" * 60)
    print("Strategy 06: Document-Structure-Aware Chunking")
    print("=" * 60)

    try:
        html_chunks = html_structure_chunk(sample_html)
        print(f"\n[HTML] → {len(html_chunks)} chunks")
        for c in html_chunks:
            print(f"  [{c['heading_path']}] {c['text'][:70]}...")
    except ImportError as e:
        print(f"[HTML] Skipped: {e}")

    print("\n[PDF]  Requires a .pdf file path — see pdf_structure_chunk()")
    print("[DOCX] Requires a .docx file path — see docx_structure_chunk()")
