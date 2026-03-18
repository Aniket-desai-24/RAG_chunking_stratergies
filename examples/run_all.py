"""
examples/run_all.py
====================
Run all 10 chunking strategies on a sample text and print a summary table.
Shows which strategies work without any API key and which need one.

Usage:
    python examples/run_all.py
"""

import sys
import os
import time

# Make sure the repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SAMPLE_TEXT = """
# The History and Future of Artificial Intelligence

## Origins

Artificial intelligence as a formal field was founded at the Dartmouth Conference in 1956.
John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon organised the event.
They believed that every aspect of human intelligence could be described precisely enough
for a machine to simulate it.

Early AI research focused on symbolic reasoning and problem-solving through rule-based systems.
These systems were brittle: they worked well in narrow domains but failed to generalise.
The first AI winter occurred in the 1970s when funding dried up after over-promising.

## The Rise of Machine Learning

The second wave of AI interest came in the 1980s with expert systems and neural networks.
Geoffrey Hinton, Yann LeCun, and Yoshua Bengio pioneered backpropagation for training neural nets.
However, hardware limitations prevented deep networks from being trained at scale.

The real breakthrough came in 2012 when AlexNet won the ImageNet competition by a large margin.
This demonstrated that deep convolutional neural networks could achieve superhuman accuracy on images.
The GPU revolution made training large models feasible for the first time.

## Transformers and the Modern Era

In 2017, Google researchers published "Attention Is All You Need", introducing the Transformer.
The Transformer architecture relies entirely on attention mechanisms, discarding recurrence.
It enabled models to be trained in parallel and scale to billions of parameters efficiently.

GPT-3, released in 2020, showed that language models could perform few-shot learning.
ChatGPT launched in November 2022 and reached 100 million users in just two months.
Large language models now power assistants, code generation, and scientific research.

## Challenges and Future Directions

Despite rapid progress, AI systems still face significant challenges.
Hallucination — generating plausible but incorrect information — remains an open problem.
Alignment research aims to ensure AI systems behave in accordance with human values.

AI is expected to transform every major industry over the next decade.
Healthcare, education, scientific research, and creative industries will all be affected.
The question is not whether AI will change the world, but how we will manage that change.
"""


def hr(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def run_strategy(name: str, fn, *args, **kwargs):
    """Run a chunking function and print results summary."""
    try:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000

        # Normalise: list of str or list of dict
        if result and isinstance(result[0], dict):
            texts = [r.get("text", "") for r in result]
        else:
            texts = result

        avg_len = sum(len(t) for t in texts) / max(len(texts), 1)
        print(f"  ✅  {len(texts):>3} chunks  |  avg {avg_len:>6.0f} chars  |  {elapsed:>6.1f} ms")
        if texts:
            preview = texts[0][:80].replace("\n", " ")
            print(f"       First chunk: {preview}...")
    except ImportError as e:
        print(f"  ⚠️  Skipped — missing library: {e}")
    except Exception as e:
        print(f"  ❌  Error: {e}")


# ── Strategy runners ──────────────────────────────────────────────────

def demo_s01():
    from chunking.s01_fixed_size import fixed_char_chunk
    hr("01 · Fixed-Size Chunking  [no libraries needed]")
    run_strategy("fixed", fixed_char_chunk, SAMPLE_TEXT, chunk_size=300, overlap=50)


def demo_s02():
    from chunking.s02_sentence_based import sentence_chunk
    hr("02 · Sentence-Based Chunking  [needs: nltk]")
    run_strategy("sentence", sentence_chunk, SAMPLE_TEXT, sentences_per_chunk=3, sentence_overlap=1)


def demo_s03():
    from chunking.s03_paragraph_section import paragraph_chunk, markdown_section_chunk
    hr("03 · Paragraph / Section Chunking  [no libraries needed]")
    print("  [paragraph]")
    run_strategy("paragraph", paragraph_chunk, SAMPLE_TEXT, max_chars=400)
    print("  [markdown sections]")
    run_strategy("markdown", markdown_section_chunk, SAMPLE_TEXT)


def demo_s04():
    from chunking.s04_recursive_character import RecursiveCharacterSplitter
    hr("04 · Recursive Character Splitting  [no libraries needed]")
    splitter = RecursiveCharacterSplitter(chunk_size=400, chunk_overlap=60)
    run_strategy("recursive", splitter.split_text, SAMPLE_TEXT)


def demo_s05():
    from chunking.s05_semantic import semantic_chunk
    hr("05 · Semantic Chunking  [needs: nltk + sentence-transformers or openai]")
    run_strategy("semantic", semantic_chunk, SAMPLE_TEXT, similarity_threshold=0.72)


def demo_s06():
    from chunking.s06_document_structure import html_structure_chunk
    hr("06 · Document-Structure-Aware  [needs: beautifulsoup4]")
    # Convert the sample to minimal HTML for the demo
    html = "<html><body>"
    for line in SAMPLE_TEXT.strip().split("\n"):
        line = line.strip()
        if line.startswith("# "):
            html += f"<h1>{line[2:]}</h1>"
        elif line.startswith("## "):
            html += f"<h2>{line[3:]}</h2>"
        elif line:
            html += f"<p>{line}</p>"
    html += "</body></html>"
    run_strategy("html", html_structure_chunk, html)


def demo_s07():
    from chunking.s07_proposition_based import proposition_chunk
    hr("07 · Proposition-Based (Agentic)  [needs: openai or anthropic API key]")
    # Use just the first paragraph to keep demo fast
    short_text = SAMPLE_TEXT.strip().split("\n\n")[2]
    run_strategy("proposition", proposition_chunk, short_text, passage_size=400)


def demo_s08():
    from chunking.s08_sliding_window import sliding_window_chunk
    hr("08 · Sliding Window  [no libraries needed]")
    run_strategy("sliding", sliding_window_chunk, SAMPLE_TEXT, window_size=300, stride=150)


def demo_s09():
    from chunking.s09_small_to_big import ParentChildChunker
    hr("09 · Small-to-Big (Parent-Child)  [no libraries needed]")
    chunker = ParentChildChunker(parent_chunk_size=400, child_chunk_size=100, child_overlap=20)
    run_strategy("parent-child", chunker.chunk, SAMPLE_TEXT)


def demo_s10():
    from chunking.s10_hierarchical import HierarchicalChunker
    hr("10 · Hierarchical Chunking  [needs: nltk]")
    chunker = HierarchicalChunker()
    root = chunker.chunk_markdown(SAMPLE_TEXT)
    print(f"  ✅  {len(chunker.store):>3} chunks across 4 levels")
    print(f"       L0={len(chunker.get_all_at_level(0))} doc  "
          f"L1={len(chunker.get_all_at_level(1))} sections  "
          f"L2={len(chunker.get_all_at_level(2))} paragraphs  "
          f"L3={len(chunker.get_all_at_level(3))} sentences")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Chunking Strategies — Demo Runner")
    print("=" * 60)
    print(f"\n  Sample document: {len(SAMPLE_TEXT)} characters")

    demo_s01()
    demo_s02()
    demo_s03()
    demo_s04()
    demo_s05()
    demo_s06()
    demo_s07()
    demo_s08()
    demo_s09()
    demo_s10()

    print("\n" + "=" * 60)
    print("  Done! Open visualizer/index.html in your browser for")
    print("  the interactive theory + code masterclass.")
    print("=" * 60)
