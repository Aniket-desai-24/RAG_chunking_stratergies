"""
Strategy 08: Sliding Window Chunking
======================================
Dense overlapping windows ensure every token is covered by multiple chunks,
maximising recall at the cost of index size and some redundancy.

No external libraries required.
No API key needed.
"""

import hashlib
from typing import List, Dict


def sliding_window_chunk(
    text: str,
    window_size: int = 512,
    stride: int = 256,
) -> List[Dict]:
    """
    Create overlapping windows across the text.

    Args:
        text:        Input text
        window_size: Characters per window
        stride:      Characters to advance per step.
                     stride < window_size → overlap  (stride = window_size → no overlap)

    Returns:
        List of {"text", "window_id", "start_char", "end_char",
                 "overlap_ratio", "chunk_hash"}
    """
    if not text.strip():
        return []
    if stride <= 0:
        raise ValueError("stride must be > 0")

    chunks: List[Dict] = []
    pos = 0
    window_id = 0

    while pos < len(text):
        window_text = text[pos: pos + window_size]
        if not window_text.strip():
            break

        chunks.append({
            "text": window_text,
            "window_id": window_id,
            "start_char": pos,
            "end_char": pos + len(window_text),
            "overlap_ratio": round((window_size - stride) / window_size, 3),
            "chunk_hash": hashlib.md5(window_text.encode()).hexdigest()[:8],
        })

        pos += stride
        window_id += 1

    return chunks


def deduplicate_retrieved_windows(
    retrieved: List[Dict],
    iou_threshold: float = 0.5,
) -> List[Dict]:
    """
    After retrieval, remove windows that overlap heavily with a higher-ranked window.
    Uses Intersection-over-Union (IoU) on character positions.

    Args:
        retrieved:     Top-k retrieved window dicts (ordered by relevance descending)
        iou_threshold: Windows with IoU above this with a kept window are dropped

    Returns:
        De-duplicated list of window dicts
    """
    kept: List[Dict] = []
    for candidate in retrieved:
        c_start, c_end = candidate["start_char"], candidate["end_char"]
        is_dup = False
        for k in kept:
            k_start, k_end = k["start_char"], k["end_char"]
            intersection = max(0, min(c_end, k_end) - max(c_start, k_start))
            union = max(c_end, k_end) - min(c_start, k_start)
            if union > 0 and intersection / union > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(candidate)
    return kept


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = ("The quick brown fox jumps over the lazy dog. " * 10).strip()

    print("=" * 60)
    print("Strategy 08: Sliding Window Chunking")
    print("=" * 60)

    windows = sliding_window_chunk(sample, window_size=80, stride=40)
    print(f"\n[window=80, stride=40] → {len(windows)} windows (50% overlap)")
    for w in windows[:4]:
        print(f"  Window {w['window_id']:>2} chars {w['start_char']:>3}–{w['end_char']:>3}  "
              f"hash={w['chunk_hash']}  {w['text'][:40]}...")

    print(f"\n  Storage overhead vs no-overlap: "
          f"~{round(80/40, 1)}× more chunks")
