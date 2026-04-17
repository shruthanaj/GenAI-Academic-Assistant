"""Utilities for cleaning and chunking academic text before indexing."""
from __future__ import annotations

import re
from typing import List


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize noisy OCR/PDF text while preserving useful section breaks."""
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _CONTROL_CHAR_RE.sub("", normalized)
    normalized = _MULTISPACE_RE.sub(" ", normalized)

    # Collapse repeated visual separators often present in exported notes.
    normalized = re.sub(r"[-_=]{4,}", "\n", normalized)
    normalized = _MULTINEWLINE_RE.sub("\n\n", normalized)
    return normalized.strip()


def _split_sentences(paragraph: str) -> List[str]:
    """Best-effort sentence segmentation using punctuation boundaries."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", paragraph.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 900,
    overlap_sentences: int = 1,
) -> List[str]:
    """Create sentence-aware chunks from cleaned text.

    The chunk size is character-based to keep behavior deterministic
    without requiring a tokenizer dependency.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current_sentences: List[str] = []
    current_len = 0

    for paragraph in paragraphs:
        sentences = _split_sentences(paragraph)
        if not sentences:
            sentences = [paragraph]

        for sentence in sentences:
            sentence_len = len(sentence)
            if sentence_len >= chunk_size:
                if current_sentences:
                    chunks.append(" ".join(current_sentences).strip())
                    current_sentences = []
                    current_len = 0
                chunks.append(sentence.strip())
                continue

            if current_len + sentence_len + 1 <= chunk_size:
                current_sentences.append(sentence)
                current_len += sentence_len + 1
                continue

            if current_sentences:
                chunks.append(" ".join(current_sentences).strip())
                overlap = current_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
                current_sentences = overlap + [sentence]
                current_len = sum(len(s) + 1 for s in current_sentences)
            else:
                current_sentences = [sentence]
                current_len = sentence_len + 1

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return [chunk for chunk in chunks if chunk]