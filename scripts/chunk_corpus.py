"""
Corpus Cleaning, Normalization, and Configurable Chunking.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System

Processes data/processed/normalized_corpus.json, applies clinical text normalization,
splits into structured overlapping chunks, validates corpus quality, and generates:
- data/processed/chunks.json
- results/corpus_statistics.json
- results/corpus_validation.csv
"""

import csv
import json
import logging
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Tuple
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/retrieval.yaml")
RESULTS_DIR = Path("results")
PROCESSED_DIR = Path("data/processed")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "chunking": {"chunk_size": 400, "overlap": 50, "min_chunk_length": 30}
    }


def normalize_clinical_text(text: str) -> str:
    """
    Applies deterministic clinical text normalization while preserving
    medical terms, units, dosages, abbreviations, and chemical notations.
    """
    if not text:
        return ""
    # Normalize unicode (NFKC)
    text = unicodedata.normalize("NFKC", text)
    # Replace non-breaking spaces and irregular whitespace
    text = text.replace("\xa0", " ").replace("\u200b", "")
    # Normalize excessive newlines while preserving paragraph separation
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize repeated horizontal spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Normalize repeated dash/separator lines
    text = re.sub(r"[-_=]{4,}", "---", text)
    return text.strip()


def recursive_sentence_chunking(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks respecting sentence boundaries and paragraph structures.
    chunk_size and overlap are specified in characters / approximate token length.
    """
    if len(text) <= chunk_size:
        return [text]

    # Split into sentences using punctuation boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sent_len = len(sentence)
        if current_length + sent_len > chunk_size and current_chunk:
            chunk_str = " ".join(current_chunk).strip()
            if chunk_str:
                chunks.append(chunk_str)
            # Apply overlap: carry over last sentence(s) if within overlap budget
            overlap_chunk = []
            overlap_len = 0
            for prev_sent in reversed(current_chunk):
                if overlap_len + len(prev_sent) <= overlap:
                    overlap_chunk.insert(0, prev_sent)
                    overlap_len += len(prev_sent)
                else:
                    break
            current_chunk = overlap_chunk
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += sent_len

    if current_chunk:
        chunk_str = " ".join(current_chunk).strip()
        if chunk_str:
            chunks.append(chunk_str)

    return chunks


def process_corpus():
    config = load_config()
    chunk_cfg = config.get("chunking", {})
    chunk_size = chunk_cfg.get("chunk_size", 400)
    overlap = chunk_cfg.get("overlap", 50)
    min_length = chunk_cfg.get("min_chunk_length", 30)

    corpus_file = PROCESSED_DIR / "normalized_corpus.json"
    if not corpus_file.exists():
        raise FileNotFoundError(f"Normalized corpus not found at {corpus_file}. Run ingest_corpus.py first.")

    with open(corpus_file, "r", encoding="utf-8") as f:
        documents = json.load(f)

    logger.info(f"Loaded {len(documents)} raw normalized documents for chunking.")

    all_chunks = []
    validation_records = []
    seen_texts = set()
    duplicate_count = 0
    empty_count = 0
    source_distribution = {}

    for doc_idx, doc in enumerate(documents):
        doc_id = doc.get("document_id", f"doc_{doc_idx:04d}")
        title = doc.get("title", "")
        raw_text = doc.get("text", "")
        source = doc.get("source", "unknown")
        meta = doc.get("metadata", {})

        source_distribution[source] = source_distribution.get(source, 0) + 1

        cleaned_text = normalize_clinical_text(raw_text)
        if not cleaned_text:
            empty_count += 1
            validation_records.append({
                "item_id": doc_id,
                "type": "document",
                "source": source,
                "char_length": 0,
                "word_count": 0,
                "status": "REJECTED_EMPTY",
                "reason": "Text is empty after normalization"
            })
            continue

        raw_chunks = recursive_sentence_chunking(cleaned_text, chunk_size=chunk_size, overlap=overlap)

        for pos, c_text in enumerate(raw_chunks):
            c_text_clean = c_text.strip()
            if len(c_text_clean) < min_length:
                continue

            chunk_id = f"{doc_id}_chunk_{pos:03d}"
            text_hash = hash(c_text_clean)
            is_dup = text_hash in seen_texts
            if is_dup:
                duplicate_count += 1

            seen_texts.add(text_hash)
            word_count = len(c_text_clean.split())

            chunk_obj = {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "title": title,
                "text": c_text_clean,
                "source": source,
                "chunk_position": pos,
                "char_length": len(c_text_clean),
                "word_count": word_count,
                "metadata": meta
            }
            all_chunks.append(chunk_obj)

            validation_records.append({
                "item_id": chunk_id,
                "type": "chunk",
                "source": source,
                "char_length": len(c_text_clean),
                "word_count": word_count,
                "status": "VALID" if not is_dup else "DUPLICATE_FLAGGED",
                "reason": "Passed normalization quality check" if not is_dup else "Duplicate text chunk detected"
            })

    # Output chunks
    chunks_out = PROCESSED_DIR / "chunks.json"
    with open(chunks_out, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {len(all_chunks)} chunks to {chunks_out}")

    # Compute Statistics
    lengths = [c["char_length"] for c in all_chunks]
    words = [c["word_count"] for c in all_chunks]
    lengths.sort()
    median_len = lengths[len(lengths) // 2] if lengths else 0
    avg_len = round(sum(lengths) / len(lengths), 2) if lengths else 0
    avg_words = round(sum(words) / len(words), 2) if words else 0

    stats = {
        "number_of_documents": len(documents),
        "number_of_chunks": len(all_chunks),
        "average_chunk_char_length": avg_len,
        "median_chunk_char_length": median_len,
        "minimum_chunk_char_length": min(lengths) if lengths else 0,
        "maximum_chunk_char_length": max(lengths) if lengths else 0,
        "average_word_count": avg_words,
        "duplicate_chunk_count": duplicate_count,
        "empty_chunk_count": empty_count,
        "source_distribution": source_distribution,
        "chunking_parameters": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "min_length": min_length,
            "strategy": chunk_cfg.get("strategy", "recursive_sentence")
        }
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_out = RESULTS_DIR / "corpus_statistics.json"
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Corpus statistics saved to {stats_out}")

    val_csv = RESULTS_DIR / "corpus_validation.csv"
    if validation_records:
        with open(val_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(validation_records[0].keys()))
            writer.writeheader()
            writer.writerows(validation_records)
        logger.info(f"Corpus validation CSV saved to {val_csv}")


if __name__ == "__main__":
    process_corpus()
