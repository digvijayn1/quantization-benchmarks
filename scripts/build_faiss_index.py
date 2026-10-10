"""
FAISS Vector Index Construction and Persistence Validation.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System

Constructs exact inner-product (Cosine Similarity on L2 normalized vectors) FAISS indexes
for both FP32 and INT8 embedding variants, maintains explicit metadata mappings,
and validates index persistence, reload consistency, and retrieval integrity.
Outputs:
- indexes/faiss/fp32/index.faiss + metadata.json
- indexes/faiss/int8_embedding/index.faiss + metadata.json
- results/faiss_persistence_validation.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import faiss
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/retrieval.yaml")
PROCESSED_DIR = Path("data/processed")
INDEXES_DIR = Path("indexes/faiss")
RESULTS_DIR = Path("results")
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {"index": {"index_type": "IndexFlatIP"}}


def build_and_persist_index(precision: str = "fp32") -> Dict[str, Any]:
    norm_precision = "int8_embedding" if "int8" in precision.lower() else "fp32"
    emb_file = PROCESSED_DIR / f"embeddings_{'int8' if 'int8' in precision.lower() else 'fp32'}.npy"
    out_dir = INDEXES_DIR / norm_precision
    out_dir.mkdir(parents=True, exist_ok=True)

    if not emb_file.exists():
        raise FileNotFoundError(f"Embeddings not found at {emb_file}. Run embed_corpus.py first.")
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks not found at {CHUNKS_FILE}. Run chunk_corpus.py first.")

    embeddings = np.load(emb_file).astype(np.float32)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({embeddings.shape[0]})")

    n_vecs, dim = embeddings.shape
    logger.info("=" * 70)
    logger.info(f"Building FAISS Index: {norm_precision} | Vectors: {n_vecs} | Dimension: {dim}")
    logger.info("=" * 70)

    # Construct Inner Product Flat Index (IndexFlatIP) for exact Cosine search
    t0 = time.time()
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    build_time = time.time() - t0
    logger.info(f"FAISS Index built in {build_time*1000:.2f}ms (Total indexed vectors: {index.ntotal})")

    # Persist Index
    index_path = out_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    index_size_mb = round(index_path.stat().st_size / (1024 * 1024), 2)
    logger.info(f"Saved FAISS index to {index_path} ({index_size_mb} MB)")

    # Build and persist explicit metadata mapping
    metadata_records = []
    for idx, c in enumerate(chunks):
        metadata_records.append({
            "faiss_id": idx,
            "chunk_id": c["chunk_id"],
            "document_id": c["document_id"],
            "title": c.get("title", ""),
            "source": c.get("source", ""),
            "chunk_position": c.get("chunk_position", 0),
            "text": c["text"],
            "metadata": c.get("metadata", {})
        })

    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "index_type": "IndexFlatIP",
            "precision": norm_precision,
            "vector_dimension": dim,
            "vector_count": n_vecs,
            "metric": "cosine_similarity (Inner Product on L2 normalized vectors)",
            "chunks": metadata_records
        }, f, indent=2, ensure_ascii=False)

    meta_size_mb = round(metadata_path.stat().st_size / (1024 * 1024), 2)
    logger.info(f"Saved metadata mapping to {metadata_path} ({meta_size_mb} MB)")

    # Validate Persistence and Reload Consistency
    logger.info("Validating Index Reload & Search Consistency...")
    del index
    reloaded_index = faiss.read_index(str(index_path))
    if reloaded_index.ntotal != n_vecs:
        raise ValueError(f"Reloaded vector count mismatch: expected {n_vecs}, got {reloaded_index.ntotal}")

    # Perform consistency query using vector 0
    query_vec = embeddings[0:1]
    dists, indices = reloaded_index.search(query_vec, k=5)
    top_id = int(indices[0][0])
    top_score = float(dists[0][0])
    consistency_passed = (top_id == 0) and (abs(top_score - 1.0) < 1e-4)

    logger.info(f"Reload test query result: Top ID={top_id}, Score={top_score:.6f} | Consistent: {consistency_passed}")

    return {
        "variant": norm_precision,
        "index_type": "IndexFlatIP",
        "vector_dimension": dim,
        "number_of_vectors": n_vecs,
        "index_file_size_mb": index_size_mb,
        "metadata_file_size_mb": meta_size_mb,
        "index_build_time_seconds": round(build_time, 4),
        "reload_success": True,
        "retrieval_consistency": "CONSISTENT" if consistency_passed else "INCONSISTENT",
        "index_path": str(index_path).replace("\\", "/"),
        "metadata_path": str(metadata_path).replace("\\", "/")
    }


def main():
    parser = argparse.ArgumentParser(description="Build and validate FAISS indexes for medical corpus.")
    parser.add_argument("--precision", type=str, default="all", choices=["fp32", "int8", "all"], help="Embedding precision")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    precisions = ["fp32", "int8"] if args.precision == "all" else [args.precision]

    validation_reports = []
    for p in precisions:
        rep = build_and_persist_index(precision=p)
        validation_reports.append(rep)

    val_json_path = RESULTS_DIR / "faiss_persistence_validation.json"
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "indexes": validation_reports
        }, f, indent=2)

    logger.info(f"FAISS persistence validation report saved to {val_json_path}")


if __name__ == "__main__":
    main()
