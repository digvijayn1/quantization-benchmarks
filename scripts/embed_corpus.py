"""
Corpus Batch Embedding Generation.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System

Generates normalized dense vector embeddings for all corpus chunks using OpenVINO IR
embedding models (FP32 and INT8 variants), preserving chunk IDs and metadata mapping.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import openvino as ov
from transformers import AutoTokenizer
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/retrieval.yaml")
PROCESSED_DIR = Path("data/processed")
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    input_mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


def normalize_vectors(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vecs / norms


def generate_embeddings(
    precision: str = "fp32",
    batch_size: int = 32,
    device: str = "CPU",
    force_recompute: bool = False
) -> Tuple[np.ndarray, Path]:
    config = load_config()
    emb_cfg = config.get("embedding", {})
    max_seq_len = emb_cfg.get("max_sequence_length", 256)

    model_dir = Path(f"models/embeddings/{precision.lower()}")
    model_xml = model_dir / "openvino_model.xml"

    if not model_xml.exists():
        raise FileNotFoundError(f"OpenVINO embedding model not found at {model_xml}. Run export_embedding_model.py first.")

    out_npy = PROCESSED_DIR / f"embeddings_{precision.lower()}.npy"
    manifest_out = PROCESSED_DIR / f"embeddings_{precision.lower()}_manifest.json"

    if out_npy.exists() and not force_recompute:
        logger.info(f"Existing embeddings found at {out_npy}. Loading cached embeddings...")
        embeddings = np.load(out_npy)
        logger.info(f"Loaded cached embeddings shape: {embeddings.shape}")
        return embeddings, out_npy

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks not found at {CHUNKS_FILE}. Run chunk_corpus.py first.")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info("=" * 70)
    logger.info(f"Generating Corpus Embeddings: Precision={precision.upper()}, Device={device}, Batches={len(chunks)//batch_size + 1}")
    logger.info(f"Model Directory: {model_dir}")
    logger.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    core = ov.Core()
    model = core.read_model(model_xml)
    compiled_model = core.compile_model(model, device)

    input_names = [i.any_name for i in model.inputs]
    logger.info(f"Model compiled on {device}. Model inputs: {input_names}")

    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    all_embeddings = []

    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="np"
        )
        feed_dict = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        if "token_type_ids" in input_names:
            feed_dict["token_type_ids"] = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))

        res = compiled_model(feed_dict)
        token_embs = list(res.values())[0] # [batch_size, seq_len, 384]

        pooled = mean_pooling(token_embs, inputs["attention_mask"])
        normed = normalize_vectors(pooled)
        all_embeddings.append(normed)

        if (i // batch_size) % 25 == 0 or (i + batch_size) >= len(texts):
            logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks...")

    total_time = time.time() - t0
    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    logger.info(f"Generated embeddings matrix shape: {embeddings_matrix.shape} in {total_time:.2f}s ({len(texts)/total_time:.1f} chunks/sec)")

    np.save(out_npy, embeddings_matrix)
    logger.info(f"Saved embeddings array to {out_npy} ({out_npy.stat().st_size / (1024*1024):.2f} MB)")

    manifest = {
        "precision": precision.lower(),
        "model_dir": str(model_dir).replace("\\", "/"),
        "total_vectors": embeddings_matrix.shape[0],
        "vector_dimension": embeddings_matrix.shape[1],
        "total_generation_time_seconds": round(total_time, 2),
        "throughput_chunks_per_sec": round(len(texts) / total_time, 2),
        "normalized": True,
        "chunk_ids": chunk_ids
    }
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return embeddings_matrix, out_npy


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for medical corpus chunks.")
    parser.add_argument("--precision", type=str, default="all", choices=["fp32", "int8", "all"], help="Embedding precision")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument("--device", type=str, default="CPU", help="Inference device (CPU, GPU.0, GPU.1)")
    parser.add_argument("--force", action="store_true", help="Force recomputation of embeddings")
    args = parser.parse_args()

    precisions = ["fp32", "int8"] if args.precision == "all" else [args.precision]
    for p in precisions:
        generate_embeddings(precision=p, batch_size=args.batch_size, device=args.device, force_recompute=args.force)


if __name__ == "__main__":
    main()
