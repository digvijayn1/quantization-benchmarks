"""
FAISS Index Save Utility.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System
"""

import argparse
import logging
import sys
from pathlib import Path
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def save_faiss_index(embeddings_path: str, output_path: str):
    emb_path = Path(embeddings_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(emb_path).astype(np.float32)
    n_vecs, dim = embeddings.shape
    logger.info(f"Creating FAISS IndexFlatIP with {n_vecs} vectors of dimension {dim}...")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(out_path))
    logger.info(f"FAISS index successfully saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="Path to .npy embeddings array")
    parser.add_argument("--output", required=True, help="Target path for .faiss index file")
    args = parser.parse_args()
    save_faiss_index(args.embeddings, args.output)
