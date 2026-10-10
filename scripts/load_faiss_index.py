"""
FAISS Index Load and Query Utility.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_faiss_index(index_dir: str) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    idx_dir = Path(index_dir)
    idx_file = idx_dir / "index.faiss"
    meta_file = idx_dir / "metadata.json"

    if not idx_file.exists():
        raise FileNotFoundError(f"FAISS index file not found at {idx_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"FAISS metadata file not found at {meta_file}")

    index = faiss.read_index(str(idx_file))
    with open(meta_file, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    chunks = meta_data.get("chunks", [])
    logger.info(f"Loaded FAISS index with {index.ntotal} vectors and {len(chunks)} metadata chunks from {idx_dir}")
    return index, chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True, help="Directory containing index.faiss and metadata.json")
    args = parser.parse_args()
    index, chunks = load_faiss_index(args.index_dir)
    print(f"Index successfully loaded: Total vectors={index.ntotal}, Total chunks={len(chunks)}")
