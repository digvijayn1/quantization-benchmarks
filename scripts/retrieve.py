"""
Top-k Dense Retrieval Engine with Granular Latency Instrumentation.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System

Performs exact dense vector similarity retrieval over FAISS indexes, measures
fine-grained stage latencies (preprocessing, embedding generation, FAISS search, metadata lookup),
and returns fully resolved chunk records with similarity scores and ranking.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import faiss
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
RESULTS_DIR = Path("results/retrieval")


class MedicalRetriever:
    def __init__(self, precision: str = "fp32", device: str = "CPU", index_dir: Optional[str] = None):
        self.precision = precision.lower()
        self.device = device
        
        norm_prec = "int8_embedding" if "int8" in self.precision else "fp32"
        self.model_dir = Path(f"models/embeddings/{'int8' if 'int8' in self.precision else 'fp32'}")
        self.index_dir = Path(index_dir) if index_dir else Path(f"indexes/faiss/{norm_prec}")
        
        self._load_embedding_model()
        self._load_index()

    def _load_embedding_model(self):
        model_xml = self.model_dir / "openvino_model.xml"
        if not model_xml.exists():
            raise FileNotFoundError(f"Embedding model not found at {model_xml}. Run export_embedding_model.py first.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.core = ov.Core()
        ov_model = self.core.read_model(model_xml)
        self.compiled_model = self.core.compile_model(ov_model, self.device)
        self.input_names = [i.any_name for i in ov_model.inputs]
        logger.info(f"Retriever initialized embedding model [{self.precision.upper()}] on {self.device}")

    def _load_index(self):
        index_file = self.index_dir / "index.faiss"
        meta_file = self.index_dir / "metadata.json"
        
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_file}. Run build_faiss_index.py first.")
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata mapping not found at {meta_file}")
            
        self.index = faiss.read_index(str(index_file))
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.metadata_chunks = meta.get("chunks", [])
        logger.info(f"Retriever loaded FAISS index [{self.index.ntotal} vectors] from {self.index_dir}")

    def embed_query(self, query: str) -> Tuple[np.ndarray, float, float]:
        """Tokenizes, executes OpenVINO model, mean-pools, and normalizes query vector."""
        t_pre_start = time.perf_counter()
        inputs = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="np"
        )
        feed_dict = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        if "token_type_ids" in self.input_names:
            feed_dict["token_type_ids"] = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))
        prep_latency = (time.perf_counter() - t_pre_start) * 1000.0

        t_emb_start = time.perf_counter()
        res = self.compiled_model(feed_dict)
        token_embs = list(res.values())[0]

        # Mean pooling
        mask = inputs["attention_mask"][:, :, np.newaxis].astype(np.float32)
        sum_embs = np.sum(token_embs * mask, axis=1)
        sum_mask = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = sum_embs / sum_mask

        # L2 normalize
        norm = np.linalg.norm(pooled, axis=-1, keepdims=True)
        norm = np.clip(norm, a_min=1e-12, a_max=None)
        query_vec = (pooled / norm).astype(np.float32)
        emb_latency = (time.perf_counter() - t_emb_start) * 1000.0

        return query_vec, prep_latency, emb_latency

    def retrieve(self, query: str, top_k: int = 5, query_id: str = "q_unknown") -> Dict[str, Any]:
        """
        Retrieves top_k matching chunks for query with high-resolution latency timing.
        """
        t_total_start = time.perf_counter()

        # 1. Query Preprocessing & Embedding
        query_vec, prep_lat, emb_lat = self.embed_query(query)

        # 2. FAISS Vector Search
        t_faiss_start = time.perf_counter()
        scores, indices = self.index.search(query_vec, top_k)
        faiss_lat = (time.perf_counter() - t_faiss_start) * 1000.0

        # 3. Metadata Lookup & Assembly
        t_meta_start = time.perf_counter()
        retrieved_chunks = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            faiss_idx = int(idx)
            if 0 <= faiss_idx < len(self.metadata_chunks):
                chunk_meta = self.metadata_chunks[faiss_idx]
                retrieved_chunks.append({
                    "rank": rank,
                    "chunk_id": chunk_meta["chunk_id"],
                    "document_id": chunk_meta["document_id"],
                    "similarity_score": round(float(score), 6),
                    "title": chunk_meta.get("title", ""),
                    "source": chunk_meta.get("source", ""),
                    "text": chunk_meta["text"],
                    "metadata": chunk_meta.get("metadata", {})
                })
            else:
                logger.warning(f"Index {faiss_idx} out of bounds for metadata list.")
        meta_lat = (time.perf_counter() - t_meta_start) * 1000.0
        total_lat = (time.perf_counter() - t_total_start) * 1000.0

        result = {
            "query_id": query_id,
            "query": query,
            "top_k": top_k,
            "precision": self.precision,
            "latency": {
                "query_preprocessing_latency_ms": round(prep_lat, 3),
                "embedding_generation_latency_ms": round(emb_lat, 3),
                "faiss_search_latency_ms": round(faiss_lat, 3),
                "metadata_lookup_latency_ms": round(meta_lat, 3),
                "total_retrieval_latency_ms": round(total_lat, 3)
            },
            "retrieved_chunks": retrieved_chunks
        }
        return result


def main():
    parser = argparse.ArgumentParser(description="Medical QA Dense Retriever")
    parser.add_argument("--query", type=str, default="What are the primary diagnostic thresholds for Type 2 Diabetes Mellitus?", help="Medical query string")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "int8"], help="Embedding precision")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--save", action="store_true", help="Save output to results/retrieval/")
    args = parser.parse_args()

    retriever = MedicalRetriever(precision=args.precision)
    # Warmup
    retriever.retrieve("Medical warmup query", top_k=1)
    
    res = retriever.retrieve(args.query, top_k=args.top_k, query_id="cli_query")
    print("\n" + "=" * 70)
    print(f"Query: {res['query']}")
    print(f"Retrieval Latency: Total={res['latency']['total_retrieval_latency_ms']:.2f}ms (Embed={res['latency']['embedding_generation_latency_ms']:.2f}ms, FAISS={res['latency']['faiss_search_latency_ms']:.2f}ms)")
    print("=" * 70)
    for c in res["retrieved_chunks"]:
        print(f"Rank {c['rank']} [Score: {c['similarity_score']:.4f}] Chunk ID: {c['chunk_id']}")
        print(f"  Source: {c['source']} | Title: {c['title'][:50]}")
        print(f"  Snippet: {c['text'][:120]}...\n")

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = RESULTS_DIR / f"sample_retrieval_{args.precision}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Saved retrieval trace to {out_file}")


if __name__ == "__main__":
    main()
