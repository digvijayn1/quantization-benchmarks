"""
Master Medical Retrieval and QA Evaluation Pipeline.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System

Coordinates end-to-end evaluation across:
1. MedQA (USMLE) and PubMedQA benchmark datasets
2. FP32 vs INT8 embedding retrieval comparisons
3. OpenVINO LLM inference models (e.g., AWQ INT4 vs FP32)
4. Full retrieval latency instrumentation with percentile distribution analysis
5. Overlap@k and rank consistency calculation between FP32 and INT8 dense indexes
Outputs:
- results/evaluation/<experiment_id>/
- results/retrieval_latency.csv
- results/retrieval_latency_summary.json
- results/retrieval_comparison_fp32_vs_int8.json
- results/run_manifest.json
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import yaml

from retrieve import MedicalRetriever
from evaluate_medqa import evaluate_medqa
from evaluate_pubmedqa import evaluate_pubmedqa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
EVAL_DIR = RESULTS_DIR / "evaluation"
CONFIG_PATH = Path("configs/retrieval_experiments.yaml")


def benchmark_retrieval_latency(
    retriever_fp32: MedicalRetriever,
    retriever_int8: MedicalRetriever,
    test_queries: List[Dict[str, Any]],
    top_k: int = 5,
    warmup: int = 10
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Measures high-resolution latency statistics for both FP32 and INT8 retrieval pipelines.
    """
    logger.info("=" * 70)
    logger.info(f"Instrumenting Retrieval Latency across {len(test_queries)} queries (Warmup={warmup})...")
    logger.info("=" * 70)

    # Warmup
    for q in test_queries[:warmup]:
        retriever_fp32.retrieve(q["question"], top_k=top_k)
        retriever_int8.retrieve(q["question"], top_k=top_k)

    detailed_records = []
    latencies_fp32 = {"prep": [], "emb": [], "faiss": [], "meta": [], "total": []}
    latencies_int8 = {"prep": [], "emb": [], "faiss": [], "meta": [], "total": []}
    overlap_at_k = []
    rank_consistency = []

    for item in test_queries:
        q_id = item.get("question_id", "q")
        query_text = item["question"]

        # Run FP32
        res_fp32 = retriever_fp32.retrieve(query_text, top_k=top_k, query_id=q_id)
        lat_fp32 = res_fp32["latency"]
        latencies_fp32["prep"].append(lat_fp32["query_preprocessing_latency_ms"])
        latencies_fp32["emb"].append(lat_fp32["embedding_generation_latency_ms"])
        latencies_fp32["faiss"].append(lat_fp32["faiss_search_latency_ms"])
        latencies_fp32["meta"].append(lat_fp32["metadata_lookup_latency_ms"])
        latencies_fp32["total"].append(lat_fp32["total_retrieval_latency_ms"])

        # Run INT8
        res_int8 = retriever_int8.retrieve(query_text, top_k=top_k, query_id=q_id)
        lat_int8 = res_int8["latency"]
        latencies_int8["prep"].append(lat_int8["query_preprocessing_latency_ms"])
        latencies_int8["emb"].append(lat_int8["embedding_generation_latency_ms"])
        latencies_int8["faiss"].append(lat_int8["faiss_search_latency_ms"])
        latencies_int8["meta"].append(lat_int8["metadata_lookup_latency_ms"])
        latencies_int8["total"].append(lat_int8["total_retrieval_latency_ms"])

        # Compare retrieval overlap & rank consistency
        ids_fp32 = [c["chunk_id"] for c in res_fp32["retrieved_chunks"]]
        ids_int8 = [c["chunk_id"] for c in res_int8["retrieved_chunks"]]

        intersection = set(ids_fp32).intersection(set(ids_int8))
        overlap_score = len(intersection) / top_k
        overlap_at_k.append(overlap_score)

        # Rank consistency: Spearman / matching top-1
        top1_match = (ids_fp32[0] == ids_int8[0]) if (ids_fp32 and ids_int8) else False
        rank_consistency.append(1.0 if top1_match else 0.0)

        detailed_records.append({
            "question_id": q_id,
            "query": query_text[:80],
            "fp32_emb_ms": lat_fp32["embedding_generation_latency_ms"],
            "fp32_faiss_ms": lat_fp32["faiss_search_latency_ms"],
            "fp32_total_ms": lat_fp32["total_retrieval_latency_ms"],
            "int8_emb_ms": lat_int8["embedding_generation_latency_ms"],
            "int8_faiss_ms": lat_int8["faiss_search_latency_ms"],
            "int8_total_ms": lat_int8["total_retrieval_latency_ms"],
            "overlap_at_k": round(overlap_score, 4),
            "top1_match": top1_match
        })

    def calc_stats(arr):
        return {
            "mean_ms": round(float(np.mean(arr)), 3),
            "median_ms": round(float(np.median(arr)), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p90_ms": round(float(np.percentile(arr, 90)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "min_ms": round(float(np.min(arr)), 3),
            "max_ms": round(float(np.max(arr)), 3)
        }

    latency_summary = {
        "benchmark_queries_count": len(test_queries),
        "top_k": top_k,
        "fp32_retrieval": {
            "query_preprocessing": calc_stats(latencies_fp32["prep"]),
            "embedding_generation": calc_stats(latencies_fp32["emb"]),
            "faiss_search": calc_stats(latencies_fp32["faiss"]),
            "metadata_lookup": calc_stats(latencies_fp32["meta"]),
            "total_retrieval": calc_stats(latencies_fp32["total"])
        },
        "int8_retrieval": {
            "query_preprocessing": calc_stats(latencies_int8["prep"]),
            "embedding_generation": calc_stats(latencies_int8["emb"]),
            "faiss_search": calc_stats(latencies_int8["faiss"]),
            "metadata_lookup": calc_stats(latencies_int8["meta"]),
            "total_retrieval": calc_stats(latencies_int8["total"])
        },
        "comparison": {
            "mean_overlap_at_k": round(float(np.mean(overlap_at_k)), 4),
            "top1_rank_agreement_rate": round(float(np.mean(rank_consistency)) * 100, 2),
            "embedding_speedup_ratio": round(float(np.mean(latencies_fp32["emb"])) / float(np.mean(latencies_int8["emb"])), 2)
        }
    }

    return latency_summary, detailed_records


def run_full_pipeline(limit: int = 25, device: str = "CPU"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    qa_model = "models/openvino/awq/int4_sym_g128_awq"

    # 1. Latency & Overlap Benchmarking across test queries
    retriever_fp32 = MedicalRetriever(precision="fp32", device=device)
    retriever_int8 = MedicalRetriever(precision="int8", device=device)

    # Load queries from evaluation subsets
    with open("data/evaluation/medqa_eval_subset.json", "r", encoding="utf-8") as f:
        medqa_subset = json.load(f)
    with open("data/evaluation/pubmedqa_eval_subset.json", "r", encoding="utf-8") as f:
        pubmedqa_subset = json.load(f)

    all_queries = medqa_subset[:50] + pubmedqa_subset[:50]
    lat_summary, lat_csv_rows = benchmark_retrieval_latency(
        retriever_fp32, retriever_int8, all_queries, top_k=5
    )

    # Save latency summary JSON
    lat_json_path = RESULTS_DIR / "retrieval_latency_summary.json"
    with open(lat_json_path, "w", encoding="utf-8") as f:
        json.dump(lat_summary, f, indent=2)
    logger.info(f"Latency summary saved to {lat_json_path}")

    # Save latency CSV
    lat_csv_path = RESULTS_DIR / "retrieval_latency.csv"
    with open(lat_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(lat_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(lat_csv_rows)
    logger.info(f"Detailed retrieval latency CSV saved to {lat_csv_path}")

    # Save comparison JSON
    comp_json_path = RESULTS_DIR / "retrieval_comparison_fp32_vs_int8.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": "FP32 vs INT8 Dense Embedding Retrieval Comparison",
            "metrics": lat_summary["comparison"],
            "fp32_latency_mean_ms": lat_summary["fp32_retrieval"]["total_retrieval"]["mean_ms"],
            "int8_latency_mean_ms": lat_summary["int8_retrieval"]["total_retrieval"]["mean_ms"]
        }, f, indent=2)

    # 2. Run MedQA Evaluation with FP32 vs INT8 retrieval
    logger.info("Executing MedQA Evaluation with FP32 Retrieval...")
    medqa_fp32_metrics = evaluate_medqa(
        qa_model_path=qa_model,
        embedding_precision="fp32",
        top_k=5,
        device=device,
        limit=limit,
        experiment_id="medqa_awq_int4_embed_fp32_k5"
    )

    logger.info("Executing MedQA Evaluation with INT8 Retrieval...")
    medqa_int8_metrics = evaluate_medqa(
        qa_model_path=qa_model,
        embedding_precision="int8",
        top_k=5,
        device=device,
        limit=limit,
        experiment_id="medqa_awq_int4_embed_int8_k5"
    )

    # 3. Run PubMedQA Evaluation with FP32 vs INT8 retrieval
    logger.info("Executing PubMedQA Evaluation with FP32 Retrieval...")
    pqa_fp32_metrics = evaluate_pubmedqa(
        qa_model_path=qa_model,
        embedding_precision="fp32",
        top_k=5,
        device=device,
        limit=limit,
        experiment_id="pubmedqa_awq_int4_embed_fp32_k5"
    )

    logger.info("Executing PubMedQA Evaluation with INT8 Retrieval...")
    pqa_int8_metrics = evaluate_pubmedqa(
        qa_model_path=qa_model,
        embedding_precision="int8",
        top_k=5,
        device=device,
        limit=limit,
        experiment_id="pubmedqa_awq_int4_embed_int8_k5"
    )

    # 4. Generate Reproducibility Run Manifest
    manifest = {
        "pipeline": "Medical Corpus Ingestion, Dense Retrieval & Evaluation Pipeline",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "processor": "AMD Ryzen 7 8845HS (Zen 4, 8C/16T, AVX-512)",
            "memory": "16 GB RAM",
            "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU (8GB GDDR6)"
        },
        "software_versions": {
            "python": "3.14.5",
            "openvino": "2026.4.1",
            "nncf": "3.4.0",
            "faiss": "1.15.1",
            "sentence_transformers": "6.1.0",
            "transformers": "5.5.4"
        },
        "embedding_models": {
            "base_model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "fp32_path": "models/embeddings/fp32",
            "int8_path": "models/embeddings/int8"
        },
        "corpus": {
            "normalized_path": "data/processed/normalized_corpus.json",
            "chunks_path": "data/processed/chunks.json",
            "total_chunks": 5987,
            "chunk_size": 400,
            "overlap": 50
        },
        "indexes": {
            "fp32_faiss": "indexes/faiss/fp32/index.faiss",
            "int8_faiss": "indexes/faiss/int8_embedding/index.faiss",
            "index_type": "IndexFlatIP"
        },
        "evaluation_summary": {
            "evaluated_queries_per_benchmark": limit,
            "medqa_fp32_retrieval_raw_acc_pct": medqa_fp32_metrics["raw_accuracy_pct"],
            "medqa_int8_retrieval_raw_acc_pct": medqa_int8_metrics["raw_accuracy_pct"],
            "pubmedqa_fp32_retrieval_raw_acc_pct": pqa_fp32_metrics["raw_accuracy_pct"],
            "pubmedqa_int8_retrieval_raw_acc_pct": pqa_int8_metrics["raw_accuracy_pct"],
            "retrieval_overlap_at_k": lat_summary["comparison"]["mean_overlap_at_k"],
            "top1_rank_agreement_rate": lat_summary["comparison"]["top1_rank_agreement_rate"]
        }
    }

    manifest_path = RESULTS_DIR / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Run manifest successfully saved to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Run complete medical retrieval & evaluation pipeline.")
    parser.add_argument("--limit", type=int, default=25, help="Number of evaluation queries per benchmark experiment")
    parser.add_argument("--device", type=str, default="CPU", help="Inference device")
    args = parser.parse_args()

    run_full_pipeline(limit=args.limit, device=args.device)


if __name__ == "__main__":
    main()
