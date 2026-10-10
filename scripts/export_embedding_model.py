"""
Sentence Embedding Model Export & INT8 Post-Training Quantization.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System

Exports sentence-transformers/all-MiniLM-L6-v2 to OpenVINO Intermediate Representation (FP32),
compresses model weights to INT8 using Intel NNCF, performs embedding equivalence validation,
and generates:
- models/embeddings/fp32/
- models/embeddings/int8/
- results/embedding_compression.json
- results/embedding_validation.csv
"""

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import nncf
import numpy as np
import openvino as ov
from optimum.intel.openvino import OVModelForFeatureExtraction
from transformers import AutoTokenizer
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/retrieval.yaml")
RESULTS_DIR = Path("results")
MODELS_BASE_DIR = Path("models/embeddings")
FP32_DIR = MODELS_BASE_DIR / "fp32"
INT8_DIR = MODELS_BASE_DIR / "int8"
CHUNKS_FILE = Path("data/processed/chunks.json")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "embedding": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "max_sequence_length": 256,
            "normalize": True
        }
    }


def mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean Pooling - Take attention mask into account for correct averaging."""
    input_mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    pooled = sum_embeddings / sum_mask
    return pooled


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """L2 normalizes embedding vectors."""
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.clip(norm, a_min=1e-12, a_max=None)
    return vec / norm


def export_and_quantize():
    config = load_config()
    emb_cfg = config.get("embedding", {})
    model_name = emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    max_seq_len = emb_cfg.get("max_sequence_length", 256)
    dim = emb_cfg.get("dimension", 384)

    FP32_DIR.mkdir(parents=True, exist_ok=True)
    INT8_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"1. Exporting FP32 Embedding Model: {model_name}")
    logger.info("=" * 70)
    t0 = time.time()
    ov_model = OVModelForFeatureExtraction.from_pretrained(model_name, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ov_model.save_pretrained(FP32_DIR)
    tokenizer.save_pretrained(FP32_DIR)
    fp32_export_time = time.time() - t0
    logger.info(f"FP32 embedding model saved to {FP32_DIR} in {fp32_export_time:.2f}s")

    # Load OpenVINO IR for NNCF INT8 Quantization
    core = ov.Core()
    fp32_xml = FP32_DIR / "openvino_model.xml"
    model = core.read_model(fp32_xml)

    # Prepare representative calibration dataset
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks not found at {CHUNKS_FILE}. Run chunk_corpus.py first.")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Use first 64 representative chunks for calibration
    calib_subset = chunks[:64]
    calib_texts = [c["text"] for c in calib_subset]

    def transform_fn(text):
        inputs = tokenizer(text, padding=True, truncation=True, max_length=max_seq_len, return_tensors="np")
        feed_dict = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        if "token_type_ids" in [i.any_name for i in model.inputs]:
            feed_dict["token_type_ids"] = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))
        return feed_dict

    calib_dataset = nncf.Dataset(calib_texts, transform_fn)

    logger.info("=" * 70)
    logger.info("2. Compressing Embedding Weights to INT8 via NNCF...")
    logger.info("=" * 70)
    t0 = time.time()
    int8_compressed = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
    int8_xml = INT8_DIR / "openvino_model.xml"
    ov.save_model(int8_compressed, int8_xml)
    tokenizer.save_pretrained(INT8_DIR)
    int8_compress_time = time.time() - t0
    logger.info(f"INT8 embedding model saved to {INT8_DIR} in {int8_compress_time:.2f}s")

    # Calculate model footprints
    fp32_xml_mb = round(fp32_xml.stat().st_size / (1024 * 1024), 2)
    fp32_bin_mb = round((FP32_DIR / "openvino_model.bin").stat().st_size / (1024 * 1024), 2)
    fp32_total_mb = round(fp32_xml_mb + fp32_bin_mb, 2)

    int8_xml_mb = round(int8_xml.stat().st_size / (1024 * 1024), 2)
    int8_bin_mb = round((INT8_DIR / "openvino_model.bin").stat().st_size / (1024 * 1024), 2)
    int8_total_mb = round(int8_xml_mb + int8_bin_mb, 2)

    compression_ratio = round(fp32_total_mb / int8_total_mb, 2) if int8_total_mb > 0 else 1.0
    size_reduction = round((1.0 - (int8_total_mb / fp32_total_mb)) * 100, 1) if fp32_total_mb > 0 else 0.0

    logger.info("=" * 70)
    logger.info("3. Validating Embedding Equivalence (FP32 vs INT8)...")
    logger.info("=" * 70)

    compiled_fp32 = core.compile_model(model, "CPU")
    compiled_int8 = core.compile_model(int8_compressed, "CPU")

    # Run on 50 fixed evaluation text chunks
    val_chunks = chunks[:50]
    validation_records = []
    cos_similarities = []
    l2_distances = []

    for c in val_chunks:
        c_id = c["chunk_id"]
        c_text = c["text"]
        inputs = tokenizer(c_text, padding=True, truncation=True, max_length=max_seq_len, return_tensors="np")
        feed = {i.any_name: inputs[i.any_name] for i in model.inputs if i.any_name in inputs}

        # FP32 inference
        out_fp32 = list(compiled_fp32(feed).values())[0]
        pooled_fp32 = mean_pooling(out_fp32, inputs["attention_mask"])
        vec_fp32 = normalize_vector(pooled_fp32)[0]

        # INT8 inference
        out_int8 = list(compiled_int8(feed).values())[0]
        pooled_int8 = mean_pooling(out_int8, inputs["attention_mask"])
        vec_int8 = normalize_vector(pooled_int8)[0]

        cos_sim = float(np.dot(vec_fp32, vec_int8))
        l2_dist = float(np.linalg.norm(vec_fp32 - vec_int8))
        cos_similarities.append(cos_sim)
        l2_distances.append(l2_dist)

        validation_records.append({
            "chunk_id": c_id,
            "char_length": len(c_text),
            "cosine_similarity": round(cos_sim, 6),
            "l2_distance": round(l2_dist, 6),
            "status": "PASS" if cos_sim >= 0.99 else "WARNING"
        })

    mean_sim = round(float(np.mean(cos_similarities)), 6)
    min_sim = round(float(np.min(cos_similarities)), 6)
    max_sim = round(float(np.max(cos_similarities)), 6)
    std_sim = round(float(np.std(cos_similarities)), 6)
    mean_l2 = round(float(np.mean(l2_distances)), 6)

    logger.info(f"Embedding Validation Summary: Mean Cosine Sim: {mean_sim}, Min: {min_sim}, Std: {std_sim}, Mean L2: {mean_l2}")

    # Write compression summary
    comp_summary = {
        "embedding_model": model_name,
        "embedding_dimension": dim,
        "max_sequence_length": max_seq_len,
        "fp32_model": {
            "xml_size_mb": fp32_xml_mb,
            "bin_size_mb": fp32_bin_mb,
            "total_size_mb": fp32_total_mb,
            "export_time_seconds": round(fp32_export_time, 2)
        },
        "int8_model": {
            "xml_size_mb": int8_xml_mb,
            "bin_size_mb": int8_bin_mb,
            "total_size_mb": int8_total_mb,
            "compress_time_seconds": round(int8_compress_time, 2),
            "compression_ratio": f"{compression_ratio}x",
            "size_reduction_percent": f"{size_reduction}%"
        },
        "calibration": {
            "dataset": str(CHUNKS_FILE),
            "calibration_samples": len(calib_texts),
            "method": "nncf.compress_weights(mode=INT8_ASYM)"
        },
        "equivalence_metrics": {
            "validation_samples": len(val_chunks),
            "mean_cosine_similarity": mean_sim,
            "min_cosine_similarity": min_sim,
            "max_cosine_similarity": max_sim,
            "std_cosine_similarity": std_sim,
            "mean_l2_distance": mean_l2,
            "validation_status": "PASSED_HIGH_FIDELITY" if min_sim >= 0.99 else "DEGRADATION_DETECTED"
        }
    }

    comp_json_path = RESULTS_DIR / "embedding_compression.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comp_summary, f, indent=2)
    logger.info(f"Compression summary saved to {comp_json_path}")

    # Write validation CSV
    val_csv_path = RESULTS_DIR / "embedding_validation.csv"
    with open(val_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(validation_records[0].keys()))
        writer.writeheader()
        writer.writerows(validation_records)
    logger.info(f"Embedding validation CSV saved to {val_csv_path}")


if __name__ == "__main__":
    export_and_quantize()
