"""
PubMedQA End-to-End Retrieval & Evaluation Harness.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System

Evaluates OpenVINO LLM variants on the fixed PubMedQA stratified subset with dense retrieval,
prompt assembly, inference generation, deterministic decision extraction (yes/no/maybe), and metrics.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import openvino as ov
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

from retrieve import MedicalRetriever
from prompt_builder import MedicalPromptBuilder
from answer_extraction import parse_and_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

SUBSET_PATH = Path("data/evaluation/pubmedqa_eval_subset.json")
RESULTS_BASE_DIR = Path("results/evaluation")


def evaluate_pubmedqa(
    qa_model_path: str = "models/openvino/awq/int4_sym_g128_awq",
    embedding_precision: str = "fp32",
    top_k: int = 5,
    device: str = "CPU",
    limit: Optional[int] = None,
    experiment_id: Optional[str] = None
) -> Dict[str, Any]:
    qa_path = Path(qa_model_path)
    if not qa_path.exists():
        raise FileNotFoundError(f"OpenVINO QA Model not found at {qa_path}")
    if not SUBSET_PATH.exists():
        raise FileNotFoundError(f"Evaluation subset not found at {SUBSET_PATH}. Run prepare_eval_subsets.py first.")

    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if limit and limit > 0:
        samples = samples[:limit]

    exp_name = experiment_id or f"pubmedqa_{qa_path.name}_{embedding_precision}_k{top_k}"
    exp_dir = RESULTS_BASE_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"Starting PubMedQA Evaluation: {exp_name}")
    logger.info(f"QA Model: {qa_path} | Embedding: {embedding_precision.upper()} | Top-k: {top_k} | Queries: {len(samples)}")
    logger.info("=" * 70)

    retriever = MedicalRetriever(precision=embedding_precision, device=device)
    prompt_builder = MedicalPromptBuilder(max_context_tokens=1536)

    logger.info(f"Loading OpenVINO QA LLM from {qa_path} on {device}...")
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(qa_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = OVModelForCausalLM.from_pretrained(
        qa_path,
        device=device,
        compile=True,
        trust_remote_code=True
    )
    llm_compile_time = time.time() - t_load
    logger.info(f"QA LLM compiled in {llm_compile_time:.2f}s")

    predictions = []
    retrieval_records = []
    
    correct_count = 0
    parsed_count = 0
    unparsed_count = 0
    retrieval_latencies = []
    generation_latencies = []

    for idx, sample in enumerate(samples, start=1):
        q_id = sample["question_id"]
        question = sample["question"]
        gold = sample["gold_answer"]

        # 1. Retrieval
        ret_result = retriever.retrieve(question, top_k=top_k, query_id=q_id)
        retrieval_records.append(ret_result)
        ret_lat = ret_result["latency"]["total_retrieval_latency_ms"]
        retrieval_latencies.append(ret_lat)

        # 2. Prompt Construction
        prompt, p_meta = prompt_builder.build_prompt(
            question=question,
            retrieved_chunks=ret_result["retrieved_chunks"],
            options=None # PubMedQA is binary/ternary reasoning: yes, no, maybe
        )

        # 3. Model Inference
        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]
        
        t_gen_start = time.time()
        output_tokens = llm.generate(
            **inputs,
            max_new_tokens=48,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        gen_time = time.time() - t_gen_start
        generation_latencies.append(gen_time * 1000.0)

        new_tokens = output_tokens[0][input_len:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 4. Answer Extraction & Scoring
        score_res = parse_and_score(generated_text, gold, task_type="pubmedqa")
        is_parsed = score_res["is_parsed"]
        is_correct = score_res["is_correct"]

        if is_parsed:
            parsed_count += 1
            if is_correct:
                correct_count += 1
        else:
            unparsed_count += 1

        if is_correct:
            failure_mode = "CORRECT"
        elif not is_parsed:
            failure_mode = "UNPARSED"
        else:
            ret_text_all = " ".join([c["text"] for c in ret_result["retrieved_chunks"]])
            if len(ret_text_all) > 100:
                failure_mode = "GENERATION_FAILURE"
            else:
                failure_mode = "RETRIEVAL_FAILURE"

        pred_record = {
            "question_id": q_id,
            "question": question,
            "gold_answer": gold,
            "predicted_answer": score_res["extracted_answer"],
            "generated_text": generated_text,
            "is_parsed": is_parsed,
            "is_correct": is_correct,
            "failure_mode": failure_mode,
            "extraction_rule": score_res["extraction_rule"],
            "retrieval_latency_ms": round(ret_lat, 2),
            "generation_latency_ms": round(gen_time * 1000.0, 2),
            "retrieved_chunk_ids": [c["chunk_id"] for c in ret_result["retrieved_chunks"]]
        }
        predictions.append(pred_record)

        if idx % 10 == 0 or idx == len(samples):
            current_raw_acc = (correct_count / idx) * 100
            current_parsed_acc = (correct_count / parsed_count * 100) if parsed_count > 0 else 0.0
            logger.info(f"[{idx}/{len(samples)}] Raw Acc: {current_raw_acc:.1f}% | Parsed Acc: {current_parsed_acc:.1f}% | Parsed: {parsed_count}/{idx}")

    total_q = len(samples)
    raw_acc = round((correct_count / total_q) * 100, 2)
    parsed_acc = round((correct_count / parsed_count) * 100, 2) if parsed_count > 0 else 0.0
    parse_rate = round((parsed_count / total_q) * 100, 2)

    # 95% Wilson Score Confidence Interval
    z = 1.96
    p = correct_count / total_q
    denom = 1 + (z**2 / total_q)
    center = (p + (z**2 / (2 * total_q))) / denom
    spread = z * np.sqrt((p * (1 - p) / total_q) + (z**2 / (4 * (total_q**2)))) / denom
    ci_lower = round(max(0.0, (center - spread) * 100), 2)
    ci_upper = round(min(100.0, (center + spread) * 100), 2)

    metrics = {
        "dataset": "PubMedQA-pqa_labeled",
        "qa_model": str(qa_path).replace("\\", "/"),
        "embedding_precision": embedding_precision,
        "top_k": top_k,
        "total_questions": total_q,
        "parsed_questions": parsed_count,
        "unparsed_questions": unparsed_count,
        "correct_answers": correct_count,
        "raw_accuracy_pct": raw_acc,
        "parsed_accuracy_pct": parsed_acc,
        "parse_rate_pct": parse_rate,
        "confidence_interval_95_pct": [ci_lower, ci_upper],
        "latency_stats": {
            "avg_retrieval_latency_ms": round(float(np.mean(retrieval_latencies)), 2),
            "p50_retrieval_latency_ms": round(float(np.percentile(retrieval_latencies, 50)), 2),
            "p95_retrieval_latency_ms": round(float(np.percentile(retrieval_latencies, 95)), 2),
            "avg_generation_latency_ms": round(float(np.mean(generation_latencies)), 2)
        },
        "failure_distribution": {
            "CORRECT": sum(1 for p in predictions if p["failure_mode"] == "CORRECT"),
            "GENERATION_FAILURE": sum(1 for p in predictions if p["failure_mode"] == "GENERATION_FAILURE"),
            "RETRIEVAL_FAILURE": sum(1 for p in predictions if p["failure_mode"] == "RETRIEVAL_FAILURE"),
            "UNPARSED": sum(1 for p in predictions if p["failure_mode"] == "UNPARSED")
        }
    }

    # 1. Predictions JSONL
    pred_path = exp_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    # 2. Retrieval Results JSONL
    ret_path = exp_dir / "retrieval_results.jsonl"
    with open(ret_path, "w", encoding="utf-8") as f:
        for r in retrieval_records:
            f.write(json.dumps(r) + "\n")

    # 3. Metrics JSON
    metrics_path = exp_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 4. Run Config JSON
    run_config = {
        "experiment_id": exp_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "qa_model": str(qa_path).replace("\\", "/"),
        "embedding_precision": embedding_precision,
        "top_k": top_k,
        "device": device,
        "limit": limit
    }
    with open(exp_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    logger.info(f"PubMedQA Evaluation finished. Metrics: Raw Acc: {raw_acc}% | Parsed Acc: {parsed_acc}% | Saved to {exp_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenVINO Medical QA model on PubMedQA subset.")
    parser.add_argument("--qa-model", type=str, default="models/openvino/awq/int4_sym_g128_awq", help="Path to OpenVINO QA model directory")
    parser.add_argument("--embedding-precision", type=str, default="fp32", choices=["fp32", "int8"], help="Embedding model precision")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval chunks")
    parser.add_argument("--device", type=str, default="CPU", help="Inference device")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation samples")
    parser.add_argument("--exp-id", type=str, default=None, help="Custom experiment ID")
    args = parser.parse_args()

    evaluate_pubmedqa(
        qa_model_path=args.qa_model,
        embedding_precision=args.embedding_precision,
        top_k=args.top_k,
        device=args.device,
        limit=args.limit,
        experiment_id=args.exp_id
    )


if __name__ == "__main__":
    main()
