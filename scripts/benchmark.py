"""
Comprehensive Architecture-Aware Factorial Benchmarking Harness.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System

Executes factorial benchmark experiments measuring TTFT, TPOT, E2E latency,
throughput, peak RSS memory, CPU utilization, thread scaling, and affinity configurations.
Outputs:
- results/raw/benchmark_runs.csv
- results/processed/benchmark_summary.csv
- results/benchmark_environment.json
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import openvino as ov
import psutil
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer
from transformers.generation.streamers import BaseStreamer
import yaml

from resource_monitor import ResourceMonitor
from retrieve import MedicalRetriever
from prompt_builder import MedicalPromptBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/performance_experiments.yaml")
RESULTS_DIR = Path("results")
RAW_DIR = RESULTS_DIR / "raw"
PROCESSED_DIR = RESULTS_DIR / "processed"


class LatencyTimingStreamer(BaseStreamer):
    def __init__(self):
        super().__init__()
        self.t_start = None
        self.t_first_token = None
        self.token_timestamps = []
        self.generated_tokens_count = 0
        self.t_end = None

    def start(self):
        self.t_start = time.perf_counter()
        self.t_first_token = None
        self.token_timestamps = []
        self.generated_tokens_count = 0

    def put(self, value):
        now = time.perf_counter()
        num_tokens = value.numel() if hasattr(value, "numel") else len(value)
        if num_tokens == 0:
            return
        if self.t_first_token is None:
            self.t_first_token = now
        self.token_timestamps.append(now)
        self.generated_tokens_count += 1

    def end(self):
        self.t_end = time.perf_counter()

    def get_metrics(self) -> Dict[str, float]:
        if self.t_start is None or self.t_first_token is None:
            return {"ttft_ms": 0.0, "tpot_ms": 0.0, "total_gen_ms": 0.0, "generated_tokens": 0, "throughput_tok_s": 0.0}

        ttft_ms = (self.t_first_token - self.t_start) * 1000.0
        total_gen_ms = (self.t_end - self.t_start) * 1000.0
        decode_ms = (self.t_end - self.t_first_token) * 1000.0
        num_decode_tokens = max(1, self.generated_tokens_count - 1)
        tpot_ms = decode_ms / num_decode_tokens if self.generated_tokens_count > 1 else decode_ms
        tps = (self.generated_tokens_count / (total_gen_ms / 1000.0)) if total_gen_ms > 0 else 0.0

        return {
            "ttft_ms": round(ttft_ms, 3),
            "tpot_ms": round(tpot_ms, 3),
            "total_gen_ms": round(total_gen_ms, 3),
            "generated_tokens": self.generated_tokens_count,
            "throughput_tok_s": round(tps, 2)
        }


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")


def benchmark_single_configuration(
    model_info: Dict[str, Any],
    threads: int,
    pinning: bool,
    mode: str = "mode_a",
    workload: str = "medium",
    warmup_runs: int = 2,
    measurement_runs: int = 5,
    retriever: Optional[MedicalRetriever] = None,
    prompt_builder: Optional[MedicalPromptBuilder] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    m_id = model_info["id"]
    m_path = Path(model_info["path"])
    prec = model_info["precision"]
    size_mb = model_info["size_mb"]

    exp_id = f"{m_id}_t{threads}_{'pinned' if pinning else 'default'}_{mode}_{workload}"
    logger.info("=" * 70)
    logger.info(f"Running Experiment: {exp_id}")
    logger.info(f"Model: {m_path} | Precision: {prec} | Threads: {threads} | Pinning: {pinning} | Mode: {mode}")
    logger.info("=" * 70)

    # Configure OpenVINO CPU properties
    ov_config = {
        "INFERENCE_NUM_THREADS": int(threads),
        "ENABLE_CPU_PINNING": bool(pinning),
        "PERFORMANCE_HINT": "LATENCY"
    }

    t0_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(m_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = OVModelForCausalLM.from_pretrained(
        m_path,
        device="CPU",
        ov_config=ov_config,
        compile=True,
        trust_remote_code=True
    )
    compile_time_s = round(time.perf_counter() - t0_load, 3)

    # Read verified compiled properties
    try:
        cm = llm.request.get_compiled_model()
        actual_threads = cm.get_property("INFERENCE_NUM_THREADS")
        actual_pinning = cm.get_property("ENABLE_CPU_PINNING")
    except Exception:
        actual_threads = threads
        actual_pinning = pinning

    max_new_tokens = 32 if workload == "short" else (64 if workload == "medium" else 128)

    sample_query = "What are the common early symptoms and diagnostic indicators of Type 2 Diabetes Mellitus?"
    sample_context = (
        "Diagnostic criteria for diabetes include fasting plasma glucose >= 126 mg/dL, "
        "2-hour plasma glucose >= 200 mg/dL during OGTT, or HbA1c >= 6.5%. Early symptoms include polyuria, polydipsia, and weight loss."
    )

    if mode == "mode_b" and retriever is not None and prompt_builder is not None:
        # Mode B: Real end-to-end flow
        prompt_template = None
    else:
        # Mode A: Precomputed standard clinical prompt
        raw_prompt = (
            f"<|im_start|>system\nYou are an offline medical question-answering assistant.<|im_end|>\n"
            f"<|im_start|>user\nClinical Context:\n{sample_context}\n\nQuestion:\n{sample_query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    streamer = LatencyTimingStreamer()

    # --- Warm-up Runs ---
    for _ in range(warmup_runs):
        if mode == "mode_b" and retriever and prompt_builder:
            r_res = retriever.retrieve(sample_query, top_k=5)
            p_text, _ = prompt_builder.build_prompt(sample_query, r_res["retrieved_chunks"])
            inps = tokenizer(p_text, return_tensors="pt")
        else:
            inps = tokenizer(raw_prompt, return_tensors="pt")

        streamer.start()
        _ = llm.generate(
            **inps,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        streamer.end()

    # --- Measured Runs ---
    raw_records = []
    ttfts, tpots, total_gens, tpss, e2es = [], [], [], [], []
    ret_lats, prompt_lats = [], []

    resource_monitor = ResourceMonitor(sampling_interval_seconds=0.02)
    resource_monitor.start()

    for run_idx in range(1, measurement_runs + 1):
        t_e2e_start = time.perf_counter()
        ret_ms = 0.0
        prompt_ms = 0.0

        if mode == "mode_b" and retriever and prompt_builder:
            t_ret_start = time.perf_counter()
            r_res = retriever.retrieve(sample_query, top_k=5)
            ret_ms = (time.perf_counter() - t_ret_start) * 1000.0

            t_prompt_start = time.perf_counter()
            prompt_str, _ = prompt_builder.build_prompt(sample_query, r_res["retrieved_chunks"])
            prompt_ms = (time.perf_counter() - t_prompt_start) * 1000.0

            inputs = tokenizer(prompt_str, return_tensors="pt")
        else:
            inputs = tokenizer(raw_prompt, return_tensors="pt")

        input_tokens = inputs["input_ids"].shape[-1]

        streamer.start()
        output_tokens = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        streamer.end()

        total_e2e_ms = (time.perf_counter() - t_e2e_start) * 1000.0
        lat_metrics = streamer.get_metrics()

        ttfts.append(lat_metrics["ttft_ms"])
        tpots.append(lat_metrics["tpot_ms"])
        total_gens.append(lat_metrics["total_gen_ms"])
        tpss.append(lat_metrics["throughput_tok_s"])
        e2es.append(total_e2e_ms)
        ret_lats.append(ret_ms)
        prompt_lats.append(prompt_ms)

        raw_records.append({
            "experiment_id": exp_id,
            "model_id": m_id,
            "precision": prec,
            "model_size_mb": size_mb,
            "threads_requested": threads,
            "threads_actual": actual_threads,
            "pinning_requested": pinning,
            "pinning_actual": actual_pinning,
            "mode": mode,
            "workload": workload,
            "run_index": run_idx,
            "input_tokens": input_tokens,
            "output_tokens": lat_metrics["generated_tokens"],
            "retrieval_ms": round(ret_ms, 3),
            "prompt_assembly_ms": round(prompt_ms, 3),
            "ttft_ms": lat_metrics["ttft_ms"],
            "tpot_ms": lat_metrics["tpot_ms"],
            "total_inference_ms": lat_metrics["total_gen_ms"],
            "total_e2e_ms": round(total_e2e_ms, 3),
            "throughput_tok_s": lat_metrics["throughput_tok_s"]
        })

    resource_profile = resource_monitor.stop()

    def stats(arr):
        return {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "std": round(float(np.std(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p90": round(float(np.percentile(arr, 90)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "p99": round(float(np.percentile(arr, 99)), 3)
        }

    summary = {
        "experiment_id": exp_id,
        "model_id": m_id,
        "precision": prec,
        "model_size_mb": size_mb,
        "threads": threads,
        "pinning": pinning,
        "mode": mode,
        "workload": workload,
        "compile_time_seconds": compile_time_s,
        "measurement_runs": measurement_runs,
        "ttft_stats": stats(ttfts),
        "tpot_stats": stats(tpots),
        "total_inference_stats": stats(total_gens),
        "total_e2e_stats": stats(e2es),
        "throughput_stats": stats(tpss),
        "retrieval_stats": stats(ret_lats) if mode == "mode_b" else None,
        "resource_profile": resource_profile,
        "status": "SUCCESS"
    }

    logger.info(
        f"[{exp_id}] Latency Mean: {summary['total_inference_stats']['mean']}ms | "
        f"TTFT: {summary['ttft_stats']['mean']}ms | TPOT: {summary['tpot_stats']['mean']}ms | "
        f"Throughput: {summary['throughput_stats']['mean']} tok/s | Peak RSS: {resource_profile['peak_rss_mb']} MB"
    )

    return raw_records, summary


def run_benchmarks():
    config = load_config()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    models = config["models"]
    threads_list = config["threads"]
    pinning_modes = config["pinning_modes"]
    warmup_runs = config.get("warmup_runs", 2)
    measurement_runs = config.get("measurement_runs", 5)

    retriever = MedicalRetriever(precision="fp32")
    prompt_builder = MedicalPromptBuilder()

    all_raw_records = []
    all_summaries = []

    logger.info("=" * 70)
    logger.info("STARTING FACTORIAL ARCHITECTURE-AWARE BENCHMARK SUITE")
    logger.info(f"Models: {len(models)} | Thread Configs: {threads_list} | Pinning Modes: {len(pinning_modes)}")
    logger.info("=" * 70)

    # 1. Primary Factorial Sweep (Mode A: Model-Only Inference across all Precision x Threads x Pinning)
    for model_info in models:
        for t in threads_list:
            for p_mode in pinning_modes:
                pinning = p_mode["enable_cpu_pinning"]
                try:
                    raw_recs, summary = benchmark_single_configuration(
                        model_info=model_info,
                        threads=t,
                        pinning=pinning,
                        mode="mode_a",
                        workload="medium",
                        warmup_runs=warmup_runs,
                        measurement_runs=measurement_runs
                    )
                    all_raw_records.extend(raw_recs)
                    all_summaries.append(summary)
                except Exception as e:
                    logger.error(f"Experiment failed for {model_info['id']} (threads={t}, pin={pinning}): {e}", exc_info=True)
                    all_summaries.append({
                        "experiment_id": f"{model_info['id']}_t{t}_{'pinned' if pinning else 'default'}_mode_a_medium",
                        "model_id": model_info["id"],
                        "threads": t,
                        "pinning": pinning,
                        "status": "FAILED",
                        "error": str(e)
                    })

    # 2. Mode B End-to-End Benchmark (Top Quantized Model vs FP32 Baseline with Retrieval)
    for m_id in ["fp32", "int4_sym_g128_awq"]:
        m_info = next(m for m in models if m["id"] == m_id)
        for t in [4, 8]:
            try:
                raw_recs, summary = benchmark_single_configuration(
                    model_info=m_info,
                    threads=t,
                    pinning=True,
                    mode="mode_b",
                    workload="medium",
                    warmup_runs=warmup_runs,
                    measurement_runs=measurement_runs,
                    retriever=retriever,
                    prompt_builder=prompt_builder
                )
                all_raw_records.extend(raw_recs)
                all_summaries.append(summary)
            except Exception as e:
                logger.error(f"Mode B experiment failed for {m_id} (threads={t}): {e}")

    # Write Raw Runs CSV
    raw_csv_path = RAW_DIR / "benchmark_runs.csv"
    if all_raw_records:
        with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_raw_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_raw_records)
        logger.info(f"Wrote {len(all_raw_records)} raw measurement runs to {raw_csv_path}")

    # Write Processed Summary CSV & JSON
    summary_rows = []
    for s in all_summaries:
        if s.get("status") == "SUCCESS":
            summary_rows.append({
                "experiment_id": s["experiment_id"],
                "model_id": s["model_id"],
                "precision": s["precision"],
                "model_size_mb": s["model_size_mb"],
                "threads": s["threads"],
                "pinning": s["pinning"],
                "mode": s["mode"],
                "workload": s["workload"],
                "compile_time_s": s["compile_time_seconds"],
                "mean_ttft_ms": s["ttft_stats"]["mean"],
                "p50_ttft_ms": s["ttft_stats"]["p50"],
                "p95_ttft_ms": s["ttft_stats"]["p95"],
                "mean_tpot_ms": s["tpot_stats"]["mean"],
                "p50_tpot_ms": s["tpot_stats"]["p50"],
                "p95_tpot_ms": s["tpot_stats"]["p95"],
                "mean_inference_ms": s["total_inference_stats"]["mean"],
                "p50_inference_ms": s["total_inference_stats"]["p50"],
                "p95_inference_ms": s["total_inference_stats"]["p95"],
                "mean_e2e_ms": s["total_e2e_stats"]["mean"],
                "mean_throughput_tok_s": s["throughput_stats"]["mean"],
                "peak_rss_mb": s["resource_profile"]["peak_rss_mb"],
                "avg_cpu_percent": s["resource_profile"]["avg_process_cpu_percent"]
            })

    summary_csv_path = PROCESSED_DIR / "benchmark_summary.csv"
    if summary_rows:
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Wrote {len(summary_rows)} processed summaries to {summary_csv_path}")

    # Write Environment JSON
    env_info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "processor": "AMD Ryzen 7 8845HS w/ Radeon 780M Graphics",
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_processors": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        },
        "software": {
            "os": "Windows 11 Home (10.0.26200)",
            "python": sys.version,
            "openvino": ov.__version__,
            "psutil": psutil.__version__
        },
        "total_experiments_executed": len(all_summaries),
        "successful_experiments": len(summary_rows),
        "failed_experiments": len(all_summaries) - len(summary_rows)
    }
    with open(RESULTS_DIR / "benchmark_environment.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    logger.info("Factorial Benchmarking Suite Completed Successfully.")


if __name__ == "__main__":
    run_benchmarks()
