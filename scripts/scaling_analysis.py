"""
Thread Scaling & Parallel Efficiency Analysis.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System

Calculates speedup (S) relative to the baseline (FP32, 1 thread, default pinning)
and parallel efficiency (E = S / N), generating results/processed/scaling_summary.csv.
"""

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUMMARY_CSV = Path("results/processed/benchmark_summary.csv")
SCALING_CSV = Path("results/processed/scaling_summary.csv")


def compute_scaling_metrics():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Summary CSV not found at {SUMMARY_CSV}. Run benchmark.py first.")

    df = pd.read_csv(SUMMARY_CSV)
    df_mode_a = df[df["mode"] == "mode_a"].copy()

    # Identify Baseline: FP32 with 1 thread and pinning=False
    baseline_match = df_mode_a[(df_mode_a["model_id"] == "fp32") & (df_mode_a["threads"] == 1) & (df_mode_a["pinning"] == False)]
    if baseline_match.empty:
        # Fallback to any FP32 1 thread
        baseline_match = df_mode_a[(df_mode_a["model_id"] == "fp32") & (df_mode_a["threads"] == 1)]

    if baseline_match.empty:
        raise ValueError("Could not locate baseline run (FP32, threads=1) in benchmark summary.")

    baseline_latency = float(baseline_match.iloc[0]["mean_inference_ms"])
    baseline_tpot = float(baseline_match.iloc[0]["mean_tpot_ms"])
    baseline_ttft = float(baseline_match.iloc[0]["mean_ttft_ms"])

    logger.info(f"Baseline (FP32, 1 Thread): Total Inference Latency={baseline_latency:.2f}ms, TPOT={baseline_tpot:.2f}ms, TTFT={baseline_ttft:.2f}ms")

    records = []
    for _, row in df_mode_a.iterrows():
        lat = float(row["mean_inference_ms"])
        tpot = float(row["mean_tpot_ms"])
        ttft = float(row["mean_ttft_ms"])
        threads = int(row["threads"])
        
        # Speedup vs FP32 1-thread baseline
        speedup_total = round(baseline_latency / lat, 3) if lat > 0 else 0.0
        speedup_tpot = round(baseline_tpot / tpot, 3) if tpot > 0 else 0.0
        speedup_ttft = round(baseline_ttft / ttft, 3) if ttft > 0 else 0.0
        
        # Model-specific self-speedup (speedup relative to same model at 1 thread)
        same_model_t1 = df_mode_a[(df_mode_a["model_id"] == row["model_id"]) & (df_mode_a["threads"] == 1) & (df_mode_a["pinning"] == row["pinning"])]
        if not same_model_t1.empty:
            model_t1_lat = float(same_model_t1.iloc[0]["mean_inference_ms"])
            self_speedup = round(model_t1_lat / lat, 3) if lat > 0 else 0.0
        else:
            self_speedup = speedup_total

        # Parallel efficiency E = Self_Speedup / Threads
        parallel_eff_pct = round((self_speedup / threads) * 100, 2)

        records.append({
            "experiment_id": row["experiment_id"],
            "model_id": row["model_id"],
            "precision": row["precision"],
            "threads": threads,
            "pinning": row["pinning"],
            "mean_inference_ms": lat,
            "mean_tpot_ms": tpot,
            "mean_ttft_ms": ttft,
            "throughput_tok_s": row["mean_throughput_tok_s"],
            "speedup_vs_fp32_t1": speedup_total,
            "speedup_tpot": speedup_tpot,
            "speedup_ttft": speedup_ttft,
            "self_speedup": self_speedup,
            "parallel_efficiency_pct": parallel_eff_pct,
            "peak_rss_mb": row["peak_rss_mb"],
            "avg_cpu_percent": row["avg_cpu_percent"]
        })

    scaling_df = pd.DataFrame(records)
    SCALING_CSV.parent.mkdir(parents=True, exist_ok=True)
    scaling_df.to_csv(SCALING_CSV, index=False)
    logger.info(f"Scaling summary written to {SCALING_CSV} ({len(records)} records)")
    return scaling_df


if __name__ == "__main__":
    compute_scaling_metrics()
