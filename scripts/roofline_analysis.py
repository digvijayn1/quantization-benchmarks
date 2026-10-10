"""
Roofline Performance Modelling & Operational Intensity Analysis.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System

Models theoretical compute ceilings, memory bandwidth ceilings, operational arithmetic intensity,
and machine balance for AMD Ryzen 7 8845HS running Qwen2.5-0.5B-Instruct across quantization precisions.
Outputs:
- results/processed/roofline_summary.json
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/processed")
SUMMARY_CSV = RESULTS_DIR / "benchmark_summary.csv"
ROOFLINE_JSON = RESULTS_DIR / "roofline_summary.json"


def perform_roofline_analysis():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Hardware Specifications (AMD Ryzen 7 8845HS)
    hw_spec = {
        "cpu_model": "AMD Ryzen 7 8845HS",
        "cores": 8,
        "threads": 16,
        "max_boost_clock_ghz": 5.1,
        "peak_fp32_tflops": 1.3056, # 8 cores * 32 ops/cycle * 5.1 GHz = 1,305.6 GFLOP/s
        "peak_fp32_gflops": 1305.6,
        "peak_int8_gops": 2611.2,    # 2x FP32 with VNNI / AVX-512 INT8
        "peak_memory_bandwidth_gb_s": 89.6, # LPDDR5x-7500 dual-channel theoretical limit
        "effective_memory_bandwidth_gb_s": 55.0, # Realistic effective DRAM throughput
        "machine_balance_flop_per_byte": round(1305.6 / 89.6, 2) # ~14.57
    }

    # 2. Model Architecture Parameters (Qwen2.5-0.5B)
    model_params = 494_032_768 # ~494M parameters
    flops_per_token_decode = 2 * model_params # ~0.988 GFLOPs

    # Precision configurations
    prec_specs = {
        "fp32": {"bytes_per_param": 4.0, "arithmetic_intensity_decode": 0.50},
        "fp16": {"bytes_per_param": 2.0, "arithmetic_intensity_decode": 1.00},
        "int8_asym": {"bytes_per_param": 1.0, "arithmetic_intensity_decode": 2.00},
        "int4_sym_g128": {"bytes_per_param": 0.5, "arithmetic_intensity_decode": 4.00},
        "int4_asym_g128": {"bytes_per_param": 0.5, "arithmetic_intensity_decode": 4.00},
        "int4_sym_g128_awq": {"bytes_per_param": 0.5, "arithmetic_intensity_decode": 4.00},
        "int4_sym_g128_scale_est": {"bytes_per_param": 0.5, "arithmetic_intensity_decode": 4.00}
    }

    # 3. Load empirical benchmark measurements if available
    empirical_records = []
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        # Select 8-thread pinned runs for primary roofline alignment
        df_8t = df[(df["threads"] == 8) & (df["pinning"] == True) & (df["mode"] == "mode_a")]
        if df_8t.empty:
            df_8t = df[(df["threads"] == 8) & (df["mode"] == "mode_a")]

        for _, row in df_8t.iterrows():
            m_id = row["model_id"]
            spec = prec_specs.get(m_id, {"bytes_per_param": 1.0, "arithmetic_intensity_decode": 2.0})
            tps = float(row["mean_throughput_tok_s"])
            
            # Attained GFLOP/s = tokens/sec * GFLOPs/token
            attained_gflops = round(tps * (flops_per_token_decode / 1e9), 3)
            # Attained Memory Bandwidth = tokens/sec * (model_params * bytes_per_param / 1e9)
            attained_bw_gb_s = round(tps * (model_params * spec["bytes_per_param"] / 1e9), 2)
            
            # Theoretical ceiling at this arithmetic intensity
            theoretical_ceiling_gflops = round(min(hw_spec["peak_fp32_gflops"], hw_spec["peak_memory_bandwidth_gb_s"] * spec["arithmetic_intensity_decode"]), 2)
            bandwidth_utilization_pct = round((attained_bw_gb_s / hw_spec["peak_memory_bandwidth_gb_s"]) * 100, 2)

            empirical_records.append({
                "model_id": m_id,
                "precision": row["precision"],
                "arithmetic_intensity_flops_per_byte": spec["arithmetic_intensity_decode"],
                "throughput_tok_s": tps,
                "attained_gflops": attained_gflops,
                "attained_bandwidth_gb_s": attained_bw_gb_s,
                "theoretical_ceiling_gflops": theoretical_ceiling_gflops,
                "bandwidth_utilization_pct": bandwidth_utilization_pct,
                "regime": "Memory-Bandwidth Bound (I < Machine Balance)"
            })

    roofline_data = {
        "hardware_specification": hw_spec,
        "model_parameters_count": model_params,
        "gflops_per_token_decode": round(flops_per_token_decode / 1e9, 4),
        "machine_balance": hw_spec["machine_balance_flop_per_byte"],
        "precision_analytical_profiles": prec_specs,
        "empirical_attained_performance": empirical_records,
        "analytical_conclusion": (
            "Because the arithmetic intensity during autoregressive token generation (0.5 to 4.0 FLOP/Byte) "
            "is strictly below the CPU machine balance (14.57 FLOP/Byte), the workload operates in the memory-bound "
            "slanted ceiling of the Roofline model. Quantization to INT8 and INT4 reduces bytes fetched per token by 2x-4x, "
            "proportionally relaxing memory bandwidth saturation and yielding measured speedups."
        )
    }

    with open(ROOFLINE_JSON, "w", encoding="utf-8") as f:
        json.dump(roofline_data, f, indent=2)
    logger.info(f"Roofline analysis saved to {ROOFLINE_JSON}")
    return roofline_data


if __name__ == "__main__":
    perform_roofline_analysis()
