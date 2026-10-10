"""
Comprehensive Architecture-Aware Visualization Suite.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System

Generates 12 publication-ready, high-resolution figures in results/plots/ using Matplotlib & Seaborn:
1. Latency vs Thread Count
2. Speedup vs Thread Count
3. Parallel Efficiency vs Thread Count
4. TTFT Comparison across Model Variants
5. TPOT Comparison across Model Variants
6. Peak RSS Memory Comparison
7. CPU Utilization Comparison
8. Model Size vs Latency
9. Accuracy vs Latency
10. Accuracy vs Memory
11. Pareto Frontier Trade-offs
12. Roofline Performance Model
"""

import json
import logging
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Style Configuration
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

PROCESSED_DIR = Path("results/processed")
PLOTS_DIR = Path("results/plots")
SUMMARY_CSV = PROCESSED_DIR / "benchmark_summary.csv"
SCALING_CSV = PROCESSED_DIR / "scaling_summary.csv"
PARETO_CSV = PROCESSED_DIR / "pareto_frontier.csv"
ROOFLINE_JSON = PROCESSED_DIR / "roofline_summary.json"


def generate_all_plots():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not SUMMARY_CSV.exists() or not SCALING_CSV.exists():
        logger.warning("Summary CSVs not found. Please run benchmark.py and scaling_analysis.py first.")
        return

    df_summary = pd.read_csv(SUMMARY_CSV)
    df_scaling = pd.read_csv(SCALING_CSV)
    df_pareto = pd.read_csv(PARETO_CSV) if PARETO_CSV.exists() else df_scaling

    palette = sns.color_palette("tab10", n_colors=len(df_summary["model_id"].unique()))
    model_colors = dict(zip(df_summary["model_id"].unique(), palette))

    # --- Plot 1: Latency vs Threads ---
    plt.figure(figsize=(9, 5.5))
    for m_id, group in df_scaling[df_scaling["pinning"] == True].groupby("model_id"):
        group_sorted = group.sort_values("threads")
        plt.plot(group_sorted["threads"], group_sorted["mean_inference_ms"], marker="o", linewidth=2, label=m_id, color=model_colors.get(m_id))
    plt.title("Total Inference Latency vs. OpenVINO Thread Count (Pinned)")
    plt.xlabel("Thread Count (N)")
    plt.ylabel("Mean Latency (ms)")
    plt.xticks([1, 2, 4, 8, 16])
    plt.legend(title="Model Variant", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(PLOTS_DIR / "01_latency_vs_threads.png")
    plt.close()

    # --- Plot 2: Speedup vs Threads ---
    plt.figure(figsize=(9, 5.5))
    threads_ref = np.array([1, 2, 4, 8, 16])
    plt.plot(threads_ref, threads_ref, "--", color="gray", alpha=0.7, label="Ideal Linear Speedup")
    for m_id, group in df_scaling[df_scaling["pinning"] == True].groupby("model_id"):
        group_sorted = group.sort_values("threads")
        plt.plot(group_sorted["threads"], group_sorted["speedup_vs_fp32_t1"], marker="s", linewidth=2, label=f"{m_id}", color=model_colors.get(m_id))
    plt.title("Speedup vs. FP32 Baseline (1 Thread, Default)")
    plt.xlabel("Thread Count (N)")
    plt.ylabel("Speedup Multiplier (x)")
    plt.xticks([1, 2, 4, 8, 16])
    plt.legend(title="Model Variant", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(PLOTS_DIR / "02_speedup_vs_threads.png")
    plt.close()

    # --- Plot 3: Parallel Efficiency vs Threads ---
    plt.figure(figsize=(9, 5.5))
    plt.axhline(100, color="gray", linestyle="--", alpha=0.7, label="100% Ideal Efficiency")
    for m_id, group in df_scaling[df_scaling["pinning"] == True].groupby("model_id"):
        group_sorted = group.sort_values("threads")
        plt.plot(group_sorted["threads"], group_sorted["parallel_efficiency_pct"], marker="^", linewidth=2, label=m_id, color=model_colors.get(m_id))
    plt.title("Parallel Efficiency vs. Thread Scaling")
    plt.xlabel("Thread Count (N)")
    plt.ylabel("Parallel Efficiency (%)")
    plt.xticks([1, 2, 4, 8, 16])
    plt.legend(title="Model Variant", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(PLOTS_DIR / "03_parallel_efficiency_vs_threads.png")
    plt.close()

    # --- Plot 4: TTFT Comparison ---
    plt.figure(figsize=(9, 5))
    df_8t = df_summary[(df_summary["threads"] == 8) & (df_summary["pinning"] == True) & (df_summary["mode"] == "mode_a")]
    if df_8t.empty:
        df_8t = df_summary[(df_summary["threads"] == 8) & (df_summary["mode"] == "mode_a")]
    sns.barplot(data=df_8t, x="model_id", y="mean_ttft_ms", palette="Blues_d")
    plt.title("Time To First Token (TTFT) Across Model Variants (8 Threads)")
    plt.xlabel("Model Variant")
    plt.ylabel("TTFT (ms)")
    plt.xticks(rotation=25, ha="right")
    plt.savefig(PLOTS_DIR / "04_ttft_comparison.png")
    plt.close()

    # --- Plot 5: TPOT Comparison ---
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_8t, x="model_id", y="mean_tpot_ms", palette="Greens_d")
    plt.title("Time Per Output Token (TPOT) Across Model Variants (8 Threads)")
    plt.xlabel("Model Variant")
    plt.ylabel("TPOT (ms/token)")
    plt.xticks(rotation=25, ha="right")
    plt.savefig(PLOTS_DIR / "05_tpot_comparison.png")
    plt.close()

    # --- Plot 6: Peak RSS Comparison ---
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_8t, x="model_id", y="peak_rss_mb", palette="Oranges_d")
    plt.title("Peak RSS Physical Memory Footprint (8 Threads)")
    plt.xlabel("Model Variant")
    plt.ylabel("Peak RSS (MB)")
    plt.xticks(rotation=25, ha="right")
    plt.savefig(PLOTS_DIR / "06_peak_rss_comparison.png")
    plt.close()

    # --- Plot 7: CPU Utilization ---
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=df_summary[df_summary["mode"] == "mode_a"], x="threads", y="avg_cpu_percent", hue="model_id", palette="tab10")
    plt.title("Average Process CPU Utilization by Thread Allocation")
    plt.xlabel("Thread Count")
    plt.ylabel("CPU Utilization (%)")
    plt.legend(title="Model Variant", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(PLOTS_DIR / "07_cpu_utilization.png")
    plt.close()

    # --- Plot 8: Model Size vs Latency ---
    plt.figure(figsize=(8, 5.5))
    for _, row in df_8t.iterrows():
        plt.scatter(row["model_size_mb"], row["mean_inference_ms"], s=180, color=model_colors.get(row["model_id"]), edgecolors="black", linewidth=1.5, zorder=5)
        plt.annotate(row["model_id"], (row["model_size_mb"] + 10, row["mean_inference_ms"] + 10), fontsize=9)
    plt.title("Model Storage Footprint vs. Inference Latency (8 Threads)")
    plt.xlabel("Model Size (MB)")
    plt.ylabel("Mean Latency (ms)")
    plt.savefig(PLOTS_DIR / "08_model_size_vs_latency.png")
    plt.close()

    # --- Plot 9: Accuracy vs Latency ---
    plt.figure(figsize=(8.5, 5.5))
    acc_df = df_pareto.copy()
    for _, row in acc_df.iterrows():
        is_pareto = row.get("pareto_latency_vs_accuracy", False)
        marker = "*" if is_pareto else "o"
        size = 250 if is_pareto else 100
        plt.scatter(row["mean_inference_ms"], row["accuracy_pct"], s=size, marker=marker, color=model_colors.get(row["model_id"]), edgecolors="black", linewidth=1.2)
        if is_pareto and row["threads"] in [4, 8, 16]:
            plt.annotate(f"{row['model_id']}_t{row['threads']}", (row["mean_inference_ms"] + 20, row["accuracy_pct"] + 0.3), fontsize=8)
    plt.title("Clinical QA Accuracy vs. Total Inference Latency")
    plt.xlabel("Inference Latency (ms) [Lower is Better]")
    plt.ylabel("Accuracy (%) [Higher is Better]")
    plt.savefig(PLOTS_DIR / "09_accuracy_vs_latency.png")
    plt.close()

    # --- Plot 10: Accuracy vs Memory ---
    plt.figure(figsize=(8.5, 5.5))
    for _, row in acc_df.iterrows():
        is_pareto = row.get("pareto_memory_vs_accuracy", False)
        marker = "*" if is_pareto else "o"
        size = 250 if is_pareto else 100
        plt.scatter(row["peak_rss_mb"], row["accuracy_pct"], s=size, marker=marker, color=model_colors.get(row["model_id"]), edgecolors="black", linewidth=1.2)
        if is_pareto and row["threads"] == 8:
            plt.annotate(row["model_id"], (row["peak_rss_mb"] + 15, row["accuracy_pct"] + 0.3), fontsize=9)
    plt.title("Clinical QA Accuracy vs. Peak RSS Memory")
    plt.xlabel("Peak RSS (MB) [Lower is Better]")
    plt.ylabel("Accuracy (%) [Higher is Better]")
    plt.savefig(PLOTS_DIR / "10_accuracy_vs_memory.png")
    plt.close()

    # --- Plot 11: Pareto Frontier ---
    plt.figure(figsize=(9, 5.5))
    pareto_points = acc_df[acc_df["pareto_optimal_3d"] == True].sort_values("mean_inference_ms")
    non_pareto = acc_df[acc_df["pareto_optimal_3d"] == False]
    
    plt.scatter(non_pareto["mean_inference_ms"], non_pareto["peak_rss_mb"], c="lightgray", s=60, alpha=0.7, label="Dominated Configurations")
    plt.plot(pareto_points["mean_inference_ms"], pareto_points["peak_rss_mb"], "--", color="crimson", linewidth=1.5, label="Pareto Frontier")
    plt.scatter(pareto_points["mean_inference_ms"], pareto_points["peak_rss_mb"], c="crimson", marker="*", s=220, edgecolors="black", linewidth=1.2, label="Pareto Optimal (3D)")

    for _, row in pareto_points.iterrows():
        plt.annotate(f"{row['model_id']}_t{row['threads']}", (row["mean_inference_ms"] + 30, row["peak_rss_mb"] + 20), fontsize=8)

    plt.title("Pareto Frontier: Inference Latency vs. Peak RSS Memory")
    plt.xlabel("Mean Inference Latency (ms) [Minimize]")
    plt.ylabel("Peak Physical Memory RSS (MB) [Minimize]")
    plt.legend()
    plt.savefig(PLOTS_DIR / "11_pareto_frontier.png")
    plt.close()

    # --- Plot 12: Roofline Model Plot ---
    plt.figure(figsize=(9, 6))
    # Roofline ceilings
    peak_gflops = 1305.6
    peak_bw = 89.6
    machine_balance = peak_gflops / peak_bw # ~14.57

    x_intensity = np.logspace(-1, 2, 200) # 0.1 to 100 FLOP/Byte
    y_attainable = np.minimum(peak_gflops, peak_bw * x_intensity)

    plt.loglog(x_intensity, y_attainable, "k-", linewidth=2.5, label=f"Roofline Ceiling (Peak={peak_gflops:.0f} GFLOP/s, BW={peak_bw:.1f} GB/s)")
    plt.axvline(machine_balance, color="gray", linestyle=":", label=f"Machine Balance ({machine_balance:.2f} FLOP/B)")

    # Model points
    if ROOFLINE_JSON.exists():
        with open(ROOFLINE_JSON, "r", encoding="utf-8") as f:
            roof_data = json.load(f)
        for pt in roof_data.get("empirical_attained_performance", []):
            ai = pt["arithmetic_intensity_flops_per_byte"]
            gflops = pt["attained_gflops"]
            m_id = pt["model_id"]
            plt.scatter(ai, gflops, s=160, color=model_colors.get(m_id, "blue"), edgecolors="black", linewidth=1.5, zorder=5, label=f"{m_id} ({gflops:.1f} GFLOP/s)")
            plt.annotate(m_id, (ai * 1.05, gflops * 1.1), fontsize=8)

    plt.title("Roofline Model: Attained Performance vs. Operational Intensity (AMD Ryzen 7 8845HS)")
    plt.xlabel("Operational Arithmetic Intensity (FLOPs / Byte) [Log Scale]")
    plt.ylabel("Attained Performance (GFLOP/s) [Log Scale]")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(PLOTS_DIR / "12_roofline_model.png")
    plt.close()

    logger.info(f"Successfully generated all 12 publication plots in {PLOTS_DIR}")


if __name__ == "__main__":
    generate_all_plots()
