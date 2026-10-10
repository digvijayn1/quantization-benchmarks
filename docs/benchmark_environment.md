# Architecture-Aware Benchmark Environment & Hardware Profile

This document records the exact hardware architecture, operating system configuration, OpenVINO CPU runtime properties, and machine learning library versions utilized for the reproducible performance benchmarking and performance-modelling stage.

---

## 1. Host Compute Topology & System Profile

| Subsystem / Property | Specification | Source / Verification |
| :--- | :--- | :--- |
| **Processor (CPU)** | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics | AMD Zen 4 Architecture (`AuthenticAMD`) |
| **Microarchitecture** | Zen 4 (4nm process, TSMC) | AVX-512, AVX2, FMA3, BMI2, SHA |
| **Physical CPU Cores** | **8 Cores** | `psutil.cpu_count(logical=False)` |
| **Logical Processors (Threads)** | **16 Threads** (Simultaneous Multithreading / SMT enabled) | `psutil.cpu_count(logical=True)` |
| **Base / Boost Frequency** | 3.8 GHz Base / 5.1 GHz Max Boost | AMD Specification |
| **L1 Cache** | 512 KB Total (32 KB I-cache + 32 KB D-cache per core) | AMD Zen 4 Hierarchy |
| **L2 Cache** | 8 MB Total (1 MB dedicated per core) | AMD Zen 4 Hierarchy |
| **L3 Cache (Shared LLC)** | 16 MB Shared across 8-core CCX | AMD Zen 4 Hierarchy |
| **System Memory (RAM)** | 16.0 GB Total Physical (~15.31 GB addressable) | LPDDR5x-7500 / DDR5-5600 Dual-Channel |
| **Theoretical Memory Bandwidth** | **~89.6 GB/s** (LPDDR5x) / ~50–60 GB/s Effective | Hardware Specification |
| **Theoretical Peak CPU FP32 Compute** | **~1.30 TFLOP/s** ($8\text{ cores} \times 32\text{ SP FLOPs/cycle} \times 5.1\text{ GHz}$) | Vector AVX-512 FMA Fused Multiply-Add |
| **Discrete GPU (dGPU)** | NVIDIA GeForce RTX 4060 Laptop GPU (8GB GDDR6, OpenVINO device: `GPU.0`) | Verified OpenVINO device list |
| **Integrated GPU (iGPU)** | AMD Radeon 780M Graphics (`gfx1103`, OpenVINO device: `GPU.1`) | Verified OpenVINO device list |
| **Operating System** | Microsoft Windows 11 Home Single Language (64-bit) | Build `10.0.26200` |

---

## 2. Software & Runtime Stack

| Software Package | Installed Version | Purpose |
| :--- | :--- | :--- |
| **Python** | `3.14.5` (64-bit AMD64) | Virtual environment (`openvino_env`) |
| **Intel OpenVINO Runtime** | `2026.4.1-22982-e213a147257-releases/2026/4` | High-performance graph inference engine |
| **Intel NNCF** | `3.4.0` | Post-training weight compression framework |
| **Optimum Intel (`optimum-intel`)** | `2.2.0` | OpenVINO Hugging Face integration |
| **PyTorch (`torch`)** | `2.14.1+cpu` | Base model loading and token tensor operations |
| **Hugging Face Transformers** | `5.5.4` | Tokenization & generation coordination |
| **FAISS (`faiss-cpu`)** | `1.15.1` | Vector index retrieval |
| **psutil** | `7.2.2` | High-frequency process CPU and RSS memory monitoring |
| **Matplotlib** | `3.11.2` | Publication-grade chart generation |
| **Seaborn** | `0.13.2` | Statistical data visualization |
| **NumPy** | `2.4.6` | Array math and timing statistics |
| **Pandas** | `3.0.6` | Benchmark summary aggregation |

---

## 3. OpenVINO CPU Plugin Configuration Properties

The Intel OpenVINO 2026.4.1 CPU plugin on Windows 11 AMD Zen 4 exposes the following runtime properties:

| OpenVINO Property | Type / Range | Configured Values in Benchmark | Description |
| :--- | :--- | :--- | :--- |
| `INFERENCE_NUM_THREADS` | Integer ($1 \dots 16$) | $1, 2, 4, 8, 16$ | Number of worker execution threads allocated to CPU inference pool. |
| `ENABLE_CPU_PINNING` | Boolean (`True` / `False`) | `True` (Pinned), `False` (Default OS scheduler) | Binds execution threads to dedicated physical/logical cores to prevent context switching. |
| `PERFORMANCE_HINT` | Enum | `LATENCY` | Minimizes response time for single-stream generation workloads. |
| `NUM_STREAMS` | Integer / String | `1` | Single execution stream for deterministic latency benchmarking. |

---

## 4. Validated Model Variants in Benchmarking Matrix

| Model Identifier | Precision Paradigm | Compression Details | Binary Footprint (MB) | Size Reduction vs FP32 |
| :--- | :--- | :--- | :--- | :--- |
| **`fp32`** | Full Precision Baseline | Uncompressed IEEE 754 32-bit | 944.08 MB | 0.0% (1.00x) |
| **`fp16`** | Half Precision Baseline | Uncompressed IEEE 754 16-bit | 944.12 MB | 0.0% (1.00x) |
| **`int8_asym`** | Uniform INT8 Asymmetric | Per-channel zero-point + scale | 474.74 MB | 49.7% (1.99x) |
| **`int4_sym_g128`** | Uniform INT4 Symmetric | Block group size $g=128$ | 308.51 MB | 67.3% (3.06x) |
| **`int4_asym_g128`** | Uniform INT4 Asymmetric | Block group size $g=128$ | 309.84 MB | 67.2% (3.05x) |
| **`int4_sym_g128_awq`** | Activation-Aware (AWQ) | Outlier-protected symmetric INT4 ($g=128$) | 308.51 MB | 67.3% (3.06x) |
| **`int4_sym_g128_scale_est`**| Scale Estimation | Optimization-based scale INT4 ($g=128$) | 308.51 MB | 67.3% (3.06x) |
