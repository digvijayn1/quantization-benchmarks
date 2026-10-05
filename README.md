# Architecture-Aware Post-Training Quantization of Offline Medical Question-Answering Systems

[![OpenVINO](https://img.shields.io/badge/OpenVINO-2026.4-blue.svg)](https://docs.openvino.ai/)
[![NNCF](https://img.shields.io/badge/NNCF-3.4-green.svg)](https://github.com/openvinotoolkit/nncf)
[![Model](https://img.shields.io/badge/Base_Model-Qwen2.5--0.5B--Instruct-orange.svg)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

Empirical research benchmarking suite and interactive dashboard exploring **Architecture-Aware Post-Training Quantization (PTQ)** and Weight Compression on `Qwen/Qwen2.5-0.5B-Instruct` for resource-constrained, offline clinical triage and medical QA scenarios.

---

## 📊 Key Highlights & Results

- **17 Quantization Variants**: Covers FP32/FP16 baselines, uniform INT8, INT4 group-size sweeps ($g \in \{32, 64, 128, 256\}$), mixed-precision ratios (0.5, 0.8), and data-aware AWQ & Scale Estimation.
- **2.32x Speedup**: AWQ INT4 ($g=128$) boosts generation throughput from **20.29 tok/s** to **47.10 tok/s**.
- **67.3% Footprint Reduction**: Model size reduced from **944.08 MB** to **308.51 MB**.
- **100% Medical Prompt Stability**: Zero degradation on clinical reasoning across 5 key clinical domains.

| Model Variant | Quantization Paradigm | Group Size ($g$) | Total Size | Compression Ratio | Size Reduction | Throughput (tok/s) | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`fp32`** | Full Precision Baseline | — | 944.08 MB | 1.00x | 0.0% | 20.29 | 1.00x |
| **`fp16`** | Half Precision Baseline | — | 944.12 MB | 1.00x | 0.0% | 20.64 | 1.02x |
| **`int8_asym`** | Uniform INT8 Asymmetric | Per-Channel | 474.74 MB | 1.99x | 49.7% | 35.22 | 1.74x |
| **`int4_sym_g128`** | Uniform INT4 Symmetric | 128 | 308.51 MB | 3.06x | 67.3% | 45.93 | 2.26x |
| **`int4_asym_g128`** | Uniform INT4 Asymmetric | 128 | 309.84 MB | 3.05x | 67.2% | 45.38 | 2.24x |
| **`int4_sym_g128_awq`** | Activation-Aware (AWQ) | 128 | 308.51 MB | 3.06x | 67.3% | **47.10** | **2.32x** |
| **`int4_sym_g128_scale_est`**| Scale Estimation | 128 | 308.51 MB | 3.06x | 67.3% | 46.22 | 2.28x |

---

## 🖥️ Interactive Dashboard

Open [`index.html`](index.html) in your browser to explore the interactive visual analytics dashboard featuring:
- Live latency and throughput trade-off scatter plots
- Group size & mixed-precision sensitivity curves
- Side-by-side clinical generation outputs and repetition metric audits
- Configurable Pareto frontier exploration

---

## 📁 Repository Structure

```text
├── configs/
│   └── quantization_experiments.yaml     # 17 experiment definitions & hyperparameters
├── data/
│   └── calibration/                      # Clinical calibration dataset & manifest
├── docs/
│   ├── quantization_research_report.md   # Comprehensive empirical research report
│   ├── manual_execution_guide.md         # End-to-end reproduction instructions
│   └── post_training_quantization_literature.md # Theoretical foundations
├── results/
│   ├── benchmarks/                       # CSV & JSON benchmark outputs
│   └── validation/                       # Domain prompt validation traces
├── scripts/
│   ├── export_openvino.py                # Base FP32/FP16 OpenVINO IR export
│   ├── prepare_calibration_data.py       # Calibration tokenization & dataset prep
│   ├── run_quantization.py               # NNCF weight compression pipeline
│   ├── benchmark_and_evaluate.py         # Latency, throughput & evaluation script
│   └── validate_baselines.py             # Validation checks across medical prompts
├── ui/                                   # Dashboard CSS and JS logic
├── index.html                            # Interactive benchmarking UI
└── .gitignore
```

---

## 🚀 Reproduction & Setup

### 1. Installation
```powershell
pip install -U "openvino>=2026.4.0" "nncf>=3.4.0" "optimum-intel>=1.22.0" "transformers>=4.48.0" pyyaml pandas tabulate
```

### 2. Export & Quantize
```powershell
# Export base FP32 / FP16 models
python scripts/export_openvino.py

# Prepare calibration dataset
python scripts/prepare_calibration_data.py

# Run all 17 quantization experiments
python scripts/run_quantization.py --config configs/quantization_experiments.yaml
```

### 3. Benchmark & Evaluate
```powershell
# Benchmark throughput, latency, and clinical stability
python scripts/benchmark_and_evaluate.py
```

---

## 📜 License
Licensed under the [Apache 2.0 License](LICENSE).
