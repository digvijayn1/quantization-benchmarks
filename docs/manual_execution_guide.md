# Manual Execution & Continuity Guide
**Project**: Architecture-Aware Optimization of an Offline Medical Question-Answering System  
**Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`  
**Host Architecture**: AMD Ryzen 7 8845HS (16 threads) | NVIDIA RTX 4060 Laptop (8GB) | OpenVINO 2026.4.1 | NNCF 3.4.0

This guide provides explicit, copy-paste terminal instructions to execute, validate, and benchmark all remaining stages of the quantization pipeline independently without AI assistance.

---

## 1. Environment & Path Setup

All commands must be executed within PowerShell from the root directory: `E:\quantization`.  
The dedicated Python virtual environment is located at:
```powershell
$PYTHON = "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe"
```

Verify your environment anytime:
```powershell
& $PYTHON -c "import openvino as ov, nncf, torch, transformers; print('Environment OK')"
```

---

## 2. Current Progress & Completed Assets

Before proceeding, confirm the existing artifacts:

| Asset | File / Directory | Status | Purpose |
| :--- | :--- | :--- | :--- |
| **FP32 Baseline** | `models/openvino/fp32/` | **COMPLETE** (944.08 MB) | Uncompressed reference IR |
| **FP16 Baseline** | `models/openvino/fp16/` | **COMPLETE** (944.12 MB) | Half-precision reference IR |
| **Baseline Validation** | `results/validation/fp32_validation.json`<br>`results/validation/fp16_validation.json` | **COMPLETE** (100% PASS) | Baseline outputs on 5 medical prompts |
| **Calibration Corpus** | `data/calibration/medical_calibration_data.json` | **COMPLETE** (64 samples) | Stratified de-identified clinical QA scenarios |
| **Experiment Matrix** | `configs/quantization_experiments.yaml` | **COMPLETE** (17 experiments) | Declarative configuration manifest |
| **Runner Scripts** | `scripts/run_quantization.py`<br>`scripts/benchmark_and_evaluate.py`<br>`scripts/validate_baselines.py` | **COMPLETE** | Modular automation scripts |

---

## 3. Step-by-Step Manual Execution

### Step 3.1: Execute Weight Quantization Matrix

You can run the entire experiment suite in a single command or execute individual target experiments.

#### Option A: Run ALL Experiments (Automated Batch)
```powershell
Set-Location -Path "E:\quantization"
& "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/run_quantization.py
```
*Note: Baselines (FP32/FP16) are automatically skipped. The script compresses each variant in sequence, copies tokenizers/configs, and writes `results/quantization_summary.json`.*

#### Option B: Run Specific Experiments
To run a specific experiment defined in `configs/quantization_experiments.yaml`, use the `--experiment` flag:

1. **INT8 Asymmetric Baseline**:
   ```powershell
   & "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/run_quantization.py --experiment int8_asym
   ```
2. **INT4 Symmetric Group Size Sweep ($g \in \{32, 64, 128, 256\}$)**:
   ```powershell
   & "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/run_quantization.py --experiment int4_sym_g32
   & "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/run_quantization.py --experiment int4_sym_g64
   & "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/run_quantization.py --experiment int4_sym_g128
   & "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/run_quantization.py --experiment int4_sym_g256
   ```
3. **INT4 Asymmetric Group Size Sweep**:
   ```powershell
   & "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/run_quantization.py --experiment int4_asym_g32
   & "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/run_quantization.py --experiment int4_asym_g64
   & "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/run_quantization.py --experiment int4_asym_g128
   & "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/run_quantization.py --experiment int4_asym_g256
   ```
4. **Mixed-Precision Ratio Sweep**:
   ```powershell
   & "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/run_quantization.py --experiment int4_sym_g128_r0.8
   & "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/run_quantization.py --experiment int4_sym_g128_r0.5
   ```
5. **Activation-Aware Weight Quantization (AWQ)**:
   ```powershell
   & "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/run_quantization.py --experiment int4_sym_g128_awq
   & "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/run_quantization.py --experiment int4_asym_g128_awq
   ```
6. **Data-Aware Scale Estimation**:
   ```powershell
   & "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/run_quantization.py --experiment int4_sym_g128_scale_est
   & "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/run_quantization.py --experiment int4_asym_g128_scale_est
   ```

---

### Step 3.2: Automated Benchmarking & Quality Evaluation

Once quantization is complete, execute the automated benchmark across all variants.

#### Benchmark on Host CPU (Ryzen 7 8845HS AVX-512):
```powershell
& "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/benchmark_and_evaluate.py --device CPU
```

#### Optional: Benchmark on NVIDIA RTX 4060 dGPU or Radeon 780M iGPU:
```powershell
# Discrete GPU:
& "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/benchmark_and_evaluate.py --device GPU.0

# Integrated GPU:
& "C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe" scripts/benchmark_and_evaluate.py --device GPU.1
```

---

### Step 3.3: Inspect & Analyze Results

#### View Summary Table in Terminal:
```powershell
Import-Csv results/benchmarks/benchmark_summary_cpu.csv | Format-Table -AutoSize
```

#### Inspect Detailed Outputs:
- Benchmarking summary table: `results/benchmarks/benchmark_summary_cpu.csv`
- Full generation logs and prompt latencies: `results/benchmarks/benchmark_summary_cpu.json`
- Quantization run metadata: `results/quantization_summary.json`

---

## 4. Troubleshooting & Gotchas

1. **Memory / RAM Headroom**:
   - The base model is ~494M parameters (~1 GB weights). FP32 compilation consumes ~2–3 GB RAM.
   - All compression algorithms operate safely within the 16 GB system memory.
2. **Missing Tokenizer in Quantized Folders**:
   - `scripts/run_quantization.py` automatically copies `tokenizer.json`, `tokenizer_config.json`, and `config.json` to every variant directory. If running manually outside the script, ensure these files are copied alongside `openvino_model.xml` and `openvino_model.bin`.
3. **Repetition / Degradation Checks**:
   - The benchmark script flags any output with a repetition ratio $> 0.35$ as `WARNING`. Watch specifically for aggressive low-bit variants ($g=256$, asymmetric without calibration) to see if group size causes hallucinations or looping.
4. **Literature Citations Compliance (Rule 2)**:
   - When compiling final academic reports, remember that bibliographic records for *References 1–7 and 16–19* were not present in the workspace. In accordance with Engineering Rule 2, do NOT fabricate external citation keys; report them as missing if needed.
