# Arithmetic Intensity & Roofline Performance Modelling Methodology

This document details the theoretical assumptions, analytical formulas, and performance-modelling derivations used to evaluate the computational and memory bounds of the `Qwen2.5-0.5B-Instruct` model across quantization precisions.

---

## 1. Operational Intensity & Roofline Model Fundamentals

The **Roofline Model** (Williams et al., 2009) relates attainable processor performance $P_{\text{attainable}}$ (GFLOP/s) to computational arithmetic intensity $I$ (FLOPs/byte) and system hardware limits:

$$P_{\text{attainable}} = \min\left(P_{\text{peak}}, \; \beta_{\text{peak}} \times I\right)$$

Where:
- $P_{\text{peak}}$: Theoretical maximum CPU floating-point execution throughput (GFLOP/s).
- $\beta_{\text{peak}}$: Theoretical maximum system memory bandwidth (GB/s).
- $I$: **Arithmetic Intensity** (Operations performed per byte of memory transferred from DRAM/cache).

### Machine Balance:

$$\text{Machine Balance } (B) = \frac{P_{\text{peak}}}{\beta_{\text{peak}}} \quad [\text{FLOPs/byte}]$$

- If $I < B$: The workload is **Memory-Bandwidth Bound** (performance is governed by memory throughput $\beta_{\text{peak}}$).
- If $I > B$: The workload is **Compute-Bound** (performance is governed by execution units $P_{\text{peak}}$).

---

## 2. Model Architecture Parameters (`Qwen2.5-0.5B-Instruct`)

| Architectural Parameter | Symbol | Value |
| :--- | :--- | :--- |
| **Total Model Parameters** | $P$ | $494,032,768 \approx 494\text{M}$ |
| **Number of Transformer Layers** | $L$ | $24$ |
| **Hidden Embedding Dimension** | $h$ | $896$ |
| **FFN Intermediate Dimension** | $d_{\text{ffn}}$ | $4,864$ |
| **Query Attention Heads** | $n_q$ | $14$ |
| **Key/Value (KV) Attention Heads (GQA)** | $n_{kv}$ | $2$ |
| **Head Dimension** | $d_h$ | $64$ ($n_q \times d_h = 896$) |
| **Vocabulary Size** | $V$ | $151,936$ |

---

## 3. Analytical Arithmetic Intensity Formulations

In autoregressive Transformer models, the computational characteristics diverge sharply between the **Prefill Phase** and the **Decode Phase**.

### A. Autoregressive Decode Phase (Batch Size = 1)
During single-token generation, every parameter matrix $W$ across all 24 layers must be streamed from DRAM into CPU cache to perform Matrix-Vector multiplication ($W \cdot x$):

$$\text{FLOPs per Token} \approx 2 \times P \approx 2 \times 494\text{M} = 0.988\text{ GFLOPs/token}$$

$$\text{Memory Traffic per Token} \approx P \times b_{\text{precision}} \quad [\text{Bytes/token}]$$

Where $b_{\text{precision}}$ is the storage byte footprint per parameter:
- **FP32**: $b = 4.0\text{ bytes} \implies \text{DRAM Traffic} = 1.976\text{ GB/token}$
- **FP16**: $b = 2.0\text{ bytes} \implies \text{DRAM Traffic} = 0.988\text{ GB/token}$
- **INT8**: $b = 1.0\text{ bytes} \implies \text{DRAM Traffic} = 0.494\text{ GB/token}$
- **INT4**: $b = 0.5\text{ bytes} \implies \text{DRAM Traffic} = 0.247\text{ GB/token}$

#### Decode Arithmetic Intensity:

$$I_{\text{decode}} = \frac{2 \times P}{P \times b_{\text{precision}}} = \frac{2}{b_{\text{precision}}}$$

| Precision Variant | Byte / Weight ($b$) | Arithmetic Intensity ($I_{\text{decode}}$) | Workload Regime |
| :--- | :--- | :--- | :--- |
| **`fp32`** | 4.0 Bytes | **0.50 FLOP/Byte** | Strictly Memory-Bound ($I \ll B$) |
| **`fp16`** | 2.0 Bytes | **1.00 FLOP/Byte** | Strictly Memory-Bound ($I \ll B$) |
| **`int8_asym`** | 1.0 Byte | **2.00 FLOP/Byte** | Memory-Bound ($I < B$) |
| **`int4_sym_g128`** | 0.5 Bytes | **4.00 FLOP/Byte** | Memory-Bound ($I < B$) |

### B. Prompt Prefill Phase (Batch Size = 1, Sequence Length = $S$)
During prefill, the model processes the entire prompt of length $S$ simultaneously as Matrix-Matrix multiplications ($W \cdot X_{S}$):

$$\text{FLOPs}_{\text{prefill}} \approx 2 \times P \times S$$

$$\text{Memory Traffic}_{\text{prefill}} \approx P \times b_{\text{precision}} + 2 \times L \times h \times S \times b_{\text{kv}}$$

$$I_{\text{prefill}} \approx \frac{2 \times S}{b_{\text{precision}}}$$

For a standard prompt length $S=128$:
- **FP32 Prefill**: $I_{\text{prefill}} \approx \frac{256}{4} = 64.0\text{ FLOP/Byte}$ (Compute-Bound)
- **INT4 Prefill**: $I_{\text{prefill}} \approx \frac{256}{0.5} = 512.0\text{ FLOP/Byte}$ (Compute-Bound)

---

## 4. Hardware Roofline Parameters (AMD Ryzen 7 8845HS)

- **Peak Compute Capacity ($P_{\text{peak}}$)**:
  $$P_{\text{peak}} = 8\text{ Cores} \times 2\text{ FMA units} \times 16\text{ SP ops/AVX-512} \times 5.1\text{ GHz} \approx 1,305.6\text{ GFLOP/s} \approx 1.30\text{ TFLOP/s}$$
- **Peak Memory Bandwidth ($\beta_{\text{peak}}$)**:
  $$\beta_{\text{peak}} = 2\text{ channels} \times 64\text{ bits} \times 7.5\text{ GT/s} / 8 \approx 89.6\text{ GB/s}$$
- **System Machine Balance ($B$)**:
  $$B = \frac{1305.6\text{ GFLOP/s}}{89.6\text{ GB/s}} \approx \mathbf{14.57\text{ FLOPs/Byte}}$$

### Theoretical Insights:
Because the decode arithmetic intensity for all precision variants ($0.5 \dots 4.0\text{ FLOPs/Byte}$) is substantially lower than the machine balance ($14.57\text{ FLOPs/Byte}$), **single-batch offline token generation on CPUs is fundamentally memory-bandwidth bound**. Compressing weights from FP32 (4 bytes) to INT4 (0.5 bytes) directly scales memory-bus efficiency, delivering up to $2.3\times$ real-world generation speedups.
