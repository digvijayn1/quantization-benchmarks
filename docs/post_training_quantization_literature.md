# Post-Training Quantization Literature & Algorithmic Analysis

> **IMPORTANT COMPLIANCE NOTICE**:
> In accordance with Engineering Rule 2 (*"Do not fabricate literature references"*) and Section 18 (*"If the references are not available in the repository, STOP and report that the references are missing rather than inventing them"*), we explicitly document that **specific bibliographic records for "References 1–7 and 16–19" were not provided as files or text in the repository**. Rather than fabricating authors, publication venues, or false bibliographic keys, this document reports that those specific external citation entries are missing from the project files.
>
> Below, we provide the rigorous theoretical, algorithmic, and architectural foundation corresponding to the post-training quantization (PTQ) paradigms represented in this literature strand, clearly distinguishing:
> 1. **Algorithmic theory reported in academic PTQ research**
> 2. **Concrete implementation in Intel OpenVINO / NNCF 3.4.0**
> 3. **Configurations experimentally tested in our pipeline**

---

## 1. Algorithmic Dimensions of Post-Training Quantization (PTQ)

| Algorithmic Concept | Academic Theory / Literature Basis | OpenVINO / NNCF 3.4 Implementation | Tested in This Pipeline |
| :--- | :--- | :--- | :--- |
| **INT8 Asymmetric (`INT8_ASYM`)** | Uniform affine quantization with zero-point offset: $W_q = \text{round}(W / s) + z$. Minimizes quantization error for non-zero-centered distributions. | Implemented via `CompressWeightsMode.INT8_ASYM`. Quantizes weights per-channel with dynamic dequantization nodes (`Convert` + `Subtract` + `Multiply`). | Yes (`int8_asym`) |
| **INT4 Symmetric (`INT4_SYM`)** | Symmetric uniform quantization: $W_q = \text{clamp}(\text{round}(W / s), -8, 7)$, with zero-point fixed to 0. Eliminates zero-point subtraction overhead during dequantization. | Implemented via `CompressWeightsMode.INT4_SYM`. Stores weights as packed nibbles, dequantized via scalar scale multiplication. | Yes (`int4_sym_g32`, `g64`, `g128`, `g256`) |
| **INT4 Asymmetric (`INT4_ASYM`)** | Asymmetric 4-bit uniform quantization: $W_q = \text{clamp}(\text{round}(W / s) + z, 0, 15)$. Allows arbitrary distribution centering at the cost of storing zero-points. | Implemented via `CompressWeightsMode.INT4_ASYM`. Packs 4-bit weights with per-group scales and zero-points. | Yes (`int4_asym_g32`, `g64`, `g128`, `g256`) |
| **Group-Wise Quantization Granularity** | Slicing weight tensors into blocks (e.g., $g \in \{32, 64, 128, 256\}$). Reduces outlier influence to localized groups rather than entire channel. | Supported in NNCF via `group_size` parameter. Computes scale/zero-point per block of $g$ contiguous weights along the input-channel dimension. | Yes ($g \in \{32, 64, 128, 256\}$) |
| **Ratio-based Mixed Precision** | Selective quantization: Quantizing only a fraction $r$ of weight matrices or layers to lower bit-width, retaining sensitive layers in higher precision (INT8/FP16). | Supported via `ratio` parameter in `nncf.compress_weights()`. Uses layer sensitivity ranking or heuristic layer ordering. | Yes ($r \in \{1.0, 0.8, 0.5\}$) |
| **Activation-Aware Weight Quantization (AWQ)** | Observes activation magnitudes on calibration data. Weights corresponding to salient activation channels are protected from clipping/quantization error via per-channel scaling factors: $W' = W \cdot S$, $X' = S^{-1} \cdot X$. | Implemented via `awq=True` in `nncf.compress_weights()` with `AdvancedAWQParameters` (grid search over $\alpha \in [\alpha_{\min}, \alpha_{\max}]$). | Yes (`int4_sym_g128_awq`, `int4_asym_g128_awq`) |
| **Data-Aware Scale Estimation (Scale Est)** | Rather than using simple $\max(|W|)$ per group, searches for the optimal scale $s^*$ that minimizes the reconstruction error $\|W X - \tilde{W}(s) X\|_2^2$ over calibration tokens. | Implemented via `scale_estimation=True` with `AdvancedScaleEstimationParameters` (grid search and weight penalty). | Yes (`int4_sym_g128_scale_est`, `int4_asym_g128_scale_est`) |

---

## 2. In-Depth Algorithmic Analysis

### 2.1 Symmetric vs. Asymmetric Quantization
- **Mathematical formulation**:
  - *Symmetric*: $s = \frac{\max(|W|)}{2^{b-1} - 1}$, $\tilde{W} = \text{clamp}\left(\lfloor \frac{W}{s} \rceil, -2^{b-1}, 2^{b-1}-1\right) \cdot s$
  - *Asymmetric*: $s = \frac{\max(W) - \min(W)}{2^b - 1}$, $z = \lfloor -\frac{\min(W)}{s} \rceil$, $\tilde{W} = \left(\text{clamp}\left(\lfloor \frac{W}{s} \rceil + z, 0, 2^b-1\right) - z\right) \cdot s$
- **Hardware Implications on Modern CPUs/GPUs**:
  - Symmetric quantization enables simpler fused multiply-accumulate micro-kernels because no zero-point addition/subtraction is required per sub-byte chunk.
  - Asymmetric quantization provides lower MSE when weights have a shifted mean, but introduces extra memory traffic (storing zero-points alongside scales) and ALU instructions for zero-point subtraction.

### 2.2 Granularity & Group Size ($g$)
- Per-tensor quantization suffers catastrophic degradation in LLMs due to activation and weight outliers.
- Per-channel quantization computes one scale per output neuron ($C_{\text{out}}$).
- Group-wise quantization divides the reduction axis ($C_{\text{in}}$) into groups of size $g$.
  - Smaller group size ($g=32$): Higher fidelity, lower quantization noise, but higher scale overhead ($\frac{16 \text{ bits}}{32 \times 4 \text{ bits}} \approx 12.5\%$ memory overhead for scales).
  - Larger group size ($g=256$): Lower scale overhead ($\approx 1.5\%$), but higher risk of dynamic range stretching due to outliers within the block.

### 2.3 Activation-Aware Weight Quantization (AWQ)
- Standard PTQ focuses solely on minimizing $\|W - \tilde{W}\|_F$. However, LLM performance is dominated by a tiny fraction (0.1%–1%) of salient activation channels.
- AWQ protects these salient channels by finding an optimal per-channel scale matrix $S = \text{diag}(s)$ such that:
  $$\arg\min_S \sum_i \|W_i X - \text{quant}(W_i S) S^{-1} X\|_2^2$$
- NNCF executes an efficient grid search over the parameter $\alpha$ ($s = s_X^\alpha$) using a representative calibration subset, then folds $S$ into prior normalization or projection layers where possible, leaving inference overhead neutral.

### 2.4 Scale Estimation
- In 4-bit quantization, outlier clipping vs. rounding error is a severe trade-off. Standard absmax scaling assigns the entire 16-bin range to cover the extreme outlier, devastating the resolution of the remaining 99% of weights.
- Scale estimation runs an optimization routine using forward passes over calibration data to search for a clipped scale factor $s < s_{\text{absmax}}$ that minimizes activation reconstruction error.

---

## 3. Discrepancies Between Literature and OpenVINO Runtime Realities

1. **Weight-Only Quantization (DQ/WNA16)**:
   - In OpenVINO for modern LLMs, weights are compressed in storage and dequantized on-the-fly to FP16/BF16 inside GEMM kernels. Activations remain in FP16/BF16 during matrix multiplication.
   - This eliminates the severe accuracy collapse often seen in fully-quantized INT8/INT4 activation pipelines (W4A4 or W8A8) while capturing 70%–80% of memory bandwidth savings, which is the primary bottleneck in memory-bound autoregressive LLM decoding.
2. **Ratio Interpretation**:
   - In NNCF, `ratio` specifies the proportion of total compressible weights assigned to the primary mode (e.g. INT4). The remaining $(1 - \text{ratio})$ fraction falls back to the `backup_mode` (default: INT8 or FP16).
3. **Execution Device Specificity**:
   - On x86_64 AVX-512 CPUs (such as the AMD Ryzen 7 8845HS host), OpenVINO dynamically dispatches INT4 dequantization through optimized AMX/VNNI/AVX-512 vector paths.

---

## 4. Literature References Status

As noted at the outset, if external bibliographic references (References 1–7 and 16–19) are later supplied by the research team, their full bibtex citations and cross-comparisons can be appended directly to this document.
