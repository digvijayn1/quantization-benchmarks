# Quantization Environment & Hardware Profile

This document captures the hardware, operating system, and machine learning software environment used for the reproducible quantization pipeline of the offline medical question-answering system.

---

## 1. Operating System & Host Environment

| Property | Value |
| :--- | :--- |
| **Operating System** | Microsoft Windows 11 Home Single Language |
| **OS Version / Build** | 10.0.26200 (64-bit) |
| **Host Architecture** | x86_64 / AMD64 |
| **Active Python Path** | `C:\Users\Digvijay Nigvekar\openvino_env\Scripts\python.exe` |
| **Python Version** | `3.14.5` |

---

## 2. Hardware Compute Capabilities

| Subsystem | Specification |
| :--- | :--- |
| **Processor (CPU)** | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (Zen 4 architecture, AVX-512, AVX2, FMA3) |
| **Physical CPU Cores** | 8 |
| **Logical Processors (Threads)** | 16 |
| **System Memory (RAM)** | 16.0 GB Total Physical (~15.3 GB visible) |
| **Discrete GPU (dGPU)** | NVIDIA GeForce RTX 4060 Laptop GPU (8GB GDDR6, OpenVINO device: `GPU.0`) |
| **Integrated GPU (iGPU)** | AMD Radeon 780M Graphics (`gfx1103`, OpenVINO device: `GPU.1`) |
| **OpenVINO Available Devices** | `['CPU', 'GPU.0', 'GPU.1']` |

---

## 3. Installed ML & Optimization Framework Versions

| Package / Library | Installed Version | Source / Notes |
| :--- | :--- | :--- |
| **PyTorch (`torch`)** | `2.14.1+cpu` | Standard wheel |
| **Torchvision (`torchvision`)** | `0.29.1` | Standard wheel |
| **Hugging Face Transformers** | `5.5.4` | Hugging Face ecosystem |
| **Hugging Face Optimum** | `2.3.0` | Hardware acceleration hub |
| **Optimum Intel (`optimum-intel`)**| `2.2.0` | OpenVINO model export & integration |
| **OpenVINO Runtime (`openvino`)** | `2026.4.1-22982-e213a147257-releases/2026/4` | Intel OpenVINO Toolkit |
| **OpenVINO Tokenizers** | `2026.4.1.0` | Fast C++ tokenizers for OpenVINO |
| **NNCF (`nncf`)** | `3.4.0` | Neural Network Compression Framework |
| **PyYAML** | `6.0.3` | Configuration parser |

---

## 4. Base Model Profile

| Property | Value |
| :--- | :--- |
| **Hugging Face Model ID** | `Qwen/Qwen2.5-0.5B-Instruct` |
| **Model Family / Type** | Qwen 2.5 (`qwen2`) |
| **Architectural Class** | Decoder-only Causal Language Model (`Qwen2ForCausalLM`) |
| **Parameter Count** | ~494,032,768 parameters (~494M) |
| **Original Model Format** | Hugging Face `safetensors` (bfloat16 weights) |
| **Context Window** | 32,768 tokens (native) |
| **Vocab Size** | 151,936 tokens |
| **Hidden Dimension / Layers** | Hidden Size: 896, Intermediate Size: 4864, Layers: 24, Attention Heads: 14, Key-Value Heads: 2 |
| **Application Domain** | Offline Medical Question-Answering & Clinical Triage Assistance |
| **Optimum Intel Compatibility** | **Supported** (natively exported via `OVModelForCausalLM.from_pretrained(..., export=True)`) |
| **OpenVINO Runtime Support** | **Supported** (OpenVINO GenAI / OVModelForCausalLM graph execution) |

---

## 5. Notes on Existing Project State

- Prior to initialization of this research pipeline, workspace `e:\quantization` contained no pre-existing code, models, or configurations.
- An isolated Python environment (`C:\Users\Digvijay Nigvekar\openvino_env`) was verified and updated with the exact production ML stack.
- No existing assets were overwritten. All subsequent scripts and artifacts are built in `e:\quantization` with full reproducibility.
