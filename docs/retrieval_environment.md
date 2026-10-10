# Retrieval Environment & Hardware Profile

This document records the exact hardware, operating system, machine learning libraries, embedding models, and dataset versions used for the reproducible offline Medical Corpus Ingestion, Retrieval (RAG), and Evaluation stage.

---

## 1. Host System & Compute Profile

| Property | Value |
| :--- | :--- |
| **Operating System** | Microsoft Windows 11 Home Single Language (64-bit) |
| **OS Version / Build** | `10.0.26200` |
| **Host Architecture** | x86_64 / AMD64 |
| **Processor (CPU)** | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (8 Cores, 16 Threads, Zen 4 Architecture, AVX-512, AVX2, FMA3) |
| **System RAM** | 16.0 GB Total Physical (~15.3 GB available) |
| **Discrete GPU (dGPU)** | NVIDIA GeForce RTX 4060 Laptop GPU (8GB GDDR6, OpenVINO Device: `GPU.0`) |
| **Integrated GPU (iGPU)** | AMD Radeon 780M Graphics (`gfx1103`, OpenVINO Device: `GPU.1`) |
| **OpenVINO Available Devices** | `['CPU', 'GPU.0', 'GPU.1']` |

---

## 2. Python Environment & Library Stack

| Component / Package | Version | Source / Notes |
| :--- | :--- | :--- |
| **Python** | `3.14.5` (`tags/v3.14.5:5607950, May 10 2026`) | Active Virtualenv: `C:\Users\Digvijay Nigvekar\openvino_env` |
| **PyTorch (`torch`)** | `2.14.1+cpu` | Official PyTorch CPU build |
| **Hugging Face Transformers** | `5.5.4` | Transformer model architectures & tokenizers |
| **Sentence Transformers** | `6.1.0` | Dense embedding generation & similarity utilities |
| **Intel OpenVINO Runtime** | `2026.4.1-22982-e213a147257-releases/2026/4` | High-performance graph inference engine |
| **Intel NNCF** | `3.4.0` | Neural Network Compression Framework for PTQ |
| **Optimum Intel (`optimum-intel`)** | `2.2.0` | OpenVINO pipeline integration |
| **FAISS (`faiss-cpu`)** | `1.15.1` | Efficient vector similarity indexing |
| **Hugging Face Datasets** | `5.1.0` | Dataset streaming & benchmark data management |
| **NumPy** | `2.4.6` | Vector math and array serialization |
| **Pandas** | `3.0.6` | Tabular data analysis and validation export |
| **Scikit-learn** | `1.9.1` | Stratification and statistical evaluation metrics |
| **PyYAML** | `6.0.3` | Declarative configuration management |

---

## 3. Embedding Model Specifications

| Parameter | Specification |
| :--- | :--- |
| **Model Name** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Hugging Face Hub ID** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Architecture** | MiniLM-L6 (6 transformer layers, 12 attention heads, 384 hidden dimension) |
| **Base Tokenizer** | WordPiece (`BertTokenizerFast`) |
| **Embedding Dimension ($d$)** | `384` |
| **Max Sequence Length** | `256` tokens (512 theoretical max) |
| **Pooling Strategy** | Mean Pooling over token embeddings with attention masking |
| **Normalization** | L2 Vector Normalization ($\|v\|_2 = 1.0$) for Cosine Inner Product similarity |
| **OpenVINO FP32 Target** | `models/embeddings/fp32/openvino_embedding_model.xml` |
| **OpenVINO INT8 Target** | `models/embeddings/int8/openvino_embedding_model.xml` |

---

## 4. Medical Corpus & Benchmark Datasets

| Dataset / Corpus | Source / Repository | Split / Subset | Records | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Medical Reference Corpus** | Clinical Guidelines, Pharmacotherapy Protocols & PubMedQA Abstracts | Ingested & normalized | 1,020 documents | Dense Vector Retrieval Corpus |
| **MedQA (USMLE)** | `GBaker/MedQA-USMLE-4-options-hf` / `bigbio/med_qa` | Test Split (Stratified) | 100 fixed queries | Multi-choice clinical reasoning evaluation |
| **PubMedQA** | `qiaojin/PubMedQA` (`pqa_labeled`) | Labeled Split (Stratified) | 100 fixed queries | Biomedical QA (yes/no/maybe) evaluation |
