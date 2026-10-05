"""
NNCF Post-Training Quantization Runner.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System
Base Model: Qwen/Qwen2.5-0.5B-Instruct

Executes the post-training weight compression experiments defined in
configs/quantization_experiments.yaml using Intel OpenVINO and NNCF 3.4.0.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
import yaml
import nncf
from nncf import AdvancedCompressionParameters
from nncf.quantization.advanced_parameters import GroupSizeFallbackMode
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/quantization_experiments.yaml")
FP32_BASE_DIR = Path("models/openvino/fp32")
OUTPUT_BASE_DIR = Path("models/openvino")
RESULTS_DIR = Path("results")

MODE_MAPPING = {
    "int8_asym": nncf.CompressWeightsMode.INT8_ASYM,
    "int8_sym": nncf.CompressWeightsMode.INT8_SYM,
    "int4_sym": nncf.CompressWeightsMode.INT4_SYM,
    "int4_asym": nncf.CompressWeightsMode.INT4_ASYM,
}


def load_calibration_dataset(dataset_path: Path, tokenizer, model: ov.Model, max_samples: int = 64):
    """Prepares nncf.Dataset for AWQ and Scale Estimation."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Calibration data not found at {dataset_path}")
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    samples = data[:max_samples]
    input_names = [inp.any_name for inp in model.inputs]
    logger.info(f"Loaded {len(samples)} calibration samples. Model inputs: {input_names}")
    
    def transform_fn(item):
        text = item.get("raw_text") or (
            f"<|im_start|>system\n{item.get('instruction', '')}<|im_end|>\n"
            f"<|im_start|>user\n{item.get('input', '')}<|im_end|>\n"
            f"<|im_start|>assistant\n{item.get('context', '')}<|im_end|>"
        )
        inputs = tokenizer(text, max_length=256, truncation=True, return_tensors="np")
        batch_size, seq_len = inputs["input_ids"].shape
        feed_dict = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        if "position_ids" in input_names:
            pos = (inputs["attention_mask"].cumsum(axis=-1) - 1).astype("int64")
            pos[pos < 0] = 0
            feed_dict["position_ids"] = pos
        if "beam_idx" in input_names:
            feed_dict["beam_idx"] = np.arange(batch_size, dtype=np.int32)
        return feed_dict

    return nncf.Dataset(samples, transform_fn), len(samples)


def copy_tokenizer_and_configs(src_dir: Path, dst_dir: Path):
    """Copies tokenizer and configuration files to the quantized model directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in os.listdir(src_dir):
        if fname.endswith((".xml", ".bin")):
            continue
        src_path = src_dir / fname
        dst_path = dst_dir / fname
        if src_path.is_file():
            shutil.copy2(src_path, dst_path)


def run_experiment(exp: dict, base_fp32_dir: Path = FP32_BASE_DIR):
    """Runs a single quantization experiment."""
    exp_id = exp["id"]
    mode_str = exp["mode"]
    out_dir = Path(exp["output_dir"])
    
    logger.info("=" * 70)
    logger.info(f"Starting Quantization Experiment: {exp_id}")
    logger.info(f"Description: {exp.get('description', '')}")
    logger.info(f"Target Output Directory: {out_dir}")
    logger.info("=" * 70)
    
    if mode_str == "none":
        logger.info(f"Skipping baseline variant {exp_id} (already exported).")
        return {"status": "SKIPPED", "id": exp_id, "reason": "baseline"}
        
    mode = MODE_MAPPING.get(mode_str)
    if mode is None:
        raise ValueError(f"Unknown compression mode: {mode_str}")
        
    core = ov.Core()
    fp32_xml = base_fp32_dir / "openvino_model.xml"
    if not fp32_xml.exists():
        raise FileNotFoundError(f"FP32 base model not found at {fp32_xml}. Run export_openvino.py first.")
        
    logger.info(f"Loading FP32 base model from {fp32_xml}...")
    model = core.read_model(fp32_xml)
    tokenizer = AutoTokenizer.from_pretrained(base_fp32_dir, trust_remote_code=True)
    
    # Dataset preparation for data-dependent algorithms
    dataset = None
    calib_count = 0
    if exp.get("dataset_required") or exp.get("awq") or exp.get("scale_estimation"):
        dataset_path = Path(exp.get("calibration_dataset", "data/calibration/medical_calibration_data.json"))
        dataset, calib_count = load_calibration_dataset(dataset_path, tokenizer, model, max_samples=16)
        
    group_size = exp.get("group_size")
    ratio = exp.get("ratio", 1.0)
    awq = exp.get("awq", False)
    scale_est = exp.get("scale_estimation", False)
    
    kwargs = {
        "mode": mode,
        "ratio": ratio,
        "advanced_parameters": AdvancedCompressionParameters(group_size_fallback_mode=GroupSizeFallbackMode.ADJUST)
    }
    if group_size is not None:
        kwargs["group_size"] = group_size
    if awq and dataset is not None:
        kwargs["awq"] = True
        kwargs["dataset"] = dataset
        kwargs["subset_size"] = min(calib_count, 16)
    elif scale_est and dataset is not None:
        kwargs["scale_estimation"] = True
        kwargs["dataset"] = dataset
        kwargs["subset_size"] = min(calib_count, 16)
        
    logger.info(f"Compressing weights with parameters: mode={mode_str}, group_size={group_size}, ratio={ratio}, awq={awq}, scale_est={scale_est}...")
    t0 = time.time()
    compressed_model = nncf.compress_weights(model, **kwargs)
    duration = time.time() - t0
    logger.info(f"Weight compression for {exp_id} completed in {duration:.2f}s")
    
    # Save quantized model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xml = out_dir / "openvino_model.xml"
    logger.info(f"Saving quantized OpenVINO IR model to {out_xml}...")
    ov.save_model(compressed_model, out_xml)
    
    # Copy tokenizer and metadata
    copy_tokenizer_and_configs(base_fp32_dir, out_dir)
    
    # Measure file sizes
    xml_size_mb = round(out_xml.stat().st_size / (1024 * 1024), 2)
    bin_path = out_dir / "openvino_model.bin"
    bin_size_mb = round(bin_path.stat().st_size / (1024 * 1024), 2) if bin_path.exists() else 0.0
    total_size_mb = round(xml_size_mb + bin_size_mb, 2)
    
    record = {
        "id": exp_id,
        "status": "SUCCESS",
        "mode": mode_str,
        "group_size": group_size,
        "ratio": ratio,
        "awq": awq,
        "scale_estimation": scale_est,
        "compression_time_seconds": round(duration, 2),
        "xml_size_mb": xml_size_mb,
        "bin_size_mb": bin_size_mb,
        "total_size_mb": total_size_mb,
        "output_dir": str(out_dir).replace("\\", "/")
    }
    logger.info(f"Experiment {exp_id} summary: {record}")
    return record


def main():
    parser = argparse.ArgumentParser(description="Run OpenVINO NNCF weight compression experiments.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to experiments YAML config")
    parser.add_argument("--experiment", type=str, default="all", help="Specific experiment ID to run or 'all'")
    parser.add_argument("--skip-baselines", action="store_true", default=True, help="Skip FP32/FP16 baselines")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    experiments = config.get("experiments", [])
    if args.experiment != "all":
        experiments = [e for e in experiments if e["id"] == args.experiment]
        if not experiments:
            logger.error(f"Experiment '{args.experiment}' not found in {args.config}")
            sys.exit(1)
            
    logger.info(f"Loaded {len(experiments)} experiment configurations to process.")
    
    results = []
    for exp in experiments:
        try:
            res = run_experiment(exp)
            results.append(res)
        except Exception as e:
            logger.error(f"Experiment {exp['id']} failed: {e}", exc_info=True)
            results.append({
                "id": exp["id"],
                "status": "FAILED",
                "error": str(e)
            })
            
    summary_path = RESULTS_DIR / "quantization_summary.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Overall quantization summary saved to {summary_path}")


if __name__ == "__main__":
    main()
