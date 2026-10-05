"""
Hugging Face -> Optimum Intel -> OpenVINO IR Export Pipeline.

Exports a Hugging Face causal language model to OpenVINO Intermediate Representation (IR),
generating both FP32 and FP16 baseline variants.
Programmatically detects and validates generated IR files.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import openvino as ov
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, AutoConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_BASE_DIR = Path("models/openvino")


def detect_ir_files(directory: Path):
    """Programmatically detect XML and BIN files in the exported directory."""
    xml_files = list(directory.glob("*.xml"))
    bin_files = list(directory.glob("*.bin"))
    
    if not xml_files or not bin_files:
        raise FileNotFoundError(
            f"Expected OpenVINO IR files (*.xml, *.bin) not found in {directory}. "
            f"Found XMLs: {xml_files}, BINs: {bin_files}"
        )
    
    xml_path = xml_files[0]
    bin_path = bin_files[0]
    return {
        "xml_path": str(xml_path).replace("\\", "/"),
        "bin_path": str(bin_path).replace("\\", "/"),
        "xml_size_bytes": xml_path.stat().st_size,
        "bin_size_bytes": bin_path.stat().st_size,
        "xml_size_mb": round(xml_path.stat().st_size / (1024 * 1024), 2),
        "bin_size_mb": round(bin_path.stat().st_size / (1024 * 1024), 2),
        "total_size_mb": round((xml_path.stat().st_size + bin_path.stat().st_size) / (1024 * 1024), 2)
    }


def export_baselines(model_id: str = DEFAULT_MODEL_ID, base_dir: Path = OUTPUT_BASE_DIR):
    """
    Exports the model to FP32 and FP16 OpenVINO IR baselines.
    """
    logger.info(f"Initiating OpenVINO IR export for model: {model_id}")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    fp32_dir = base_dir / "fp32"
    fp16_dir = base_dir / "fp16"
    fp32_dir.mkdir(parents=True, exist_ok=True)
    fp16_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading tokenizer and configuration...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # Step 1: Export base PyTorch model to OpenVINO graph via Optimum Intel
    t0 = time.time()
    logger.info("Exporting PyTorch model to OpenVINO IR representation (Optimum Intel)...")
    ov_model_wrapper = OVModelForCausalLM.from_pretrained(
        model_id,
        export=True,
        compile=False,
        load_in_8bit=False,
        trust_remote_code=True
    )
    export_duration = time.time() - t0
    logger.info(f"OpenVINO graph conversion completed in {export_duration:.2f}s")
    
    # Step 2: Save FP32 Baseline (compress_to_fp16=False)
    logger.info(f"Saving FP32 baseline to {fp32_dir}...")
    fp32_xml = fp32_dir / "openvino_model.xml"
    ov.save_model(ov_model_wrapper.model, fp32_xml, compress_to_fp16=False)
    
    # Save tokenizer and configs to FP32 directory
    tokenizer.save_pretrained(fp32_dir)
    config.save_pretrained(fp32_dir)
    if hasattr(ov_model_wrapper, "generation_config") and ov_model_wrapper.generation_config is not None:
        ov_model_wrapper.generation_config.save_pretrained(fp32_dir)
        
    fp32_info = detect_ir_files(fp32_dir)
    logger.info(f"FP32 Baseline saved successfully: {fp32_info}")
    
    # Step 3: Save FP16 Baseline (compress_to_fp16=True)
    logger.info(f"Saving FP16 baseline to {fp16_dir}...")
    fp16_xml = fp16_dir / "openvino_model.xml"
    ov.save_model(ov_model_wrapper.model, fp16_xml, compress_to_fp16=True)
    
    # Save tokenizer and configs to FP16 directory
    tokenizer.save_pretrained(fp16_dir)
    config.save_pretrained(fp16_dir)
    if hasattr(ov_model_wrapper, "generation_config") and ov_model_wrapper.generation_config is not None:
        ov_model_wrapper.generation_config.save_pretrained(fp16_dir)
        
    fp16_info = detect_ir_files(fp16_dir)
    logger.info(f"FP16 Baseline saved successfully: {fp16_info}")
    
    summary = {
        "model_id": model_id,
        "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "export_duration_seconds": round(export_duration, 2),
        "fp32_baseline": fp32_info,
        "fp16_baseline": fp16_info
    }
    
    manifest_path = base_dir / "export_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Baseline export manifest written to {manifest_path}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Hugging Face model to OpenVINO IR FP32/FP16 baselines.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model ID or path")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_BASE_DIR), help="Output directory")
    args = parser.parse_args()
    
    export_baselines(model_id=args.model_id, base_dir=Path(args.output_dir))
