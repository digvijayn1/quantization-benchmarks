"""
FP32 and FP16 Baseline Validation Script.

Executes deterministic inference across fixed medical QA prompts to establish
the reference baseline outputs before weight compression.

IMPORTANT NOTICE:
These prompts and outputs are solely for technical inference validation,
runtime stability verification, and baseline comparison. They are NOT
intended to evaluate clinical correctness or offer medical advice.
"""

import argparse
import difflib
import json
import logging
import re
import sys
import time
from pathlib import Path

import openvino as ov
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

from validation_prompts import VALIDATION_PROMPTS, GENERATION_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/validation")


def calculate_repetition_ratio(text: str, n: int = 3) -> float:
    """
    Computes repetition ratio based on n-gram uniqueness.
    Returns value between 0.0 (all unique) and 1.0 (completely repetitive).
    """
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    unique_ngrams = set(ngrams)
    # Ratio of repeated n-grams
    return round(1.0 - (len(unique_ngrams) / len(ngrams)), 4)


def validate_variant(model_path: Path, variant_name: str, device: str = "CPU"):
    """
    Validates a single OpenVINO IR model variant on the fixed medical QA prompt set.
    """
    logger.info(f"--- Starting Baseline Validation for {variant_name} on {device} ---")
    logger.info(f"Model path: {model_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
        
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = OVModelForCausalLM.from_pretrained(
        model_path,
        device=device,
        compile=True,
        trust_remote_code=True
    )
    load_time = time.time() - t0
    logger.info(f"Model loaded and compiled in {load_time:.2f}s")
    
    prompt_results = []
    
    for item in VALIDATION_PROMPTS:
        p_id = item["id"]
        raw_prompt = item["prompt"]
        
        # Format with ChatML template
        formatted_input = (
            f"<|im_start|>system\nYou are an offline medical question-answering assistant. "
            f"Provide an informative, factual response.<|im_end|>\n"
            f"<|im_start|>user\n{raw_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        inputs = tokenizer(formatted_input, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]
        
        t_start = time.time()
        runtime_success = False
        generated_text = ""
        error_msg = None
        
        try:
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                temperature=GENERATION_CONFIG["temperature"],
                do_sample=GENERATION_CONFIG["do_sample"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            # Slice generated tokens only
            new_tokens = output_tokens[0][input_len:]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            runtime_success = True
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Inference error on {p_id}: {e}")
            
        latency = time.time() - t_start
        output_present = len(generated_text) > 0
        rep_ratio = calculate_repetition_ratio(generated_text)
        
        # Validation status categorization
        if not runtime_success or not output_present or len(generated_text) < 15:
            val_status = "FAIL"
        elif rep_ratio > 0.35:
            val_status = "WARNING"
        else:
            val_status = "PASS"
            
        record = {
            "prompt_id": p_id,
            "domain": item["domain"],
            "input_text": raw_prompt,
            "generated_output": generated_text,
            "output_present": output_present,
            "output_length_chars": len(generated_text),
            "output_length_words": len(generated_text.split()),
            "repetition_ratio": rep_ratio,
            "runtime_success": runtime_success,
            "latency_seconds": round(latency, 3),
            "validation_status": val_status,
            "generation_parameters": GENERATION_CONFIG,
            "model_variant": variant_name,
            "execution_device": device,
            "error_message": error_msg
        }
        prompt_results.append(record)
        logger.info(f"[{variant_name}] {p_id} ({item['domain']}): {val_status} | Length: {len(generated_text)} chars | Repetition: {rep_ratio} | Latency: {latency:.2f}s")
        
    overall_status = "PASS"
    if any(r["validation_status"] == "FAIL" for r in prompt_results):
        overall_status = "FAIL"
    elif any(r["validation_status"] == "WARNING" for r in prompt_results):
        overall_status = "WARNING"
        
    summary_data = {
        "model_variant": variant_name,
        "execution_device": device,
        "load_time_seconds": round(load_time, 2),
        "overall_status": overall_status,
        "total_prompts": len(prompt_results),
        "successful_prompts": sum(1 for r in prompt_results if r["runtime_success"]),
        "generation_config": GENERATION_CONFIG,
        "prompts": prompt_results
    }
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"{variant_name}_validation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Validation summary saved to {out_file}")
    
    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Validate OpenVINO FP32 and FP16 baselines.")
    parser.add_argument("--variants", nargs="+", default=["fp32", "fp16"], help="Variants to validate")
    parser.add_argument("--device", type=str, default="CPU", help="Inference device (CPU, GPU.0, GPU.1)")
    args = parser.parse_args()
    
    for var in args.variants:
        m_path = Path("models/openvino") / var
        validate_variant(m_path, var, device=args.device)


if __name__ == "__main__":
    main()
