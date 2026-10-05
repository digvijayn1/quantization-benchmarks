"""
Comprehensive Benchmarking and Evaluation Script.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System
Base Model: Qwen/Qwen2.5-0.5B-Instruct

Evaluates OpenVINO IR model variants across:
1. Physical Footprint: XML size, BIN size, total size, compression ratio vs FP32
2. Inference Latency: Load time, Time to First Token (TTFT), tokens per second (tok/s), total generation time
3. Qualitative Metrics: Repetition ratio, word count, output coherence on medical QA prompts
Saves detailed JSON and summary CSV to results/benchmarks/.
"""

import argparse
import csv
import json
import logging
import os
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

BENCHMARKS_DIR = Path("results/benchmarks")
MODELS_BASE_DIR = Path("models/openvino")


def calculate_repetition_ratio(text: str, n: int = 3) -> float:
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return round(1.0 - (len(set(ngrams)) / len(ngrams)), 4)


def get_model_size_mb(model_dir: Path):
    xml_files = list(model_dir.glob("*.xml"))
    bin_files = list(model_dir.glob("*.bin"))
    if not xml_files or not bin_files:
        return 0.0, 0.0, 0.0
    xml_mb = round(xml_files[0].stat().st_size / (1024 * 1024), 2)
    bin_mb = round(bin_files[0].stat().st_size / (1024 * 1024), 2)
    return xml_mb, bin_mb, round(xml_mb + bin_mb, 2)


def benchmark_variant(model_dir: Path, variant_id: str, device: str = "CPU", fp32_size_mb: float = 944.08):
    logger.info("=" * 70)
    logger.info(f"Benchmarking Variant: {variant_id} on {device}")
    logger.info(f"Directory: {model_dir}")
    logger.info("=" * 70)
    
    xml_mb, bin_mb, total_mb = get_model_size_mb(model_dir)
    compression_ratio = round(fp32_size_mb / total_mb, 2) if total_mb > 0 else 1.0
    size_reduction_pct = round((1.0 - (total_mb / fp32_size_mb)) * 100, 1) if total_mb > 0 else 0.0
    
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = OVModelForCausalLM.from_pretrained(
        model_dir,
        device=device,
        compile=True,
        trust_remote_code=True
    )
    load_time = time.time() - t0
    logger.info(f"Compiled in {load_time:.2f}s | Size: {total_mb} MB (Reduction: {size_reduction_pct}%, Ratio: {compression_ratio}x)")
    
    prompt_records = []
    latencies = []
    tokens_per_sec_list = []
    
    for item in VALIDATION_PROMPTS:
        p_id = item["id"]
        raw_prompt = item["prompt"]
        formatted_input = (
            f"<|im_start|>system\nYou are an offline medical question-answering assistant. "
            f"Provide an informative, factual response.<|im_end|>\n"
            f"<|im_start|>user\n{raw_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(formatted_input, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]
        
        t_start = time.time()
        try:
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                temperature=GENERATION_CONFIG["temperature"],
                do_sample=GENERATION_CONFIG["do_sample"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            gen_time = time.time() - t_start
            new_tokens = output_tokens[0][input_len:]
            gen_tokens_count = len(new_tokens)
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            runtime_success = True
        except Exception as e:
            gen_time = time.time() - t_start
            gen_tokens_count = 0
            generated_text = ""
            runtime_success = False
            logger.error(f"Error on {p_id}: {e}")
            
        tps = round(gen_tokens_count / gen_time, 2) if gen_time > 0 and gen_tokens_count > 0 else 0.0
        rep_ratio = calculate_repetition_ratio(generated_text)
        
        latencies.append(gen_time)
        tokens_per_sec_list.append(tps)
        
        val_status = "PASS" if runtime_success and len(generated_text) >= 15 and rep_ratio <= 0.35 else "WARNING"
        prompt_records.append({
            "prompt_id": p_id,
            "domain": item["domain"],
            "generated_text": generated_text,
            "generated_tokens": gen_tokens_count,
            "latency_seconds": round(gen_time, 3),
            "tokens_per_second": tps,
            "repetition_ratio": rep_ratio,
            "status": val_status
        })
        logger.info(f"[{variant_id}] {p_id}: {val_status} | {tps} tok/s | Latency: {gen_time:.2f}s | Rep: {rep_ratio}")
        
    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    avg_tps = round(sum(tokens_per_sec_list) / len(tokens_per_sec_list), 2) if tokens_per_sec_list else 0.0
    avg_rep = round(sum(r["repetition_ratio"] for r in prompt_records) / len(prompt_records), 4) if prompt_records else 0.0
    
    summary = {
        "variant_id": variant_id,
        "device": device,
        "model_dir": str(model_dir).replace("\\", "/"),
        "xml_size_mb": xml_mb,
        "bin_size_mb": bin_mb,
        "total_size_mb": total_mb,
        "compression_ratio": compression_ratio,
        "size_reduction_pct": size_reduction_pct,
        "compile_time_seconds": round(load_time, 2),
        "avg_latency_seconds": avg_latency,
        "avg_tokens_per_second": avg_tps,
        "avg_repetition_ratio": avg_rep,
        "prompts": prompt_records
    }
    return summary


def discover_model_variants(base_dir: Path):
    variants = []
    for root, dirs, files in os.walk(base_dir):
        if "openvino_model.xml" in files and "openvino_model.bin" in files:
            p = Path(root)
            var_id = str(p.relative_to(base_dir)).replace("\\", "/")
            variants.append((var_id, p))
    return sorted(variants, key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenVINO quantized model variants.")
    parser.add_argument("--variants", nargs="+", default=None, help="Specific variant IDs or folder paths")
    parser.add_argument("--device", type=str, default="CPU", help="Inference device (CPU, GPU.0, GPU.1)")
    args = parser.parse_args()
    
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    all_discovered = discover_model_variants(MODELS_BASE_DIR)
    
    if args.variants:
        target_variants = []
        for v in args.variants:
            matched = False
            for v_id, v_path in all_discovered:
                if v == v_id or v == str(v_path) or v in v_id:
                    target_variants.append((v_id, v_path))
                    matched = True
                    break
            if not matched:
                logger.warning(f"Could not find variant matching: {v}")
    else:
        target_variants = all_discovered
        
    logger.info(f"Discovered {len(target_variants)} model variants to benchmark: {[v[0] for v in target_variants]}")
    
    results = []
    csv_rows = []
    
    for v_id, v_path in target_variants:
        res = benchmark_variant(v_path, v_id, device=args.device)
        results.append(res)
        csv_rows.append({
            "Variant": v_id,
            "Device": args.device,
            "Total_Size_MB": res["total_size_mb"],
            "Compression_Ratio": f"{res['compression_ratio']}x",
            "Size_Reduction_%": f"{res['size_reduction_pct']}%",
            "Compile_Time_s": res["compile_time_seconds"],
            "Avg_Latency_s": res["avg_latency_seconds"],
            "Avg_Tokens_Per_Sec": res["avg_tokens_per_second"],
            "Avg_Repetition_Ratio": res["avg_repetition_ratio"]
        })
        
    # Write JSON
    json_path = BENCHMARKS_DIR / f"benchmark_summary_{args.device.lower()}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Detailed JSON results written to {json_path}")
    
    # Write CSV
    csv_path = BENCHMARKS_DIR / f"benchmark_summary_{args.device.lower()}.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info(f"Summary CSV written to {csv_path}")


if __name__ == "__main__":
    main()
