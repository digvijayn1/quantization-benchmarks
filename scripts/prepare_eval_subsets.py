"""
Fixed Stratified Evaluation Subsets Generator.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System

Extracts fixed, reproducible, stratified subsets (100 samples each) for MedQA (USMLE)
and PubMedQA benchmarks using seed 42, ensuring balanced answer label distributions.
Outputs:
- data/evaluation/medqa_eval_subset.json
- data/evaluation/pubmedqa_eval_subset.json
- results/evaluation_subset_manifest.json
"""

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

EVAL_DIR = Path("data/evaluation")
RESULTS_DIR = Path("results")
SEED = 42
SUBSET_SIZE = 100


def prepare_medqa_subset() -> List[Dict[str, Any]]:
    logger.info("Loading MedQA test dataset...")
    # Load MedQA test split
    medqa = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="test")
    logger.info(f"Loaded MedQA total test samples: {len(medqa)}")

    # Group by gold label for stratification
    by_label = defaultdict(list)
    for idx, item in enumerate(medqa):
        label_idx = item.get("label", 0)
        label_letter = chr(ord('A') + label_idx) if isinstance(label_idx, int) else str(label_idx)
        
        q_text = item.get("sent1", "") or item.get("question", "")
        options = {
            "A": item.get("ending0", ""),
            "B": item.get("ending1", ""),
            "C": item.get("ending2", ""),
            "D": item.get("ending3", "")
        }
        
        record = {
            "question_id": f"medqa_test_{idx:04d}",
            "question": q_text,
            "options": options,
            "gold_answer": label_letter,
            "dataset": "MedQA-USMLE-4-options",
            "metadata": {
                "original_id": item.get("id", str(idx)),
                "split": "test"
            }
        }
        by_label[label_letter].append(record)

    # Stratified sampling
    rng = random.Random(SEED)
    samples_per_label = SUBSET_SIZE // len(by_label)
    selected = []
    
    for lbl in sorted(by_label.keys()):
        items = by_label[lbl]
        rng.shuffle(items)
        selected.extend(items[:samples_per_label])

    # If remainder, fill deterministically
    remainder = SUBSET_SIZE - len(selected)
    if remainder > 0:
        for lbl in sorted(by_label.keys()):
            items = by_label[lbl]
            if len(items) > samples_per_label:
                selected.append(items[samples_per_label])
                remainder -= 1
                if remainder == 0:
                    break

    rng.shuffle(selected)
    logger.info(f"Created stratified MedQA subset: {len(selected)} samples. Label distribution: {dict((lbl, sum(1 for s in selected if s['gold_answer'] == lbl)) for lbl in sorted(by_label.keys()))}")
    return selected


def prepare_pubmedqa_subset() -> List[Dict[str, Any]]:
    logger.info("Loading PubMedQA labeled dataset...")
    pqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    logger.info(f"Loaded PubMedQA total labeled samples: {len(pqa)}")

    by_decision = defaultdict(list)
    for idx, item in enumerate(pqa):
        pubid = str(item.get("pubid", f"pqa_{idx}"))
        question = item.get("question", "")
        decision = str(item.get("final_decision", "")).lower()
        context_dict = item.get("context", {})
        contexts = context_dict.get("contexts", [])
        long_answer = item.get("long_answer", "")

        record = {
            "question_id": f"pubmedqa_{pubid}",
            "question": question,
            "reference_abstract": "\n".join(contexts),
            "reference_conclusion": long_answer,
            "gold_answer": decision,
            "dataset": "PubMedQA-pqa_labeled",
            "metadata": {
                "pubid": pubid,
                "mesh_terms": item.get("mesh_terms", [])
            }
        }
        by_decision[decision].append(record)

    # Stratified sampling across yes / no / maybe
    rng = random.Random(SEED)
    selected = []
    # PubMedQA labeled distribution is approx 55% yes, 33% no, 12% maybe
    targets = {
        "yes": 55,
        "no": 33,
        "maybe": 12
    }
    for dec, count in targets.items():
        items = by_decision[dec]
        rng.shuffle(items)
        selected.extend(items[:count])

    rng.shuffle(selected)
    logger.info(f"Created stratified PubMedQA subset: {len(selected)} samples. Label distribution: {dict((dec, sum(1 for s in selected if s['gold_answer'] == dec)) for dec in sorted(targets.keys()))}")
    return selected


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    medqa_subset = prepare_medqa_subset()
    pubmedqa_subset = prepare_pubmedqa_subset()

    medqa_file = EVAL_DIR / "medqa_eval_subset.json"
    with open(medqa_file, "w", encoding="utf-8") as f:
        json.dump(medqa_subset, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved MedQA evaluation subset to {medqa_file}")

    pubmedqa_file = EVAL_DIR / "pubmedqa_eval_subset.json"
    with open(pubmedqa_file, "w", encoding="utf-8") as f:
        json.dump(pubmedqa_subset, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved PubMedQA evaluation subset to {pubmedqa_file}")

    manifest = {
        "random_seed": SEED,
        "subsets": {
            "medqa": {
                "source_dataset": "GBaker/MedQA-USMLE-4-options-hf (test split)",
                "total_source_samples": 1273,
                "subset_size": len(medqa_subset),
                "stratification_variable": "gold_answer (A, B, C, D)",
                "samples_per_stratum": {
                    "A": sum(1 for s in medqa_subset if s["gold_answer"] == "A"),
                    "B": sum(1 for s in medqa_subset if s["gold_answer"] == "B"),
                    "C": sum(1 for s in medqa_subset if s["gold_answer"] == "C"),
                    "D": sum(1 for s in medqa_subset if s["gold_answer"] == "D")
                },
                "filepath": str(medqa_file).replace("\\", "/")
            },
            "pubmedqa": {
                "source_dataset": "qiaojin/PubMedQA (pqa_labeled)",
                "total_source_samples": 1000,
                "subset_size": len(pubmedqa_subset),
                "stratification_variable": "final_decision (yes, no, maybe)",
                "samples_per_stratum": {
                    "yes": sum(1 for s in pubmedqa_subset if s["gold_answer"] == "yes"),
                    "no": sum(1 for s in pubmedqa_subset if s["gold_answer"] == "no"),
                    "maybe": sum(1 for s in pubmedqa_subset if s["gold_answer"] == "maybe")
                },
                "filepath": str(pubmedqa_file).replace("\\", "/")
            }
        }
    }

    manifest_file = RESULTS_DIR / "evaluation_subset_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Evaluation subset manifest saved to {manifest_file}")


if __name__ == "__main__":
    main()
