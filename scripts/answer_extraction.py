"""
Deterministic Answer Extraction Module for Medical QA Benchmarks.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System

Extracts structured predictions (option letters for MedQA, decision labels for PubMedQA)
from generated model text using a strict, multi-stage deterministic regex hierarchy.
Ambiguous, malformed, or contradictory responses are flagged as 'UNPARSED'.
"""

import logging
import re
import sys
from typing import Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_medqa_answer(text: str, valid_options: Tuple[str, ...] = ("A", "B", "C", "D", "E")) -> Tuple[str, str]:
    """
    Extracts multiple-choice option letter (A, B, C, D, E) from LLM output.
    Returns: (extracted_choice, extraction_rule)
    If no unambiguous choice can be extracted, returns ("UNPARSED", "failed_all_rules").
    """
    if not text or not text.strip():
        return "UNPARSED", "empty_text"

    clean_text = text.strip()

    # Rule 1: Explicit conclusion indicators at end or start of sentences
    # e.g., "The correct answer is (B)", "Answer: C", "Correct option: **A**", "Conclusion: D"
    pattern_rule1 = [
        r"(?:the\s+)?(?:correct|best)\s+(?:answer|option|choice)\s+is\s*[:\*\(\[]*\s*([A-E])\b",
        r"(?:answer|choice|option)\s*[:\*\(\[]*\s*([A-E])\b",
        r"(?:therefore|thus|hence|consequently)[,\s]+(?:the\s+)?(?:answer|choice|option\s+is)?\s*[:\*\(\[]*\s*([A-E])\b",
        r"\b([A-E])\s*(?:\.|\))\s+(?:is\s+the\s+correct|is\s+correct)"
    ]

    for p in pattern_rule1:
        matches = re.findall(p, clean_text, re.IGNORECASE)
        if matches:
            cand = matches[-1].upper() # Take the final conclusion if multiple
            if cand in valid_options:
                return cand, "rule1_explicit_conclusion"

    # Rule 2: Markdown emphasized or parenthesized single options near the beginning or end
    # e.g., "**B**", "(C)", "[A]"
    lines = [ln.strip() for ln in clean_text.splitlines() if ln.strip()]
    if lines:
        first_line = lines[0]
        last_line = lines[-1]
        
        # Check last line first
        last_match = re.search(r"[\*\(\[]+([A-E])[\*\)\]]+", last_line)
        if last_match and last_match.group(1).upper() in valid_options:
            return last_match.group(1).upper(), "rule2_last_line_emphasis"
            
        # Check first line
        first_match = re.search(r"^[\*\(\[]*([A-E])[\*\)\]\.:\s]", first_line)
        if first_match and first_match.group(1).upper() in valid_options:
            return first_match.group(1).upper(), "rule2_first_line_start"

    # Rule 3: Single isolated option letter on its own line
    for line in reversed(lines):
        if line.strip().upper() in valid_options:
            return line.strip().upper(), "rule3_isolated_line"

    # Rule 4: Standalone option pattern in short responses
    if len(clean_text) < 50:
        short_match = re.search(r"\b([A-E])\b", clean_text)
        if short_match and short_match.group(1).upper() in valid_options:
            return short_match.group(1).upper(), "rule4_short_text_standalone"

    return "UNPARSED", "failed_all_rules"


def extract_pubmedqa_answer(text: str) -> Tuple[str, str]:
    """
    Extracts PubMedQA decision label ('yes', 'no', 'maybe') from LLM output.
    Returns: (extracted_label, extraction_rule)
    """
    if not text or not text.strip():
        return "UNPARSED", "empty_text"

    clean_text = text.strip()

    # Rule 1: Explicit decision indicator
    pattern_rule1 = [
        r"(?:the\s+)?(?:answer|decision|conclusion)\s+is\s*[:\*\(\[]*\s*(yes|no|maybe)\b",
        r"(?:therefore|thus|hence|in conclusion)[,\s]+(?:the\s+)?(?:answer\s+is\s*)?[:\*\(\[]*\s*(yes|no|maybe)\b",
        r"^(?:yes|no|maybe)\b"
    ]

    for p in pattern_rule1:
        matches = re.findall(p, clean_text, re.IGNORECASE)
        if matches:
            return matches[-1].lower(), "rule1_explicit_decision"

    # Rule 2: Check last line for isolated or emphasized yes/no/maybe
    lines = [ln.strip() for ln in clean_text.splitlines() if ln.strip()]
    if lines:
        last_line = lines[-1].lower()
        for label in ["yes", "no", "maybe"]:
            if re.search(rf"\b{label}\b", last_line):
                return label, "rule2_last_line"

    # Rule 3: First line start
    if lines:
        first_line = lines[0].lower()
        for label in ["yes", "no", "maybe"]:
            if re.search(rf"^{label}\b", first_line):
                return label, "rule3_first_line"

    return "UNPARSED", "failed_all_rules"


def parse_and_score(
    generated_text: str,
    gold_answer: str,
    task_type: str = "medqa"
) -> Dict[str, Any]:
    """
    Extracts prediction and evaluates correctness against ground truth.
    """
    if task_type == "medqa":
        pred, rule = extract_medqa_answer(generated_text)
        gold_norm = gold_answer.strip().upper()
        is_parsed = (pred != "UNPARSED")
        is_correct = (pred == gold_norm) if is_parsed else False
    elif task_type == "pubmedqa":
        pred, rule = extract_pubmedqa_answer(generated_text)
        gold_norm = gold_answer.strip().lower()
        is_parsed = (pred != "UNPARSED")
        is_correct = (pred == gold_norm) if is_parsed else False
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return {
        "extracted_answer": pred,
        "gold_answer": gold_norm,
        "is_parsed": is_parsed,
        "is_correct": is_correct,
        "extraction_rule": rule
    }


if __name__ == "__main__":
    # Test cases
    test_medqa = [
        ("The clinical findings indicate acute myocardial infarction. The correct answer is B.", "B"),
        ("Based on the symptoms, Option D is the most effective therapy.", "D"),
        ("**A** is the diagnostic test of choice.", "A"),
        ("I am unsure between A and C, but it could be complex.", "A"), # Should be UNPARSED
        ("Answer: C", "C")
    ]
    print("Testing MedQA Answer Extraction:")
    for txt, gold in test_medqa:
        res = parse_and_score(txt, gold, "medqa")
        print(f"Text: '{txt[:45]}...' -> Pred: {res['extracted_answer']} | Gold: {res['gold_answer']} | Parsed: {res['is_parsed']} | Correct: {res['is_correct']}")

    test_pubmed = [
        ("The study demonstrates a clear therapeutic effect. The answer is yes.", "yes"),
        ("Conclusion: No significant association was observed.", "no"),
        ("Evidence remains contradictory, therefore maybe.", "maybe"),
        ("We analyzed 500 patients over 10 years.", "yes") # Unparsed
    ]
    print("\nTesting PubMedQA Answer Extraction:")
    for txt, gold in test_pubmed:
        res = parse_and_score(txt, gold, "pubmedqa")
        print(f"Text: '{txt[:45]}...' -> Pred: {res['extracted_answer']} | Gold: {res['gold_answer']} | Parsed: {res['is_parsed']} | Correct: {res['is_correct']}")
