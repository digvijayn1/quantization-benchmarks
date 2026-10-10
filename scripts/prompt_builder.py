"""
Prompt Assembly and Context Budget Management Module.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System

Builds formatted, tokenizer-aligned chat prompts combining retrieved clinical chunks,
question vignettes, and multiple-choice options with strict context-window budgeting,
delimiter protection, and deterministic truncation reporting.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = Path("prompts/medical_qa_prompt.txt")


class MedicalPromptBuilder:
    def __init__(
        self,
        template_path: Path = PROMPT_TEMPLATE_PATH,
        max_context_tokens: int = 1536,
        chunk_separator: str = "\n\n---\n\n",
        include_metadata: bool = True
    ):
        self.template_path = Path(template_path)
        self.max_context_tokens = max_context_tokens
        self.chunk_separator = chunk_separator
        self.include_metadata = include_metadata
        self._load_template()

    def _load_template(self):
        if not self.template_path.exists():
            raise FileNotFoundError(f"Prompt template not found at {self.template_path}")
        with open(self.template_path, "r", encoding="utf-8") as f:
            self.template = f.read()

    def format_options(self, options: Dict[str, str]) -> str:
        """Formats multiple-choice options dictionary into a clean block."""
        if not options:
            return ""
        lines = ["Options:"]
        for key in sorted(options.keys()):
            lines.append(f"{key}) {options[key]}")
        return "\n".join(lines)

    def assemble_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        approx_chars_per_token: int = 4
    ) -> Tuple[str, List[str], int, int]:
        """
        Assembles ranked context chunks, enforcing maximum context token budget.
        Returns: (formatted_context_str, used_chunk_ids, total_chunks_used, truncated_tokens)
        """
        if not retrieved_chunks:
            return "No additional external context retrieved.", [], 0, 0

        max_char_budget = self.max_context_tokens * approx_chars_per_token
        assembled_blocks = []
        used_ids = []
        current_chars = 0
        truncated_chars = 0

        for chunk in retrieved_chunks:
            rank = chunk.get("rank", len(assembled_blocks) + 1)
            source = chunk.get("source", "Medical Corpus")
            title = chunk.get("title", "")
            text = chunk.get("text", "").strip()
            c_id = chunk.get("chunk_id", f"chunk_{rank}")

            header = f"[Source {rank}: {source} | {title}]" if self.include_metadata else f"[Source {rank}]"
            block = f"{header}\n{text}"
            block_len = len(block) + len(self.chunk_separator)

            if current_chars + block_len <= max_char_budget:
                assembled_blocks.append(block)
                used_ids.append(c_id)
                current_chars += block_len
            else:
                # Truncate or omit
                remaining_budget = max_char_budget - current_chars
                if remaining_budget > 150: # Partial inclusion if meaningful
                    trimmed_text = text[: remaining_budget - len(header) - 30] + "... [TRUNCATED]"
                    partial_block = f"{header}\n{trimmed_text}"
                    assembled_blocks.append(partial_block)
                    used_ids.append(c_id)
                    truncated_chars += (len(text) - len(trimmed_text))
                else:
                    truncated_chars += len(text)

        formatted_context = self.chunk_separator.join(assembled_blocks)
        truncated_tokens = truncated_chars // approx_chars_per_token

        return formatted_context, used_ids, len(used_ids), truncated_tokens

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        options: Optional[Dict[str, str]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Builds full prompt string with assembled context and metadata.
        """
        formatted_context, used_ids, num_used, trunc_tokens = self.assemble_context(retrieved_chunks)
        options_block = self.format_options(options) if options else ""

        full_prompt = self.template.format(
            retrieved_context=formatted_context,
            question=question.strip(),
            options_block=options_block
        )

        meta = {
            "num_chunks_retrieved": len(retrieved_chunks),
            "num_chunks_used": num_used,
            "used_chunk_ids": used_ids,
            "context_truncated_tokens": trunc_tokens,
            "max_context_budget_tokens": self.max_context_tokens
        }
        return full_prompt, meta


if __name__ == "__main__":
    builder = MedicalPromptBuilder()
    sample_chunks = [
        {
            "rank": 1,
            "chunk_id": "doc_001_chunk_001",
            "source": "Clinical Guidelines",
            "title": "Type 2 Diabetes Diagnostic Thresholds",
            "text": "Diagnostic criteria: Fasting plasma glucose >= 126 mg/dL (7.0 mmol/L), 2-hour plasma glucose >= 200 mg/dL during OGTT, or HbA1c >= 6.5%."
        }
    ]
    prompt, meta = builder.build_prompt(
        question="What is the diagnostic threshold for fasting blood glucose in diabetes?",
        retrieved_chunks=sample_chunks,
        options={"A": ">= 100 mg/dL", "B": ">= 126 mg/dL", "C": ">= 140 mg/dL", "D": ">= 200 mg/dL"}
    )
    print("Generated Prompt Preview:")
    print("=" * 70)
    print(prompt)
    print("=" * 70)
    print("Metadata:", meta)
