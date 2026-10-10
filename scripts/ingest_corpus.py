"""
Medical Corpus Ingestion Pipeline.
Project: Architecture-Aware Optimization of an Offline Medical Question-Answering System

Ingests clinical reference contexts, domain-specific medical guidelines, and PubMed biomedical abstracts
into a standardized internal document schema:
{
    "document_id": "...",
    "title": "...",
    "text": "...",
    "source": "...",
    "metadata": {}
}
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
CORPUS_DIR = DATA_DIR / "corpus"
CALIBRATION_FILE = DATA_DIR / "calibration" / "medical_calibration_data.json"
OUTPUT_CORPUS = PROCESSED_DIR / "normalized_corpus.json"


def ingest_calibration_data() -> List[Dict[str, Any]]:
    """Ingests curated clinical guidelines and reference scenarios from local calibration set."""
    documents = []
    if not CALIBRATION_FILE.exists():
        logger.warning(f"Calibration data not found at {CALIBRATION_FILE}")
        return documents

    with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        s_id = item.get("sample_id", "doc_calib")
        domain = item.get("domain", "General Clinical")
        query = item.get("query", "")
        context = item.get("reference_context", "")

        # Rich medical document representation
        doc_text = f"Clinical Scenario & Query: {query}\n\nClinical Guidance: {context}"
        doc = {
            "document_id": f"clinical_{s_id}",
            "title": f"Clinical Reference: {query[:60]}...",
            "text": doc_text.strip(),
            "source": "clinical_calibration_guidelines",
            "metadata": {
                "domain": domain,
                "original_sample_id": s_id,
                "type": "clinical_guideline"
            }
        }
        documents.append(doc)

    logger.info(f"Ingested {len(documents)} clinical reference documents from calibration dataset.")
    return documents


def ingest_pubmedqa_abstracts() -> List[Dict[str, Any]]:
    """Ingests biomedical and clinical abstracts from PubMedQA corpus."""
    documents = []
    try:
        logger.info("Loading PubMedQA labeled corpus abstracts...")
        pqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        for item in pqa:
            pubid = str(item.get("pubid", ""))
            question = item.get("question", "")
            context_dict = item.get("context", {})
            contexts = context_dict.get("contexts", [])
            labels = context_dict.get("labels", [])
            long_answer = item.get("long_answer", "")
            final_decision = item.get("final_decision", "")

            context_text = "\n".join(
                [f"[{lbl}] {txt}" for lbl, txt in zip(labels, contexts)]
            ) if labels and contexts else "\n".join(contexts)

            full_text = f"Title/Question: {question}\n\nAbstract Context:\n{context_text}\n\nFindings & Conclusion: {long_answer}"

            doc = {
                "document_id": f"pubmed_{pubid}",
                "title": question,
                "text": full_text.strip(),
                "source": "pubmed_biomedical_abstracts",
                "metadata": {
                    "pubid": pubid,
                    "final_decision": final_decision,
                    "mesh_terms": item.get("mesh_terms", []),
                    "type": "peer_reviewed_abstract"
                }
            }
            documents.append(doc)
        logger.info(f"Ingested {len(documents)} biomedical abstracts from PubMedQA.")
    except Exception as e:
        logger.error(f"Failed to ingest PubMedQA abstracts: {e}")

    return documents


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Medical Corpus Ingestion...")
    clinical_docs = ingest_calibration_data()
    pubmed_docs = ingest_pubmedqa_abstracts()

    all_docs = clinical_docs + pubmed_docs
    logger.info(f"Total documents assembled: {len(all_docs)}")

    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)

    logger.info(f"Normalized corpus successfully saved to {OUTPUT_CORPUS} ({OUTPUT_CORPUS.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
