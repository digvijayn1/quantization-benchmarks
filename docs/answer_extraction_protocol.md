# Answer Extraction & Evaluation Protocol

This document defines the deterministic protocol used to extract structured benchmark answers from free-form model generations in the offline medical question-answering evaluation pipeline.

---

## 1. Principles of Automated Medical Evaluation

1. **Deterministic & Rule-Governed**: Evaluation must never rely on human subjective interpretation or LLM-as-a-judge prompting which introduces nondeterminism.
2. **Strict Hierarchy**: High-confidence conclusion markers take precedence over tokens mentioned in background chain-of-thought reasoning.
3. **No Guessing**: If a model generates ambiguous, non-committal, or multiple conflicting options, the response must be tagged as `UNPARSED`.
4. **Transparent Failure Categorization**: Distinguish between model reasoning failures, retrieval failures, and extraction failures.

---

## 2. MedQA (USMLE) Multi-Choice Extraction Protocol

MedQA queries present a clinical vignette and 4-5 answer choices labeled `A`, `B`, `C`, `D`, `E`.

### Rule Hierarchy:
1. **Rule 1: Explicit Conclusion Indicators**:
   - Matches: `"The correct answer is (X)"`, `"Answer: X"`, `"Correct option: **X**"`, `"Therefore, option X is correct"`.
   - If multiple conclusions appear, the final stated conclusion is selected.
2. **Rule 2: Emphasized Options on First or Last Line**:
   - Matches bold or parenthesized letters (e.g., `**B**`, `(C)`, `[A]`) occurring in the initial or concluding line.
3. **Rule 3: Isolated Line Token**:
   - Matches a standalone letter on its own line (e.g. `B`).
4. **Rule 4: Short-Text Standalone**:
   - For outputs shorter than 50 characters with an unambiguous single choice token.
5. **Fallback**:
   - Returns `UNPARSED`.

### Examples:

| Generated Output | Extracted Option | Rule Applied | Status |
| :--- | :--- | :--- | :--- |
| `Based on clinical symptoms of polyuria and polydipsia, the correct answer is B.` | **`B`** | `rule1_explicit_conclusion` | **PARSED** |
| `Option **A** is the standard first-line therapy.` | **`A`** | `rule2_first_line_start` | **PARSED** |
| `Diagnostic findings indicate stage 2 hypertension.\n\nC` | **`C`** | `rule3_isolated_line` | **PARSED** |
| `We must consider both A and C depending on creatinine clearance.` | **`UNPARSED`** | `failed_all_rules` | **UNPARSED** |

---

## 3. PubMedQA Biomedical Decision Extraction Protocol

PubMedQA requires classifying biomedical hypotheses as `yes`, `no`, or `maybe`.

### Rule Hierarchy:
1. **Rule 1: Explicit Decision Marker**:
   - Matches: `"The answer is yes"`, `"Conclusion: no"`, `"Therefore, maybe"`.
2. **Rule 2: Concluding Line Decision Token**:
   - Scans the final sentence/line for the decisive term.
3. **Rule 3: Initial Decision Token**:
   - Scans the first word/phrase of the response.
4. **Fallback**:
   - Returns `UNPARSED`.

---

## 4. Evaluation Metrics Definition

For each dataset, two accuracy metrics are reported:

$$\text{Raw Accuracy} = \frac{\text{Correct Answers}}{\text{Total Questions}}$$

$$\text{Parsed Accuracy} = \frac{\text{Correct Answers}}{\text{Parsed Questions}}$$

$$\text{Parsing Rate} = \frac{\text{Parsed Questions}}{\text{Total Questions}} \times 100\%$$

---

## 5. Error Taxonomy & Failure Attribution

Every incorrect prediction is categorized into one of the following failure modes:

1. **`CORRECT`**: The extracted answer matches the gold ground truth.
2. **`GENERATION_FAILURE`**: Context contains relevant clinical evidence, but the model reasoned incorrectly or chose the wrong option.
3. **`RETRIEVAL_FAILURE`**: Top-$k$ retrieved chunks do not contain the necessary clinical facts to answer the question.
4. **`ANSWER_EXTRACTION_FAILURE` / `UNPARSED`**: Model generated ambiguous, repetitive, or unparseable text.
