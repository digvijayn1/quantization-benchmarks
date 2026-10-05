"""
Standard Medical QA Validation Prompts.

IMPORTANT NOTICE:
These prompts are strictly designed for TECHNICAL VALIDATION of model inference,
runtime stability, tokenizer alignment, output coherence, and repetition metrics.
They are NOT intended to evaluate clinical correctness or provide medical advice.
"""

VALIDATION_PROMPTS = [
    {
        "id": "val_med_01",
        "domain": "Endocrinology",
        "prompt": "What are the common early symptoms and diagnostic indicators of Type 2 Diabetes Mellitus?"
    },
    {
        "id": "val_med_02",
        "domain": "Cardiology / Pulmonology",
        "prompt": "Explain why non-selective beta-blockers should be used with extreme caution in patients with asthma."
    },
    {
        "id": "val_med_03",
        "domain": "Cardiovascular Therapeutics",
        "prompt": "What are the primary first-line pharmacological drug classes recommended for managing essential hypertension?"
    },
    {
        "id": "val_med_04",
        "domain": "Nephrology / Pharmacokinetics",
        "prompt": "How does reduced renal clearance alter the dosing strategy for medications excreted primarily by the kidneys?"
    },
    {
        "id": "val_med_05",
        "domain": "Emergency Medicine / Trauma",
        "prompt": "Describe the three behavioral response components evaluated in the Glasgow Coma Scale (GCS)."
    }
]

GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "temperature": 0.0,
    "do_sample": False,
    "seed": 42
}
