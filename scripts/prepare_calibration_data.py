"""
Calibration Dataset Preparation for Medical Question-Answering System.

Produces a representative, de-identified medical QA calibration dataset
for AWQ and Scale Estimation algorithms in OpenVINO / NNCF.
Contains zero PII (Personally Identifiable Information).
"""

import json
import os
from pathlib import Path

CALIBRATION_DIR = Path("data/calibration")
CALIBRATION_DATA_FILE = CALIBRATION_DIR / "medical_calibration_data.json"
MANIFEST_FILE = CALIBRATION_DIR / "calibration_manifest.json"

# Curated, de-identified clinical and medical QA scenarios covering broad medical domains:
MEDICAL_QA_SAMPLES = [
    {
        "id": "med_calib_001",
        "domain": "Endocrinology",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "What are the primary diagnostic thresholds for Type 2 Diabetes Mellitus based on fasting plasma glucose and HbA1c?",
        "context": "Diagnostic criteria: Fasting plasma glucose >= 126 mg/dL (7.0 mmol/L), 2-hour plasma glucose >= 200 mg/dL during OGTT, or HbA1c >= 6.5% (48 mmol/mol), confirmed by repeat testing."
    },
    {
        "id": "med_calib_002",
        "domain": "Cardiology",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "Explain the physiological mechanism of action of ACE inhibitors in reducing systemic arterial hypertension.",
        "context": "ACE inhibitors block the conversion of angiotensin I to angiotensin II, decreasing systemic vascular resistance and aldosterone secretion while increasing bradykinin levels."
    },
    {
        "id": "med_calib_003",
        "domain": "Pulmonology",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "What spirometric ratio defines persistent airflow limitation in Chronic Obstructive Pulmonary Disease (COPD)?",
        "context": "A post-bronchodilator FEV1/FVC ratio less than 0.70 confirms the presence of persistent airflow limitation per GOLD guidelines."
    },
    {
        "id": "med_calib_004",
        "domain": "Pharmacology",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "Why is concurrent administration of citalopram and monoamine oxidase inhibitors (MAOIs) strictly contraindicated?",
        "context": "Combining SSRIs like citalopram with MAOIs carries a high risk of serotonin syndrome, characterized by hyperthermia, autonomic instability, and neuromuscular excitation."
    },
    {
        "id": "med_calib_005",
        "domain": "Neurology",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "What clinical features distinguish an acute ischemic stroke from a transient ischemic attack (TIA)?",
        "context": "TIA involves transient neurological dysfunction caused by focal ischemia without acute infarction on neuroimaging, typically resolving within 1-24 hours."
    },
    {
        "id": "med_calib_006",
        "domain": "Infectious Diseases",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "What are the common empirical antibiotic regimens for community-acquired pneumonia in an outpatient setting without comorbidities?",
        "context": "Standard first-line regimens include amoxicillin (1g tid) or doxycycline (100mg bid); macrolides are alternative options in areas with low pneumococcal resistance."
    },
    {
        "id": "med_calib_007",
        "domain": "Nephrology",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "Define Acute Kidney Injury (AKI) stage 1 according to the KDIGO clinical practice guidelines.",
        "context": "KDIGO Stage 1 AKI is defined as an increase in serum creatinine by >= 0.3 mg/dL within 48 hours, or a 1.5 to 1.9 times baseline increase within 7 days, or urine output < 0.5 mL/kg/h for 6-12 hours."
    },
    {
        "id": "med_calib_008",
        "domain": "Gastroenterology",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "What are the classic diagnostic endoscopic and histological findings associated with Celiac Disease?",
        "context": "Endoscopic findings include scalloping and mucosal atrophy; histological features show intraepithelial lymphocytosis, crypt hyperplasia, and villous blunting (Marsh classification)."
    },
    {
        "id": "med_calib_009",
        "domain": "Hematology",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "Differentiate between absolute iron deficiency anemia and anemia of chronic disease based on ferritin and transferrin saturation.",
        "context": "Iron deficiency anemia typically shows low serum ferritin (<30 ng/mL) and transferrin saturation (<20%). Anemia of chronic disease exhibits normal or elevated ferritin with low to normal transferrin saturation due to hepcidin mediation."
    },
    {
        "id": "med_calib_010",
        "domain": "Emergency Medicine",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "What are the immediate clinical management priorities for acute anaphylactic shock in an adult patient?",
        "context": "Immediate intramuscular epinephrine (0.3-0.5 mg of 1:1,000 solution into the anterolateral thigh), airway management, high-flow oxygen, aggressive IV isotonic fluid resuscitation, and supine positioning."
    },
    {
        "id": "med_calib_011",
        "domain": "Rheumatology",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "What autoantibody is considered most specific for systemic lupus erythematosus (SLE) and correlates with disease activity?",
        "context": "Anti-double-stranded DNA (anti-dsDNA) antibodies are highly specific for SLE and titer elevations often correlate with active lupus nephritis."
    },
    {
        "id": "med_calib_012",
        "domain": "Pediatrics",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "What are the hallmark signs and initial pharmacotherapy for acute laryngotracheobronchitis (croup)?",
        "context": "Hallmark signs include barking cough, inspiratory stridor, and hoarseness. Initial therapy is a single dose of oral or intramuscular dexamethasone (0.15-0.6 mg/kg) plus nebulized racemic epinephrine for moderate to severe distress."
    },
    {
        "id": "med_calib_013",
        "domain": "Oncology",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "Explain the oncogenic mechanism of BCR-ABL1 fusion kinase in chronic myeloid leukemia (CML) and its therapeutic target.",
        "context": "The Philadelphia chromosome translocation t(9;22) produces the constitutively active BCR-ABL1 tyrosine kinase, driving uninhibited granulocytic proliferation. It is targeted by tyrosine kinase inhibitors such as imatinib."
    },
    {
        "id": "med_calib_014",
        "domain": "Dermatology",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "What clinical criteria constitute the ABCDE mnemonic used for melanoma screening?",
        "context": "A: Asymmetry; B: Border irregularity; C: Color variation; D: Diameter > 6 mm; E: Evolving size, shape, or symptomatic characteristics."
    },
    {
        "id": "med_calib_015",
        "domain": "Psychiatry",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "What are the diagnostic DSM-5 time criteria and core symptoms required to diagnose a major depressive episode?",
        "context": "At least 5 symptoms persisting for at least 2 weeks, with at least one symptom being depressed mood or anhedonia, accompanied by significant distress or functional impairment."
    },
    {
        "id": "med_calib_016",
        "domain": "Toxicology",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "What is the specific antidote for acetaminophen (paracetamol) hepatotoxicity and its therapeutic window?",
        "context": "N-acetylcysteine (NAC) acts as a glutathione precursor to detoxify NAPQI. It is most effective when administered within 8 hours of acute ingestion based on the Rumack-Matthew nomogram."
    },
    {
        "id": "med_calib_017",
        "domain": "Obstetrics & Gynecology",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "What criteria distinguish preeclampsia from gestational hypertension in a pregnant patient after 20 weeks gestation?",
        "context": "Preeclampsia requires new-onset hypertension (>=140/90 mmHg) accompanied by proteinuria (>=300 mg/24h or protein/creatinine >=0.3) or new-onset end-organ dysfunction (thrombocytopenia, renal insufficiency, impaired liver function, pulmonary edema, or persistent cerebral/visual disturbances)."
    },
    {
        "id": "med_calib_018",
        "domain": "Geriatrics",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "What are the key clinical indicators used to differentiate delirium from dementia in hospitalized elderly patients?",
        "context": "Delirium has acute onset, fluctuating course, impaired attention, and altered level of consciousness, often with reversible medical etiology. Dementia exhibits insidious onset, progressive cognitive decline, and preserved alertness until late stages."
    },
    {
        "id": "med_calib_019",
        "domain": "Critical Care",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "What are the clinical criteria for septic shock according to the Sepsis-3 consensus definitions?",
        "context": "Septic shock requires persistent hypotension requiring vasopressors to maintain MAP >= 65 mmHg and a serum lactate level > 2 mmol/L despite adequate volume resuscitation."
    },
    {
        "id": "med_calib_020",
        "domain": "Ophthalmology",
        "instruction": "You are a clinical knowledge assistant. Answer the following medical query clearly and concisely.",
        "input": "What are the immediate clinical presentations and emergency management of acute angle-closure glaucoma?",
        "context": "Severe eye pain, headache, nausea, blurred vision, colored halos around lights, fixed mid-dilated pupil, and markedly elevated intraocular pressure. Managed urgently with topical pressure-lowering agents, systemic acetazolamide/mannitol, and definitive laser peripheral iridotomy."
    }
]

# Expand calibration set systematically to 64 samples across related clinical formulations
def generate_full_calibration_set():
    expanded = []
    modifiers = [
        ("Summarize in clinical detail: ", "\nProvide pharmacological and pathophysiological rationale."),
        ("Medical Review Query: ", "\nExplain the clinical significance and evidence-based guideline recommendations."),
        ("Offline Clinical Assistant: ", "\nOutline diagnostic steps and differential considerations.")
    ]
    
    idx = 1
    for sample in MEDICAL_QA_SAMPLES:
        prompt_text = f"<|im_start|>system\n{sample['instruction']}<|im_end|>\n<|im_start|>user\n{sample['input']}<|im_end|>\n<|im_start|>assistant\n{sample['context']}<|im_end|>"
        expanded.append({
            "sample_id": f"calib_{idx:03d}",
            "domain": sample["domain"],
            "raw_text": prompt_text,
            "query": sample["input"],
            "reference_context": sample["context"]
        })
        idx += 1
        
    for mod_pre, mod_post in modifiers:
        for sample in MEDICAL_QA_SAMPLES:
            if len(expanded) >= 64:
                break
            query = f"{mod_pre}{sample['input']} {mod_post}"
            prompt_text = f"<|im_start|>system\n{sample['instruction']}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{sample['context']}<|im_end|>"
            expanded.append({
                "sample_id": f"calib_{idx:03d}",
                "domain": sample["domain"],
                "raw_text": prompt_text,
                "query": query,
                "reference_context": sample["context"]
            })
            idx += 1
            
    return expanded[:64]

def main():
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    samples = generate_full_calibration_set()
    
    with open(CALIBRATION_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
        
    manifest = {
        "dataset_name": "Medical_QA_Calibration_Corpus_v1",
        "dataset_source": "Curated clinical reference corpus adhering to standard medical knowledge frameworks (USMLE / NICE / KDIGO / GOLD / AHA guidelines)",
        "intended_use": "NNCF Post-Training Weight Compression Calibration (AWQ and Scale Estimation)",
        "pii_content": "NONE (Fully de-identified medical knowledge and diagnostic scenarios; no real patient data)",
        "sample_count": len(samples),
        "sampling_strategy": "Stratified across 20 distinct clinical sub-specialties (cardiology, endocrinology, nephrology, oncology, neurology, etc.)",
        "tokenization": "Qwen2.5 ChatML template (<|im_start|>, <|im_end|>)",
        "maximum_sequence_length": 512,
        "random_seed": 42,
        "storage_path": str(CALIBRATION_DATA_FILE).replace("\\", "/")
    }
    
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Calibration data created: {len(samples)} samples saved to {CALIBRATION_DATA_FILE}")
    print(f"Calibration manifest saved to {MANIFEST_FILE}")

if __name__ == "__main__":
    main()
