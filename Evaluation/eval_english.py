import pandas as pd
import re
import requests
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Filtering 
REMOVE_MISSING_ICD10 = True    # Remove entries with missing/empty clinician ICD-10 codes
REMOVE_R99_CODES = True        # Remove entries with R99 (ill-defined causes) codes

ENABLE_SEMANTIC_COMPARISON = True  # Enable/disable semantic comparison
OLLAMA_API_URL = "http://localhost:11434/api/generate"
COMPARISON_MODEL = "llama3:latest"  # Model for semantic comparison

LOG_FILE = "/spaces/25G05/CODlist/evaluation_english_log.txt"
LLM_RESULTS_CSV = "/spaces/25G05/CODlist/llama3_zeroshot_COD_results_20251008_203255.csv" 
CLINICIAN_CSV   = "/dataA/madiva/va/student/madiva_va_clinician_COD_20250926.csv"

# Generate output filename with current date and time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_EVAL_CSV = f"/spaces/25G05/CODlist/llama3_8b_COD_evaluation_english_{timestamp}.csv"

ICD10_PATTERN = re.compile(r"[A-Z][0-9]{2}(?:\.[0-9])?")

def extract_icd10_roots(text: str) -> List[str]:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    matches = ICD10_PATTERN.findall(str(text).upper())
    roots = {m[:3] for m in matches}  
    return sorted(roots)

def icd_root_or_none(text: str) -> Optional[str]:
    # Return the 3-char root of the first ICD-10 code in text, or None if none found.
    roots = extract_icd10_roots(text)
    return roots[0] if roots else None

def semantic_comparison(llm_description: str, clinician_description: str) -> bool:
    # Use another LLM to compare semantic similarity between LLM and clinician descriptions.
    if not llm_description.strip() or not clinician_description.strip():
        return False
    
    prompt = f"""You are a medical expert. Compare these two cause of death descriptions and determine if they represent the same general cause or very similar, even if worded differently.

LLM Diagnosis: "{llm_description.strip()}"
Clinician Diagnosis: "{clinician_description.strip()}"

Respond with ONLY "YES" if they represent the same/similar condition, or "NO" if they are different conditions."""

    try:
        payload = {
            "model": COMPARISON_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent responses
                "num_ctx": 2048
            }
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        answer = result.get("response", "").strip().upper()
        
        # Return True if response contains "YES"
        return "YES" in answer
        
    except Exception as e:
        print(f"Error in semantic comparison: {e}")
        return False

llm_df = pd.read_csv(LLM_RESULTS_CSV)
clin_df = pd.read_csv(CLINICIAN_CSV)

# Load InterVA predictions from main VA dataset
INTERVA_CSV = "/dataA/madiva/va/student/madiva_va_dataset_20250924.csv"
interva_df = pd.read_csv(INTERVA_CSV)

# Select only the InterVA prediction columns we need
interva_cols = ['individual_id', 'cause1', 'prob1', 'cause2', 'prob2', 'cause3', 'prob3']
interva_df = interva_df[interva_cols]

# Normalize ID column name
llm_df = llm_df.rename(columns={"id": "individual_id"})

# First merge LLM results with clinician data
df = pd.merge(llm_df, clin_df, on="individual_id", how="inner")

# Then merge with InterVA predictions
df = pd.merge(df, interva_df, on="individual_id", how="left")  
print(f"Merged with InterVA predictions. Final dataset has {len(df)} entries.")

initial_count = len(df)
filter_reasons = []

if REMOVE_MISSING_ICD10:
    before_missing = len(df)
    df = df[df["ICD10Code"].notna()]  # Remove NaN values
    df = df[df["ICD10Code"].astype(str).str.strip() != ""]  # Remove empty strings
    missing_filtered = before_missing - len(df)
    if missing_filtered > 0:
        filter_reasons.append(f"{missing_filtered} missing ICD-10 codes")

if REMOVE_R99_CODES:
    before_r99 = len(df)
    df = df[~df["ICD10Code"].astype(str).str.upper().str.startswith("R99")]  # Remove R99 codes
    r99_filtered = before_r99 - len(df)
    if r99_filtered > 0:
        filter_reasons.append(f"{r99_filtered} R99 codes")

filtered_count = len(df)
total_filtered = initial_count - filtered_count

if total_filtered > 0:
    print(f"Filtered out {total_filtered} entries ({', '.join(filter_reasons)})")
else:
    print("No entries filtered")
print(f"Remaining entries for evaluation: {filtered_count}")

filter_description = []
if REMOVE_MISSING_ICD10:
    filter_description.append("missing ICD-10")
if REMOVE_R99_CODES:
    filter_description.append("R99 codes")
filter_desc = " and ".join(filter_description) if filter_description else "none"

df["llm_icd10_raw"] = df["icd10_code"].fillna("")

# LLM cleaned codes (all roots, joined)
df["llm_icd10_roots"] = df["icd10_code"].apply(extract_icd10_roots)
df["llm_icd10_code"] = df["llm_icd10_roots"].apply(lambda xs: ";".join(xs) if xs else "")

# Clinician root code (cleaned for comparison)
df["clinician_icd10_code"] = df["ICD10Code"].apply(lambda x: icd_root_or_none(x) or "")

df["llm_english_description"] = df["cause_of_death"].fillna("")
df["clinician_english_description"] = df["CauseofDeath"].fillna("")

# Correct if clinician root is among LLM roots (lenient match)
def is_correct(row) -> bool:
    clin = row["clinician_icd10_code"]
    preds = set(row["llm_icd10_roots"])
    return bool(clin) and (clin in preds)

# Correct if first letter matches (less strict)
def is_correct_first_letter(row) -> bool:
    clin = row["clinician_icd10_code"]
    preds = row["llm_icd10_roots"]
    if not clin or not preds:
        return False
    clin_first_letter = clin[0] if clin else ""
    pred_first_letters = {pred[0] for pred in preds if pred}
    return clin_first_letter in pred_first_letters

df["correct"] = df.apply(is_correct, axis=1)
df["correct_first_letter"] = df.apply(is_correct_first_letter, axis=1)

# Semantic comparison
if ENABLE_SEMANTIC_COMPARISON:
    print("Performing semantic comparison of English descriptions...")
    print(f"This may take a while for {len(df)} entries...")
    
    semantic_results = []
    for idx, row in df.iterrows():
        if idx % 10 == 0:  
            print(f"Processed {idx}/{len(df)} entries...")
        
        result = semantic_comparison(
            row["llm_english_description"], 
            row["clinician_english_description"]
        )
        semantic_results.append(result)
    
    df["correct_semantic"] = semantic_results
    print("Semantic comparison completed.")
else:
    df["correct_semantic"] = False
    print("Semantic comparison disabled.")

total = len(df)
correct_n = int(df["correct"].sum())
correct_first_letter_n = int(df["correct_first_letter"].sum())
correct_semantic_n = int(df["correct_semantic"].sum()) if ENABLE_SEMANTIC_COMPARISON else 0

accuracy = correct_n / total if total else 0.0
accuracy_first_letter = correct_first_letter_n / total if total else 0.0
accuracy_semantic = correct_semantic_n / total if total else 0.0

print(f"\nEVALUATION RESULTS:")
print(f"==================")
print(f"Total cases compared: {total}")
print(f"Correct matches (exact ICD-10 root): {correct_n} ({accuracy:.2%})")
print(f"Correct matches (first letter): {correct_first_letter_n} ({accuracy_first_letter:.2%})")
if ENABLE_SEMANTIC_COMPARISON:
    print(f"Correct matches (semantic similarity): {correct_semantic_n} ({accuracy_semantic:.2%})")

# log results
log_entry = f"""
{'='*60}
English Evaluation Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
LLM Results File: {LLM_RESULTS_CSV}
Clinician File: {CLINICIAN_CSV}
Output File: {OUTPUT_EVAL_CSV}

Filtering Settings:
- Remove missing ICD-10: {REMOVE_MISSING_ICD10}
- Remove R99 codes: {REMOVE_R99_CODES}

Semantic Comparison:
- Enabled: {ENABLE_SEMANTIC_COMPARISON}
- Model: {COMPARISON_MODEL if ENABLE_SEMANTIC_COMPARISON else 'N/A'}

Initial entries: {initial_count}
Filtered out: {total_filtered} ({filter_desc})
Final entries evaluated: {total}

Results:
- Correct matches (exact ICD-10 root): {correct_n}/{total} = {accuracy:.2%}
- Correct matches (first letter): {correct_first_letter_n}/{total} = {accuracy_first_letter:.2%}
- Correct matches (semantic similarity): {correct_semantic_n}/{total} = {accuracy_semantic:.2%}

"""

# Append to log file
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(log_entry)

print(f"\nResults logged to: {LOG_FILE}")

# output CSV
out_cols = [
    "individual_id",
    "llm_icd10_raw",               # original 
    "llm_icd10_code",              # cleaned 
    "llm_english_description",
    "clinician_english_description",
    "clinician_icd10_code",        # cleaned clinician root 
    "correct",                     # exact ICD-10 match
    "correct_first_letter",        # first letter match
    "correct_semantic",            # semantic similarity match
    # InterVA predictions
    "cause1",                  
    "prob1",                      
    "cause2",                      
    "prob2",                       
    "cause3",                      
    "prob3",                       
]
out_df = df[out_cols].rename(columns={"individual_id": "ID"})
out_df.to_csv(OUTPUT_EVAL_CSV, index=False)

print(f"\nOutput saved to: {Path(OUTPUT_EVAL_CSV).resolve()}")
print("\nEvaluation complete!")