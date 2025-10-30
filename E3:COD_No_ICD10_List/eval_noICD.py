import pandas as pd
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

# Filtering 
REMOVE_MISSING_ICD10 = True    # Remove entries with missing/empty clinician ICD-10 codes
REMOVE_R99_CODES = True       # Remove entries with R99 (ill-defined causes) codes

LOG_FILE = "/spaces/25G05/Aaliyah/evaluation_log.txt"
LLM_RESULTS_CSV = "/spaces/25G05/Rizwaanah/ZeroShot/llama3_8b_zeroshot_COD_no_results_20251030_133204.csv" 
CLINICIAN_CSV   = "/dataA/madiva/va/student/madiva_va_clinician_COD_20250926.csv"
CLINICAL_COD_PAIRS_FILE = "/spaces/25G05/Aaliyah/Clinical_COD_pairs.txt"

# Generate output filename with current date and time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_EVAL_CSV = f"/spaces/25G05/Rizwaanah/Evaluation/llama3_8b_zeroshot_COD_evaluation_noICD_{timestamp}.csv"

ICD10_PATTERN = re.compile(r"[A-Z][0-9]{2}(?:\.[0-9])?")

def load_clinical_cod_pairs(file_path: str) -> Dict[int, Dict[str, str]]:
    # Load the clinical COD pairs from the text file.
    cod_pairs = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # each line that starts with a number
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('Use the following'):
            continue
            
        # Match pattern
        match = re.match(r'^(\d+)\.\s*([A-Z]\d{2}):\s*(.+)$', line)
        if match:
            number = int(match.group(1))
            icd10_code = match.group(2)
            description = match.group(3).strip()
            
            cod_pairs[number] = {
                'icd10_code': icd10_code,
                'description': description
            }
    
    return cod_pairs

def extract_numbers_from_text(text: str) -> List[int]:

    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    
    # Find all numbers in the text
    numbers = re.findall(r'\b(\d+)\b', str(text))
    return [int(n) for n in numbers if 1 <= int(n) <= 233]  

def extract_icd10_roots(text: str) -> List[str]:
 
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    matches = ICD10_PATTERN.findall(str(text).upper())
    roots = {m[:3] for m in matches}  # unique 3-char roots
    return sorted(roots)

def icd_root_or_none(text: str) -> Optional[str]:
    roots = extract_icd10_roots(text)
    return roots[0] if roots else None

# Load clinical COD pairs mapping
print("Loading clinical COD pairs...")
cod_pairs = load_clinical_cod_pairs(CLINICAL_COD_PAIRS_FILE)
print(f"Loaded {len(cod_pairs)} clinical COD pairs")

# Load datasets
llm_df = pd.read_csv(LLM_RESULTS_CSV)
clin_df = pd.read_csv(CLINICIAN_CSV)

# Load insilicoVA predictions from main VA dataset
insilicova_CSV = "/dataA/madiva/va/student/madiva_va_dataset_20250924.csv"
insilicova_df = pd.read_csv(insilicova_CSV)

insilicova_cols = ['individual_id', 'cause1', 'prob1', 'cause2', 'prob2', 'cause3', 'prob3']
insilicova_df = insilicova_df[insilicova_cols]

# Normalize ID column name
llm_df = llm_df.rename(columns={"id": "individual_id"})

# First merge LLM results with clinician data
df = pd.merge(llm_df, clin_df, on="individual_id", how="inner")

# Then merge with insilicova predictions
df = pd.merge(df, insilicova_df, on="individual_id", how="left")  
print(f"Merged with insilicova predictions. Final dataset has {len(df)} entries.")

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

# Use the direct number column from LLM output
df["llm_number"] = df["number"].fillna(0).astype(int)  
df["llm_number"] = df["llm_number"].apply(lambda x: x if x > 0 else None)  

# LLM cause of death description
df["llm_description"] = df["cause_of_death"].fillna("")

# Map LLM numbers to ICD-10 codes and descriptions using the COD pairs
def map_number_to_icd10(number):
    if number and number in cod_pairs:
        return cod_pairs[number]['icd10_code']
    return ""

def map_number_to_description(number):
    if number and number in cod_pairs:
        return cod_pairs[number]['description']
    return ""

df["llm_mapped_icd10"] = df["llm_number"].apply(map_number_to_icd10)
df["llm_mapped_description"] = df["llm_number"].apply(map_number_to_description)

df["clinician_icd10_code"] = df["ICD10Code"].apply(lambda x: icd_root_or_none(x) or "")
df["clinician_english_description"] = df["CauseofDeath"].fillna("")

# Find the correct number and description for each clinician ICD-10 code
def find_correct_number_for_icd10(icd10_code):
    if not icd10_code:
        return None
    for num, data in cod_pairs.items():
        if data['icd10_code'] == icd10_code:
            return num
    return None

def find_correct_description_for_icd10(icd10_code):
    if not icd10_code:
        return ""
    for num, data in cod_pairs.items():
        if data['icd10_code'] == icd10_code:
            return data['description']
    return ""

df["correct_number"] = df["clinician_icd10_code"].apply(find_correct_number_for_icd10)
df["correct_description"] = df["clinician_icd10_code"].apply(find_correct_description_for_icd10)

# Does LLM number match the correct number for clinician ICD-10?
def is_number_correct(row):
    return (row["llm_number"] is not None and 
            row["correct_number"] is not None and 
            row["llm_number"] == row["correct_number"])

# Does LLM description match the correct description from clinician data?
def is_description_correct(row):
    if not row["correct_description"]: 
        return False
    
    expected_desc = str(row["correct_description"]).lower().strip()
    llm_desc = str(row["llm_description"]).lower().strip()
    
    # string matching 
    return expected_desc in llm_desc or llm_desc in expected_desc

# Both number and description are correct
def is_both_correct(row):
    return row["number_correct"] and row["description_correct"]

# evaluation functions
df["number_correct"] = df.apply(is_number_correct, axis=1)
df["description_correct"] = df.apply(is_description_correct, axis=1) 
df["both_correct"] = df.apply(is_both_correct, axis=1)

df["correct"] = df["number_correct"] 
df["valid_number"] = df["llm_number"].apply(lambda x: x is not None and x in cod_pairs)

def is_icd10_correct(row):
    clin = row["clinician_icd10_code"]
    llm_mapped = row["llm_mapped_icd10"]
    return bool(clin) and bool(llm_mapped) and (clin == llm_mapped)

def is_icd10_first_letter_correct(row):
    clin = row["clinician_icd10_code"]
    llm_mapped = row["llm_mapped_icd10"]
    if not clin or not llm_mapped:
        return False
    return clin[0] == llm_mapped[0]

df["icd10_correct"] = df.apply(is_icd10_correct, axis=1)
df["icd10_first_letter_correct"] = df.apply(is_icd10_first_letter_correct, axis=1)

total = len(df)
valid_numbers = int(df["valid_number"].sum())

number_correct_n = int(df["number_correct"].sum())
description_correct_n = int(df["description_correct"].sum()) 
both_correct_n = int(df["both_correct"].sum())
icd10_correct_n = int(df["icd10_correct"].sum())
icd10_first_letter_correct_n = int(df["icd10_first_letter_correct"].sum())

number_accuracy = number_correct_n / total if total else 0.0
description_accuracy = description_correct_n / total if total else 0.0
both_accuracy = both_correct_n / total if total else 0.0
icd10_accuracy = icd10_correct_n / total if total else 0.0
icd10_first_letter_accuracy = icd10_first_letter_correct_n / total if total else 0.0
valid_number_rate = valid_numbers / total if total else 0.0

number_accuracy_among_valid = number_correct_n / valid_numbers if valid_numbers else 0.0
description_accuracy_among_valid = description_correct_n / valid_numbers if valid_numbers else 0.0
both_accuracy_among_valid = both_correct_n / valid_numbers if valid_numbers else 0.0

print(f"Total cases compared: {total}")
print(f"\n=== FINAL RESULTS ===")
print(f"Number correct: {number_correct_n}/{total} = {number_accuracy:.2%}")
print(f"Description correct: {description_correct_n}/{total} = {description_accuracy:.2%}")
print(f"Both correct: {both_correct_n}/{total} = {both_accuracy:.2%}")

#log results
log_entry = f"""
{'='*60}
Evaluation Run (Number Mapping): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
LLM Results File: {LLM_RESULTS_CSV}
Clinician File: {CLINICIAN_CSV}
Clinical COD Pairs File: {CLINICAL_COD_PAIRS_FILE}
Output File: {OUTPUT_EVAL_CSV}

Filtering Settings:
- Remove missing ICD-10: {REMOVE_MISSING_ICD10}
- Remove R99 codes: {REMOVE_R99_CODES}

Initial entries: {initial_count}
Filtered out: {total_filtered} ({filter_desc})
Final entries evaluated: {total}

Results:
- Number correct: {number_correct_n}/{total} = {number_accuracy:.2%}
- Description correct: {description_correct_n}/{total} = {description_accuracy:.2%}
- Both correct: {both_correct_n}/{total} = {both_accuracy:.2%}

"""

with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(log_entry)

print(f"Results logged to: {LOG_FILE}")

# output CSV
out_cols = [
    "individual_id",
    "llm_number",                      
    "llm_description",                 
    "correct_number",                 
    "correct_description",            
    "number_correct",                 
    "description_correct",            
]
out_df = df[out_cols].rename(columns={"individual_id": "id", "llm_description": "llm_english_description", "description_correct": "english_correct"})
out_df.to_csv(OUTPUT_EVAL_CSV, index=False)

print(f"Wrote: {Path(OUTPUT_EVAL_CSV).resolve()}")
