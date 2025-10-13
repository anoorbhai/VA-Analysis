import pandas as pd
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# ---------- CONFIG ----------
# Filtering options
REMOVE_MISSING_ICD10 = True    # Remove entries with missing/empty clinician ICD-10 codes
REMOVE_R99_CODES = False        # Remove entries with R99 (ill-defined causes) codes
# REMOVE_R99_CODES = True        # Remove entries with R99 (ill-defined causes) codes

LOG_FILE = "/spaces/25G05/CODlist/evaluation_log.txt"
LLM_RESULTS_CSV = "/spaces/25G05/CODlist/llama3_8b_fewshot_COD_results_20251013_113000.csv" 
LOG_FILE = "/spaces/25G05/CODlist/evaluation_log.txt"
# LLM_RESULTS_CSV = "/spaces/25G05/CODlist/llama3_8b_fewshot_COD_results_20251008_203255.csv" 
CLINICIAN_CSV   = "/dataA/madiva/va/student/madiva_va_clinician_COD_20250926.csv"

# Generate output filename with current date and time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_EVAL_CSV = f"/spaces/25G05/CODlist/llama3_8b_fewshot_COD_evaluation_{timestamp}.csv"
# OUTPUT_EVAL_CSV = f"/spaces/25G05/CODlist/llama3_8b_COD_evaluation_{timestamp}.csv"

# ---------- HELPERS ----------
ICD10_PATTERN = re.compile(r"[A-Z][0-9]{2}(?:\.[0-9])?")

def extract_icd10_roots(text: str) -> List[str]:
    """
    Find all ICD-10 codes in a string and return unique 3-char roots (e.g., J18.9 -> J18).
    Handles multiple codes and extra prose.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    matches = ICD10_PATTERN.findall(str(text).upper())
    roots = {m[:3] for m in matches}  # unique 3-char roots
    return sorted(roots)

def icd_root_or_none(text: str) -> Optional[str]:
    """Return the 3-char root of the first ICD-10 code in text, or None if none found."""
    roots = extract_icd10_roots(text)
    return roots[0] if roots else None

# ---------- LOAD ----------
llm_df = pd.read_csv(LLM_RESULTS_CSV)
clin_df = pd.read_csv(CLINICIAN_CSV)

# Load insilicova predictions from main VA dataset
insilicova_CSV = "/dataA/madiva/va/student/madiva_va_dataset_20250924.csv"
insilicova_df = pd.read_csv(insilicova_CSV)

# Select only the insilicova prediction columns we need
insilicova_cols = ['individual_id', 'cause1', 'prob1', 'cause2', 'prob2', 'cause3', 'prob3']
insilicova_df = insilicova_df[insilicova_cols]

# Normalize ID column name
llm_df = llm_df.rename(columns={"id": "individual_id"})

# ---------- MERGE ----------
# First merge LLM results with clinician data
df = pd.merge(llm_df, clin_df, on="individual_id", how="inner")

# Then merge with insilicova predictions
df = pd.merge(df, insilicova_df, on="individual_id", how="left")  # left join to keep all LLM results
print(f"Merged with insilicova predictions. Final dataset has {len(df)} entries.")

# ---------- FILTER DATA ----------
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

# Update filter description for logging
filter_description = []
if REMOVE_MISSING_ICD10:
    filter_description.append("missing ICD-10")
if REMOVE_R99_CODES:
    filter_description.append("R99 codes")
filter_desc = " and ".join(filter_description) if filter_description else "none"

# ---------- COMPUTE FIELDS ----------
# Preserve original LLM ICD-10 text (non-cleaned)
df["llm_icd10_raw"] = df["icd10_code"].fillna("")

# LLM cleaned codes (all roots, joined)
df["llm_icd10_roots"] = df["icd10_code"].apply(extract_icd10_roots)
df["llm_icd10_code"] = df["llm_icd10_roots"].apply(lambda xs: ";".join(xs) if xs else "")

# Clinician root code (cleaned for comparison)
df["clinician_icd10_code"] = df["ICD10Code"].apply(lambda x: icd_root_or_none(x) or "")

# Friendly descriptions
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

# ---------- SUMMARY (optional print) ----------
total = len(df)
correct_n = int(df["correct"].sum())
correct_first_letter_n = int(df["correct_first_letter"].sum())
accuracy = correct_n / total if total else 0.0
accuracy_first_letter = correct_first_letter_n / total if total else 0.0

print(f"Total cases compared: {total}")
print(f"Correct matches (exact root): {correct_n}")
print(f"Accuracy (exact root match): {accuracy:.2%}")
print(f"Correct matches (first letter): {correct_first_letter_n}")
print(f"Accuracy (less strict - first letter): {accuracy_first_letter:.2%}")

# ---------- LOG RESULTS ----------
log_entry = f"""
{'='*60}
Evaluation Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
LLM Results File: {LLM_RESULTS_CSV}
Clinician File: {CLINICIAN_CSV}
Output File: {OUTPUT_EVAL_CSV}

Filtering Settings:
- Remove missing ICD-10: {REMOVE_MISSING_ICD10}
- Remove R99 codes: {REMOVE_R99_CODES}

Initial entries: {initial_count}
Filtered out: {total_filtered} ({filter_desc})
Final entries evaluated: {total}

Results:
- Correct matches (exact root): {correct_n}/{total} = {accuracy:.2%}
- Correct matches (first letter): {correct_first_letter_n}/{total} = {accuracy_first_letter:.2%}

"""

# Append to log file
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(log_entry)

print(f"Results logged to: {LOG_FILE}")

# ---------- WRITE OUTPUT ----------
out_cols = [
    "individual_id",
    "llm_icd10_raw",               # original non-cleaned LLM field
    "llm_icd10_code",              # cleaned roots (e.g., J18;L51)
    "llm_english_description",
    "clinician_english_description",
    "clinician_icd10_code",        # cleaned clinician root used for comparison
    "correct",
    "correct_first_letter",        # less strict matching (first letter only)
    # insilicova predictions
    "cause1",                      # insilicova top prediction
    "prob1",                       # insilicova top prediction probability
    "cause2",                      # insilicova second prediction
    "prob2",                       # insilicova second prediction probability
    "cause3",                      # insilicova third prediction
    "prob3",                       # insilicova third prediction probability
]
out_df = df[out_cols].rename(columns={"individual_id": "ID"})
out_df.to_csv(OUTPUT_EVAL_CSV, index=False)

print(f"Wrote: {Path(OUTPUT_EVAL_CSV).resolve()}")
