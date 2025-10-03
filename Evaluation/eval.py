import pandas as pd
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# ---------- CONFIG ----------
LLM_RESULTS_CSV = "/spaces/25G05/ZeroShot/llama3_zeroshot_results_20251001.csv" 
CLINICIAN_CSV   = "/dataA/madiva/va/student/madiva_va_clinician_COD_20250926.csv"

# Generate output filename with current date and time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_EVAL_CSV = f"/spaces/25G05/ZeroShot/llama3_evaluation_{timestamp}.csv"

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

# Normalize ID column name
llm_df = llm_df.rename(columns={"id": "individual_id"})

# ---------- MERGE ----------
df = pd.merge(llm_df, clin_df, on="individual_id", how="inner")

# ---------- FILTER DATA ----------
# Exclude entries with R99 or missing/invalid clinician ICD-10 codes
initial_count = len(df)
df = df[df["ICD10Code"].notna()]  # Remove NaN values
df = df[df["ICD10Code"].astype(str).str.strip() != ""]  # Remove empty strings
df = df[~df["ICD10Code"].astype(str).str.upper().str.startswith("R99")]  # Remove R99 codes
filtered_count = len(df)
print(f"Filtered out {initial_count - filtered_count} entries (R99 codes or missing ICD-10 codes)")
print(f"Remaining entries for evaluation: {filtered_count}")

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

# ---------- WRITE OUTPUT ----------
out_cols = [
    "individual_id",
    "llm_icd10_raw",               # NEW: original non-cleaned LLM field
    "llm_icd10_code",              # cleaned roots (e.g., J18;L51)
    "llm_english_description",
    "clinician_english_description",
    "clinician_icd10_code",        # cleaned clinician root used for comparison
    "correct",
    "correct_first_letter",        # NEW: less strict matching (first letter only)
]
out_df = df[out_cols].rename(columns={"individual_id": "ID"})
out_df.to_csv(OUTPUT_EVAL_CSV, index=False)

print(f"Wrote: {Path(OUTPUT_EVAL_CSV).resolve()}")
