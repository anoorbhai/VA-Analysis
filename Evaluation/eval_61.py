#!/usr/bin/env python3
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# ---------- CONFIG ----------
# Inputs
LLM_RESULTS_CSV = "/spaces/25G05/FewShot/llama3_8b_fewshot_61_results_20251015_122022.csv"
CLINICIAN_CSV   = "/dataA/madiva/va/student/madiva_va_clinician_COD_20250926.csv"
# Mapping from clinician assignment -> your 61-code scheme
MAPPING_CSV     = "/spaces/25G05/61COD/clinician_to_scheme_mapping.csv"

LOG_FILE = "/spaces/25G05/61COD/evaluation_log.txt"

# Output (auto-dated)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_EVAL_CSV = f"/spaces/25G05/61COD/llama3_8b_few_61_evaluation_{timestamp}.csv"

# Filtering toggles
REMOVE_MISSING_SCHEME = True     # drop if LLM CODE is missing
REMOVE_UNKNOWN_CODE99 = True    # drop if LLM CODE == 99.00 (unknown)

# ---------- PATTERNS & HELPERS ----------
SCHEME_CODE_RE = re.compile(r"^\d{2}\.\d{2}$")  # e.g., 04.02
ICD10_ROOT_RE  = re.compile(r"[A-Z][0-9]{2}")

def norm(s: Optional[str]) -> str:
    """Normalize strings for strict-but-sane equality (casefold + collapses spaces)."""
    if s is None:
        return ""
    s = str(s)
    s = s.strip().casefold()
    s = re.sub(r"\s+", " ", s)
    return s

def scheme_chapter(code: str) -> str:
    """Return the 2-digit chapter (e.g., '04' from '04.02') or ''."""
    if not isinstance(code, str):
        return ""
    m = SCHEME_CODE_RE.match(code.strip())
    if not m:
        return ""
    return code[:2]

def icd10_root(text: Optional[str]) -> Optional[str]:
    """Extract the first ICD-10 root (e.g., 'J18') if present."""
    if not isinstance(text, str):
        return None
    m = ICD10_ROOT_RE.search(text.upper())
    return m.group(0) if m else None

# ---------- LOAD ----------
llm_df = pd.read_csv(LLM_RESULTS_CSV)

llm_df = pd.read_csv(LLM_RESULTS_CSV)

# Rename columns to match expected format
llm_df = llm_df.rename(columns={
    "cause_of_death": "CAUSE_SHORT",   # English description of COD
    "code": "CODE",                    # The numeric 61-code
    "confidence": "CONFIDENCE"         # Confidence percentage (if present)
})

clin_df = pd.read_csv(CLINICIAN_CSV)
map_df  = pd.read_csv(MAPPING_CSV)

# Expected LLM columns (rename to a canonical schema if needed)
# Expect: individual_id, CODE, CAUSE_SHORT, CONFIDENCE
# (rename if your LLM CSV used different headers)
llm_df = llm_df.rename(columns={
    "id": "individual_id",
    "code": "CODE",
    "cause_short": "CAUSE_SHORT",
    "confidence": "CONFIDENCE"
})

# Sanity: keep only needed columns & coerce to str
llm_df["individual_id"] = llm_df["individual_id"].astype(str)
llm_df["CODE"]          = llm_df["CODE"].astype(str)
llm_df["CAUSE_SHORT"]   = llm_df["CAUSE_SHORT"].astype(str)
llm_df["CONFIDENCE"]    = llm_df["CONFIDENCE"].astype(str)

# Clinician columns: keep these if present
# Expect: individual_id, ICD10Code, CauseofDeath
clin_df["individual_id"] = clin_df["individual_id"].astype(str)
for col in ["ICD10Code", "CauseofDeath"]:
    if col not in clin_df.columns:
        clin_df[col] = ""

# ---------- MAPPING PREP ----------
# We support two mapping routes to your 61-code scheme:
# 1) By ICD-10 root  -> scheme code/desc
# 2) By clinician English description -> scheme code/desc
# The mapping file should provide at least one of these keys per row.

# Expected columns in mapping CSV (flexible, but these are recommended):
#   icd10_root, clinician_desc, scheme_code, scheme_cause
for req in ["scheme_code", "scheme_cause"]:
    if req not in map_df.columns:
        raise ValueError(f"Mapping CSV must include column: {req}")

# Normalize mapping keys for robust joins
if "icd10_root" not in map_df.columns:
    map_df["icd10_root"] = ""
if "clinician_desc" not in map_df.columns:
    map_df["clinician_desc"] = ""

map_df["icd10_root_norm"]   = map_df["icd10_root"].astype(str).str.upper().str.strip()
map_df["clin_desc_norm"]    = map_df["clinician_desc"].apply(norm)
map_df["scheme_code"]       = map_df["scheme_code"].astype(str).str.strip()
map_df["scheme_cause"]      = map_df["scheme_cause"].astype(str).str.strip()

# ---------- MAP CLINICIAN -> SCHEME ----------
# Strategy:
# A) Try ICD-10 root join
# B) For rows not yet mapped, try description join

clin_df["icd10_root"]        = clin_df["ICD10Code"].apply(icd10_root).fillna("")
clin_df["icd10_root_norm"]   = clin_df["icd10_root"].str.upper().str.strip()
clin_df["clin_desc_norm"]    = clin_df["CauseofDeath"].apply(norm)

# A) ICD-10 root merge
merged_a = pd.merge(
    clin_df,
    map_df[["icd10_root_norm", "scheme_code", "scheme_cause"]],
    on="icd10_root_norm",
    how="left",
)

# Identify unmapped
unmapped_mask = merged_a["scheme_code"].isna()

# B) Description merge for unmapped
fallback = pd.merge(
    merged_a.loc[unmapped_mask, ["individual_id", "ICD10Code", "CauseofDeath", "icd10_root", "icd10_root_norm", "clin_desc_norm"]],
    map_df[["clin_desc_norm", "scheme_code", "scheme_cause"]],
    on="clin_desc_norm",
    how="left",
)

# Stitch back
merged_a.loc[unmapped_mask, ["scheme_code", "scheme_cause"]] = fallback[["scheme_code", "scheme_cause"]].values

# Final clinician→scheme mapped DF
clin_mapped_df = merged_a.rename(columns={
    "scheme_code": "clin_scheme_code",
    "scheme_cause": "clin_scheme_cause"
})

# ---------- MERGE LLM + CLINICIAN (scheme) ----------
df = pd.merge(llm_df, clin_mapped_df, on="individual_id", how="inner")

print(f"Merged LLM results with clinician data. Entries: {len(df)}")

# ---------- FILTER ----------
initial_count = len(df)
reasons = []

if REMOVE_MISSING_SCHEME:
    before = len(df)
    df = df[df["CODE"].astype(str).str.strip() != ""]
    removed = before - len(df)
    if removed:
        reasons.append(f"{removed} missing LLM CODE")

if REMOVE_UNKNOWN_CODE99:
    before = len(df)
    df = df[df["CODE"].astype(str).str.strip() != "99.00"]
    removed = before - len(df)
    if removed:
        reasons.append(f"{removed} unknown (99.00)")

filtered_count = len(df)
print(f"Filtered: {initial_count - filtered_count} ({'; '.join(reasons) if reasons else 'none'})")
print(f"Remaining entries for evaluation: {filtered_count}")

# ---------- COMPUTE FIELDS ----------
# Normalize strings for comparisons
df["llm_code"]         = df["CODE"].astype(str).str.strip()
df["llm_cause"]        = df["CAUSE_SHORT"].astype(str).str.strip()
df["llm_confidence"]   = df["CONFIDENCE"].astype(str).str.strip()

df["llm_chapter"]      = df["llm_code"].apply(scheme_chapter)
df["clin_code"]        = df["clin_scheme_code"].fillna("").astype(str).str.strip()
df["clin_cause"]       = df["clin_scheme_cause"].fillna("").astype(str).str.strip()
df["clin_chapter"]     = df["clin_code"].apply(scheme_chapter)

# Evaluation rules
def exact_code_match(row) -> bool:
    return bool(row["llm_code"]) and (row["llm_code"] == row["clin_code"])

def chapter_match(row) -> bool:
    return bool(row["llm_chapter"]) and (row["llm_chapter"] == row["clin_chapter"])

def desc_exact_match(row) -> bool:
    return norm(row["llm_cause"]) == norm(row["clin_cause"]) and row["llm_cause"] != ""

df["correct_exact"]   = df.apply(exact_code_match, axis=1)
df["correct_chapter"] = df.apply(chapter_match, axis=1)
df["desc_match"]      = df.apply(desc_exact_match, axis=1)

# ---------- SUMMARY ----------
total = len(df)
acc_exact   = (df["correct_exact"].sum()   / total) if total else 0.0
acc_chapter = (df["correct_chapter"].sum() / total) if total else 0.0
acc_desc    = (df["desc_match"].sum()      / total) if total else 0.0

print(f"Total cases compared: {total}")
print(f"Accuracy (exact code): {acc_exact:.2%}")
print(f"Accuracy (chapter):    {acc_chapter:.2%}")
print(f"Accuracy (desc eq.):   {acc_desc:.2%}")

# ---------- LOG ----------
filter_desc = []
if REMOVE_MISSING_SCHEME: filter_desc.append("missing LLM CODE")
if REMOVE_UNKNOWN_CODE99: filter_desc.append("99.00")
filter_desc = " and ".join(filter_desc) if filter_desc else "none"

log_entry = f"""
{'='*60}
Evaluation Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
LLM Results File: {LLM_RESULTS_CSV}
Clinician File: {CLINICIAN_CSV}
Mapping File: {MAPPING_CSV}
Output File: {OUTPUT_EVAL_CSV}

Filtering Settings:
- Remove missing LLM CODE: {REMOVE_MISSING_SCHEME}
- Remove 99.00 (unknown):   {REMOVE_UNKNOWN_CODE99}

Initial entries: {initial_count}
Filtered out: {initial_count - filtered_count} ({filter_desc})
Final entries evaluated: {total}

Results:
- Accuracy (exact code): {acc_exact:.2%}
- Accuracy (chapter):    {acc_chapter:.2%}
- Accuracy (desc eq.):   {acc_desc:.2%}

"""

with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write(log_entry)

print(f"Results logged to: {LOG_FILE}")

# ---------- WRITE OUTPUT ----------
out_cols = [
    "individual_id",
    # LLM
    "llm_code",
    "llm_cause",
    "llm_confidence",
    "llm_chapter",
    # Clinician mapped -> scheme
    "clin_code",
    "clin_cause",
    "clin_chapter",
    # Original clinician fields (for audit)
    "ICD10Code",
    "CauseofDeath",
    "icd10_root",
    # Matches
    "correct_exact",
    "correct_chapter",
    "desc_match",
]

out_df = df[out_cols].rename(columns={"individual_id": "ID"})
out_df.to_csv(OUTPUT_EVAL_CSV, index=False)
print(f"Wrote: {Path(OUTPUT_EVAL_CSV).resolve()}")
