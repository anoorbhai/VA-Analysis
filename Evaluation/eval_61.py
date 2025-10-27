#!/usr/bin/env python3
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

WS_RE = re.compile(r"\s+")
NON_PRINT_RE = re.compile(r"[\u2000-\u200b\u202f\u00a0]") 
SCHEME_CODE_RE = re.compile(r"^\d{2}\.\d{2}$")            
ICD10_ROOT_RE  = re.compile(r"[A-Z][0-9]{2}")

def clean_text(s):
    """remove invisible spaces, collapse whitespace, drop surrounding quotes."""
    if s is None:
        return ""
    s = str(s)
    s = NON_PRINT_RE.sub("", s)
    s = s.strip()
    s = WS_RE.sub(" ", s)
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s

def normalize_scheme_code(x):
    """
    Normalize scheme codes to NN.NN
    """
    s = clean_text(x)
    m = re.search(r"(\d{1,2})\.(\d{1,2})", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return f"{a:02d}.{b:02d}"
    # 3-4 digits
    digits = re.sub(r"\D", "", s)
    if len(digits) == 4:
        a, b = int(digits[:2]), int(digits[2:])
        return f"{a:02d}.{b:02d}"
    if len(digits) == 3:
        a, b = int(digits[0]), int(digits[1:])
        return f"{a:02d}.{b:02d}"
    return s

def scheme_chapter_from_norm(code_norm: str) -> str:
    """Given a normalized NN.NN, return NN (chapter) or ''."""
    return code_norm[:2] if SCHEME_CODE_RE.match(code_norm) else ""

def icd10_root(text: Optional[str]) -> Optional[str]:
    """Extract first ICD-10 root (e.g., 'J18') if present."""
    if not isinstance(text, str):
        return None
    m = ICD10_ROOT_RE.search(text.upper())
    return m.group(0) if m else None

def norm_str_for_eq(s: Optional[str]) -> str:
    """Lowercase, strip, collapse spaces for exact text equality."""
    if s is None:
        return ""
    s = clean_text(s)
    s = s.casefold()
    s = re.sub(r"\s+", " ", s)
    return s

LLM_RESULTS_CSV = "/spaces/25G05/Aaliyah/FewShot/llama3_70b_fewshot_61_results_20251017_151059.csv"
CLINICIAN_CSV   = "/dataA/madiva/va/student/madiva_va_clinician_COD_20250926.csv"
MAPPING_CSV     = "/spaces/25G05/61_codes.csv"

LOG_FILE = "/spaces/25G05/Rizwaanah/evaluation_log.txt"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_EVAL_CSV = f"/spaces/25G05/Rizwaanah/Evaluation/llama3_70b_few_61_evaluation_{timestamp}.csv"

REMOVE_MISSING_SCHEME = True
REMOVE_UNKNOWN_CODE99 = True

llm_df = pd.read_csv(LLM_RESULTS_CSV)
clin_df = pd.read_csv(CLINICIAN_CSV)
map_df  = pd.read_csv(MAPPING_CSV)

# Accept common variants
llm_df = llm_df.rename(columns={
    "id": "individual_id",
    "ID": "individual_id",
    "code": "CODE",
    "Code": "CODE",
    "cause_short": "CAUSE_SHORT",
    "cause_of_death": "CAUSE_SHORT",
    "confidence": "CONFIDENCE",
    "Confidence": "CONFIDENCE",
})

for c in ["individual_id", "CODE", "CAUSE_SHORT", "CONFIDENCE"]:
    if c in llm_df.columns:
        llm_df[c] = llm_df[c].astype(str)
    else:
        # create empty if truly missing
        llm_df[c] = ""

for req in ["scheme_code", "scheme_cause"]:
    if req not in map_df.columns:
        raise ValueError(f"Mapping CSV must include column: {req}")

if "icd10_root" not in map_df.columns:
    map_df["icd10_root"] = ""
if "clinician_desc" not in map_df.columns:
    map_df["clinician_desc"] = ""

map_df["icd10_root_norm"] = map_df["icd10_root"].astype(str).str.upper().str.strip()
map_df["clin_desc_norm"]  = map_df["clinician_desc"].apply(norm_str_for_eq)
map_df["scheme_code"]     = map_df["scheme_code"].astype(str).str.strip()
map_df["scheme_cause"]    = map_df["scheme_cause"].astype(str).str.strip()

clin_df["individual_id"] = clin_df["individual_id"].astype(str)
for col in ["ICD10Code", "CauseofDeath"]:
    if col not in clin_df.columns:
        clin_df[col] = ""

clin_df["icd10_root"]      = clin_df["ICD10Code"].apply(icd10_root).fillna("")
clin_df["icd10_root_norm"] = clin_df["icd10_root"].str.upper().str.strip()
clin_df["clin_desc_norm"]  = clin_df["CauseofDeath"].apply(norm_str_for_eq)

# ICD-10 root merge
merged_a = pd.merge(
    clin_df,
    map_df[["icd10_root_norm", "scheme_code", "scheme_cause"]],
    on="icd10_root_norm",
    how="left",
)

# Fallback: clinician description merge where not mapped yet
unmapped_mask = merged_a["scheme_code"].isna()
fallback = pd.merge(
    merged_a.loc[unmapped_mask, ["individual_id", "ICD10Code", "CauseofDeath",
                                 "icd10_root", "icd10_root_norm", "clin_desc_norm"]],
    map_df[["clin_desc_norm", "scheme_code", "scheme_cause"]],
    on="clin_desc_norm",
    how="left",
)

merged_a.loc[unmapped_mask, ["scheme_code", "scheme_cause"]] = \
    fallback[["scheme_code", "scheme_cause"]].values

clin_mapped_df = merged_a.rename(columns={
    "scheme_code": "clin_scheme_code",
    "scheme_cause": "clin_scheme_cause"
})

df = pd.merge(llm_df, clin_mapped_df, on="individual_id", how="inner")
print(f"Merged LLM results with clinician data. Entries: {len(df)}")

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

# Raw
df["llm_code_raw"]   = df["CODE"].astype(str)
df["clin_code_raw"]  = df["clin_scheme_code"].fillna("").astype(str)
df["llm_cause_raw"]  = df.get("CAUSE_SHORT", "").astype(str)
df["clin_cause_raw"] = df.get("clin_scheme_cause", "").astype(str)

# Normalized codes
df["llm_code_norm"]  = df["llm_code_raw"].apply(normalize_scheme_code)
df["clin_code_norm"] = df["clin_code_raw"].apply(normalize_scheme_code)

df["llm_chapter"]  = df["llm_code_norm"].apply(scheme_chapter_from_norm)
df["clin_chapter"] = df["clin_code_norm"].apply(scheme_chapter_from_norm)

# Normalized causes (text)
df["llm_cause_norm"]  = df["llm_cause_raw"].apply(norm_str_for_eq)
df["clin_cause_norm"] = df["clin_cause_raw"].apply(norm_str_for_eq)

# Metrics
df["correct_exact"]   = (df["llm_code_norm"] == df["clin_code_norm"]) & (df["llm_code_norm"] != "")
df["correct_chapter"] = (df["llm_chapter"]   == df["clin_chapter"])   & (df["llm_chapter"]   != "")
df["desc_match"]      = (df["llm_cause_norm"] == df["clin_cause_norm"]) & (df["llm_cause_norm"] != "")

total = len(df)
acc_exact   = (df["correct_exact"].sum()   / total) if total else 0.0
acc_chapter = (df["correct_chapter"].sum() / total) if total else 0.0
acc_desc    = (df["desc_match"].sum()      / total) if total else 0.0

print(f"Total cases compared: {total}")
print(f"Accuracy (exact code): {acc_exact:.2%}")
print(f"Accuracy (chapter):    {acc_chapter:.2%}")
print(f"Accuracy (desc eq.):   {acc_desc:.2%}")

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

# Output columns 
out_cols = [
    "individual_id",
    "llm_code_raw","llm_code_norm","llm_chapter",
    "llm_cause_raw",
    "clin_code_raw","clin_code_norm","clin_chapter",
    "clin_cause_raw",
    # Original clinician fields (for audit)
    "ICD10Code","CauseofDeath","icd10_root",
    # Matches
    "correct_exact","correct_chapter","desc_match",
]

out_df = df[out_cols].rename(columns={"individual_id": "ID"})
out_df.to_csv(OUTPUT_EVAL_CSV, index=False)
print(f"Wrote: {Path(OUTPUT_EVAL_CSV).resolve()}")
