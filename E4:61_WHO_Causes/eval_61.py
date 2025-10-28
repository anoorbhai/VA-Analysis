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
    s = clean_text(x)
    m = re.search(r"(\d{1,2})\.(\d{1,2})", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return f"{a:02d}.{b:02d}"
    digits = re.sub(r"\D", "", s)
    if len(digits) == 4:
        a, b = int(digits[:2]), int(digits[2:])
        return f"{a:02d}.{b:02d}"
    if len(digits) == 3:
        a, b = int(digits[0]), int(digits[1:])
        return f"{a:02d}.{b:02d}"
    return s

def scheme_chapter_from_norm(code_norm: str) -> str:
    return code_norm[:2] if SCHEME_CODE_RE.match(code_norm) else ""

def icd10_root(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = ICD10_ROOT_RE.search(text.upper())
    return m.group(0) if m else None

def norm_str_for_eq(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = clean_text(s)
    s = s.casefold()
    s = re.sub(r"\s+", " ", s)
    return s

def compute_csmf_accuracy(
    df: pd.DataFrame,
    pred_col: str = "llm_code_norm",
    true_col: str = "clin_code_norm",
    exclude_codes = {"99.00", "NA", "", None}
):
    tmp = df[[pred_col, true_col]].copy()
    # Normalise to strings
    tmp["pred"] = tmp[pred_col].astype(str)
    tmp["true"] = tmp[true_col].astype(str)

    if exclude_codes:
        tmp = tmp[~tmp["true"].isin(exclude_codes)]
        tmp = tmp[~tmp["pred"].isin(exclude_codes)]

    n = len(tmp)
    if n == 0:
        return 0.0, pd.DataFrame(columns=["cause","p_true","p_pred","abs_diff"])

    # Fractions by cause
    p_true = (tmp["true"].value_counts(dropna=False) / n).sort_index()
    p_pred = (tmp["pred"].value_counts(dropna=False) / n).sort_index()

    causes = sorted(set(p_true.index).union(set(p_pred.index)))
    p_true = p_true.reindex(causes, fill_value=0.0)
    p_pred = p_pred.reindex(causes, fill_value=0.0)

    abs_diffs = (p_pred - p_true).abs()
    numerator = abs_diffs.sum()
    denom = 2.0 * (1.0 - p_true.min())

    if denom == 0.0:
        csmf_acc = 1.0 if numerator == 0.0 else 0.0
    else:
        csmf_acc = 1.0 - (numerator / denom)

    per_cause = pd.DataFrame({
        "cause": causes,
        "p_true": p_true.values,
        "p_pred": p_pred.values,
        "abs_diff": abs_diffs.values
    }).sort_values("p_true", ascending=False).reset_index(drop=True)

    return float(csmf_acc), per_cause

def chance_corrected_concordance(true_labels, pred_labels, all_causes=None):
    # CCC_j = ( (TP_j / (TP_j + FN_j)) - 1/C ) / (1 - 1/C)
    n = len(true_labels)
    if n == 0 or len(pred_labels) != n:
        return float('nan')

    # Determine cause set and C
    if all_causes is not None:
        causes = list(dict.fromkeys(all_causes))  
    else:
        causes = sorted(set(true_labels))

    C = len(causes)
    if C <= 1:
        return float('nan')

    # Per-cause TP and totals
    tp = {c: 0 for c in causes}
    true_totals = {c: 0 for c in causes}
    for t, p in zip(true_labels, pred_labels):
        if t in true_totals:
            true_totals[t] += 1
            if p == t:
                tp[t] += 1

    denom_c = (1.0 - 1.0 / C)
    if denom_c == 0.0:
        return float('nan')

    ccc_vals = []
    for c in causes:
        N_c = true_totals[c]  
        if N_c == 0:
            continue
        recall_c = tp[c] / N_c
        ccc_c = (recall_c - (1.0 / C)) / denom_c
        ccc_vals.append(ccc_c)

    if not ccc_vals:
        return float('nan')

    return sum(ccc_vals) / len(ccc_vals)

LLM_RESULTS_CSV = "/spaces/25G05/Aaliyah/ZeroShot/llama3_8b_zeroshot_61_results_20251018_150616.csv"
CLINICIAN_CSV   = "/dataA/madiva/va/student/madiva_va_clinician_COD_20250926.csv"
MAPPING_CSV     = "/spaces/25G05/61_codes.csv"

LOG_FILE = "/spaces/25G05/Rizwaanah/evaluation_log.txt"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_EVAL_CSV = f"/spaces/25G05/Rizwaanah/Evaluation/llama3_70b_few_61_evaluation_{timestamp}.csv"

OUTPUT_TOP5_RECALL_CSV = f"/spaces/25G05/Rizwaanah/Evaluation/llama3_70b_few_61_top5_recall_{timestamp}.csv"
OUTPUT_CSMF_SUMMARY_TXT = f"/spaces/25G05/Rizwaanah/Evaluation/llama3_70b_few_61_csmf_summary_{timestamp}.txt"

REMOVE_MISSING_SCHEME = True
REMOVE_UNKNOWN_CODE99 = False

llm_df = pd.read_csv(LLM_RESULTS_CSV)
clin_df = pd.read_csv(CLINICIAN_CSV)
map_df  = pd.read_csv(MAPPING_CSV)

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
df_prefilter = df.copy()

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

# Normalised codes
df["llm_code_norm"]  = df["llm_code_raw"].apply(normalize_scheme_code)
df["clin_code_norm"] = df["clin_code_raw"].apply(normalize_scheme_code)

df["llm_chapter"]  = df["llm_code_norm"].apply(scheme_chapter_from_norm)
df["clin_chapter"] = df["clin_code_norm"].apply(scheme_chapter_from_norm)

exclude_set = set()
if REMOVE_UNKNOWN_CODE99:
    exclude_set.add("99.00")
exclude_set.update({"", "NA", None})

csmf_acc, csmf_table = compute_csmf_accuracy(
    df,
    pred_col="llm_code_norm",
    true_col="clin_code_norm",
    exclude_codes=exclude_set
)
print(f"CSMF Accuracy (evaluated set): {csmf_acc:.4f}")

pref = df_prefilter.copy()
pref["llm_code_norm_pref"]  = pref.get("CODE", "").astype(str).apply(normalize_scheme_code)
pref["clin_code_norm_pref"] = pref.get("clin_scheme_code", "").astype(str).apply(normalize_scheme_code)

total_pref = len(pref)
pred_99_mask = pref["llm_code_norm_pref"].astype(str).str.strip() == "99.00"
pred_99_count = int(pred_99_mask.sum())
pred_99_pct   = (pred_99_count / total_pref * 100.0) if total_pref else 0.0

true_99_mask  = pref["clin_code_norm_pref"].astype(str).str.strip() == "99.00"
correct_99    = int((pred_99_mask & true_99_mask).sum())
correct_99_pct_of_pred = (correct_99 / pred_99_count * 100.0) if pred_99_count else 0.0
correct_99_pct_of_total = (correct_99 / total_pref * 100.0) if total_pref else 0.0

print(f'Predicted "99.00" count: {pred_99_count} ({pred_99_pct:.2f}% of all merged cases)')
print(f'Correct "99.00" (pred==true): {correct_99} '
      f'({correct_99_pct_of_pred:.2f}% of 99.00 predictions; {correct_99_pct_of_total:.2f}% of total)')

counts = df["clin_code_norm"].value_counts()
top5_codes = list(counts.index[:6])

rows = []
for code in top5_codes:
    support = int((df["clin_code_norm"] == code).sum())
    tp = int(((df["clin_code_norm"] == code) & (df["llm_code_norm"] == code)).sum())
    recall = (tp / support) if support else 0.0
    rows.append({"cause": code, "support": support, "true_pos": tp, "recall": recall})

top5_recall_df = pd.DataFrame(rows).sort_values("support", ascending=False)
print("\nTop-6 cause-specific recall (evaluated set):")
for _, r in top5_recall_df.iterrows():
    print(f"- {r['cause']}: recall={r['recall']:.2%} (TP={r['true_pos']}, N={r['support']})")

_mask = (
    df["clin_code_norm"].astype(str).str.strip() != ""
) & (
    df["llm_code_norm"].astype(str).str.strip() != ""
)
true_seq = df.loc[_mask, "clin_code_norm"].tolist()
pred_seq = df.loc[_mask, "llm_code_norm"].tolist()
ccc_macro = chance_corrected_concordance(true_seq, pred_seq)

print(f"\nCCC (macro, equal-weight; Murray 2011): {ccc_macro:.4f}")


with open(OUTPUT_CSMF_SUMMARY_TXT, "w", encoding="utf-8") as fh:
    fh.write(
        f"CSMF Accuracy (evaluated set): {csmf_acc:.6f}\n"
        f'Predicted "99.00": {pred_99_count} / {total_pref} ({pred_99_pct:.2f}%)\n'
        f'Correct "99.00": {correct_99} '
        f'({correct_99_pct_of_pred:.2f}% of 99.00 preds; {correct_99_pct_of_total:.2f}% of total)\n\n'
        "Top-5 cause-specific recall:\n"
    )
    for _, r in top5_recall_df.iterrows():
        fh.write(f"- {r['cause']}: recall={r['recall']:.6f} (TP={r['true_pos']}, N={r['support']})\n")

print(f"\nCCC (macro, equal-weight; Murray 2011): {ccc_macro:.4f}")

top5_recall_df.to_csv(OUTPUT_TOP5_RECALL_CSV, index=False)

# Normalised causes (text)
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

CSMF Accuracy (evaluated set): {csmf_acc:.6f}
Predicted "99.00": {pred_99_count} / {total_pref} ({pred_99_pct:.2f}%)
Correct "99.00": {correct_99} ({correct_99_pct_of_pred:.2f}% of 99.00 preds; {correct_99_pct_of_total:.2f}% of total)

Top-5 cause-specific recall:
{chr(10).join([f"- {r['cause']}: recall={r['recall']:.2%} (TP={r['true_pos']}, N={r['support']})" for _, r in top5_recall_df.iterrows()])}

CCC (macro, equal-weight; Murray 2011): {ccc_macro:.6f}
"""

with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write(log_entry)

print(f"Results logged to: {LOG_FILE}")

out_cols = [
    "individual_id",
    "llm_code_raw","llm_code_norm","llm_chapter",
    "llm_cause_raw",
    "clin_code_raw","clin_code_norm","clin_chapter",
    "clin_cause_raw",
    "ICD10Code","CauseofDeath","icd10_root",
    "correct_exact","correct_chapter","desc_match",
]

out_df = df[out_cols].rename(columns={"individual_id": "ID"})
out_df.to_csv(OUTPUT_EVAL_CSV, index=False)
print(f"Wrote: {Path(OUTPUT_EVAL_CSV).resolve()}")
