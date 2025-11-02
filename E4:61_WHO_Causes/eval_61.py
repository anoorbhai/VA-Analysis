
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

LLM_RESULTS_CSV = "/spaces/25G05/Aaliyah/FewShot/llama3_70b_fewshot_61_results_20251029_002216.csv"
CLINICIAN_CSV   = "/dataA/madiva/va/student/madiva_va_clinician_COD_20250926.csv"
MAPPING_CSV     = "/spaces/25G05/61_codes.csv"

# Output Configuration
OUTPUT_DIR = "/spaces/25G05/Aaliyah/Evaluation"
OUTPUT_FILENAME = None  

# Filtering 
REMOVE_MISSING_SCHEME = True

# Evaluation Parameters
LABEL_FIELD = "scheme_code"  
TOPK = 5  
UNCERTAINTY_CODE = "99.00"  
EXCLUDE_UNCERTAINTY = True 


WS_RE = re.compile(r"\s+")
NON_PRINT_RE = re.compile(r"[\u2000-\u200b\u202f\u00a0]")
SCHEME_CODE_RE = re.compile(r"^\d{2}\.\d{2}$")
ICD10_ROOT_RE = re.compile(r"[A-Z][0-9]{2}")


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


def extract_chapter(code: str) -> str:
    if SCHEME_CODE_RE.match(code):
        return code[:2]
    return ""


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


def norm_text(x: Optional[str]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().strip('"').strip("'").lower()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_code(x: Optional[str]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip().strip('"').strip("'")


def csmf_accuracy(true_labels: List[str], pred_labels: List[str]) -> float:
    # Compute dataset-level CSMF accuracy using standard Murray/PHMRC formula
    N = len(true_labels)
    if N == 0:
        return float('nan')
    labels = sorted(set(true_labels) | set(pred_labels))
    true_counts = {l: 0 for l in labels}
    pred_counts = {l: 0 for l in labels}
    for t in true_labels:
        true_counts[t] += 1
    for p in pred_labels:
        pred_counts[p] += 1
    pi_true = {l: true_counts[l] / N for l in labels}
    pi_pred = {l: pred_counts[l] / N for l in labels}
    abs_diff_sum = sum(abs(pi_pred[l] - pi_true[l]) for l in labels)
    min_true = min(pi_true.values()) if len(pi_true) > 0 else 0.0
    denom = 2.0 * (1.0 - min_true)
    if denom == 0:
        return float('nan')
    return 1.0 - (abs_diff_sum / denom)

def chance_corrected_concordance(true_labels, pred_labels, all_causes=None):
    n = len(true_labels)
    if n == 0 or len(pred_labels) != n:
        return float('nan')

    if all_causes is not None:
        causes = list(dict.fromkeys(all_causes))  
    else:
        causes = sorted(set(true_labels))

    C = len(causes)
    if C <= 1:
        return float('nan')

    # Precompute per-cause TP and total true counts (TP+FN)
    tp = {c: 0 for c in causes}
    true_totals = {c: 0 for c in causes}

    for t, p in zip(true_labels, pred_labels):
        if t in true_totals:
            true_totals[t] += 1
            if p == t:
                tp[t] += 1

    # Compute cause-specific CCC_j
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

    # Equal-weight mean of CCC_j across causes with true instances
    return sum(ccc_vals) / len(ccc_vals)



def per_cause_recall(true_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
    # Compute recall per label = TP / (TP + FN)
    labels = sorted(set(true_labels))
    recalls = {}
    for l in labels:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == l and p == l)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == l and p != l)
        denom = tp + fn
        recalls[l] = (tp / denom) if denom > 0 else float('nan')
    return recalls


def topk_by_true_prevalence(true_labels: List[str], k: int, exclude_label: Optional[str] = None) -> List[str]:
    counts = {}
    for t in true_labels:
        if exclude_label is not None and t == exclude_label:
            continue
        counts[t] = counts.get(t, 0) + 1
    return [l for l, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))][:k]


def main():
    print("="*70)
    print("COMPREHENSIVE VA EVALUATION")
    print("="*70)
    
    print(f"\nLoading LLM results from: {LLM_RESULTS_CSV}")
    llm_df = pd.read_csv(LLM_RESULTS_CSV)
    
    print(f"Loading clinician data from: {CLINICIAN_CSV}")
    clin_df = pd.read_csv(CLINICIAN_CSV)
    
    print(f"Loading mapping from: {MAPPING_CSV}")
    map_df = pd.read_csv(MAPPING_CSV)
    
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
    
    # Validate mapping file
    for req in ["scheme_code", "scheme_cause"]:
        if req not in map_df.columns:
            raise ValueError(f"Mapping CSV must include column: {req}")
    
    if "icd10_root" not in map_df.columns:
        map_df["icd10_root"] = ""
    if "clinician_desc" not in map_df.columns:
        map_df["clinician_desc"] = ""
    
    # Normalize mapping data
    map_df["icd10_root_norm"] = map_df["icd10_root"].astype(str).str.upper().str.strip()
    map_df["clin_desc_norm"] = map_df["clinician_desc"].apply(norm_str_for_eq)
    map_df["scheme_code"] = map_df["scheme_code"].astype(str).str.strip()
    map_df["scheme_cause"] = map_df["scheme_cause"].astype(str).str.strip()
    
    # Normalize clinician data
    clin_df["individual_id"] = clin_df["individual_id"].astype(str)
    for col in ["ICD10Code", "CauseofDeath"]:
        if col not in clin_df.columns:
            clin_df[col] = ""
    
    clin_df["icd10_root"] = clin_df["ICD10Code"].apply(icd10_root).fillna("")
    clin_df["icd10_root_norm"] = clin_df["icd10_root"].str.upper().str.strip()
    clin_df["clin_desc_norm"] = clin_df["CauseofDeath"].apply(norm_str_for_eq)
    
    # Map clinician data to scheme codes (ICD-10 root merge)
    merged_a = pd.merge(
        clin_df,
        map_df[["icd10_root_norm", "scheme_code", "scheme_cause"]],
        on="icd10_root_norm",
        how="left",
    )
    
    # Fallback: clinician description merge
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
    
    # Normalize clinician scheme codes first 
    df["true_code_norm"] = df["clin_scheme_code"].fillna("").astype(str).apply(normalize_scheme_code)
    
    initial_count = len(df)
    reasons = []
    
    if REMOVE_MISSING_SCHEME:
        before = len(df)
        df = df[df["CODE"].astype(str).str.strip() != ""]
        removed = before - len(df)
        if removed:
            reasons.append(f"{removed} missing LLM CODE")
    
    if EXCLUDE_UNCERTAINTY:
        before = len(df)
        df = df[df["true_code_norm"] != "99.00"]
        removed = before - len(df)
        if removed:
            reasons.append(f"{removed} clinician assigned 99.00")
    
    filtered_count = len(df)
    print(f"Filtered: {initial_count - filtered_count} ({'; '.join(reasons) if reasons else 'none'})")
    print(f"Remaining entries for evaluation: {filtered_count}")
    
    # Normalize for evaluation 
    df["pred_code_norm"] = df["CODE"].astype(str).apply(normalize_scheme_code)
    df["pred_cause_norm"] = df.get("CAUSE_SHORT", "").astype(str).apply(norm_text)
    df["true_cause_norm"] = df.get("clin_scheme_cause", "").astype(str).apply(norm_text)
    
    # Extract chapters
    df["pred_chapter"] = df["pred_code_norm"].apply(extract_chapter)
    df["true_chapter"] = df["true_code_norm"].apply(extract_chapter)
    
    unc = str(UNCERTAINTY_CODE)
    df["pred_is_unc"] = (df["pred_code_norm"] == unc)
    df["true_is_unc"] = (df["true_code_norm"] == unc)
    
    df["code_match"] = (df["pred_code_norm"] == df["true_code_norm"]) & (df["pred_code_norm"] != "")
    df["cause_match"] = (df["pred_cause_norm"] == df["true_cause_norm"]) & (df["pred_cause_norm"] != "")
    df["chapter_match"] = (df["pred_chapter"] == df["true_chapter"]) & (df["pred_chapter"] != "")
    
    total = len(df)
    code_acc = df["code_match"].mean() if total > 0 else float('nan')
    cause_acc = df["cause_match"].mean() if total > 0 else float('nan')
    both_acc = (df["code_match"] & df["cause_match"]).mean() if total > 0 else float('nan')
    chapter_acc = df["chapter_match"].mean() if total > 0 else float('nan')
    
    base_mask = (df["pred_code_norm"] != "") & (df["true_code_norm"] != "")
    
    sub = df[base_mask].copy()
    
    if LABEL_FIELD == "scheme_code":
        true_labels = sub["true_code_norm"].tolist()
        pred_labels = sub["pred_code_norm"].tolist()
    else:
        true_labels = sub["true_cause_norm"].tolist()
        pred_labels = sub["pred_cause_norm"].tolist()
    
    csmf_acc = csmf_accuracy(true_labels, pred_labels)
    ccc = chance_corrected_concordance(true_labels, pred_labels)
    
    # Per-cause recall 
    labels_for_topk = true_labels
    topk = topk_by_true_prevalence(labels_for_topk, TOPK, exclude_label=None)
    
    recalls = per_cause_recall(true_labels, pred_labels)
    topk_recalls = {c: recalls.get(c, float('nan')) for c in topk}
    macro_topk_recall = np.nanmean(list(topk_recalls.values())) if len(topk_recalls) > 0 else float('nan')
    
    # Uncertainty metrics
    valid_pred_mask = (df["pred_code_norm"] != "")
    uncertainty_rate = df.loc[valid_pred_mask, "pred_is_unc"].mean() if valid_pred_mask.any() else float('nan')
    
    # Uncertainty accuracy
    pred_unc_mask = df["pred_is_unc"]
    if pred_unc_mask.any():
        uncertainty_accuracy = df.loc[pred_unc_mask, "true_is_unc"].mean()
    else:
        uncertainty_accuracy = float('nan')
    
    summary = {
        "llm_results_file": str(LLM_RESULTS_CSV),
        "clinician_file": str(CLINICIAN_CSV),
        "mapping_file": str(MAPPING_CSV),
        "n_rows_initial": int(initial_count),
        "n_rows_filtered": int(initial_count - filtered_count),
        "n_rows_evaluated": int(total),
        "n_rows_for_distribution_metrics": int(len(sub)),
        "remove_missing_scheme": bool(REMOVE_MISSING_SCHEME),
        "exclude_uncertainty": bool(EXCLUDE_UNCERTAINTY),
        "uncertainty_code": unc,
        "label_field": LABEL_FIELD,
        "exact_code_match_accuracy": float(code_acc) if code_acc == code_acc else None,
        "exact_description_match_accuracy": float(cause_acc) if cause_acc == cause_acc else None,
        "both_code_and_description_accuracy": float(both_acc) if both_acc == both_acc else None,
        "correct_chapter_accuracy": float(chapter_acc) if chapter_acc == chapter_acc else None,
        "csmf_accuracy": float(csmf_acc) if csmf_acc == csmf_acc else None,
        "ccc": float(ccc) if ccc == ccc else None,
        "uncertainty_rate": float(uncertainty_rate) if uncertainty_rate == uncertainty_rate else None,
        "uncertainty_accuracy": float(uncertainty_accuracy) if uncertainty_accuracy == uncertainty_accuracy else None,
        "topk": int(TOPK),
        "topk_causes": ",".join(topk),
        "topk_macro_recall": float(macro_topk_recall) if macro_topk_recall == macro_topk_recall else None,
    }
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if OUTPUT_FILENAME is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_txt = output_dir / f"comprehensive_eval_{timestamp}.txt"
        output_csv = output_dir / f"comprehensive_eval_{timestamp}.csv"
        output_recalls_csv = output_dir / f"comprehensive_eval_{timestamp}_per_cause_recalls.csv"
        output_detailed_csv = output_dir / f"comprehensive_eval_{timestamp}_detailed.csv"
    else:
        base_name = Path(OUTPUT_FILENAME).stem
        output_txt = output_dir / f"{base_name}.txt"
        output_csv = output_dir / f"{base_name}.csv"
        output_recalls_csv = output_dir / f"{base_name}_per_cause_recalls.csv"
        output_detailed_csv = output_dir / f"{base_name}_detailed.csv"
    
    # Build output text
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("COMPREHENSIVE VERBAL AUTOPSY EVALUATION RESULTS")
    output_lines.append("=" * 70)
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"LLM Results: {LLM_RESULTS_CSV}")
    output_lines.append(f"Clinician Data: {CLINICIAN_CSV}")
    output_lines.append(f"Mapping File: {MAPPING_CSV}")
    output_lines.append("")
    output_lines.append("CONFIGURATION")
    output_lines.append("-" * 70)
    output_lines.append(f"Label field for distribution metrics: {LABEL_FIELD}")
    output_lines.append(f"Top-K causes for recall: {TOPK}")
    output_lines.append(f"Uncertainty code: {unc}")
    output_lines.append(f"Remove missing LLM CODE: {REMOVE_MISSING_SCHEME}")
    output_lines.append(f"Exclude clinician-assigned uncertainty (code 99): {EXCLUDE_UNCERTAINTY}")
    output_lines.append("")
    output_lines.append("DATASET SUMMARY")
    output_lines.append("-" * 70)
    output_lines.append(f"Initial entries: {initial_count}")
    output_lines.append(f"Filtered out: {initial_count - filtered_count} ({'; '.join(reasons) if reasons else 'none'})")
    output_lines.append(f"Rows evaluated (accuracy metrics): {total}")
    output_lines.append(f"Rows for distribution metrics: {len(sub)} (excludes cases with missing codes)")
    output_lines.append("")
    output_lines.append("ACCURACY METRICS (computed on all {0} rows)".format(total))
    output_lines.append("-" * 70)
    output_lines.append(f"Exact scheme code match: {summary['exact_code_match_accuracy']*100:.2f}%" if summary['exact_code_match_accuracy'] is not None else "Exact scheme code match: N/A")
    output_lines.append(f"Exact cause description match: {summary['exact_description_match_accuracy']*100:.2f}%" if summary['exact_description_match_accuracy'] is not None else "Exact cause description match: N/A")
    output_lines.append(f"Both code & description correct: {summary['both_code_and_description_accuracy']*100:.2f}%" if summary['both_code_and_description_accuracy'] is not None else "Both code & description correct: N/A")
    output_lines.append(f"Correct chapter (code prefix): {summary['correct_chapter_accuracy']*100:.2f}%" if summary['correct_chapter_accuracy'] is not None else "Correct chapter (code prefix): N/A")
    output_lines.append("")
    output_lines.append("DISTRIBUTION-BASED METRICS (computed on {0} rows with valid codes)".format(len(sub)))
    output_lines.append("-" * 70)
    output_lines.append(f"CSMF Accuracy: {summary['csmf_accuracy']*100:.2f}%" if summary['csmf_accuracy'] is not None else "CSMF Accuracy: N/A")
    output_lines.append(f"Chance-Corrected Concordance (CCC): {summary['ccc']:.3f}" if summary['ccc'] is not None else "Chance-Corrected Concordance (CCC): N/A")
    output_lines.append("")
    output_lines.append("UNCERTAINTY METRICS")
    output_lines.append("-" * 70)
    output_lines.append(f"Uncertainty rate (% predictions = uncertain): {summary['uncertainty_rate']*100:.2f}%" if summary['uncertainty_rate'] is not None else "Uncertainty rate: N/A")
    output_lines.append(f"Uncertainty accuracy (when predicting uncertain, % correct): {summary['uncertainty_accuracy']*100:.2f}%" if summary['uncertainty_accuracy'] is not None else "Uncertainty accuracy: N/A")
    output_lines.append("")
    output_lines.append(f"TOP-{TOPK} CAUSE-SPECIFIC RECALLS (by true prevalence)")
    output_lines.append("-" * 70)
    for c in topk:
        recall_val = topk_recalls.get(c, float('nan'))
        if recall_val == recall_val:
            output_lines.append(f"  {c:30s}: {recall_val*100:.2f}%")
        else:
            output_lines.append(f"  {c:30s}: N/A")
    output_lines.append("")
    output_lines.append(f"Macro-average recall (top-{TOPK}): {summary['topk_macro_recall']*100:.2f}%" if summary['topk_macro_recall'] is not None else f"Macro-average recall (top-{TOPK}): N/A")
    output_lines.append("")
    output_lines.append("=" * 70)
    
    with open(output_txt, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print('\n'.join(output_lines))
    
    def format_numeric(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return str(val).replace('.', ',')
    
    # Save summary CSV with metrics as rows
    csv_rows = []
    
    csv_rows.append({"Metric": "LLM results csv", "Value": str(LLM_RESULTS_CSV)})

    csv_rows.append({"Metric": "CSMF", "Value": format_numeric(float(csmf_acc)) if csmf_acc == csmf_acc else None})
    csv_rows.append({"Metric": "CCC", "Value": format_numeric(float(ccc)) if ccc == ccc else None})
    
    # Only include uncertainty metrics if we're not excluding uncertainty
    if not EXCLUDE_UNCERTAINTY:
        csv_rows.append({"Metric": "Uncertainty Rate", "Value": format_numeric(float(uncertainty_rate)) if uncertainty_rate == uncertainty_rate else None})
        csv_rows.append({"Metric": "Uncertainty Precision", "Value": format_numeric(float(uncertainty_accuracy)) if uncertainty_accuracy == uncertainty_accuracy else None})
    
    for i, cause in enumerate(topk, 1):
        recall_val = topk_recalls.get(cause, float('nan'))
        csv_rows.append({"Metric": f"Top {i} Cause", "Value": cause})  # Keep scheme code format unchanged
        csv_rows.append({"Metric": f"Top {i} Recall", "Value": format_numeric(float(recall_val)) if recall_val == recall_val else None})
    
    # Add macro average
    csv_rows.append({"Metric": "Top 5 Macro Recall", "Value": format_numeric(float(macro_topk_recall)) if macro_topk_recall == macro_topk_recall else None})
    
    pd.DataFrame(csv_rows).to_csv(output_csv, index=False)
    
    # Save per-cause recalls 
    recalls_df = pd.DataFrame({
        'cause': list(recalls.keys()),
        'recall': [format_numeric(v) for v in recalls.values()]
    })
    recalls_df.to_csv(output_recalls_csv, index=False)
    
    detailed_cols = [
        "individual_id",
        "CODE", "pred_code_norm", "pred_chapter",
        "CAUSE_SHORT", "pred_cause_norm",
        "clin_scheme_code", "true_code_norm", "true_chapter",
        "clin_scheme_cause", "true_cause_norm",
        "ICD10Code", "CauseofDeath", "icd10_root",
        "code_match", "cause_match", "chapter_match",
    ]
    detailed_df = df[[c for c in detailed_cols if c in df.columns]]
    detailed_df.to_csv(output_detailed_csv, index=False)
    
    print(f"\nResults saved to:")
    print(f"  Text summary: {output_txt}")
    print(f"  CSV summary: {output_csv}")
    print(f"  Per-cause recalls: {output_recalls_csv}")
    print(f"  Detailed results: {output_detailed_csv}")


if __name__ == "__main__":
    main()