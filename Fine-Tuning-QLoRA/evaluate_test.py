#!/usr/bin/env python3
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import re
import sys
from pathlib import Path
from datetime import datetime

# ==================== CONFIGURATION ====================

INPUT_CSV = "/spaces/25G05/Fine-Tuning/results/match_analysis_inference_results_20251027_225130.csv"
OUTPUT_DIR = "/spaces/25G05/Fine-Tuning/results" 
OUTPUT_FILENAME = None  # None = auto-generate with timestamp

# Evaluation parameters
LABEL_FIELD = "scheme_code"
TOPK = 6
UNCERTAINTY_CODE = "99.0"  
EXCLUDE_UNCERTAINTY = True

# =======================================================


def norm_text(x: Optional[str]) -> str:
    """Normalize free-text for comparison (case/whitespace/quotes)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().strip('"').strip("'").lower()
    # collapse internal whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def norm_code(x: Optional[str]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip().strip('"').strip("'")


def extract_chapter(code: str) -> str:
    c = norm_code(code)
    if c == "":
        return ""
    if '.' in c:
        return c.split('.', 1)[0]
    return c


def csmf_accuracy(true_labels: List[str], pred_labels: List[str]) -> float:
    """Compute dataset-level CSMF accuracy using standard Murray/PHMRC formula."""
    N = len(true_labels)
    if N == 0:
        return float('nan')
    # Build distributions over union of labels
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
    # Defensive checks
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
        N_c = true_totals[c]  # TP + FN for cause c
        if N_c == 0:
            # No true instances of this cause in the evaluation set; skip in average
            continue
        recall_c = tp[c] / N_c
        ccc_c = (recall_c - (1.0 / C)) / denom_c
        ccc_vals.append(ccc_c)

    if not ccc_vals:
        return float('nan')

    # Equal-weight mean of CCC_j across causes with true instances
    return sum(ccc_vals) / len(ccc_vals)


def per_cause_recall(true_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
    """Compute recall per label = TP / (TP + FN)."""
    labels = sorted(set(true_labels))  # only causes that appear in truth
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
    input_path = INPUT_CSV
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if OUTPUT_FILENAME is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_txt = output_dir / f"evaluation_results_{timestamp}.txt"
        output_csv = output_dir / f"evaluation_summary_{timestamp}.csv"
        output_recalls_csv = output_dir / f"evaluation_per_cause_recalls_{timestamp}.csv"
    else:
        base_name = Path(OUTPUT_FILENAME).stem
        output_txt = output_dir / f"{base_name}.txt"
        output_csv = output_dir / f"{base_name}.csv"
        output_recalls_csv = output_dir / f"{base_name}_per_cause_recalls.csv"

    df = pd.read_csv(input_path)

    pred_code_col = "predicted_scheme_code"
    true_code_col = "expected_scheme_code"
    pred_cause_col = "predicted_cause"
    true_cause_col = "expected_cause"

    valid = df.copy()

    valid["pred_code_norm"] = valid[pred_code_col].apply(norm_code)
    valid["true_code_norm"] = valid[true_code_col].apply(norm_code)
    valid["pred_cause_norm"] = valid[pred_cause_col].apply(norm_text)
    valid["true_cause_norm"] = valid[true_cause_col].apply(norm_text)

    unc = str(UNCERTAINTY_CODE)
    valid["pred_is_unc"] = (valid["pred_code_norm"] == unc)
    valid["true_is_unc"] = (valid["true_code_norm"] == unc)

    valid["pred_chapter"] = valid["pred_code_norm"].apply(extract_chapter)
    valid["true_chapter"] = valid["true_code_norm"].apply(extract_chapter)

    base_mask = (~valid["pred_code_norm"].isna()) & (~valid["true_code_norm"].isna())
    base_mask &= (valid["pred_code_norm"] != "") & (valid["true_code_norm"] != "")

    if EXCLUDE_UNCERTAINTY:
        base_mask &= (~valid["pred_is_unc"]) & (~valid["true_is_unc"])

    sub = valid[base_mask].copy()

    # Exact code match accuracy
    sub["code_match"] = (sub["pred_code_norm"] == sub["true_code_norm"])
    code_acc = sub["code_match"].mean() if len(sub) > 0 else float('nan')

    # Exact description match accuracy

    sub["cause_match"] = (sub["pred_cause_norm"] == sub["true_cause_norm"])
    cause_acc = sub["cause_match"].mean() if len(sub) > 0 else float('nan')

    # Both code & description correct
    both_acc = (sub["code_match"] & sub["cause_match"]).mean() if len(sub) > 0 else float('nan')

    # Correct chapter accuracy (uses code-based chapters)
    sub["chapter_match"] = (sub["pred_chapter"] == sub["true_chapter"]) & (sub["pred_chapter"] != "") & (sub["true_chapter"] != "")
    chapter_acc = sub["chapter_match"].mean() if len(sub) > 0 else float('nan')

    # Build labels for distribution-based metrics (CSMF/CCC/Recall)
    if LABEL_FIELD == "scheme_code":
        true_labels_all = sub["true_code_norm"].tolist()
        pred_labels_all = sub["pred_code_norm"].tolist()
    else:
        true_labels_all = sub["true_cause_norm"].tolist()
        pred_labels_all = sub["pred_cause_norm"].tolist()

    # CSMF accuracy (dataset-level)
    csmf_acc = csmf_accuracy(true_labels_all, pred_labels_all)

    # Chance-corrected concordance (CCC)
    ccc = chance_corrected_concordance(true_labels_all, pred_labels_all)

    # Cause-specific recall for top-K causes
    labels_for_topk = [t for t in true_labels_all if (not EXCLUDE_UNCERTAINTY or t != unc)]
    topk = topk_by_true_prevalence(labels_for_topk, TOPK, exclude_label=(unc if EXCLUDE_UNCERTAINTY else None))

    recalls = per_cause_recall(true_labels_all, pred_labels_all)
    topk_recalls = {c: recalls.get(c, float('nan')) for c in topk}
    macro_topk_recall = np.nanmean(list(topk_recalls.values())) if len(topk_recalls) > 0 else float('nan')

    # Uncertainty rate: fraction of predicted == uncertainty code among rows with a prediction
    valid_pred_mask = (~valid["pred_code_norm"].isna()) & (valid["pred_code_norm"] != "")
    uncertainty_rate = valid.loc[valid_pred_mask, "pred_is_unc"].mean() if valid_pred_mask.any() else float('nan')

    # Uncertainty accuracy: Of predictions where LLM said uncertain (99), 
    # how often did clinician also say uncertain?
    pred_unc_mask = valid["pred_is_unc"]
    if pred_unc_mask.any():
        uncertainty_accuracy = valid.loc[pred_unc_mask, "true_is_unc"].mean()
    else:
        uncertainty_accuracy = float('nan')

    # Build summary dict
    summary = {
        "input_file": str(input_path),
        "n_rows_input": int(len(df)),
        "n_rows_evaluated": int(len(sub)),
        "exclude_uncertainty": bool(EXCLUDE_UNCERTAINTY),
        "uncertainty_code": unc,
        "label_field_for_distribution_metrics": LABEL_FIELD,
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

    # Build output text
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("VERBAL AUTOPSY MODEL EVALUATION RESULTS")
    output_lines.append("=" * 70)
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Input file: {input_path}")
    output_lines.append("")
    output_lines.append("CONFIGURATION")
    output_lines.append("-" * 70)
    output_lines.append(f"Label field for distribution metrics: {LABEL_FIELD}")
    output_lines.append(f"Top-K causes for recall: {TOPK}")
    output_lines.append(f"Uncertainty code: {unc}")
    output_lines.append(f"Exclude uncertainty from metrics: {EXCLUDE_UNCERTAINTY}")
    output_lines.append("")
    output_lines.append("DATASET SUMMARY")
    output_lines.append("-" * 70)
    output_lines.append(f"Total rows in input: {summary['n_rows_input']}")
    output_lines.append(f"Rows evaluated (valid predictions): {summary['n_rows_evaluated']}")
    output_lines.append("")
    output_lines.append("ACCURACY METRICS")
    output_lines.append("-" * 70)
    output_lines.append(f"Exact scheme code match: {summary['exact_code_match_accuracy']:.4f}" if summary['exact_code_match_accuracy'] is not None else "Exact scheme code match: N/A")
    output_lines.append(f"Exact cause description match: {summary['exact_description_match_accuracy']:.4f}" if summary['exact_description_match_accuracy'] is not None else "Exact cause description match: N/A")
    output_lines.append(f"Both code & description correct: {summary['both_code_and_description_accuracy']:.4f}" if summary['both_code_and_description_accuracy'] is not None else "Both code & description correct: N/A")
    output_lines.append(f"Correct chapter (code prefix): {summary['correct_chapter_accuracy']:.4f}" if summary['correct_chapter_accuracy'] is not None else "Correct chapter (code prefix): N/A")
    output_lines.append("")
    output_lines.append("DISTRIBUTION-BASED METRICS")
    output_lines.append("-" * 70)
    output_lines.append(f"CSMF Accuracy: {summary['csmf_accuracy']:.4f}" if summary['csmf_accuracy'] is not None else "CSMF Accuracy: N/A")
    output_lines.append(f"Chance-Corrected Concordance (CCC): {summary['ccc']:.4f}" if summary['ccc'] is not None else "Chance-Corrected Concordance (CCC): N/A")
    output_lines.append("")
    output_lines.append("UNCERTAINTY METRICS")
    output_lines.append("-" * 70)
    output_lines.append(f"Uncertainty rate (% predictions = uncertain): {summary['uncertainty_rate']:.4f}" if summary['uncertainty_rate'] is not None else "Uncertainty rate: N/A")
    output_lines.append(f"Uncertainty accuracy (recall for uncertain class): {summary['uncertainty_accuracy']:.4f}" if summary['uncertainty_accuracy'] is not None else "Uncertainty accuracy: N/A")
    output_lines.append("")
    output_lines.append(f"TOP-{TOPK} CAUSE-SPECIFIC RECALLS (by true prevalence)")
    output_lines.append("-" * 70)
    for c in topk:
        recall_val = topk_recalls.get(c, float('nan'))
        if recall_val == recall_val:  # check not NaN
            output_lines.append(f"  {c:30s}: {recall_val:.4f}")
        else:
            output_lines.append(f"  {c:30s}: N/A")
    output_lines.append("")
    output_lines.append(f"Macro-average recall (top-{TOPK}): {summary['topk_macro_recall']:.4f}" if summary['topk_macro_recall'] is not None else f"Macro-average recall (top-{TOPK}): N/A")
    output_lines.append("")
    output_lines.append("=" * 70)
    
    # Write to text file
    with open(output_txt, 'w') as f:
        f.write('\n'.join(output_lines))
    
    # Print to console
    print('\n'.join(output_lines))
    
    # Save summary as CSV
    pd.DataFrame([summary]).to_csv(output_csv, index=False)
    
    # Save per-cause recalls
    recalls_df = pd.DataFrame({
        'cause': list(recalls.keys()),
        'recall': list(recalls.values())
    })
    recalls_df.to_csv(output_recalls_csv, index=False)
    
    print(f"\nResults saved to:")
    print(f"  Text summary: {output_txt}")
    print(f"  CSV summary: {output_csv}")
    print(f"  Per-cause recalls: {output_recalls_csv}")


if __name__ == "__main__":
    main()
