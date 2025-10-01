#!/usr/bin/env python3
"""
Single Case Verbal Autopsy Llama3 Prompt Script

This script lets you input an individual_id, extracts the corresponding row from the VA dataset,
formats all columns and values into a prompt, sends it to the Llama3 model via Ollama, and writes the full response to an output text file.

Usage:
    python va_prompt_single.py <individual_id>
"""

import sys
import pandas as pd
import requests
from pathlib import Path

# Constants
INPUT_CSV_PATH = "/dataA/madiva/va/student/madiva_va_dataset_20250924.csv"
MODEL_NAME = "llama3_VA:latest"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OUTPUT_TXT_PATH = "/spaces/25G05/ZeroShot/llama3_single_output.txt"
EXCLUDE_FIELDS = ['cause1', 'prob1', 'cause2', 'prob2', 'cause3', 'prob3']


def format_case_for_llm(row: pd.Series) -> str:
    prompt_parts = []
    individual_id = row.get('individual_id', 'Unknown')
    prompt_parts.append(f"Case ID: {individual_id}")
    prompt_parts.append("\nSYMPTOM DATA:")
    for column_name, value in row.items():
        if column_name in EXCLUDE_FIELDS:
            continue
        if pd.isna(value) or (isinstance(value, str) and value.strip() == '') or value == '-':
            continue
        cleaned_value = str(value).strip()
        prompt_parts.append(f"{column_name}: {cleaned_value}")
    narrative = row.get('narrative', '')
    if pd.isna(narrative) or str(narrative).strip() == '':
        narrative = "No narrative provided"
    prompt_parts.append("\nNARRATIVE:")
    prompt_parts.append(str(narrative).strip())
    prompt_parts.append("\nPlease analyze this verbal autopsy case and provide your diagnosis.")
    return "\n".join(prompt_parts)


def main():
    if len(sys.argv) != 2:
        print("Usage: python va_prompt_single.py <individual_id>")
        sys.exit(1)
    individual_id = sys.argv[1]
    # Load dataset
    df = pd.read_csv(INPUT_CSV_PATH)
    # Find row by individual_id
    row = df[df['individual_id'] == individual_id]
    if row.empty:
        print(f"No case found for individual_id: {individual_id}")
        sys.exit(1)
    row = row.iloc[0]
    # Format prompt
    prompt = format_case_for_llm(row)
    # Query Llama3 via Ollama
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }
    response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
    if response.status_code == 200:
        result = response.json()
        response_text = result.get('response', '')
        with open(OUTPUT_TXT_PATH, 'w') as f:
            f.write(response_text)
        print(f"Response written to {OUTPUT_TXT_PATH}")
    else:
        print(f"Ollama API error: {response.status_code} - {response.text}")
        sys.exit(1)

if __name__ == "__main__":
    main()
