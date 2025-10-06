import pandas as pd
from pathlib import Path

CSV_PATH = Path("/dataA/madiva/va/student/madiva_va_clinician_COD_20250926.csv") 
OUTPUT_SNIPPET = Path("/spaces/25G05/Clinical_COD_pairs.txt")                

MAX_ROWS = None

# Column names in the CSV
COL_CAUSE = "CauseofDeath"
COL_CODE  = "ICD10Code"

def main():
    
    # Read only the two columns as strings
    df = pd.read_csv(
        CSV_PATH,
        usecols=[COL_CAUSE, COL_CODE],
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        engine="python"
    )

    df[COL_CAUSE] = df[COL_CAUSE].astype(str).str.strip()
    df[COL_CODE]  = df[COL_CODE].astype(str).str.strip()
    df = df[(df[COL_CAUSE] != "") & (df[COL_CODE] != "")]
    df = df.drop_duplicates(subset=[COL_CAUSE, COL_CODE]).reset_index(drop=True)

    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS)

    if df.empty:
        raise ValueError("No (CauseofDeath, ICD10Code) pairs found after cleaning.")

    # To go into model file
    lines = []
    lines.append('Use the following physician-coded pairs as reference for ICD-10 mapping.')
    lines.append('')
    for _, row in df.iterrows():
        cause = row[COL_CAUSE]
        code  = row[COL_CODE]
        lines.append(f"- {code}: {cause}")
    lines.append('"""')
    snippet = "\n".join(lines)

    # Write snippet to a file
    OUTPUT_SNIPPET.write_text(snippet, encoding="utf-8")
    print(f"Wrote snippet to: {OUTPUT_SNIPPET.resolve()}")

if __name__ == "__main__":
    main()
