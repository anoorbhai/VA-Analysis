import pandas as pd
from pathlib import Path

CSV_PATH = Path("/dataA/madiva/va/student/madiva_va_clinician_COD_20250926.csv") 
OUTPUT_SNIPPET = Path("/spaces/25G05/Aaliyah/Clinical_COD_pairs.txt")                

MAX_ROWS = None

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
    # Exclude entries with "<no data>"
    df = df[~df[COL_CAUSE].str.contains("<no data>", case=False, na=False)]
    df = df[~df[COL_CODE].str.contains("<no data>", case=False, na=False)]
    # Exclude entries with "NA" in ICD10 code
    df = df[df[COL_CODE] != "NA"]
    df = df.drop_duplicates(subset=[COL_CAUSE, COL_CODE]).reset_index(drop=True)

    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS)

    if df.empty:
        raise ValueError("No (CauseofDeath, ICD10Code) pairs found after cleaning.")

    lines = []
    lines.append('Use the following physician-coded pairs as reference for ICD-10 mapping.')
    lines.append('')
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        cause = row[COL_CAUSE]
        code  = row[COL_CODE]
        lines.append(f"{i}. {code}: {cause}")
    lines.append('"""')
    snippet = "\n".join(lines)

    OUTPUT_SNIPPET.write_text(snippet, encoding="utf-8")
    print(f"Wrote snippet to: {OUTPUT_SNIPPET.resolve()}")

if __name__ == "__main__":
    main()

