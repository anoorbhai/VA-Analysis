import pandas as pd
from pathlib import Path
from functools import reduce

# --------- User settings ----------
FOLDER = Path("/dataA/madiva/va/VA")          # folder containing your CSVs
OUTPUT = Path("VA_merged_cases.csv")  # output file path
MAKE_PRESENCE_FLAGS = True   # add per-file presence boolean columns
# ----------------------------------

def read_csv_with_prefix(fp: Path, make_presence: bool = True) -> pd.DataFrame:
    """
    Reads a CSV, ensures AnonId is string, and prefixes all columns (except AnonId)
    with the file stem (e.g., 'ChildSymptoms__Fever').
    Optionally adds a presence flag for that file.
    """
    stem = fp.stem  # e.g., 'ChildSymptoms' from 'ChildSymptoms.csv'

    # Read CSV; engine='python' is more forgiving with odd delimiters/quotes
    df = pd.read_csv(
        fp,
        dtype={"AnonId": "string"},   # keep IDs safe (no numeric coercion)
        na_values=["", "NA", "NaN"],  # common NA tokens
        engine="python"
    )

    if "AnonId" not in df.columns:
        raise ValueError(f"File {fp.name} has no 'AnonId' column.")

    # Optional: handle duplicates within a file (same AnonId repeated)
    # If you know one row per AnonId is guaranteed, you can remove this block.
    # Here we keep the *first* row for each AnonId.
    if df["AnonId"].duplicated().any():
        df = (df
              .drop_duplicates(subset=["AnonId"], keep="first")
              .reset_index(drop=True))

    # Prefix all non-ID columns with <file>__
    new_cols = {}
    for c in df.columns:
        if c != "AnonId":
            new_cols[c] = f"{stem}__{c}"
    df = df.rename(columns=new_cols)

    # Add presence flag (True if the row exists in this file for that AnonId)
    if make_presence:
        df[f"{stem}__present"] = True

    return df

def outer_join_on_anonid(dfs):
    """
    Full outer-join a list of DataFrames on AnonId.
    """
    if not dfs:
        # no files found; return empty frame with just AnonId
        return pd.DataFrame(columns=["AnonId"])

    merged = reduce(lambda left, right: pd.merge(left, right, on="AnonId", how="outer"),
                    dfs)
    return merged

def main():
    csv_files = sorted(FOLDER.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {FOLDER.resolve()}")

    dfs = [read_csv_with_prefix(fp, make_presence=MAKE_PRESENCE_FLAGS) for fp in csv_files]
    merged = outer_join_on_anonid(dfs)

    # If presence flags were added, fill missing flags with False
    if MAKE_PRESENCE_FLAGS:
        presence_cols = [c for c in merged.columns if c.endswith("__present")]
        for c in presence_cols:
            merged[c] = merged[c].fillna(False)

    # Save
    merged.to_csv(OUTPUT, index=False)
    print(f"Done! Wrote merged file to: {OUTPUT.resolve()}")
    print(f"Rows (unique AnonId): {merged.shape[0]}, Columns: {merged.shape[1]}")

if __name__ == "__main__":
    main()
