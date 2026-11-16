"""
baseline_cleaning.py
Deterministic non-LLM baseline cleaning pipeline.
Applies rule-based transformations:
1. Drops exact duplicate rows.
2. Trims leading and trailing whitespace in all text columns.
3. In object columns where >= 90% of non-null values parse as numbers
   (after removing '$', ',' and mapping 'N/A', 'NA', '' to missing),
   converts the column to numeric.
4. Title-cases values of low-cardinality text columns (< 5% of rows and <= 50 unique values),
   never touching ID-like columns.
No imputation, no row dropping apart from exact duplicates, no clipping.
"""

from typing import Any
import numpy as np
import pandas as pd


MISSING_STRINGS = {"N/A", "NA", "NAN", "NULL", "NONE", ""}


def _is_id_like(col_name: str) -> bool:
    """Checks whether a column name represents an identifier."""
    c = str(col_name).strip().lower()
    return c == "id" or c.endswith("_id") or c.endswith("id") or "identifier" in c


def baseline_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Cleans a DataFrame deterministically without using an LLM.

    Args:
        df: Input DataFrame.

    Returns:
        (cleaned_df, log_lines): A tuple of the cleaned DataFrame and a list
        describing all transformation actions taken.
    """
    cleaned = df.copy(deep=True)
    log_lines: list[str] = []

    # 1. Drop exact duplicate rows
    n_before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    n_dups = n_before - len(cleaned)
    if n_dups > 0:
        log_lines.append(f"Dropped {n_dups} exact duplicate row(s) ({n_before} -> {len(cleaned)})")

    # 2. Strip whitespace in text columns
    text_cols_stripped = []
    for col in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[col]) or pd.api.types.is_string_dtype(cleaned[col]):
            mask = cleaned[col].notna()
            cleaned.loc[mask, col] = cleaned.loc[mask, col].astype(str).str.strip()
            text_cols_stripped.append(col)
    if text_cols_stripped:
        log_lines.append(f"Stripped whitespace in {len(text_cols_stripped)} text column(s)")

    # 3. Numeric conversion for string-polluted numeric columns
    # In object columns where >=90% of non-null values parse as numbers after
    # removing '$', ',' and mapping 'N/A', 'NA', '' to missing, convert to numeric.
    for col in list(cleaned.columns):
        if pd.api.types.is_object_dtype(cleaned[col]) or pd.api.types.is_string_dtype(cleaned[col]):
            non_null = cleaned[col].dropna()
            if len(non_null) == 0:
                continue

            s = non_null.astype(str).str.strip()
            is_na_str = s.str.upper().isin(MISSING_STRINGS)
            cand = s[~is_na_str].str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
            parsed = pd.to_numeric(cand, errors="coerce")

            valid_parsed = parsed.notna().sum()
            na_mapped = is_na_str.sum()
            total_convertible = valid_parsed + na_mapped

            if valid_parsed > 0 and (total_convertible / len(non_null)) >= 0.90:
                full_s = cleaned[col].astype(str).str.strip()
                full_missing = full_s.str.upper().isin(MISSING_STRINGS) | cleaned[col].isna()
                full_cleaned_str = full_s.str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
                num_col = pd.to_numeric(full_cleaned_str, errors="coerce")
                num_col[full_missing] = np.nan
                cleaned[col] = num_col
                pct = (total_convertible / len(non_null)) * 100
                log_lines.append(
                    f"Converted column '{col}' to numeric ({total_convertible}/{len(non_null)} parsed as numbers, {pct:.1f}%)"
                )

    # 4. Title-case low cardinality text columns (< 5% of rows and <= 50 unique values, never ID-like)
    n_rows = len(cleaned)
    for col in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[col]) or pd.api.types.is_string_dtype(cleaned[col]):
            if _is_id_like(col):
                continue

            n_unique = int(cleaned[col].dropna().nunique())
            if n_rows > 0 and (n_unique / n_rows < 0.05) and (n_unique <= 50):
                mask = cleaned[col].notna()
                cleaned.loc[mask, col] = cleaned.loc[mask, col].astype(str).str.title()
                pct = (n_unique / n_rows) * 100
                log_lines.append(
                    f"Title-cased low-cardinality text column '{col}' ({n_unique} unique values, {pct:.1f}% of rows)"
                )

    return cleaned, log_lines
