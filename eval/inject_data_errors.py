"""
inject_data_errors.py
Synthetic Data Corruption: Injects deterministic data quality issues (duplicates,
missing values, inconsistent text, negative values, string-contaminated numbers)
into a dataset, recording an exact audit manifest with true pre-corruption metrics.
"""

from typing import Any
import numpy as np
import pandas as pd


def make_dirty_dataset(
    df: pd.DataFrame,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Creates a deterministic dirty copy of df without modifying the input df.
    Picks columns based on dtypes and injects:
    1. Exact duplicate rows (3% of rows inserted at random positions)
    2. Missing values (3% in three numeric columns and 3% in two text columns)
    3. Inconsistent text (5% of rows in one text column: case changes, whitespace)
    4. Impossible values (0.5% negative values in a non-negative numeric column)
    5. Numbers stored as text (1% of values in a numeric column converted to strings)

    Returns:
        (dirty_df, manifest)
        manifest contains clean row count, column names, true means before corruption,
        and exact counts and row indices of all injected corruptions.
    """
    # 0. Never modify the input df
    clean_df = df.copy(deep=True)
    clean_row_count = len(clean_df)
    clean_columns = list(clean_df.columns)

    if clean_row_count == 0:
        return clean_df, {
            "clean_row_count": 0,
            "clean_columns": clean_columns,
            "true_means": {},
        }

    rng = np.random.default_rng(seed)

    # 1. Classify columns by dtype
    numeric_cols = [
        c for c in clean_df.columns
        if pd.api.types.is_numeric_dtype(clean_df[c]) and not pd.api.types.is_bool_dtype(clean_df[c])
    ]
    text_cols = [
        c for c in clean_df.columns
        if pd.api.types.is_string_dtype(clean_df[c]) or pd.api.types.is_object_dtype(clean_df[c])
    ]

    # Select columns for corruptions
    num_cols_for_missing = numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    text_cols_for_missing = text_cols[:2] if len(text_cols) >= 2 else text_cols

    inc_text_col = text_cols[2] if len(text_cols) >= 3 else (text_cols[0] if text_cols else None)

    # Non-negative numeric column for impossible values
    non_neg_candidates = [
        c for c in numeric_cols
        if (clean_df[c].dropna() >= 0).all() and len(clean_df[c].dropna()) > 0
    ]
    impossible_num_col = non_neg_candidates[-1] if non_neg_candidates else (numeric_cols[0] if numeric_cols else None)

    # Numeric column for numbers stored as text
    text_num_candidates = [
        c for c in numeric_cols
        if c != impossible_num_col
    ]
    numbers_as_text_col = text_num_candidates[0] if text_num_candidates else (numeric_cols[0] if numeric_cols else None)

    # Record true means before corruption for all affected numeric columns
    affected_numeric_cols = set(num_cols_for_missing)
    if impossible_num_col:
        affected_numeric_cols.add(impossible_num_col)
    if numbers_as_text_col:
        affected_numeric_cols.add(numbers_as_text_col)

    true_means = {
        col: float(clean_df[col].mean())
        for col in sorted(affected_numeric_cols)
    }

    dirty = clean_df.copy(deep=True)
    manifest: dict[str, Any] = {
        "clean_row_count": clean_row_count,
        "clean_columns": clean_columns,
        "true_means": true_means,
        "missing_values": {},
        "inconsistent_text": None,
        "impossible_values": None,
        "numbers_stored_as_text": None,
        "duplicate_rows": None,
    }

    # 2. Missing values: 3% in three numeric columns and 3% in two text columns
    for col in num_cols_for_missing:
        n_miss = max(1, int(clean_row_count * 0.03))
        indices = sorted([int(x) for x in rng.choice(clean_row_count, size=min(n_miss, clean_row_count), replace=False)])
        dirty.loc[indices, col] = np.nan
        manifest["missing_values"][col] = {
            "count": len(indices),
            "indices": indices,
            "dtype": "numeric",
        }

    for col in text_cols_for_missing:
        n_miss = max(1, int(clean_row_count * 0.03))
        indices = sorted([int(x) for x in rng.choice(clean_row_count, size=min(n_miss, clean_row_count), replace=False)])
        dirty.loc[indices, col] = None
        manifest["missing_values"][col] = {
            "count": len(indices),
            "indices": indices,
            "dtype": "text",
        }

    # 3. Inconsistent text: 5% of rows in one text column
    if inc_text_col:
        n_inc = max(1, int(clean_row_count * 0.05))
        indices = sorted([int(x) for x in rng.choice(clean_row_count, size=min(n_inc, clean_row_count), replace=False)])
        for pos, idx in enumerate(indices):
            val = str(dirty.loc[idx, inc_text_col])
            mod_choice = pos % 4
            if mod_choice == 0:
                dirty.loc[idx, inc_text_col] = f"  {val}"
            elif mod_choice == 1:
                dirty.loc[idx, inc_text_col] = f"{val}  "
            elif mod_choice == 2:
                dirty.loc[idx, inc_text_col] = f" {val.upper()} "
            else:
                dirty.loc[idx, inc_text_col] = val.lower()
        manifest["inconsistent_text"] = {
            "column": inc_text_col,
            "count": len(indices),
            "indices": indices,
        }

    # 4. Impossible values: 0.5% negative values in one non-negative numeric column
    if impossible_num_col:
        n_imp = max(1, int(clean_row_count * 0.005))
        indices = sorted([int(x) for x in rng.choice(clean_row_count, size=min(n_imp, clean_row_count), replace=False)])
        for idx in indices:
            val = dirty.loc[idx, impossible_num_col]
            if pd.isna(val) or val == 0:
                dirty.loc[idx, impossible_num_col] = -15.0
            else:
                dirty.loc[idx, impossible_num_col] = -abs(float(val))
        manifest["impossible_values"] = {
            "column": impossible_num_col,
            "count": len(indices),
            "indices": indices,
        }

    # 5. Numbers stored as text: 1% of values in one numeric column turned into strings
    if numbers_as_text_col:
        n_text = max(1, int(clean_row_count * 0.01))
        indices = sorted([int(x) for x in rng.choice(clean_row_count, size=min(n_text, clean_row_count), replace=False)])
        dirty[numbers_as_text_col] = dirty[numbers_as_text_col].astype(object)
        for pos, idx in enumerate(indices):
            val = dirty.loc[idx, numbers_as_text_col]
            mod_choice = pos % 3
            if mod_choice == 0:
                dirty.loc[idx, numbers_as_text_col] = "N/A"
            elif mod_choice == 1:
                dirty.loc[idx, numbers_as_text_col] = f"${float(val):.2f}" if (not pd.isna(val) and isinstance(val, (int, float))) else "$12.50"
            else:
                dirty.loc[idx, numbers_as_text_col] = "null"
        manifest["numbers_stored_as_text"] = {
            "column": numbers_as_text_col,
            "count": len(indices),
            "indices": indices,
        }

    # 6. Exact duplicate rows: copies of 3% of the rows appended at random positions
    n_dups = max(1, int(clean_row_count * 0.03))
    dup_source_indices = sorted([int(x) for x in rng.choice(clean_row_count, size=min(n_dups, clean_row_count), replace=False)])
    dup_rows = dirty.iloc[dup_source_indices].copy()

    # Interleave at random insertion positions
    insert_positions = sorted([int(x) for x in rng.choice(clean_row_count + 1, size=len(dup_rows), replace=False)])
    pieces: list[pd.DataFrame] = []
    injected_indices: list[int] = []
    prev_pos = 0
    curr_offset = 0

    for i, pos in enumerate(insert_positions):
        chunk = dirty.iloc[prev_pos:pos]
        pieces.append(chunk)
        curr_offset += len(chunk)

        injected_indices.append(curr_offset)
        pieces.append(dup_rows.iloc[[i]])
        curr_offset += 1
        prev_pos = pos

    pieces.append(dirty.iloc[prev_pos:])
    dirty = pd.concat(pieces, ignore_index=True)

    manifest["duplicate_rows"] = {
        "count": len(injected_indices),
        "source_indices": dup_source_indices,
        "injected_indices": injected_indices,
    }

    return dirty, manifest
