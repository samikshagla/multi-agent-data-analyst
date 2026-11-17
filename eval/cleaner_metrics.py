"""
cleaner_metrics.py
Evaluation metrics for Data Cleaning:
Compares a cleaned dataset against the original clean truth and corruption manifest.
Calculates:
1. Row preservation and duplicate removal rate
2. Missing value handling and rows sacrificed
3. Inconsistent text normalization share
4. Negative value handling in non-negative columns
5. Numbers-stored-as-text recovery share
6. Collateral damage (unintended change rate on uncorrupted cells)
7. Statistical mean drift on numeric columns
All functions are pure Python and pandas without LLM dependencies.
"""

from typing import Any
import numpy as np
import pandas as pd


def _is_numeric_value(val: Any) -> bool:
    """Checks if a value is a numeric type (excluding booleans)."""
    if isinstance(val, bool) or isinstance(val, (np.bool_,)):
        return False
    return isinstance(val, (int, float, np.number)) and np.isfinite(val)


def _cells_differ(v_clean: Any, v_cleaned: Any) -> bool:
    """
    Determines whether a cell value was unintentionally modified.
    Treated as UNCHANGED:
    - Both missing / NaN
    - Text differences only in casing or surrounding whitespace
    - Numeric values within 1e-6 relative difference
    - Datatype changes preserving values (e.g., int -> float, string -> category)
    """
    is_clean_na = bool(pd.api.types.is_scalar(v_clean) and pd.isna(v_clean))
    is_cleaned_na = bool(pd.api.types.is_scalar(v_cleaned) and pd.isna(v_cleaned))

    if is_clean_na and is_cleaned_na:
        return False
    if is_clean_na != is_cleaned_na:
        return True

    # Numeric comparison
    if _is_numeric_value(v_clean) and _is_numeric_value(v_cleaned):
        f_clean = float(v_clean)
        f_cleaned = float(v_cleaned)
        denom = max(abs(f_clean), abs(f_cleaned))
        if denom == 0:
            return False
        rel_diff = abs(f_clean - f_cleaned) / denom
        return rel_diff >= 1e-6

    # Boolean comparison
    if isinstance(v_clean, (bool, np.bool_)) and isinstance(v_cleaned, (bool, np.bool_)):
        return bool(v_clean) != bool(v_cleaned)
    if isinstance(v_clean, (bool, np.bool_)) or isinstance(v_cleaned, (bool, np.bool_)):
        return True

    # Text comparison: ignore case and surrounding whitespace
    str_clean = str(v_clean).strip().lower()
    str_cleaned = str(v_cleaned).strip().lower()
    return str_clean != str_cleaned


def cleaner_metrics(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """
    Computes rigorous evaluation metrics comparing cleaned_df to clean_df.

    Args:
        clean_df: Certified clean pre-corruption DataFrame (ground truth).
        dirty_df: Corrupted DataFrame produced by make_dirty_dataset.
        cleaned_df: Cleaned DataFrame produced by baseline or cleaner agent.
        manifest: Audit manifest dictionary from make_dirty_dataset.

    Returns:
        JSON-serializable dictionary of cleaning metrics.
    """
    # 0. Identify and validate key column to align rows
    key_col = manifest.get("key_column")
    if not key_col or key_col not in clean_df.columns:
        candidates = [
            c for c in clean_df.columns
            if clean_df[c].nunique(dropna=True) == len(clean_df) and clean_df[c].notna().all()
        ]
        if candidates:
            id_cand = [c for c in candidates if c.lower() == "id" or c.lower().endswith("_id") or "id" in c.lower()]
            key_col = id_cand[0] if id_cand else candidates[0]
        else:
            key_col = clean_df.columns[0]

    # Validate that key column in clean_df is complete and unique
    if key_col not in clean_df.columns:
        raise ValueError(f"Key column '{key_col}' not found in clean_df.")
    if clean_df[key_col].isna().any():
        raise ValueError(f"Key column '{key_col}' in clean_df contains null values.")
    if clean_df[key_col].nunique(dropna=False) != len(clean_df):
        raise ValueError(
            f"Key column '{key_col}' in clean_df is not unique "
            f"({clean_df[key_col].nunique()} unique vs {len(clean_df)} rows)."
        )

    has_key_in_cleaned = (key_col in cleaned_df.columns)

    # 1. Row preservation, duplicates, and identifier integrity
    expected_rows = int(len(clean_df))
    cleaned_rows = int(len(cleaned_df))

    # Exact duplicate rows in cleaned_df and dirty_df
    duplicate_rows_remaining = int(cleaned_df.duplicated().sum())
    dirty_exact_dups = int(dirty_df.duplicated().sum())
    if dirty_exact_dups > 0:
        raw_rate = (dirty_exact_dups - duplicate_rows_remaining) / dirty_exact_dups
        duplicate_removal_rate = float(np.clip(raw_rate, 0.0, 1.0))
    else:
        duplicate_removal_rate = 1.0

    # Identifier integrity:
    # - key_nulls_in_cleaned: null keys in cleaned_df
    # - key_duplicates_in_cleaned: duplicated NON-NULL keys in cleaned_df
    # Treat rows with null key as lost
    if has_key_in_cleaned:
        key_nulls_in_cleaned = int(cleaned_df[key_col].isna().sum())
        non_null_keys = cleaned_df[key_col].dropna()
        key_duplicates_in_cleaned = int(non_null_keys.duplicated().sum())
        clean_keys = set(clean_df[key_col])
        cleaned_valid_keys = set(non_null_keys)
        valid_rows_lost = int(len(clean_keys - cleaned_valid_keys))
    else:
        key_nulls_in_cleaned = cleaned_rows
        key_duplicates_in_cleaned = 0
        valid_rows_lost = expected_rows

    rows_metric = {
        "expected_rows": expected_rows,
        "cleaned_rows": cleaned_rows,
        "valid_rows_lost": valid_rows_lost,
        "duplicate_rows_remaining": duplicate_rows_remaining,
        "duplicate_removal_rate": duplicate_removal_rate,
        "key_nulls_in_cleaned": key_nulls_in_cleaned,
        "key_duplicates_in_cleaned": key_duplicates_in_cleaned,
    }

    # Build key-indexed lookup for cleaned_df (only valid, non-null keys)
    cleaned_by_key: dict[Any, pd.Series] = {}
    if has_key_in_cleaned:
        cleaned_valid = cleaned_df.dropna(subset=[key_col]).drop_duplicates(subset=[key_col])
        for _, row in cleaned_valid.iterrows():
            cleaned_by_key[row[key_col]] = row

    # 2. Missing values handling
    missing_metrics: dict[str, Any] = {}
    missing_manifest = manifest.get("missing_values") or {}

    for col, info in missing_manifest.items():
        injected_count = int(info.get("count", 0))
        indices = info.get("indices", [])

        filled_count = 0
        row_removed_count = 0
        still_missing_count = 0

        for idx in indices:
            row_key = clean_df.iloc[idx][key_col]
            if row_key not in cleaned_by_key:
                # Key is absent from cleaned_df (row removed or key corrupted to null)
                row_removed_count += 1
            else:
                c_row = cleaned_by_key[row_key]
                if col in c_row and not pd.isna(c_row[col]):
                    filled_count += 1
                else:
                    still_missing_count += 1

        denom = injected_count if injected_count > 0 else 1
        filled_rate = float(filled_count / denom)
        row_removed_rate = float(row_removed_count / denom)
        still_missing_rate = float(still_missing_count / denom)
        rate_handled = float((filled_count + row_removed_count) / denom)

        missing_metrics[col] = {
            "injected_count": injected_count,
            "filled_count": filled_count,
            "filled_rate": filled_rate,
            "row_removed_count": row_removed_count,
            "row_removed_rate": row_removed_rate,
            "still_missing_count": still_missing_count,
            "still_missing_rate": still_missing_rate,
            # Backward compatibility aliases
            "remaining_missing": still_missing_count,
            "valid_rows_removed": row_removed_count,
            "rate_handled": rate_handled,
        }

    # 3. Inconsistent text normalization
    inc_info = manifest.get("inconsistent_text")
    if inc_info:
        inc_col = inc_info.get("column", "")
        inc_indices = inc_info.get("indices", [])
        true_distinct = int(clean_df[inc_col].dropna().nunique()) if inc_col in clean_df.columns else 0
        cleaned_distinct = int(cleaned_df[inc_col].dropna().nunique()) if inc_col in cleaned_df.columns else 0

        normalized_count = 0
        rows_removed_count = 0
        not_normalized_count = 0

        for idx in inc_indices:
            row_key = clean_df.iloc[idx][key_col]
            if row_key not in cleaned_by_key:
                rows_removed_count += 1
            else:
                c_row = cleaned_by_key[row_key]
                true_label = str(clean_df.iloc[idx][inc_col]).strip().lower()
                if inc_col in c_row:
                    cleaned_val = str(c_row[inc_col]).strip().lower()
                    if cleaned_val == true_label:
                        normalized_count += 1
                    else:
                        not_normalized_count += 1
                else:
                    not_normalized_count += 1

        surviving_count = len(inc_indices) - rows_removed_count
        share_norm = float(normalized_count / surviving_count) if surviving_count > 0 else 1.0

        inconsistent_text_metric: dict[str, Any] | None = {
            "column": inc_col,
            "true_distinct_labels": true_distinct,
            "cleaned_distinct_labels": cleaned_distinct,
            "injected_count": len(inc_indices),
            "rows_removed": rows_removed_count,
            "normalized_count": normalized_count,
            "not_normalized_count": not_normalized_count,
            "surviving_count": surviving_count,
            "share_normalized": share_norm,
        }
    else:
        inconsistent_text_metric = None

    # 4. Negative values handling
    imp_info = manifest.get("impossible_values")
    if imp_info:
        imp_col = imp_info.get("column", "")
        imp_indices = imp_info.get("indices", [])
        injected_count = len(imp_indices)
        remaining_negs = 0
        if imp_col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[imp_col]):
            remaining_negs = int((cleaned_df[imp_col].dropna() < 0).sum())

        negative_values_metric: dict[str, Any] | None = {
            "column": imp_col,
            "injected_count": injected_count,
            "remaining_negatives": remaining_negs,
        }
    else:
        negative_values_metric = None

    # 5. Numbers stored as text
    num_text_info = manifest.get("numbers_stored_as_text")
    if num_text_info:
        nt_col = num_text_info.get("column", "")
        nt_indices = num_text_info.get("indices", [])
        is_num = bool(nt_col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[nt_col]))

        recovered_count = 0
        rows_removed_count = 0
        unrecovered_count = 0

        for pos, idx in enumerate(nt_indices):
            row_key = clean_df.iloc[idx][key_col]
            true_val = clean_df.iloc[idx][nt_col]
            mod_choice = pos % 3

            if row_key not in cleaned_by_key:
                rows_removed_count += 1
            else:
                c_row = cleaned_by_key[row_key]
                c_val = c_row.get(nt_col)

                if mod_choice == 1:
                    # Formatted currency like "$12.50"
                    if _is_numeric_value(c_val) and _is_numeric_value(true_val):
                        f_clean = float(true_val)
                        f_cleaned = float(c_val)
                        denom = max(abs(f_clean), abs(f_cleaned), 1e-9)
                        rel_diff = abs(f_clean - f_cleaned) / denom
                        if rel_diff <= 0.01:
                            recovered_count += 1
                        else:
                            unrecovered_count += 1
                    else:
                        unrecovered_count += 1
                else:
                    # Missing indicator like "N/A" or "null"
                    # Counted as handled if missing or imputed with a number
                    if pd.isna(c_val) or _is_numeric_value(c_val):
                        recovered_count += 1
                    else:
                        unrecovered_count += 1

        surviving_count = len(nt_indices) - rows_removed_count
        rec_share = float(recovered_count / surviving_count) if surviving_count > 0 else 1.0

        numbers_as_text_metric: dict[str, Any] | None = {
            "column": nt_col,
            "is_numeric": is_num,
            "injected_count": len(nt_indices),
            "rows_removed": rows_removed_count,
            "recovered_count": recovered_count,
            "unrecovered_count": unrecovered_count,
            "surviving_count": surviving_count,
            "recovery_share": rec_share,
        }
    else:
        numbers_as_text_metric = None

    # 6. Collateral Damage: Unintended changes on uncorrupted cells
    # Collect all (key, col) pairs that were injected with errors
    injected_cells: set[tuple[Any, str]] = set()

    for col, info in missing_manifest.items():
        for idx in info.get("indices", []):
            injected_cells.add((clean_df.iloc[idx][key_col], col))

    if inc_info:
        inc_c = inc_info.get("column", "")
        for idx in inc_info.get("indices", []):
            injected_cells.add((clean_df.iloc[idx][key_col], inc_c))

    if imp_info:
        imp_c = imp_info.get("column", "")
        for idx in imp_info.get("indices", []):
            injected_cells.add((clean_df.iloc[idx][key_col], imp_c))

    if num_text_info:
        nt_c = num_text_info.get("column", "")
        for idx in num_text_info.get("indices", []):
            injected_cells.add((clean_df.iloc[idx][key_col], nt_c))

    # Evaluate all uncorrupted cells among common rows
    total_uninjected = 0
    unintended_changes = 0
    changed_by_col: dict[str, int] = {}

    common_keys = [k for k in clean_df[key_col] if k in cleaned_by_key]

    # Pre-index clean_df by key for fast cell evaluation (keeping key column in columns)
    clean_indexed = clean_df.drop_duplicates(subset=[key_col]).set_index(key_col, drop=False)
    eval_cols = [c for c in clean_df.columns if c in cleaned_df.columns]

    for k in common_keys:
        clean_row = clean_indexed.loc[k]
        cleaned_row = cleaned_by_key[k]

        for col in eval_cols:
            if (k, col) in injected_cells:
                continue

            total_uninjected += 1
            v_orig = clean_row[col]
            v_clean = cleaned_row[col]

            if _cells_differ(v_orig, v_clean):
                unintended_changes += 1
                changed_by_col[col] = changed_by_col.get(col, 0) + 1

    # Any columns completely dropped from cleaned_df count as unintended changes
    missing_cols = [c for c in clean_df.columns if c not in cleaned_df.columns]
    for col in missing_cols:
        for k in common_keys:
            if (k, col) not in injected_cells:
                total_uninjected += 1
                unintended_changes += 1
                changed_by_col[col] = changed_by_col.get(col, 0) + 1

    unintended_change_rate = float(unintended_changes / total_uninjected) if total_uninjected > 0 else 0.0

    sorted_changed = sorted(changed_by_col.items(), key=lambda x: x[1], reverse=True)
    top_5_columns = [{"column": c, "changed_cells": cnt} for c, cnt in sorted_changed[:5] if cnt > 0]

    damage_metric = {
        "total_uninjected_cells": total_uninjected,
        "unintentionally_changed_cells": unintended_changes,
        "unintended_change_rate": unintended_change_rate,
        "top_changed_columns": top_5_columns,
    }

    # 7. Statistical mean drift
    drift_metrics: dict[str, Any] = {}
    true_means = manifest.get("true_means") or {}

    for col, true_mean in true_means.items():
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            series = cleaned_df[col].dropna()
            if len(series) > 0:
                c_mean = float(series.mean())
                denom = abs(true_mean) if abs(true_mean) > 0 else 1.0
                rel_drift = float(abs(c_mean - true_mean) / denom)
                drift_metrics[col] = {
                    "true_mean": true_mean,
                    "cleaned_mean": c_mean,
                    "relative_drift": rel_drift,
                }
            else:
                drift_metrics[col] = {
                    "true_mean": true_mean,
                    "cleaned_mean": None,
                    "relative_drift": None,
                }
        else:
            drift_metrics[col] = {
                "true_mean": true_mean,
                "cleaned_mean": None,
                "relative_drift": None,
            }

    return {
        "rows": rows_metric,
        "identifier_integrity": {
            "key_column": key_col,
            "key_nulls_in_cleaned": key_nulls_in_cleaned,
            "key_duplicates_in_cleaned": key_duplicates_in_cleaned,
        },
        "missing_values": missing_metrics,
        "inconsistent_text": inconsistent_text_metric,
        "negative_values": negative_values_metric,
        "numbers_stored_as_text": numbers_as_text_metric,
        "damage": damage_metric,
        "statistical_drift": drift_metrics,
    }
