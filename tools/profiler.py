"""
profiler.py
Data Profiling Tool: Computes summary statistics, missing values, column data types,
cardinality, and distribution metrics on pandas DataFrames, returning a JSON-serializable
profile dictionary and a formatted prompt-ready text block for LLMs.
"""

from typing import Any
import warnings
import numpy as np
import pandas as pd


def _to_json_safe(obj: Any) -> Any:
    """
    Recursively converts pandas/numpy data structures and scalars into
    standard JSON-serializable Python native types (int, float, bool, str, None).
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return obj.isoformat() if hasattr(obj, "isoformat") else str(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _is_possible_date_series(series: pd.Series) -> bool:
    """
    Determines whether an object/string series represents dates by inspecting
    a sample of non-null values.
    """
    non_nulls = series.dropna()
    if len(non_nulls) == 0:
        return False

    # Take up to 50 non-null values for efficient checking
    sample = non_nulls.head(50)
    str_sample = sample.astype(str).str.strip()

    # Skip if purely numeric with <= 4 digits (avoids false-flagging simple integers)
    if str_sample.str.fullmatch(r"^\d{1,4}$").all():
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(str_sample, errors="coerce")
            valid = parsed.notna()
            # If at least 70% of sample values parse into dates
            if valid.mean() >= 0.70:
                years = parsed[valid].dt.year
                # Verify plausible human calendar range
                if ((years >= 1900) & (years <= 2100)).mean() >= 0.70:
                    return True
        except Exception:
            return False

    return False


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generates a compact, JSON-serializable profile of a pandas DataFrame.

    Profile includes:
    - n_rows, n_cols
    - columns: per-column dtype, null percentage, and unique value count
    - numeric_columns: min, max, mean for numeric columns
    - categorical_columns: top 5 most frequent values for object/category columns
    - possible_date_columns: object columns where most values parse as dates
    - duplicate_rows: count of duplicate rows
    - sample_rows: first 5 rows as a list of dicts

    Args:
        df (pd.DataFrame): Input dataframe to profile.

    Returns:
        dict[str, Any]: Completely JSON-serializable dictionary profiling the dataset.
    """
    n_rows = int(len(df))
    n_cols = int(len(df.columns))
    duplicate_rows = int(df.duplicated().sum()) if n_rows > 0 else 0

    columns_meta: dict[str, dict[str, Any]] = {}
    numeric_summary: dict[str, dict[str, Any]] = {}
    categorical_summary: dict[str, dict[str, int]] = {}
    possible_date_columns: list[str] = []

    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)

        # Null percentage calculation
        if n_rows > 0:
            null_pct = round(float(series.isna().mean() * 100), 2)
            unique_count = int(series.nunique(dropna=True))
        else:
            null_pct = 0.0
            unique_count = 0

        columns_meta[str(col)] = {
            "dtype": dtype_str,
            "null_percentage": null_pct,
            "unique_values": unique_count,
        }

        # Numeric column statistics
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            valid_vals = series.dropna()
            if len(valid_vals) > 0:
                min_val = float(valid_vals.min()) if not pd.isna(valid_vals.min()) else None
                max_val = float(valid_vals.max()) if not pd.isna(valid_vals.max()) else None
                mean_val = round(float(valid_vals.mean()), 4) if not pd.isna(valid_vals.mean()) else None
            else:
                min_val, max_val, mean_val = None, None, None

            numeric_summary[str(col)] = {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
            }

        # Categorical / Object column top 5 values
        is_categorical = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )
        if is_categorical:
            # Check for possible date columns
            if _is_possible_date_series(series):
                possible_date_columns.append(str(col))

            top5 = series.value_counts(dropna=False).head(5)
            top5_dict: dict[str, int] = {}
            for val, count in top5.items():
                label = "<NA>" if pd.isna(val) else str(val)
                top5_dict[label] = int(count)
            categorical_summary[str(col)] = top5_dict

    # Sample rows (first 5)
    sample_records = df.head(5).to_dict(orient="records")
    sanitized_sample = [_to_json_safe(row) for row in sample_records]

    profile: dict[str, Any] = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "duplicate_rows": duplicate_rows,
        "columns": columns_meta,
        "numeric_columns": numeric_summary,
        "categorical_columns": categorical_summary,
        "possible_date_columns": possible_date_columns,
        "sample_rows": sanitized_sample,
    }

    # Ensure total JSON serializability
    return _to_json_safe(profile)


def profile_to_text(profile: dict[str, Any]) -> str:
    """
    Converts a dataset profile dictionary into a concise, formatted text
    block ready for inclusion in an LLM prompt.

    Args:
        profile (dict[str, Any]): Profile dictionary produced by profile_dataframe.

    Returns:
        str: Human-readable markdown-like text summary.
    """
    lines: list[str] = []

    # 1. Dataset Overview
    lines.append("Dataset Overview:")
    lines.append(f"- Rows: {profile.get('n_rows', 0):,}")
    lines.append(f"- Columns: {profile.get('n_cols', 0)}")
    lines.append(f"- Duplicate Rows: {profile.get('duplicate_rows', 0):,}")
    lines.append("")

    # 2. Columns Profile
    columns = profile.get("columns", {})
    if columns:
        lines.append("Columns Profile:")
        for col, meta in columns.items():
            lines.append(
                f"  - {col}: dtype={meta.get('dtype')}, "
                f"nulls={meta.get('null_percentage')}%, "
                f"unique_values={meta.get('unique_values')}"
            )
        lines.append("")

    # 3. Numeric Summary
    numeric_cols = profile.get("numeric_columns", {})
    if numeric_cols:
        lines.append("Numeric Columns Summary:")
        for col, stats in numeric_cols.items():
            min_v = stats.get("min")
            max_v = stats.get("max")
            mean_v = stats.get("mean")
            min_str = f"{min_v:.2f}" if isinstance(min_v, (int, float)) else str(min_v)
            max_str = f"{max_v:.2f}" if isinstance(max_v, (int, float)) else str(max_v)
            mean_str = f"{mean_v:.2f}" if isinstance(mean_v, (int, float)) else str(mean_v)
            lines.append(f"  - {col}: min={min_str}, max={max_str}, mean={mean_str}")
        lines.append("")

    # 4. Categorical Top Values
    cat_cols = profile.get("categorical_columns", {})
    if cat_cols:
        lines.append("Categorical Columns (Top Values):")
        for col, top_vals in cat_cols.items():
            vals_preview = ", ".join(f"'{k}': {v}" for k, v in top_vals.items())
            lines.append(f"  - {col}: {vals_preview}")
        lines.append("")

    # 5. Possible Date Columns
    date_cols = profile.get("possible_date_columns", [])
    if date_cols:
        lines.append(f"Possible Date Columns: {', '.join(date_cols)}")
    else:
        lines.append("Possible Date Columns: None detected")
    lines.append("")

    # 6. Sample Rows
    sample_rows = profile.get("sample_rows", [])
    if sample_rows:
        lines.append(f"Sample Rows (First {len(sample_rows)} rows):")
        for i, row in enumerate(sample_rows, start=1):
            short_row = {
                k: (str(v)[:50] + "..." if isinstance(v, str) and len(str(v)) > 50 else v)
                for k, v in row.items()
            }
            lines.append(f"  Row {i}: {short_row}")

    return "\n".join(lines).strip()
