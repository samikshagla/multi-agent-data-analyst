"""
cleaner.py
Cleaner Agent: Inspects raw uploaded CSV data, requests an intelligent cleaning plan
from Gemini LLM, deterministically applies validated data transformations (no code execution),
enforces safety rules against excessive data loss, and produces a cleaned dataset.
"""

from datetime import datetime, timezone
import json
import os
import re
from typing import Any, Callable
import numpy as np
import pandas as pd

from llm import call_llm, LLMUnavailableError, LLMConfigError
from state import AgentState
from tools.profiler import profile_dataframe, profile_to_text


# ============================================================================
# Action Functions (Deterministic transformations, no code execution)
# ============================================================================

def drop_duplicates(df: pd.DataFrame, column: str = "all", **kwargs) -> tuple[pd.DataFrame, str, bool]:
    """Drops exact duplicate rows across all or specified columns."""
    n_before = len(df)
    if column == "all" or not column:
        new_df = df.drop_duplicates()
    else:
        new_df = df.drop_duplicates(subset=[column])
    n_removed = n_before - len(new_df)
    msg = f"Removed {n_removed} duplicate row(s) ({n_before} -> {len(new_df)})"
    return new_df, msg, True


def strip_whitespace(df: pd.DataFrame, column: Any = "all", **kwargs) -> tuple[pd.DataFrame, str, bool]:
    """Trims leading and trailing whitespace from string/object columns."""
    new_df = df.copy()
    if column == "all" or not column:
        targets = [
            c for c in new_df.columns
            if pd.api.types.is_object_dtype(new_df[c]) or pd.api.types.is_string_dtype(new_df[c])
        ]
    elif isinstance(column, list):
        targets = [c for c in column if c in new_df.columns]
    else:
        targets = [column] if column in new_df.columns else []

    modified_cols: list[str] = []
    for col in targets:
        if pd.api.types.is_object_dtype(new_df[col]) or pd.api.types.is_string_dtype(new_df[col]):
            mask = new_df[col].notna()
            new_df.loc[mask, col] = new_df.loc[mask, col].astype(str).str.strip()
            modified_cols.append(col)

    if len(modified_cols) <= 5:
        names_str = f" ({', '.join(modified_cols)})" if modified_cols else ""
    else:
        names_str = f" ({', '.join(modified_cols[:5])}, and {len(modified_cols) - 5} more)"

    msg = f"Stripped whitespace in {len(modified_cols)} text column(s){names_str}"
    return new_df, msg, True


def standardize_case(
    df: pd.DataFrame, column: str, case: str = "lower", **kwargs
) -> tuple[pd.DataFrame, str, bool]:
    """Converts a text column's casing to lower, upper, or title case."""
    new_df = df.copy()
    mask = new_df[column].notna()
    case_type = str(case).strip().lower()

    if case_type == "upper":
        new_df.loc[mask, column] = new_df.loc[mask, column].astype(str).str.upper()
    elif case_type == "title":
        new_df.loc[mask, column] = new_df.loc[mask, column].astype(str).str.title()
    else:
        new_df.loc[mask, column] = new_df.loc[mask, column].astype(str).str.lower()
        case_type = "lower"

    msg = f"Standardized case of '{column}' to {case_type}"
    return new_df, msg, True


def fill_missing(
    df: pd.DataFrame, column: str, method: str = "median", value: Any = None, **kwargs
) -> tuple[pd.DataFrame, str, bool]:
    """Imputes missing values using mean, median, mode, or a specified constant value."""
    n_missing = int(df[column].isna().sum())
    if n_missing == 0:
        return df, f"No missing values in '{column}' to fill", True

    new_df = df.copy()
    method_str = str(method).strip().lower()

    if method_str == "mean":
        fill_val = new_df[column].mean()
    elif method_str == "median":
        fill_val = new_df[column].median()
    elif method_str == "mode":
        modes = new_df[column].mode()
        fill_val = modes.iloc[0] if len(modes) > 0 else "Unknown"
    elif method_str == "constant":
        fill_val = value if value is not None else "Missing"
    else:
        # Fallback to median for numeric, mode for non-numeric
        if pd.api.types.is_numeric_dtype(new_df[column]):
            fill_val = new_df[column].median()
            method_str = "median"
        else:
            modes = new_df[column].mode()
            fill_val = modes.iloc[0] if len(modes) > 0 else "Unknown"
            method_str = "mode"

    new_df[column] = new_df[column].fillna(fill_val)
    msg = f"Filled {n_missing} missing value(s) in '{column}' with {method_str} ({fill_val})"
    return new_df, msg, True


def drop_missing_rows(df: pd.DataFrame, column: str = "all", **kwargs) -> tuple[pd.DataFrame, str, bool]:
    """Drops rows containing missing values across all or in a specific column."""
    n_before = len(df)
    if column == "all" or not column:
        new_df = df.dropna()
    else:
        new_df = df.dropna(subset=[column])
    n_removed = n_before - len(new_df)
    msg = f"Dropped {n_removed} row(s) with missing values in '{column}' ({n_before} -> {len(new_df)})"
    return new_df, msg, True


def convert_dtype(
    df: pd.DataFrame, column: str, to: str = "numeric", **kwargs
) -> tuple[pd.DataFrame, str, bool]:
    """Converts a column data type to datetime, numeric, or category."""
    new_df = df.copy()
    target_type = str(to).strip().lower()

    if target_type == "datetime":
        new_df[column] = pd.to_datetime(new_df[column], errors="coerce")
    elif target_type == "numeric":
        new_df[column] = pd.to_numeric(new_df[column], errors="coerce")
    elif target_type == "category":
        col_lower = column.lower().strip()
        if col_lower == "id" or col_lower.endswith("_id"):
            return df, f"Skipped converting '{column}' to category: column is an ID column", False

        total_rows = len(df)
        n_unique = int(df[column].nunique(dropna=True))
        if total_rows > 0 and (n_unique / total_rows > 0.05) and (n_unique > 50):
            pct = (n_unique / total_rows) * 100
            return (
                df,
                f"Skipped converting '{column}' to category: high cardinality ({n_unique} unique values, {pct:.1f}% of rows)",
                False,
            )

        new_df[column] = new_df[column].astype("category")
    else:
        return df, f"Unknown target dtype '{target_type}' for column '{column}'", False

    msg = f"Converted column '{column}' to {target_type}"
    return new_df, msg, True


def clip_outliers(
    df: pd.DataFrame, column: str, method: str = "iqr", **kwargs
) -> tuple[pd.DataFrame, str, bool]:
    """Caps extreme values using IQR thresholds without deleting any rows."""
    if not pd.api.types.is_numeric_dtype(df[column]):
        return df, f"Cannot clip outliers: column '{column}' is not numeric", False

    series = df[column].dropna()
    if len(series) == 0:
        return df, f"Cannot clip outliers: column '{column}' contains only nulls", False

    q25 = float(series.quantile(0.25))
    q75 = float(series.quantile(0.75))
    iqr = q75 - q25
    if iqr == 0:
        return df, f"Skipped clipping outliers in '{column}': IQR is 0", False

    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr

    outlier_count = int(((df[column] < lower_bound) | (df[column] > upper_bound)).sum())
    total_rows = len(df)
    if total_rows > 0 and (outlier_count / total_rows > 0.02):
        pct = (outlier_count / total_rows) * 100
        return (
            df,
            f"Skipped clipping outliers in '{column}': {outlier_count} rows ({pct:.1f}%) would be clipped, exceeding 2% threshold",
            False,
        )

    new_df = df.copy()
    new_df[column] = new_df[column].clip(lower=lower_bound, upper=upper_bound)
    msg = (
        f"Clipped {outlier_count} outlier(s) in '{column}' using IQR "
        f"bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
    )
    return new_df, msg, True


def is_identifier_column(series: pd.Series, name: str) -> bool:
    """
    Identifies whether a DataFrame column represents unique record identifiers,
    keys, or UUIDs that should never be mutated by lossy transformations.

    Returns True when:
    - Lower-cased name equals "id", starts with "id_", ends with "_id",
      contains "uuid", "guid", or ends with "_key".
    - Or when the column is text, contains at least 50 rows, and every non-null value is unique.
    """
    col_name = str(name).strip().lower()
    if (
        col_name == "id"
        or col_name.startswith("id_")
        or col_name.endswith("_id")
        or "uuid" in col_name
        or "guid" in col_name
        or col_name.endswith("_key")
    ):
        return True

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        if len(series) >= 50:
            non_null = series.dropna()
            if len(non_null) > 0 and int(non_null.nunique()) == len(non_null):
                return True

    return False


# Registry of allowed cleaning actions
ALLOWED_ACTIONS: dict[str, Callable[..., tuple[pd.DataFrame, str, bool]]] = {
    "drop_duplicates": drop_duplicates,
    "strip_whitespace": strip_whitespace,
    "standardize_case": standardize_case,
    "fill_missing": fill_missing,
    "drop_missing_rows": drop_missing_rows,
    "convert_dtype": convert_dtype,
    "clip_outliers": clip_outliers,
}


# ============================================================================
# Helpers: Parsing & Validation
# ============================================================================

def _parse_cleaning_plan(llm_reply: str) -> tuple[list[dict[str, Any]], bool]:
    """
    Safely parses the LLM cleaning plan. Strips markdown fences, extracts JSON list,
    and falls back to safe default actions if parsing fails.
    """
    raw_text = llm_reply.strip()

    # Strip markdown code blocks (e.g. ```json ... ```)
    if "```" in raw_text:
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\s*```$", "", raw_text)
        raw_text = raw_text.strip()

    # Attempt direct JSON parse
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            return data, False
        if isinstance(data, dict) and "steps" in data and isinstance(data["steps"], list):
            return data["steps"], False
    except Exception:
        pass

    # Attempt regex pattern match for JSON array [ { ... } ]
    array_match = re.search(r"\[\s*\{.*\}\s*\]", raw_text, flags=re.DOTALL)
    if array_match:
        try:
            data = json.loads(array_match.group(0))
            if isinstance(data, list):
                return data, False
        except Exception:
            pass

    # Safe fallback plan when parsing fails
    fallback_plan = [
        {"action": "drop_duplicates", "column": "all", "reason": "Fallback: drop exact duplicate rows"},
        {"action": "strip_whitespace", "column": "all", "reason": "Fallback: strip whitespace in text columns"},
    ]
    return fallback_plan, True


# ============================================================================
# LangGraph Node: cleaner_node
# ============================================================================

def cleaner_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph Cleaner Node:
    1. Loads raw CSV and computes profile.
    2. Requests cleaning plan from LLM.
    3. Deterministically applies validated transformations (rejects arbitrary code).
    4. Enforces the <=20% row removal safety threshold.
    5. Saves cleaned dataset to data/cleaned/cleaned_<filename>.csv.
    6. Updates state fields: clean_df_path, profile, cleaning_log, events.
    """
    cleaning_log = list(state.get("cleaning_log", []))
    events = list(state.get("events", []))
    raw_path = state.get("raw_df_path", "")

    try:
        if not raw_path or not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data file not found at: '{raw_path}'")

        # Load raw dataframe
        df = pd.read_csv(raw_path)

        # Build initial profile
        initial_profile = profile_dataframe(df)
        profile_text_block = profile_to_text(initial_profile)

        # Build LLM prompt
        prompt = f"""You are a professional data cleaning specialist.
Analyze this dataset profile and return a cleaning plan.

{profile_text_block}

Allowed actions list:
- drop_duplicates: removes identical rows (column: 'all' or column name)
- strip_whitespace: trims spaces in text columns (column: 'all' or column name)
- standardize_case: changes case (column: column name, case: 'lower' | 'upper' | 'title')
- fill_missing: imputes nulls (column: column name, method: 'mean' | 'median' | 'mode' | 'constant', value: optional)
- drop_missing_rows: drops rows with nulls (column: column name or 'all')
- convert_dtype: casts column (column: column name, to: 'datetime' | 'numeric' | 'category')
- clip_outliers: caps extreme numbers using IQR (column: column name, method: 'iqr')

Return ONLY a JSON list of cleaning steps. Each step has an 'action' from the allowed list, a 'column' (or 'all'), optional parameters, and a short 'reason'. Do not return code. Do not clip outliers in revenue, price, count, or time columns unless the values are clearly impossible. Always list column names explicitly, never use 'all' for standardize_case. Never modify identifier columns (ids, keys, uuids) except trimming whitespace."""

        # Call Gemini LLM
        llm_reply = call_llm(prompt)
        plan, used_fallback = _parse_cleaning_plan(llm_reply)

        if used_fallback:
            fallback_msg = "LLM response could not be parsed as a JSON list. Applied safe default cleaning plan."
            cleaning_log.append(fallback_msg)
            events.append({
                "agent": "cleaner",
                "type": "warning",
                "message": fallback_msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        current_df = df.copy()

        # Consolidate multiple strip_whitespace steps into one merged step
        consolidated_plan = []
        strip_columns: list[str] = []
        has_strip_all = False

        for step in plan:
            if not isinstance(step, dict):
                continue
            act = str(step.get("action", "")).strip()
            raw_col = step.get("column", "")
            if act == "strip_whitespace":
                if raw_col == "all" or not raw_col:
                    has_strip_all = True
                elif isinstance(raw_col, list):
                    strip_columns.extend(str(c).strip() for c in raw_col)
                else:
                    strip_columns.append(str(raw_col).strip())
            else:
                consolidated_plan.append(step)

        if has_strip_all:
            consolidated_plan.insert(0, {"action": "strip_whitespace", "column": "all", "reason": "Trim whitespace in all text columns"})
        elif strip_columns:
            # Preserve order, remove duplicates
            unique_strip = list(dict.fromkeys(strip_columns))
            consolidated_plan.insert(0, {"action": "strip_whitespace", "column": unique_strip, "reason": "Trim whitespace in specified text columns"})

        plan = consolidated_plan

        # Deterministically apply each cleaning step
        for step in plan:
            if not isinstance(step, dict):
                continue

            action = str(step.get("action", "")).strip()
            raw_column = step.get("column", "")
            if isinstance(raw_column, list):
                column = raw_column
            else:
                column = str(raw_column).strip()
            reason = step.get("reason", "")

            # 1. Action validation: Must be in allowed list
            if action not in ALLOWED_ACTIONS:
                skip_msg = f"Skipped unknown action '{action}'. Reason: action not in allowed list."
                cleaning_log.append(skip_msg)
                events.append({
                    "agent": "cleaner",
                    "type": "warning",
                    "message": skip_msg,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                continue

            # 2. Column validation: Must exist in dataframe or be 'all' for supporting actions
            actions_supporting_all = {"drop_duplicates", "strip_whitespace", "drop_missing_rows"}
            if isinstance(column, list):
                column = [c for c in column if c in current_df.columns]
                if not column:
                    continue
            elif column == "all" and action not in actions_supporting_all:
                skip_msg = f"Skipped '{action}': action does not support column='all'."
                cleaning_log.append(skip_msg)
                events.append({
                    "agent": "cleaner",
                    "type": "warning",
                    "message": skip_msg,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                continue
            elif column != "all" and column not in current_df.columns:
                skip_msg = f"Skipped '{action}': column '{column}' not found in dataset."
                cleaning_log.append(skip_msg)
                events.append({
                    "agent": "cleaner",
                    "type": "warning",
                    "message": skip_msg,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                continue

            # 2b. Safety Rule: Never modify identifier columns except trimming whitespace
            if column != "all" and isinstance(column, str) and column in current_df.columns:
                if is_identifier_column(current_df[column], column):
                    should_skip_id = False
                    if action in ("standardize_case", "fill_missing", "clip_outliers"):
                        should_skip_id = True
                    elif action == "convert_dtype":
                        target_to = str(step.get("to", "numeric")).strip().lower()
                        if target_to in ("numeric", "datetime"):
                            should_skip_id = True

                    if should_skip_id:
                        warn_msg = f"Skipped '{action}' on identifier column '{column}'"
                        cleaning_log.append(warn_msg)
                        events.append({
                            "agent": "cleaner",
                            "type": "warning",
                            "message": warn_msg,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        continue

            # 3. Safety Rule: Check if row removal exceeds 20% of CURRENT rows
            current_rows = len(current_df)
            if current_rows > 0:
                rows_to_remove = 0
                if action == "drop_duplicates":
                    if column == "all" or not column:
                        rows_to_remove = int(current_df.duplicated().sum())
                    else:
                        rows_to_remove = int(current_df.duplicated(subset=[column]).sum())
                elif action == "drop_missing_rows":
                    if column == "all" or not column:
                        rows_to_remove = int(current_df.isna().any(axis=1).sum())
                    else:
                        rows_to_remove = int(current_df[column].isna().sum())

                if rows_to_remove > (0.20 * current_rows):
                    pct = (rows_to_remove / current_rows) * 100
                    warn_msg = (
                        f"Warning: Skipped '{action}' on '{column}' because it would remove {rows_to_remove} rows "
                        f"({pct:.1f}% of current data), exceeding the 20% safety threshold."
                    )
                    cleaning_log.append(warn_msg)
                    events.append({
                        "agent": "cleaner",
                        "type": "warning",
                        "message": warn_msg,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    continue

            # 4. Apply transformation
            action_fn = ALLOWED_ACTIONS[action]
            step_params = {k: v for k, v in step.items() if k not in ("action", "column", "reason")}
            new_df, action_msg, success = action_fn(current_df, column=column, **step_params)

            if success:
                current_df = new_df
                cleaning_log.append(action_msg)
                events.append({
                    "agent": "cleaner",
                    "type": "info",
                    "message": action_msg,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            else:
                cleaning_log.append(f"Warning: {action_msg}")
                events.append({
                    "agent": "cleaner",
                    "type": "warning",
                    "message": action_msg,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        # Save cleaned dataset to workspace or data/cleaned/cleaned_<filename>.csv
        workspace_dir = state.get("workspace_dir")
        if workspace_dir:
            cleaned_dir = os.path.normpath(os.path.join(workspace_dir, "cleaned"))
        else:
            cleaned_dir = os.path.join("data", "cleaned")
        os.makedirs(cleaned_dir, exist_ok=True)
        raw_basename = os.path.basename(raw_path)
        clean_df_path = os.path.normpath(os.path.join(cleaned_dir, f"cleaned_{raw_basename}"))

        # Guarantee we never overwrite the original raw file
        if os.path.abspath(clean_df_path) == os.path.abspath(raw_path):
            clean_df_path = os.path.normpath(os.path.join(cleaned_dir, f"clean_{raw_basename}"))

        current_df.to_csv(clean_df_path, index=False)

        # Profile the cleaned data
        updated_profile = profile_dataframe(current_df)

        return {
            "clean_df_path": clean_df_path,
            "profile": updated_profile,
            "cleaning_log": cleaning_log,
            "events": events,
            "cleaning_summary": {
                "raw_rows": int(len(df)),
                "raw_cols": int(len(df.columns)),
                "clean_rows": int(len(current_df)),
                "clean_cols": int(len(current_df.columns)),
            },
        }

    except (LLMUnavailableError, LLMConfigError):
        raise
    except Exception as e:
        # Graceful failure wrapper: log error, save raw data as clean file, and continue
        err_msg = f"Cleaner agent encountered an unexpected error: {e}"
        cleaning_log.append(err_msg)
        events.append({
            "agent": "cleaner",
            "type": "error",
            "message": err_msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        try:
            workspace_dir = state.get("workspace_dir")
            if workspace_dir:
                cleaned_dir = os.path.normpath(os.path.join(workspace_dir, "cleaned"))
            else:
                cleaned_dir = os.path.join("data", "cleaned")
            os.makedirs(cleaned_dir, exist_ok=True)
            raw_basename = os.path.basename(raw_path) if raw_path else "fallback.csv"
            fallback_clean_path = os.path.normpath(os.path.join(cleaned_dir, f"cleaned_{raw_basename}"))

            if raw_path and os.path.exists(raw_path) and os.path.abspath(raw_path) != os.path.abspath(fallback_clean_path):
                import shutil
                shutil.copyfile(raw_path, fallback_clean_path)
            elif not os.path.exists(fallback_clean_path):
                pd.DataFrame().to_csv(fallback_clean_path, index=False)

            fallback_df = pd.read_csv(fallback_clean_path)
            fallback_profile = profile_dataframe(fallback_df)
        except Exception:
            fallback_clean_path = raw_path
            fallback_profile = state.get("profile", {})

        raw_r = int(len(df)) if ("df" in locals() and df is not None) else 0
        raw_c = int(len(df.columns)) if ("df" in locals() and df is not None) else 0
        clean_r = int(len(fallback_df)) if ("fallback_df" in locals() and fallback_df is not None) else raw_r
        clean_c = int(len(fallback_df.columns)) if ("fallback_df" in locals() and fallback_df is not None) else raw_c

        return {
            "clean_df_path": fallback_clean_path,
            "profile": fallback_profile,
            "cleaning_log": cleaning_log,
            "events": events,
            "cleaning_summary": {
                "raw_rows": raw_r,
                "raw_cols": raw_c,
                "clean_rows": clean_r,
                "clean_cols": clean_c,
            },
        }
