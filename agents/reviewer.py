"""
reviewer.py
Reviewer Agent: Independently verifies and validates analytical outputs through:
1. Independent recomputation via an alternative pandas implementation.
2. Rule-based invariant checks (non-negativity, nulls/infinities, aggregation sizes, takeaway alignment).
3. Advisory logic review for potential query/aggregation flaws.
Formulates structured review_feedback and audit reports for the feedback loop.
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
from tools.sandbox import run_code


# ============================================================================
# Helper: Results Matching with Tolerance
# ============================================================================

GENERIC_HEADERS = {
    "index", "0", "1", "level_0", "level_1", "unnamed: 0",
    "value", "values", "count", "counts", "proportion", "proportions",
    "pct", "percent", "percentage", "share", "shares", "total", "totals",
    "sum", "mean", "avg", "median", "result", "results", "metric", "measure",
}


def _is_generic_header(col: Any) -> bool:
    if isinstance(col, (int, float, np.number)):
        return True
    s = str(col).strip().lower()
    return s in GENERIC_HEADERS or s.isdigit()


def _extract_numeric_cells(table: Any) -> list[float]:
    """
    Extracts all numeric cells from a DataFrame or Series,
    ignoring non-numeric label cells and RangeIndex.
    """
    if isinstance(table, pd.Series):
        nums = []
        for val in table.dropna().values:
            if isinstance(val, (int, float, np.number)) and not isinstance(val, (bool, np.bool_)):
                if np.isfinite(val):
                    nums.append(float(val))
        return nums

    if isinstance(table, pd.DataFrame):
        nums = []
        for col in table.columns:
            # Ignore RangeIndex if reset as 'index' or 'level_0'
            col_str = str(col).strip().lower()
            if col_str in ("index", "level_0") and len(table) > 0:
                s = table[col]
                if pd.api.types.is_integer_dtype(s):
                    vals = s.dropna().values
                    if len(vals) > 0 and (vals == np.arange(len(vals))).all():
                        continue

            for val in table[col].dropna().values:
                if isinstance(val, (int, float, np.number)) and not isinstance(val, (bool, np.bool_)):
                    if np.isfinite(val):
                        nums.append(float(val))
        return nums

    return []


def _extract_labels_and_values(
    df: pd.DataFrame,
) -> tuple[set[str], dict[str, list[float]], list[float]]:
    """
    Extracts (labels, label_to_values_map, all_numbers) from a DataFrame.
    Supports long tables, 1D wide tables, and 2D pivoted tables.
    """
    all_numbers: list[float] = []

    # 1. Collect all numeric values (ignoring reset RangeIndex)
    for col in df.columns:
        col_str = str(col).strip().lower()
        if col_str in ("index", "level_0") and len(df) > 0:
            if pd.api.types.is_integer_dtype(df[col]):
                vals = df[col].dropna().values
                if len(vals) > 0 and (vals == np.arange(len(vals))).all():
                    continue
        for val in df[col].dropna().values:
            if isinstance(val, (int, float, np.number)) and not isinstance(val, (bool, np.bool_)):
                if np.isfinite(val):
                    all_numbers.append(float(val))

    # 2. Identify text columns and numeric columns
    text_cols = []
    num_cols = []
    for col in df.columns:
        col_str = str(col).strip().lower()
        if col_str in ("index", "level_0") and len(df) > 0:
            if pd.api.types.is_integer_dtype(df[col]):
                vals = df[col].dropna().values
                if len(vals) > 0 and (vals == np.arange(len(vals))).all():
                    continue

        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            num_cols.append(col)
        else:
            text_cols.append(col)

    labels: set[str] = set()
    label_map: dict[str, list[float]] = {}

    # Check which text columns contain actual category labels
    real_text_cols = []
    for tc in text_cols:
        vals = [str(v).strip().lower() for v in df[tc].dropna().values]
        if any(v not in GENERIC_HEADERS and not v.isdigit() for v in vals):
            real_text_cols.append(tc)

    # Which numeric columns represent category headers (wide format)
    # They are category headers if non-generic and either len(df) == 1 or multiple numeric columns
    non_generic_num_cols = [c for c in num_cols if not _is_generic_header(c)]
    is_wide_num_cols = len(non_generic_num_cols) > 0 and (len(df) == 1 or len(non_generic_num_cols) > 1)

    # A) Extract labels from text cells
    if len(real_text_cols) > 0:
        for idx in range(len(df)):
            row_labels = []
            for tc in real_text_cols:
                v = df[tc].iloc[idx]
                if pd.notna(v):
                    lbl = str(v).strip().lower()
                    if lbl not in GENERIC_HEADERS and not lbl.isdigit():
                        row_labels.append(lbl)

            if not row_labels:
                continue

            # Row's numeric values
            row_nums = []
            for nc in num_cols:
                v = df[nc].iloc[idx]
                if isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)):
                    if np.isfinite(v):
                        row_nums.append(float(v))

            for rlbl in row_labels:
                labels.add(rlbl)
                if rlbl not in label_map:
                    label_map[rlbl] = []
                label_map[rlbl].extend(row_nums)

    # B) Extract labels from wide numeric column headers
    if is_wide_num_cols:
        for nc in non_generic_num_cols:
            col_lbl = str(nc).strip().lower()
            labels.add(col_lbl)
            col_nums = []
            for v in df[nc].dropna().values:
                if isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)):
                    if np.isfinite(v):
                        col_nums.append(float(v))
            if col_lbl not in label_map:
                label_map[col_lbl] = []
            label_map[col_lbl].extend(col_nums)

    return labels, label_map, all_numbers


def results_match_with_trace(a: Any, b: Any, tolerance: float = 0.01) -> tuple[bool, dict[str, Any]]:
    """
    Compares two analytical calculation results for equality within a given tolerance.
    Returns a tuple of (matched: bool, trace: dict) where trace contains:
    - 'branch': 'scalar vs scalar', 'scalar vs table', 'scalar vs table (single numeric cell)',
                'table same layout', 'layout-independent fallback', 'shape mismatch', or 'type mismatch'
    - 'reason': detailed diagnostic reason explaining why it matched or mismatched
    - 'matched': boolean equal to the matched flag
    """
    if a is None or b is None:
        matched = (a is b)
        reason = "both values are None" if matched else f"one value is None ({'a' if a is None else 'b'})"
        return matched, {"branch": "scalar vs scalar", "reason": reason, "matched": matched}

    is_a_scalar = isinstance(a, (int, float, str, bool, np.number, np.bool_))
    is_b_scalar = isinstance(b, (int, float, str, bool, np.number, np.bool_))

    # 1. Scalar vs Scalar comparison
    if is_a_scalar and is_b_scalar:
        branch = "scalar vs scalar"
        if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
            if pd.isna(a) and pd.isna(b):
                return True, {"branch": branch, "reason": "both values are NaN", "matched": True}
            if pd.isna(a) or pd.isna(b):
                return False, {"branch": branch, "reason": f"one value is NaN: {a} vs {b}", "matched": False}
            denom = max(abs(float(a)), abs(float(b)))
            if denom == 0:
                return True, {"branch": branch, "reason": "both values are 0.0", "matched": True}
            rel_diff = abs(float(a) - float(b)) / denom
            if rel_diff <= tolerance:
                return True, {
                    "branch": branch,
                    "reason": f"numeric values match within {tolerance:.1%} tolerance ({float(a):.4g} vs {float(b):.4g}, rel_diff={rel_diff:.4f})",
                    "matched": True,
                }
            else:
                return False, {
                    "branch": branch,
                    "reason": f"numeric values differ ({float(a):.4g} vs {float(b):.4g}, rel_diff={rel_diff:.4f} > {tolerance:.1%})",
                    "matched": False,
                }
        else:
            str_a = str(a).strip().lower()
            str_b = str(b).strip().lower()
            if str_a == str_b:
                return True, {"branch": branch, "reason": f"scalar text strings match: '{str(a).strip()}'", "matched": True}
            else:
                return False, {"branch": branch, "reason": f"scalar text strings differ: '{str(a).strip()}' vs '{str(b).strip()}'", "matched": False}

    # 2. Scalar vs Table comparison (Series or DataFrame)
    is_a_table = isinstance(a, (pd.DataFrame, pd.Series))
    is_b_table = isinstance(b, (pd.DataFrame, pd.Series))

    if (is_a_scalar and is_b_table) or (is_b_scalar and is_a_table):
        scalar_val = a if is_a_scalar else b
        table_val = b if is_a_scalar else a

        numeric_cells = _extract_numeric_cells(table_val)
        if len(numeric_cells) == 1:
            target_num = numeric_cells[0]
            branch = "scalar vs table (single numeric cell)"
            if isinstance(scalar_val, (int, float, np.number)) and not isinstance(scalar_val, (bool, np.bool_)):
                num_scalar = float(scalar_val)
                denom = max(abs(num_scalar), abs(target_num))
                rel_diff = 0.0 if denom == 0 else abs(num_scalar - target_num) / denom
                matched = (rel_diff <= tolerance)
                if matched:
                    reason = (
                        f"numeric scalar matches single numeric table cell within {tolerance:.1%} tolerance "
                        f"({num_scalar:.4g} vs {target_num:.4g}, rel_diff={rel_diff:.4f})"
                    )
                else:
                    reason = (
                        f"numeric scalar differs from single numeric table cell "
                        f"({num_scalar:.4g} vs {target_num:.4g}, rel_diff={rel_diff:.4f} > {tolerance:.1%})"
                    )
                return matched, {"branch": branch, "reason": reason, "matched": matched}
            elif isinstance(scalar_val, str):
                try:
                    num_scalar = float(scalar_val.strip())
                    denom = max(abs(num_scalar), abs(target_num))
                    rel_diff = 0.0 if denom == 0 else abs(num_scalar - target_num) / denom
                    matched = (rel_diff <= tolerance)
                    if matched:
                        reason = (
                            f"numeric scalar matches single numeric table cell within {tolerance:.1%} tolerance "
                            f"({num_scalar:.4g} vs {target_num:.4g}, rel_diff={rel_diff:.4f})"
                        )
                    else:
                        reason = (
                            f"numeric scalar differs from single numeric table cell "
                            f"({num_scalar:.4g} vs {target_num:.4g}, rel_diff={rel_diff:.4f} > {tolerance:.1%})"
                        )
                    return matched, {"branch": branch, "reason": reason, "matched": matched}
                except ValueError:
                    return False, {"branch": branch, "reason": f"scalar string '{scalar_val}' is not numeric", "matched": False}
            else:
                return False, {"branch": branch, "reason": f"scalar type {type(scalar_val).__name__} is not numeric", "matched": False}
        else:
            branch = "scalar vs table"
            cell_count = table_val.size
            reason = f"scalar vs table with {cell_count} cells, not comparable"
            return False, {"branch": branch, "reason": reason, "matched": False}

    # Convert Series to DataFrame for table-vs-table comparison
    if isinstance(a, pd.Series):
        a = a.reset_index()
    if isinstance(b, pd.Series):
        b = b.reset_index()

    # 3. DataFrame vs DataFrame comparison
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        if len(a) == 0 and len(b) == 0:
            return True, {"branch": "table same layout", "reason": "both DataFrames are empty", "matched": True}

        shapes_match = (len(a) == len(b) and a.shape[1] == b.shape[1])
        diff_reason = ""

        # Standard elementwise comparison if shapes match
        if shapes_match:
            # Align columns by name if set of columns is identical
            if set(a.columns) == set(b.columns):
                b_aligned = b[list(a.columns)]
            else:
                b_aligned = b

            # Sort both by first column and reset index
            try:
                a_sorted = a.sort_values(by=a.columns[0]).reset_index(drop=True)
                b_sorted = b_aligned.sort_values(by=b_aligned.columns[0]).reset_index(drop=True)
            except Exception:
                a_sorted = a.reset_index(drop=True)
                b_sorted = b_aligned.reset_index(drop=True)

            match_standard = True
            for i in range(a_sorted.shape[1]):
                col_a = a_sorted.iloc[:, i]
                col_b = b_sorted.iloc[:, i]

                is_a_num = pd.api.types.is_numeric_dtype(col_a)
                is_b_num = pd.api.types.is_numeric_dtype(col_b)

                if is_a_num and is_b_num:
                    diff = np.abs(col_a.astype(float) - col_b.astype(float))
                    denom = np.maximum(np.abs(col_a.astype(float)), np.abs(col_b.astype(float)))
                    denom = np.where(denom == 0, 1.0, denom)
                    rel_diff = diff / denom
                    if (rel_diff > tolerance).any():
                        match_standard = False
                        max_d = float(rel_diff.max())
                        diff_reason = f"numeric column '{col_a.name}' values differ (max rel_diff={max_d:.4f} > {tolerance:.1%})"
                        break
                else:
                    str_a = col_a.astype(str).str.strip().str.lower()
                    str_b = col_b.astype(str).str.strip().str.lower()
                    if (str_a != str_b).any():
                        match_standard = False
                        diff_mask = (str_a != str_b)
                        first_a = str_a[diff_mask].iloc[:3].tolist()
                        first_b = str_b[diff_mask].iloc[:3].tolist()
                        diff_reason = f"text labels differ: {first_a} vs {first_b}"
                        break

            if match_standard:
                return True, {"branch": "table same layout", "reason": "all rows and columns match within tolerance", "matched": True}
        else:
            diff_reason = f"shape mismatch: {a.shape} vs {b.shape}"

        # Layout-independent comparison fallback:
        # Extract labels, label-to-value map, and all numeric values
        labels_a, label_map_a, all_nums_a = _extract_labels_and_values(a)
        labels_b, label_map_b, all_nums_b = _extract_labels_and_values(b)

        # If either table has text labels, require label sets to match
        if len(labels_a) > 0 or len(labels_b) > 0:
            if labels_a != labels_b:
                diff_a = sorted(labels_a - labels_b)[:3]
                diff_b = sorted(labels_b - labels_a)[:3]
                reason = f"text labels differ: {diff_a} vs {diff_b}"
                branch = "table same layout" if shapes_match else "shape mismatch"
                return False, {"branch": branch, "reason": reason, "matched": False}

            # Label sets match: compare numbers per label within tolerance
            for lbl in sorted(labels_a):
                vals_a = label_map_a.get(lbl, [])
                vals_b = label_map_b.get(lbl, [])
                if len(vals_a) != len(vals_b):
                    branch = "table same layout" if shapes_match else "shape mismatch"
                    return False, {
                        "branch": branch,
                        "reason": f"count of values for label '{lbl}' differs: {len(vals_a)} vs {len(vals_b)}",
                        "matched": False,
                    }
                sorted_a = sorted(vals_a)
                sorted_b = sorted(vals_b)
                for va, vb in zip(sorted_a, sorted_b):
                    denom = max(abs(va), abs(vb))
                    rel_diff = 0.0 if denom == 0 else abs(va - vb) / denom
                    if rel_diff > tolerance:
                        branch = "table same layout" if shapes_match else "shape mismatch"
                        return False, {
                            "branch": branch,
                            "reason": diff_reason or f"numeric values differ for label '{lbl}' ({va:.4g} vs {vb:.4g}, rel_diff={rel_diff:.4f} > {tolerance:.1%})",
                            "matched": False,
                        }

            return True, {
                "branch": "layout-independent fallback",
                "reason": f"layout-independent comparison matched {len(labels_a)} labels and numeric values within {tolerance:.1%} tolerance",
                "matched": True,
            }

        # Zero text labels in both tables: fall back to comparing sorted numeric values
        nums_a = sorted(all_nums_a)
        nums_b = sorted(all_nums_b)

        if len(nums_a) > 0 and len(nums_a) == len(nums_b):
            diffs = [
                0.0 if max(abs(va), abs(vb)) == 0 else abs(va - vb) / max(abs(va), abs(vb))
                for va, vb in zip(nums_a, nums_b)
            ]
            if all(d <= tolerance for d in diffs):
                return True, {
                    "branch": "layout-independent fallback",
                    "reason": f"layout-independent comparison matched {len(nums_a)} sorted numeric values within tolerance",
                    "matched": True,
                }

        # If layout-independent comparison did not match
        branch_name = "table same layout" if shapes_match else "shape mismatch"
        return False, {"branch": branch_name, "reason": diff_reason or "numeric values differ", "matched": False}

    return False, {"branch": "type mismatch", "reason": f"unsupported types: {type(a).__name__} vs {type(b).__name__}", "matched": False}


def results_match(a: Any, b: Any, tolerance: float = 0.01) -> bool:
    """
    Compares two analytical calculation results for equality within a given tolerance:
    - Scalars: compares numeric values within relative tolerance (default 1%),
      handling zero and NaN safely.
    - DataFrames: compares after sorting both by their first column and resetting
      index. Compares numeric columns within tolerance and text columns exactly.
      Row counts and column counts must match.
    """
    matched, _ = results_match_with_trace(a, b, tolerance=tolerance)
    return matched


# ============================================================================
# Helper: Value Shortener
# ============================================================================

def _shorten_val(val: Any) -> Any:
    """Provides a concise, JSON-friendly representation of analytical results."""
    if val is None:
        return "None"
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return round(float(val), 4)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, pd.DataFrame):
        return f"DataFrame(rows={len(val)}, cols={list(val.columns)})"
    s = str(val)
    return s[:120] + "..." if len(s) > 120 else s


# ============================================================================
# Check 2: Rule-Based Checks (No LLM)
# ============================================================================

def _extract_takeaway_numbers(takeaway: str, res: Any) -> list[str]:
    """
    Finds numbers cited in chart takeaway text and verifies their presence in the
    result data (within 1% tolerance, allowing for rounding and magnitudes like 1.2M).
    - Unit suffix bug: only treat a suffix as multiplier when directly attached (1.2M, 500k)
      or magnitude words ('thousand', 'million', 'billion'). Never treat ' m' with a space as million.
    - Ignore numbers that come from labels (column names, text cells, bracket labels like '0-10%').
    - Ignore numbers in ranges like '0-10%', '6.4-7.6'.
    - Leniently ignores years (1900-2099), small integers < 10, and percentages (%).
    """
    problems: list[str] = []

    # 1. Collect numbers from result numeric cells
    result_numbers: list[float] = []
    if isinstance(res, (int, float, np.number)):
        if not pd.isna(res) and np.isfinite(res):
            result_numbers.append(float(res))
    elif isinstance(res, pd.DataFrame):
        for col in res.select_dtypes(include=[np.number]).columns:
            for v in res[col].dropna():
                if np.isfinite(v):
                    result_numbers.append(float(v))

    # 2. Collect numbers from result labels/text columns and column names
    label_numbers: set[float] = set()
    if isinstance(res, pd.DataFrame):
        for col in res.columns:
            for num_str in re.findall(r"\d+(?:\.\d+)?", str(col)):
                try:
                    label_numbers.add(float(num_str))
                except ValueError:
                    pass
        for col in res.select_dtypes(include=["object", "category", "string"]).columns:
            for val in res[col].dropna().unique():
                for num_str in re.findall(r"\d+(?:\.\d+)?", str(val)):
                    try:
                        label_numbers.add(float(num_str))
                    except ValueError:
                        pass

    # 3. Collect numbers inside ranges in takeaway (e.g. 0-10%, 1.56–1.69, 6.4-7.6)
    range_numbers: set[float] = set()
    for rm in re.finditer(r"(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)", takeaway):
        try:
            range_numbers.add(float(rm.group(1)))
            range_numbers.add(float(rm.group(2)))
        except ValueError:
            pass

    # 4. Match numbers with precise suffix handling
    # Matches:
    # - $100, 15.71, 0.215
    # - 1.2M, 500k, 3B (suffix directly attached, no space, not followed by word chars)
    # - 1.2 million, 500 thousand, 3 billion (magnitude words)
    pattern = re.compile(
        r"(\$?\d+(?:,\d{3})*(?:\.\d+)?)(?:([kmbKMB])(?![a-zA-Z])|\s+(thousand|million|billion)\b)?(%?)",
        flags=re.IGNORECASE,
    )

    for m in pattern.finditer(takeaway):
        raw_num, direct_suffix, word_suffix, pct = m.groups()

        # Ignore percentage values
        if pct == "%":
            continue

        clean_num = raw_num.replace("$", "").replace(",", "")
        try:
            val = float(clean_num)
        except ValueError:
            continue

        # Ignore numbers in ranges (e.g. '0-10%')
        if val in range_numbers:
            continue

        # Ignore numbers that come from result labels or column names
        if val in label_numbers:
            continue

        # Ignore calendar years
        if not direct_suffix and not word_suffix and val.is_integer() and 1900 <= int(val) <= 2099:
            continue

        # Ignore small integers below 10
        if not direct_suffix and not word_suffix and val.is_integer() and 0 <= int(val) < 10:
            continue

        # Calculate multiplier
        multiplier = 1.0
        suffix_str = direct_suffix or word_suffix
        if suffix_str:
            s_low = suffix_str.lower()
            if s_low in ("k", "thousand"):
                multiplier = 1e3
            elif s_low in ("m", "million"):
                multiplier = 1e6
            elif s_low in ("b", "billion"):
                multiplier = 1e9

        scaled_val = val * multiplier
        raw_token = m.group(0).strip()

        # Check for presence in result_numbers
        matched = False
        for v in result_numbers:
            diff1 = abs(v - scaled_val) / max(abs(v), abs(scaled_val), 1e-9)
            if diff1 <= 0.01 or round(v) == round(scaled_val):
                matched = True
                break

            diff2 = abs(v - val) / max(abs(v), abs(val), 1e-9)
            if diff2 <= 0.01 or round(v) == round(val):
                matched = True
                break

        if not matched:
            problems.append(f"Takeaway mentions {raw_token} which is not in the result")

    return problems


def rule_checks(item: dict[str, Any], df: pd.DataFrame, chart: dict[str, Any] | None = None) -> list[str]:
    """
    Performs deterministic rule-based checks on an analytical result item:
    1. Checks if result is empty.
    2. Checks non-negativity on sum/count/sales/revenue columns.
    3. Checks for presence of NaN or infinite values.
    4. Checks for missing aggregation (> 50 rows).
    Note: Takeaway number checks are advisory warnings and handled separately in reviewer_node.
    """
    problems: list[str] = []
    res = item.get("result")

    # 1. Result is empty
    if res is None:
        problems.append("Result is None")
        return problems
    if isinstance(res, pd.DataFrame) and res.empty:
        problems.append("Result DataFrame is empty")
        return problems
    if isinstance(res, (list, dict)) and len(res) == 0:
        problems.append("Result is empty")
        return problems

    # 2. Non-negative column checks (count, total, quantity, sales, revenue)
    keywords = ("count", "total", "quantity", "sales", "revenue")
    if isinstance(res, pd.DataFrame):
        for col in res.select_dtypes(include=[np.number]).columns:
            col_lower = str(col).lower()
            if any(k in col_lower for k in keywords):
                if (res[col] < 0).any():
                    problems.append(f"Column '{col}' should be non-negative but contains negative values")
    elif isinstance(res, (int, float, np.number)):
        q_lower = str(item.get("question", "")).lower()
        if any(k in q_lower for k in keywords) and res < 0:
            problems.append(f"Result should be non-negative for this query but was {res}")

    # 3. NaN or Infinite values
    if isinstance(res, pd.DataFrame):
        if res.isna().any().any():
            problems.append("Result contains NaN values")
        num_cols = res.select_dtypes(include=[np.number])
        if np.isinf(num_cols).any().any():
            problems.append("Result contains infinite values")
    elif isinstance(res, float):
        if np.isnan(res) or np.isinf(res):
            problems.append("Result is NaN or infinite")

    # 4. Result size > 50 rows (likely missing aggregation)
    if isinstance(res, pd.DataFrame) and len(res) > 50:
        problems.append("Result has more than 50 rows (possible missing aggregation)")

    return problems


# ============================================================================
# Check 1: Independent Recomputation
# ============================================================================

def _clean_code_fence(text: str) -> str:
    """Strips markdown code fences from LLM responses."""
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:python)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _recompute_independently(
    question: str,
    original_code: str,
    original_res: Any,
    columns: list[str],
    df: pd.DataFrame,
    call_llm_fn: Callable[[str], str],
) -> tuple[str, Any, str, str]:
    """
    Check 1: Asks LLM to solve the question using an alternative pandas method,
    runs it via sandbox, and verifies agreement with original results.
    Returns (status: 'pass' | 'fail' | 'inconclusive', recomputed_res, note, recomp_code).
    """
    prompt = f"""Question: {question}

Original pandas code:
```python
{original_code}
```

Columns available:
{', '.join(columns)}

Solve the same question using a different pandas method than the original code
(for example, pivot_table instead of groupby, or sort_values instead of nlargest).
Store the answer in a variable called result. Use only pd and np, no imports.
Return ONLY the code."""

    reply = call_llm_fn(prompt)
    recomp_code = _clean_code_fence(reply)

    # Execute alternative code in sandbox
    res = run_code(recomp_code, df)

    # Retry once if execution encountered an error
    if res["status"] != "ok":
        retry_prompt = f"""Previous alternative code failed with: {res.get('error')}.
Fix it and return only the corrected alternative pandas code for: '{question}'.
Store the answer in a variable called result. Use only pd and np, no imports. Return ONLY the code."""
        retry_reply = call_llm_fn(retry_prompt)
        recomp_code = _clean_code_fence(retry_reply)
        res = run_code(recomp_code, df)

    if res["status"] != "ok":
        # Mark inconclusive (not a fail) per spec
        fail_trace = {"branch": "recomputation_error", "reason": f"Recomputation code failed: {res.get('error')}", "matched": False}
        return "inconclusive", None, f"Recomputation code failed: {res.get('error')}", recomp_code, fail_trace

    recomputed_val = res.get("result")
    matches, trace = results_match_with_trace(original_res, recomputed_val)

    if matches:
        note = ""
        if isinstance(original_res, pd.DataFrame) and isinstance(recomputed_val, pd.DataFrame):
            if original_res.shape != recomputed_val.shape or list(original_res.columns) != list(recomputed_val.columns):
                note = "same values, different layout"
        return "pass", recomputed_val, note, recomp_code, trace
    else:
        return "fail", recomputed_val, "Independent recomputation gave a different result", recomp_code, trace


# ============================================================================
# Check 3: Logic Review (LLM Advisory Only)
# ============================================================================

def _result_to_markdown_preview(res: Any, max_rows: int = 10) -> str:
    """Formats a DataFrame, Series, or scalar as a clean markdown table preview."""
    if isinstance(res, pd.DataFrame):
        if res.empty:
            return "*Empty DataFrame*"
        df_sub = res.head(max_rows)
        headers = [str(c) for c in df_sub.columns]
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        rows = []
        for _, row in df_sub.iterrows():
            rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
        trunc = f"\n*(Showing first {max_rows} of {len(res)} rows)*" if len(res) > max_rows else ""
        return "\n".join([header_line, sep_line] + rows) + trunc
    elif isinstance(res, pd.Series):
        if res.empty:
            return "*Empty Series*"
        df_sub = res.head(max_rows).to_frame().reset_index()
        headers = [str(c) for c in df_sub.columns]
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        rows = []
        for _, row in df_sub.iterrows():
            rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
        trunc = f"\n*(Showing first {max_rows} of {len(res)} items)*" if len(res) > max_rows else ""
        return "\n".join([header_line, sep_line] + rows) + trunc
    elif res is not None:
        return f"`{res}`"
    return "*No result generated*"


def _is_formatting_or_type_only_issue(issue_str: str) -> bool:
    """
    Returns True if the issue text discusses ONLY Series, DataFrame, index,
    dtype, or formatting without identifying real mathematical or logical calculation errors.
    """
    text = issue_str.lower()
    type_kw = [
        "series", "dataframe", "data frame", "index", "dtype",
        "data type", "formatting", "format", "column naming", "column-naming",
    ]
    has_type_mention = any(k in text for k in type_kw)

    if not has_type_mention:
        return False

    real_kw = [
        "filter", "aggregation", "aggregate", "double count", "wrong column",
        "wrong calculation", "unsupported", "calculation", "formula", "missing where",
        "missing condition", "wrong metric", "incorrect value", "math mismatch", "sum", "mean",
    ]
    has_real_issue = any(r in text for r in real_kw)
    return not has_real_issue


def filter_logic_issues(issues: list[str]) -> list[str]:
    """
    Filters out advisory issues that complain only about Series, DataFrame, index,
    dtype, or formatting, while keeping genuine logical and calculation problems.
    """
    return [
        s for s in issues
        if isinstance(s, str) and s.strip() and not _is_formatting_or_type_only_issue(s)
    ]


def _logic_review(
    question: str,
    code: str,
    res: Any,
    call_llm_fn: Callable[[str], str],
) -> list[str]:
    """
    Check 3: Solicits an advisory logic review from LLM.
    Issues returned are non-blocking warnings only.
    """
    preview = _result_to_markdown_preview(res, max_rows=10)
    prompt = f"""Review this analytical question, pandas code, and result preview for potential logical issues.
The sandbox automatically converts a pandas Series into a two-column DataFrame, so never report a Series versus DataFrame difference, or any pure formatting, type, index, or column-naming difference, as an issue. Only report problems that can change the numbers: wrong aggregation, missing filter, double counting, wrong column, or a conclusion not supported by the numbers.

Question: {question}
Pandas Code:
```python
{code}
```
Result preview:
{preview}

Return ONLY JSON: {{"issues": [list of short strings, empty if none]}}"""

    try:
        reply = call_llm_fn(prompt)
        cleaned = _clean_code_fence(reply)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "issues" in data and isinstance(data["issues"], list):
            raw_issues = [str(s).strip() for s in data["issues"] if str(s).strip()]
            return filter_logic_issues(raw_issues)
    except (LLMUnavailableError, LLMConfigError):
        raise
    except Exception:
        pass
    return []


# ============================================================================
# LangGraph Node: reviewer_node
# ============================================================================

def reviewer_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph Reviewer Node:
    - Verifies analysis results with status 'ok'.
    - Runs Check 1 (independent recomputation), Check 2 (rule checks), Check 3 (logic warnings).
    - Fails an item if Check 1 mismatches or Check 2 finds violations.
    - Updates review_feedback (failures only), review_report (complete audit),
      caught_errors, and events.
    """
    events = list(state.get("events", []))
    caught_errors = list(state.get("caught_errors", []))
    retry_round = state.get("retry_count", 0)
    rerun_ids = state.get("rerun_ids", [])
    previous_report = state.get("review_report", [])
    prev_report_by_id = {
        r["id"]: r for r in previous_report if isinstance(r, dict) and "id" in r
    }

    analysis_results = state.get("analysis_results", [])
    charts = state.get("charts", [])
    charts_by_id = {c.get("id"): c for c in charts if isinstance(c, dict)}

    review_feedback: list[dict[str, Any]] = []
    review_report: list[dict[str, Any]] = []

    # 1. Load dataset
    csv_path = state.get("clean_df_path") or state.get("raw_df_path")
    if not csv_path or not os.path.exists(csv_path):
        events.append({
            "agent": "reviewer",
            "type": "error",
            "message": f"Reviewer could not find dataset at '{csv_path}'",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return {
            "review_feedback": [],
            "review_report": [],
            "caught_errors": caught_errors,
            "events": events,
        }

    df = pd.read_csv(csv_path)
    columns_list = list(df.columns)

    for item in analysis_results:
        # Skip items that already failed analysis
        if item.get("status") != "ok":
            continue

        item_id = item.get("id")

        # On re-run, reuse previous report entry for items not re-run by analyst
        if rerun_ids and item_id not in rerun_ids:
            if item_id in prev_report_by_id:
                reused = dict(prev_report_by_id[item_id])
                review_report.append(reused)
                if reused.get("verdict") == "fail":
                    review_feedback.append({
                        "id": item_id,
                        "verdict": "fail",
                        "reason": reused.get("reason", ""),
                        "suggested_fix": "",
                        "original_value": reused.get("original_value"),
                        "recomputed_value": reused.get("recomputed_value"),
                    })
                continue

        q = item.get("question", "")
        code = item.get("code", "")
        original_res = item.get("result")
        chart = charts_by_id.get(item_id)

        try:
            # Validate that original_res is a supported result type
            if not isinstance(
                original_res,
                (pd.DataFrame, pd.Series, int, float, str, bool, np.number, np.bool_),
            ):
                raise TypeError(f"Unsupported result object type: {type(original_res).__name__}")

            # ----------------------------------------------------------------
            # Check 1: Independent Recomputation
            # ----------------------------------------------------------------
            recomp_out = _recompute_independently(
                question=q,
                original_code=code,
                original_res=original_res,
                columns=columns_list,
                df=df,
                call_llm_fn=call_llm,
            )
            c1_status = recomp_out[0]
            recomp_res = recomp_out[1]
            c1_note = recomp_out[2]
            recomp_code = recomp_out[3]
            c1_trace = recomp_out[4] if len(recomp_out) > 4 else {}

            # ----------------------------------------------------------------
            # Check 2: Rule-Based Checks (Real result problems only)
            # ----------------------------------------------------------------
            c2_problems = rule_checks(item, df, chart=chart)

            # Takeaway number check (Warning only, does not fail item or trigger retry)
            takeaway_warning: str | None = None
            if chart and chart.get("takeaway"):
                takeaway_problems = _extract_takeaway_numbers(chart["takeaway"], original_res)
                if takeaway_problems:
                    takeaway_warning = "; ".join(takeaway_problems)

            # ----------------------------------------------------------------
            # Check 3: Logic Review (Advisory Only)
            # ----------------------------------------------------------------
            c3_warnings = _logic_review(
                question=q,
                code=code,
                res=original_res,
                call_llm_fn=call_llm,
            )

            # Consolidate warnings
            warnings: list[str] = list(c3_warnings)
            if c1_status == "inconclusive" and c1_note:
                warnings.append(c1_note)
            if c1_status == "pass" and c1_note:
                warnings.append(c1_note)
            if takeaway_warning:
                warnings.append(f"takeaway_mismatch: {takeaway_warning}")

            # Determine final verdict (only real result failures fail the item)
            is_fail = (c1_status == "fail") or (len(c2_problems) > 0)
            if is_fail:
                verdict = "fail"
            elif c1_status == "inconclusive":
                verdict = "unverified"
            else:
                verdict = "pass"

            # Formulate reason and suggested fix
            reasons: list[str] = []
            if c1_status == "fail":
                reasons.append(c1_note or "Independent recomputation gave a different result")
            if c2_problems:
                reasons.extend(c2_problems)
            if c1_status == "inconclusive" and not is_fail:
                reasons.append(c1_note or "Independent recomputation was inconclusive; could not verify")

            reason_str = "; ".join(reasons) if reasons else "All checks passed successfully."

            if c1_status == "fail":
                suggested_fix = (
                    "Re-check the aggregation and filters; an independent method gave a different value."
                )
            elif c2_problems:
                suggested_fix = f"Fix the following data issues: {'; '.join(c2_problems)}."
            else:
                suggested_fix = ""

            orig_short = _shorten_val(original_res)
            recomp_short = _shorten_val(recomp_res) if recomp_res is not None else None

            # Capping real recomputed result at 20 rows if DataFrame / Series
            capped_recomp_res = recomp_res
            if isinstance(recomp_res, (pd.DataFrame, pd.Series)):
                capped_recomp_res = recomp_res.head(20).copy()

            # Track which check produced a failure
            failed_check: str | None = None
            if verdict == "fail":
                if c1_status == "fail" and c2_problems:
                    failed_check = "recomputation_mismatch; rule_check"
                elif c1_status == "fail":
                    failed_check = "recomputation_mismatch"
                elif c2_problems:
                    failed_check = "rule_check"

            # Add to full audit report
            review_report.append({
                "id": item_id,
                "question": q,
                "verdict": verdict,
                "reason": reason_str,
                "warnings": warnings,
                "logic_warnings": c3_warnings,
                "takeaway_warning": takeaway_warning,
                "original_value": orig_short,
                "recomputed_value": recomp_short,
                "recomputed_result": capped_recomp_res,
                "recomputed_code": recomp_code,
                "recomputation_code": recomp_code,
                "comparison_trace": c1_trace,
                "trace": c1_trace,
                "failed_check": failed_check,
            })

            # Record failing items in feedback and caught_errors
            if verdict == "fail":
                # Determine specific error type
                if c1_status == "fail":
                    error_type = "recomputation_mismatch"
                elif any("empty" in p.lower() for p in c2_problems):
                    error_type = "empty_result"
                else:
                    error_type = "rule_violation"

                orig_str = str(orig_short)[:300] if orig_short is not None else ""
                recomp_str = str(recomp_short)[:300] if recomp_short is not None else ""

                review_feedback.append({
                    "id": item_id,
                    "verdict": "fail",
                    "reason": reason_str,
                    "suggested_fix": suggested_fix,
                    "original_value": orig_str,
                    "recomputed_value": recomp_str,
                })
                caught_errors.append({
                    "id": item_id,
                    "question": q,
                    "reason": reason_str,
                    "type": error_type,
                    "retry_round": retry_round,
                    "original_value": orig_str,
                    "recomputed_value": recomp_str,
                    "original_code": code,
                    "recomputed_code": recomp_code,
                })

            events.append({
                "agent": "reviewer",
                "type": "info" if verdict == "pass" else "warning",
                "message": f"Review completed for item {item_id}: verdict '{verdict}'. {reason_str}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        except (LLMUnavailableError, LLMConfigError):
            raise
        except Exception as item_err:
            # Failure isolation: treat item as unverified with reason
            warn_msg = f"reviewer could not verify this item: {item_err}"
            orig_short = _shorten_val(original_res) if "original_res" in locals() else None
            exc_trace = {"branch": "exception", "reason": str(item_err), "matched": False}
            review_report.append({
                "id": item_id,
                "question": q,
                "verdict": "unverified",
                "reason": warn_msg,
                "warnings": [warn_msg],
                "logic_warnings": [],
                "takeaway_warning": None,
                "original_value": orig_short,
                "recomputed_value": None,
                "recomputed_result": None,
                "recomputed_code": "",
                "recomputation_code": "",
                "comparison_trace": exc_trace,
                "trace": exc_trace,
                "failed_check": None,
            })
            events.append({
                "agent": "reviewer",
                "type": "error",
                "message": f"Item {item_id} evaluation error: {item_err}. Marked as unverified.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    return {
        "review_feedback": review_feedback,
        "review_report": review_report,
        "caught_errors": caught_errors,
        "events": events,
    }
