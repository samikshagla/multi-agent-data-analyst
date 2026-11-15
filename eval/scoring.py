"""
scoring.py
Independent Comparator for Multi-Agent Data Analyst Evaluation.
Compares reference and candidate results across scalars and tables with tolerance,
multiset row matching, column-name invariance, and ignorable index handling.
"""

from typing import Any
import numpy as np
import pandas as pd


def _to_dataframe(val: Any) -> pd.DataFrame | None:
    """Converts a Series or dict representation into a pandas DataFrame."""
    if isinstance(val, pd.DataFrame):
        return val.copy(deep=True)
    if isinstance(val, pd.Series):
        return val.reset_index()
    if isinstance(val, dict) and "columns" in val and "rows" in val:
        return pd.DataFrame(val["rows"], columns=val["columns"])
    return None


def _is_scalar(val: Any) -> bool:
    """Returns True if the value is a scalar (not a DataFrame, Series, list, or dict)."""
    if isinstance(val, (pd.DataFrame, pd.Series, list, dict)):
        return False
    return True


def _is_ignorable_index_col(col_name: Any, series: pd.Series) -> bool:
    """
    Checks if a column is an ignorable index column (e.g. 0..n-1 RangeIndex
    or name matching 'index', 'level_0', 'Unnamed: 0').
    """
    n = len(series)
    name_str = str(col_name).strip().lower()
    is_index_name = name_str in {"index", "level_0", "level_1", "unnamed: 0", "none", ""}

    # Check if values form sequential 0..n-1 integers
    try:
        vals = series.tolist()
        if vals == list(range(n)):
            return True
    except Exception:
        pass

    return is_index_name and n <= 1


def _drop_ignorable_index_cols(df: pd.DataFrame, target_col_count: int | None = None) -> pd.DataFrame:
    """
    Drops ignorable index columns from a DataFrame, especially when it has extra columns
    compared to the target column count.
    """
    if df.empty or df.shape[1] <= 1:
        return df

    cols_to_drop = []
    for col in df.columns:
        if _is_ignorable_index_col(col, df[col]):
            cols_to_drop.append(col)

    if target_col_count is not None:
        # Only drop as many as needed to match target_col_count
        excess = df.shape[1] - target_col_count
        if excess > 0 and len(cols_to_drop) >= excess:
            return df.drop(columns=cols_to_drop[:excess])
        elif len(cols_to_drop) > 0 and (df.shape[1] - len(cols_to_drop)) == target_col_count:
            return df.drop(columns=cols_to_drop)
        return df
    else:
        # Drop known index columns if other columns exist
        if len(cols_to_drop) > 0 and len(cols_to_drop) < df.shape[1]:
            return df.drop(columns=cols_to_drop)
        return df


def _extract_one_cell_scalar(df: pd.DataFrame) -> Any | None:
    """Extracts scalar from a one-cell table (1x1 or 1x2 with ignorable index)."""
    if not isinstance(df, pd.DataFrame) or len(df) != 1:
        return None

    if df.shape == (1, 1):
        return df.iloc[0, 0]

    # Try dropping ignorable index column
    cleaned = _drop_ignorable_index_cols(df, target_col_count=1)
    if cleaned.shape == (1, 1):
        return cleaned.iloc[0, 0]

    return None


def _compare_scalars(ref: Any, cand: Any, tol: float = 0.01) -> tuple[bool, str]:
    """Compares two scalar values with relative tolerance for numbers."""
    if ref is None and cand is None:
        return True, ""
    if ref is None or cand is None:
        return False, f"value {cand} vs {ref}"

    # Boolean check (must precede int check since bool subclasses int)
    if isinstance(ref, (bool, np.bool_)) or isinstance(cand, (bool, np.bool_)):
        if bool(ref) == bool(cand):
            return True, ""
        return False, f"value {cand} vs {ref}"

    # Numeric check
    if isinstance(ref, (int, float, np.number)) and isinstance(cand, (int, float, np.number)):
        r = float(ref)
        c = float(cand)
        if np.isnan(r) and np.isnan(c):
            return True, ""
        if np.isnan(r) or np.isnan(c):
            return False, "NaN value encountered"
        if r == 0.0:
            if abs(c) <= 1e-6:
                return True, ""
            return False, f"value {c} vs {r}"
        rel_diff = abs(c - r) / abs(r)
        if rel_diff <= tol:
            return True, ""
        return False, f"value {c} vs {r}"

    # String / other check
    s_ref = str(ref).strip().lower()
    s_cand = str(cand).strip().lower()
    if s_ref == s_cand:
        return True, ""
    return False, f"value '{cand}' vs '{ref}'"


def _normalize_row(row_series: pd.Series) -> tuple[list[str], list[float]]:
    """Separates and normalizes row values into sorted texts and sorted numbers."""
    texts: list[str] = []
    nums: list[float] = []

    for cell in row_series:
        if cell is None or pd.isna(cell):
            texts.append("nan")
        elif isinstance(cell, (bool, np.bool_)):
            texts.append(str(bool(cell)).lower())
        elif isinstance(cell, (int, float, np.number)):
            nums.append(float(cell))
        else:
            texts.append(str(cell).strip().lower())

    return sorted(texts), sorted(nums)


def _rows_match(row_ref: pd.Series, row_cand: pd.Series, tol: float) -> tuple[bool, str]:
    """Checks whether two normalized rows match within tolerance."""
    ref_texts, ref_nums = _normalize_row(row_ref)
    cand_texts, cand_nums = _normalize_row(row_cand)

    if ref_texts != cand_texts:
        return False, f"labels {cand_texts} vs {ref_texts}"

    if len(ref_nums) != len(cand_nums):
        return False, f"numeric count {len(cand_nums)} vs {len(ref_nums)}"

    for rn, cn in zip(ref_nums, cand_nums):
        if np.isnan(rn) and np.isnan(cn):
            continue
        if np.isnan(rn) or np.isnan(cn):
            return False, "NaN value encountered"
        if rn == 0.0:
            if abs(cn) > 1e-6:
                return False, f"value {cn} vs {rn}"
        else:
            if abs(cn - rn) / abs(rn) > tol:
                return False, f"value {cn} vs {rn}"

    return True, ""


def answer_correct(reference: Any, candidate: Any, tol: float = 0.01) -> tuple[bool, str]:
    """
    Independent comparator between reference answer and candidate answer.

    Rules:
    - scalars: relative tolerance tol (default 1%), handles zero.
    - tables: same number of rows; compare the multiset of rows after normalising:
      text values lowercased and stripped, numeric values compared within tolerance;
      column order and column names may differ;
      candidate with extra columns is rejected unless extra columns are ignorable index columns.
    - scalar vs a one-cell table counts as equal.
    - returns (bool, reason) where reason is a short explanation on failure.
    """
    if reference is None:
        return False, "reference is None"
    if candidate is None:
        return False, "candidate is None"

    # Convert Series and dictionary formats to DataFrames
    ref_df = _to_dataframe(reference)
    cand_df = _to_dataframe(candidate)

    # Check for one-cell table equivalence to scalar
    ref_is_scalar = _is_scalar(reference) and ref_df is None
    cand_is_scalar = _is_scalar(candidate) and cand_df is None

    # Both are scalars
    if ref_is_scalar and cand_is_scalar:
        return _compare_scalars(reference, candidate, tol)

    # Reference is scalar, candidate is table
    if ref_is_scalar and cand_df is not None:
        cand_scalar = _extract_one_cell_scalar(cand_df)
        if cand_scalar is not None:
            return _compare_scalars(reference, cand_scalar, tol)
        return False, f"reference is scalar but candidate has {len(cand_df)} rows and {cand_df.shape[1]} columns"

    # Candidate is scalar, reference is table
    if cand_is_scalar and ref_df is not None:
        ref_scalar = _extract_one_cell_scalar(ref_df)
        if ref_scalar is not None:
            return _compare_scalars(ref_scalar, candidate, tol)
        return False, f"candidate is scalar but reference has {len(ref_df)} rows and {ref_df.shape[1]} columns"

    # Both are tables
    assert ref_df is not None and cand_df is not None

    # Handle ignorable index columns in candidate if candidate has extra columns
    if cand_df.shape[1] > ref_df.shape[1]:
        cand_df = _drop_ignorable_index_cols(cand_df, target_col_count=ref_df.shape[1])

    # Check column count
    if cand_df.shape[1] != ref_df.shape[1]:
        return False, f"column count {cand_df.shape[1]} vs {ref_df.shape[1]}"

    # Check row count
    if len(cand_df) != len(ref_df):
        return False, f"row count {len(cand_df)} vs {len(ref_df)}"

    n = len(ref_df)
    if n == 0:
        return True, ""

    # Build bipartite matching graph between ref rows and cand rows
    # adj[i] = list of candidate row indices j that match reference row i
    adj: list[list[int]] = [[] for _ in range(n)]
    last_reason = "row mismatch"

    for i in range(n):
        ref_row = ref_df.iloc[i]
        for j in range(n):
            cand_row = cand_df.iloc[j]
            ok, reason = _rows_match(ref_row, cand_row, tol)
            if ok:
                adj[i].append(j)
            else:
                last_reason = reason

    # Maximum Bipartite Matching via augmenting paths DFS
    match_cand = [-1] * n

    def _dfs_augment(u: int, visited: list[bool]) -> bool:
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                if match_cand[v] < 0 or _dfs_augment(match_cand[v], visited):
                    match_cand[v] = u
                    return True
        return False

    matched_count = 0
    for u in range(n):
        visited = [False] * n
        if _dfs_augment(u, visited):
            matched_count += 1
        else:
            # First unmatched reference row
            # Provide specific reason if possible
            ref_row = ref_df.iloc[u]
            ref_texts, ref_nums = _normalize_row(ref_row)
            # Find closest candidate row with matching labels
            for j in range(n):
                c_texts, c_nums = _normalize_row(cand_df.iloc[j])
                if c_texts == ref_texts:
                    for rn, cn in zip(ref_nums, c_nums):
                        if abs(cn - rn) / (abs(rn) if rn != 0 else 1.0) > tol:
                            return False, f"value {cn} vs {rn}"
            return False, last_reason

    if matched_count == n:
        return True, ""

    return False, last_reason
