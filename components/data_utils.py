"""
components/data_utils.py
Pure utility functions for Streamlit UI data operations (no Streamlit dependencies).
Provides robust CSV loading with encoding fallbacks and size limits, dataset snapshot
profiling, and safe file upload persistence.
"""

from datetime import datetime
import io
import os
from pathlib import Path
import re
from typing import Any
import pandas as pd

from tools.profiler import _is_possible_date_series

# Demo dataset constraints
MAX_ROWS = 200_000
MAX_COLS = 100


def load_csv_safe(file_or_path: Any) -> tuple[pd.DataFrame | None, str | None]:
    """
    Safely loads a CSV file from a file path, BytesIO, or file-like upload object.
    Tries utf-8, utf-8-sig, and latin-1 encodings in sequence.

    Validates:
    - Non-empty file (size > 0 bytes)
    - Successful parsing as tabular CSV
    - At least one column
    - At least two data rows (not header-only or single-row)
    - Demo limits: <= 200,000 rows and <= 100 columns

    Args:
        file_or_path: File path (str/Path), file-like object (BytesIO/UploadedFile), or bytes.

    Returns:
        tuple[pd.DataFrame | None, str | None]: (DataFrame, None) on success,
        or (None, friendly_error_message) on failure.
    """
    if file_or_path is None:
        return None, "No file was provided."

    raw_bytes: bytes

    # 1. Read raw bytes from string/Path or file-like object
    try:
        if isinstance(file_or_path, (str, Path)):
            path_obj = Path(file_or_path)
            if not path_obj.exists():
                return None, f"File not found: '{file_or_path}'."
            if not path_obj.is_file():
                return None, f"Specified path is not a file: '{file_or_path}'."
            raw_bytes = path_obj.read_bytes()
        elif isinstance(file_or_path, bytes):
            raw_bytes = file_or_path
        elif hasattr(file_or_path, "read"):
            content = file_or_path.read()
            # Rewind file pointer if supported
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            if isinstance(content, str):
                raw_bytes = content.encode("utf-8")
            else:
                raw_bytes = bytes(content)
        elif hasattr(file_or_path, "getbuffer"):
            raw_bytes = bytes(file_or_path.getbuffer())
        elif hasattr(file_or_path, "getvalue"):
            val = file_or_path.getvalue()
            raw_bytes = val.encode("utf-8") if isinstance(val, str) else bytes(val)
        else:
            return None, f"Unsupported file type: {type(file_or_path).__name__}."
    except Exception as read_err:
        return None, f"Error reading file content: {read_err}."

    # 2. Check for empty content
    if len(raw_bytes.strip()) == 0:
        return None, "The CSV file is empty (0 bytes)."

    # Check for binary or corrupt content (null bytes are invalid in CSV text)
    if b"\x00" in raw_bytes:
        return None, "Could not parse file as CSV: binary or corrupt file format detected."

    # 3. Try parsing across candidate encodings: utf-8, utf-8-sig, latin-1
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    df: pd.DataFrame | None = None
    last_parse_error: str | None = None

    for enc in encodings:
        try:
            # Parse with pandas from bytes buffer
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
            last_parse_error = None
            break
        except pd.errors.EmptyDataError:
            return None, "The CSV file is empty or contains no parseable columns."
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as parse_err:
            last_parse_error = str(parse_err)
            # Some parser errors might be due to bad encoding bytes, continue trying next encoding
            continue

    if df is None:
        err_msg = last_parse_error or "Could not decode file content."
        return (
            None,
            f"Could not parse file as CSV. Please ensure it is valid CSV text (UTF-8 or Latin-1). Details: {err_msg}",
        )

    # 4. Validate columns
    if len(df.columns) == 0:
        return None, "The CSV file contains no columns."

    # 5. Validate row counts
    num_rows = len(df)
    if num_rows == 0:
        return None, "The CSV file contains only headers and no data rows."
    if num_rows == 1:
        return (
            None,
            "The CSV file contains only one row of data. Multi-agent analysis requires at least 2 rows to analyze trends.",
        )

    # 6. Validate demo limits
    if num_rows > MAX_ROWS:
        return (
            None,
            f"Dataset exceeds demo limits: {num_rows:,} rows found (maximum allowed is {MAX_ROWS:,} rows).",
        )
    if len(df.columns) > MAX_COLS:
        return (
            None,
            f"Dataset exceeds demo limits: {len(df.columns)} columns found (maximum allowed is {MAX_COLS} columns).",
        )

    return df, None


def dataset_snapshot(df: pd.DataFrame) -> dict[str, Any]:
    """
    Computes a comprehensive overview dictionary of a loaded DataFrame:
    - rows: total row count
    - columns: total column count
    - missing_pct: overall missing value percentage (across all cells)
    - duplicate_rows: count of identical duplicate rows
    - numeric_cols: count of numeric columns
    - text_cols: count of string/text columns
    - date_like_cols: count of datetime or parseable date-like columns

    Reuses date detection logic from tools/profiler.py.

    Args:
        df: Input pandas DataFrame.

    Returns:
        dict[str, Any]: Dataset metrics summary dictionary.
    """
    rows = int(len(df))
    cols = int(len(df.columns))
    duplicate_rows = int(df.duplicated().sum()) if rows > 0 else 0

    total_cells = rows * cols
    missing_cells = int(df.isna().sum().sum())
    missing_pct = round((missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0

    numeric_count = 0
    date_like_count = 0
    text_count = 0

    for col in df.columns:
        series = df[col]
        # Check datetime dtype
        if pd.api.types.is_datetime64_any_dtype(series):
            date_like_count += 1
        # Check numeric dtype (excluding booleans)
        elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric_count += 1
        # Check possible date string series from profiler
        elif _is_possible_date_series(series):
            date_like_count += 1
        else:
            text_count += 1

    return {
        "rows": rows,
        "columns": cols,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "numeric_cols": numeric_count,
        "text_cols": text_count,
        "date_like_cols": date_like_count,
    }


def save_upload(uploaded_file: Any, upload_dir: str | Path = "data/uploads") -> Path:
    """
    Persists an uploaded file to data/uploads/<timestamp>_<safe filename>.csv.
    - Sanitizes filename to only permit letters, numbers, dot, dash, and underscores.
    - Strips directory traversal characters (e.g. '../../x y.csv' -> 'x_y.csv').
    - Ensures target directory exists.
    - Never overwrites an existing file (appends an incremental numeric suffix if collision occurs).

    Args:
        uploaded_file: Uploaded file object (with .name and .getbuffer/.read) or file path.
        upload_dir: Directory to save the uploaded file into (default 'data/uploads').

    Returns:
        Path: Absolute or relative Path object of the saved file.
    """
    target_dir = Path(upload_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract raw filename
    raw_name = getattr(uploaded_file, "name", "uploaded_file.csv")
    if isinstance(uploaded_file, (str, Path)) and not hasattr(uploaded_file, "read"):
        raw_name = Path(uploaded_file).name

    # 2. Extract base name and strip path separators
    normalized_name = str(raw_name).replace("\\", "/")
    base_name = os.path.basename(normalized_name)

    # 3. Sanitize filename: replace spaces with underscores, allow letters, numbers, dot, dash, underscore
    clean_name = base_name.replace(" ", "_")
    safe_name = re.sub(r"[^a-zA-Z0-9.\-_]", "", clean_name)

    # Fallback if name becomes empty after sanitization
    if not safe_name or safe_name.strip(".") == "":
        safe_name = "upload.csv"

    # Ensure .csv extension
    if not safe_name.lower().endswith(".csv"):
        safe_name = f"{safe_name}.csv"

    # 4. Form candidate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_name = f"{timestamp}_{safe_name}"
    target_path = target_dir / candidate_name

    # 5. Collision avoidance: never overwrite an existing file
    stem = target_path.stem
    ext = target_path.suffix
    counter = 1
    while target_path.exists():
        target_path = target_dir / f"{stem}_{counter}{ext}"
        counter += 1

    # 6. Extract content bytes
    content_bytes: bytes
    if hasattr(uploaded_file, "getbuffer"):
        content_bytes = bytes(uploaded_file.getbuffer())
    elif hasattr(uploaded_file, "getvalue"):
        val = uploaded_file.getvalue()
        content_bytes = val.encode("utf-8") if isinstance(val, str) else bytes(val)
    elif hasattr(uploaded_file, "read"):
        val = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        content_bytes = val.encode("utf-8") if isinstance(val, str) else bytes(val)
    elif isinstance(uploaded_file, (str, Path)):
        content_bytes = Path(uploaded_file).read_bytes()
    elif isinstance(uploaded_file, bytes):
        content_bytes = uploaded_file
    else:
        content_bytes = b""

    # 7. Write to disk
    target_path.write_bytes(content_bytes)
    return target_path
