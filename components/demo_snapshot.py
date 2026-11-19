"""
components/demo_snapshot.py
Pure Python and pandas snapshot manager for recorded demo mode.
Handles saving and loading of pipeline execution snapshots without Streamlit dependencies.
Enforces secret scanning, path sanitization, file size caps, and schema versioning.
"""

from datetime import datetime, timezone
import json
import os
import re
import shutil
import subprocess
from typing import Any
import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
MAX_SNAPSHOT_SIZE = 3 * 1024 * 1024  # 3 MB

SECRET_PATTERNS = ["gsk_", "AIza", "sk-", "api_key", "Bearer "]
LOCAL_PATH_PATTERNS = [r"[a-zA-Z]:\\", r"[a-zA-Z]:/", r"/home/", r"/Users/"]


class SnapshotError(Exception):
    """Raised when snapshot saving, loading, or validation fails."""
    pass


# ============================================================================
# Value Serialization & Deserialization
# ============================================================================

def _serialize_scalar(x: Any) -> Any:
    """Serializes a single scalar value into JSON-safe standard types."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if np.isnan(x) or np.isinf(x):
            return None if np.isnan(x) else str(x)
        return float(x)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    return str(x)


def _serialize_value(val: Any) -> Any:
    """Recursively converts objects (DataFrames, Series, numbers, etc.) to JSON-safe forms."""
    if val is None:
        return None
    if isinstance(val, pd.DataFrame):
        total_rows = int(len(val))
        df_sub = val.head(50)
        truncated = total_rows > 50
        cols = [str(c) for c in df_sub.columns]
        rows = []
        for r in df_sub.itertuples(index=False):
            rows.append([_serialize_scalar(x) for x in r])
        return {
            "__type__": "dataframe",
            "columns": cols,
            "rows": rows,
            "total_rows": total_rows,
            "truncated": truncated,
        }
    if isinstance(val, pd.Series):
        total_rows = int(len(val))
        s_sub = val.head(50)
        truncated = total_rows > 50
        cols = ["index", str(s_sub.name or "value")]
        rows = []
        for idx, item in s_sub.items():
            rows.append([_serialize_scalar(idx), _serialize_scalar(item)])
        return {
            "__type__": "series",
            "columns": cols,
            "rows": rows,
            "total_rows": total_rows,
            "truncated": truncated,
        }
    if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
        return int(val)
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return None if np.isnan(val) else str(val)
        return float(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    if isinstance(val, dict):
        return {str(k): _serialize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize_value(x) for x in val]
    return str(val)


def _deserialize_value(val: Any) -> Any:
    """Recursively reconstructs DataFrames and native types from serialized snapshot data."""
    if isinstance(val, dict):
        if val.get("__type__") in ("dataframe", "series") or (
            "columns" in val and "rows" in val and "total_rows" in val
        ):
            cols = val.get("columns", [])
            rows = val.get("rows", [])
            try:
                return pd.DataFrame(rows, columns=cols)
            except Exception:
                return pd.DataFrame()
        return {k: _deserialize_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_deserialize_value(x) for x in val]
    return val


# ============================================================================
# Security & Secret Scanning
# ============================================================================

def _find_violation(obj: Any, path: str = "") -> tuple[str, str] | None:
    """
    Recursively scans obj for secret patterns or local filesystem paths.
    Returns (violation_reason, field_name) if detected, else None.
    """
    if isinstance(obj, str):
        for pat in SECRET_PATTERNS:
            if pat in obj:
                return (f"secret pattern '{pat}'", path or "root")
        for pat in LOCAL_PATH_PATTERNS:
            if re.search(pat, obj):
                return (f"local path pattern '{pat}'", path or "root")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            k_str = str(k)
            for pat in SECRET_PATTERNS:
                if pat in k_str:
                    return (f"secret pattern '{pat}' in key", f"{path}.{k_str}" if path else k_str)
            sub = _find_violation(v, f"{path}.{k_str}" if path else k_str)
            if sub:
                return sub
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            sub = _find_violation(item, f"{path}[{idx}]")
            if sub:
                return sub
    return None


def _sanitize_string_paths(text: str) -> str:
    """Replaces absolute filesystem paths in messages with base filenames."""
    if not isinstance(text, str):
        return text
    # Replace Windows paths: e.g. E:\... or C:\...
    cleaned = re.sub(r"[a-zA-Z]:\\[^ '\"\n]+", lambda m: os.path.basename(m.group(0)), text)
    # Replace POSIX paths: e.g. /home/... or /Users/...
    cleaned = re.sub(r"/(?:home|Users)/[^ '\"\n]+", lambda m: os.path.basename(m.group(0)), cleaned)
    return cleaned


# ============================================================================
# Git Commit & Metadata Helpers
# ============================================================================

def _get_short_git_commit() -> str:
    """Extracts short git commit hash; returns 'unknown' on failure."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        out = res.stdout.strip()
        if out and len(out) >= 4:
            return out
    except Exception:
        pass
    return "unknown"


# ============================================================================
# Public Snapshot API
# ============================================================================

def save_snapshot(
    final_state: dict[str, Any],
    steps: list[dict[str, Any]],
    path: str,
    meta_extra: dict[str, Any] | None = None,
    force: bool = False,
) -> None:
    """
    Saves an execution snapshot to a single JSON file.

    Args:
        final_state: Final pipeline AgentState dictionary.
        steps: Recorded streaming node updates with elapsed times.
        path: Target file path for the snapshot.
        meta_extra: Optional additional metadata fields.
        force: If True, allows overwriting an existing file.
    """
    if os.path.exists(path) and not force:
        raise SnapshotError(
            f"Snapshot file '{path}' already exists. Use force=True to overwrite."
        )

    # 1. Dataset Before/After Metrics
    raw_path = final_state.get("raw_df_path", "")
    clean_path = final_state.get("clean_df_path", "")
    raw_rows, raw_cols = None, None
    clean_rows, clean_cols = None, None

    if raw_path and os.path.exists(raw_path):
        try:
            df_r = pd.read_csv(raw_path)
            raw_rows, raw_cols = int(len(df_r)), int(len(df_r.columns))
        except Exception:
            pass

    if clean_path and os.path.exists(clean_path):
        try:
            df_c = pd.read_csv(clean_path)
            clean_rows, clean_cols = int(len(df_c)), int(len(df_c.columns))
        except Exception:
            pass

    # Fallback to profile or stats if file is missing
    if raw_rows is None:
        prof = final_state.get("profile", {})
        raw_rows = prof.get("rows")
        raw_cols = prof.get("columns")

    # 2. Extract Run Stats
    stats = final_state.get("run_stats", {})
    by_model = stats.get("by_model", {})
    models_used = list(by_model.keys()) if by_model else stats.get("models_used", [])

    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "models_used": models_used,
        "git_commit": _get_short_git_commit(),
        "dataset_name": os.path.basename(raw_path) if raw_path else "dataset.csv",
        "raw_rows": raw_rows,
        "raw_cols": raw_cols,
        "clean_rows": clean_rows,
        "clean_cols": clean_cols,
        "tokens": int(stats.get("tokens", 0)),
        "llm_calls": int(stats.get("llm_calls", 0)),
        "seconds": float(stats.get("time_seconds", 0.0)),
        "retries_used": int(stats.get("retries_used", final_state.get("retry_count", 0))),
    }
    if meta_extra:
        meta.update(meta_extra)

    # 3. Compact Steps
    compact_steps: list[dict[str, Any]] = []
    for s in steps:
        node_name = s.get("node_name", "")
        elapsed = round(float(s.get("seconds", s.get("elapsed_seconds", 0.0))), 2)

        raw_evs = s.get("events", [])
        clean_evs = []
        for e in raw_evs:
            if isinstance(e, dict):
                clean_evs.append({
                    "agent": e.get("agent", ""),
                    "type": e.get("type", "info"),
                    "message": str(e.get("message", "")),
                    "timestamp": e.get("timestamp", ""),
                })

        has_fb = bool(s.get("review_feedback"))
        retry_cnt = int(s.get("retry_count", 0))

        step_dict: dict[str, Any] = {
            "node_name": node_name,
            "seconds": elapsed,
            "events": clean_evs,
            "has_review_feedback": has_fb,
            "review_feedback": [{"id": 0}] if has_fb else [],
            "retry_count": retry_cnt,
        }

        # Include status check info for analyst node
        if "analysis_results" in s:
            step_dict["analysis_results"] = [
                {"status": r.get("status", "ok")} for r in s["analysis_results"] if isinstance(r, dict)
            ]

        compact_steps.append(step_dict)

    # 4. Final State Data
    cleaning_summary = {
        "raw_rows": raw_rows,
        "raw_cols": raw_cols,
        "clean_rows": clean_rows,
        "clean_cols": clean_cols,
    }

    final_data: dict[str, Any] = {
        "final_report": str(final_state.get("final_report", "")),
        "cleaning_log": [str(item) for item in final_state.get("cleaning_log", [])],
        "questions": list(final_state.get("questions", [])),
        "analysis_results": _serialize_value(final_state.get("analysis_results", [])),
        "charts": _serialize_value(final_state.get("charts", [])),
        "review_report": _serialize_value(final_state.get("review_report", [])),
        "caught_errors": _serialize_value(final_state.get("caught_errors", [])),
        "run_stats": _serialize_value(stats),
        "events": list(final_state.get("events", [])),
        "cleaning_summary": cleaning_summary,
        "raw_df_path": os.path.basename(raw_path) if raw_path else "dataset.csv",
        "clean_df_path": os.path.basename(clean_path) if clean_path else "cleaned_dataset.csv",
        "retry_count": int(final_state.get("retry_count", 0)),
    }

    snapshot = {
        "meta": meta,
        "steps": compact_steps,
        "final": final_data,
    }

    # 5. Scan for Secrets & Absolute Local Paths
    violation = _find_violation(snapshot)
    if violation:
        reason, field = violation
        raise SnapshotError(
            f"Refusing to save snapshot: {reason} detected in field '{field}'."
        )

    # 6. Check Size Cap & Write
    json_text = json.dumps(snapshot, indent=2, ensure_ascii=False)
    json_bytes = json_text.encode("utf-8")
    if len(json_bytes) > MAX_SNAPSHOT_SIZE:
        raise SnapshotError(
            f"Snapshot size ({len(json_bytes)} bytes) exceeds the maximum allowed limit of 3 MB."
        )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(json_bytes)


def save_cleaned_csv_for_demo(final_state: dict[str, Any], path: str) -> bool:
    """
    Copies the cleaned CSV file next to the snapshot if it exists and is under 3 MB.
    Returns True on success, False otherwise.
    """
    clean_path = final_state.get("clean_df_path")
    if not clean_path or not os.path.exists(clean_path):
        return False
    try:
        size = os.path.getsize(clean_path)
        if size > MAX_SNAPSHOT_SIZE:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        shutil.copyfile(clean_path, path)
        return True
    except Exception:
        return False


def load_snapshot(path: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """
    Loads and validates an execution snapshot from JSON.

    Returns:
        tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
            (final_state, steps, meta)

    Raises:
        SnapshotError: If the file is missing, corrupted, lacks required sections,
                       or has an unsupported schema version.
    """
    if not os.path.exists(path):
        raise SnapshotError(f"Snapshot file not found at '{path}'.")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise SnapshotError(f"Corrupted or invalid JSON snapshot in '{path}': {e}") from e

    if not isinstance(data, dict):
        raise SnapshotError(f"Invalid snapshot structure in '{path}': expected JSON object.")

    for req_key in ("meta", "steps", "final"):
        if req_key not in data:
            raise SnapshotError(f"Corrupted snapshot in '{path}': missing required section '{req_key}'.")

    meta = data["meta"]
    if not isinstance(meta, dict):
        raise SnapshotError(f"Invalid 'meta' section in '{path}'.")

    schema_ver = meta.get("schema_version")
    if schema_ver is None:
        raise SnapshotError(f"Snapshot '{path}' is missing 'schema_version'.")
    if schema_ver > SCHEMA_VERSION:
        raise SnapshotError(
            f"Snapshot schema version {schema_ver} is newer than supported version {SCHEMA_VERSION}."
        )

    steps = data["steps"]
    if not isinstance(steps, list):
        raise SnapshotError(f"Invalid 'steps' section in '{path}': expected list.")

    final_raw = data["final"]
    if not isinstance(final_raw, dict):
        raise SnapshotError(f"Invalid 'final' section in '{path}': expected object.")

    final_state = _deserialize_value(final_raw)
    return final_state, steps, meta
