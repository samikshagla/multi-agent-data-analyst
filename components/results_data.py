"""
components/results_data.py
Pure Python and pandas data preparation functions for analysis results tabs.
Zero Streamlit imports to ensure reliable, modular, unit-testable execution.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any
import numpy as np
import pandas as pd

from components.data_utils import load_csv_safe


# ============================================================================
# 1. Item Verification Tagging
# ============================================================================

def item_tag(final_state: dict[str, Any], item_id: int) -> str:
    """
    Determines the verification status tag for an analytical item.

    Return values:
        - "Failed": if the analysis status is failed.
        - "Unresolved": if the latest review verdict is fail.
        - "Unverified": if the verdict is unverified.
        - "Fixed after retry": if the id is in caught_errors and latest review verdict is pass.
        - "Verified": if pass and never caught.

    Args:
        final_state: Full pipeline execution state dictionary.
        item_id: Integer identifier of the analysis item.

    Returns:
        str: One of ("Verified", "Fixed after retry", "Unverified", "Unresolved", "Failed").
    """
    if not isinstance(final_state, dict):
        return "Failed"

    analysis_results = final_state.get("analysis_results", [])
    if not isinstance(analysis_results, list):
        analysis_results = []

    review_report = final_state.get("review_report", [])
    if not isinstance(review_report, list):
        review_report = []

    caught_errors = final_state.get("caught_errors", [])
    if not isinstance(caught_errors, list):
        caught_errors = []

    # 1. Check if the analysis itself failed
    ar_item = next(
        (a for a in analysis_results if isinstance(a, dict) and a.get("id") == item_id),
        None,
    )
    if ar_item is not None and ar_item.get("status") == "failed":
        return "Failed"

    # 2. Extract latest review verdict for this item
    item_reviews = [
        r for r in review_report if isinstance(r, dict) and r.get("id") == item_id
    ]
    if item_reviews:
        latest_verdict = str(item_reviews[-1].get("verdict", "")).strip().lower()
        if latest_verdict == "fail":
            return "Unresolved"
        elif latest_verdict == "unverified":
            return "Unverified"
        elif latest_verdict == "pass":
            # Check if this id was intercepted in caught_errors
            was_caught = any(
                isinstance(e, dict) and e.get("id") == item_id for e in caught_errors
            )
            return "Fixed after retry" if was_caught else "Verified"

    # 3. If no review was performed, infer from analysis status
    if ar_item is not None:
        if ar_item.get("status") == "ok":
            return "Unverified"
        return "Failed"

    return "Failed"


# ============================================================================
# 2. Chart Items Extraction
# ============================================================================

def build_chart_items(final_state: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Builds a display list of chart items including generated Plotly specs,
    metric cards, and skipped/failed items with explanatory notes.

    Args:
        final_state: State dictionary from final pipeline execution.

    Returns:
        list[dict[str, Any]]: Display items containing id, question, title,
        type, takeaway, figure_json, tag, note, status, and takeaway_warning.
    """
    if not isinstance(final_state, dict):
        return []

    analysis_results = final_state.get("analysis_results", [])
    if not isinstance(analysis_results, list):
        analysis_results = []

    charts = final_state.get("charts", [])
    if not isinstance(charts, list):
        charts = []

    review_report = final_state.get("review_report", [])
    if not isinstance(review_report, list):
        review_report = []

    charts_by_id = {
        c["id"]: c for c in charts if isinstance(c, dict) and "id" in c
    }
    reviews_by_id = {
        r["id"]: r for r in review_report if isinstance(r, dict) and "id" in r
    }

    seen_ids: set[int] = set()
    items: list[dict[str, Any]] = []

    # Iterate through analysis_results to preserve question order
    for ar in analysis_results:
        if not isinstance(ar, dict) or "id" not in ar:
            continue
        iid = ar["id"]
        seen_ids.add(iid)
        question = ar.get("question", "")
        chart = charts_by_id.get(iid)
        rev = reviews_by_id.get(iid, {})
        tag = item_tag(final_state, iid)
        takeaway_warning = rev.get("takeaway_warning")

        if chart:
            title = chart.get("title") or question or f"Question {iid + 1}"
            ctype = chart.get("type", "unknown")
            takeaway = chart.get("takeaway", "")
            fig_json = chart.get("figure_json")
            note = chart.get("note", "")
            status = chart.get("status", "ok")
        else:
            title = question or f"Question {iid + 1}"
            ctype = "unknown"
            takeaway = ""
            fig_json = None
            status = "skipped"
            note = ar.get("error") or "No chart generated"

        items.append({
            "id": iid,
            "question": question,
            "title": title,
            "type": ctype,
            "takeaway": takeaway,
            "figure_json": fig_json,
            "tag": tag,
            "note": note,
            "status": status,
            "takeaway_warning": takeaway_warning,
        })

    # Include any charts whose id was not in analysis_results
    for chart in charts:
        if not isinstance(chart, dict) or "id" not in chart:
            continue
        iid = chart["id"]
        if iid not in seen_ids:
            seen_ids.add(iid)
            rev = reviews_by_id.get(iid, {})
            tag = item_tag(final_state, iid)
            title = chart.get("title") or f"Chart {iid + 1}"
            items.append({
                "id": iid,
                "question": title,
                "title": title,
                "type": chart.get("type", "unknown"),
                "takeaway": chart.get("takeaway", ""),
                "figure_json": chart.get("figure_json"),
                "tag": tag,
                "note": chart.get("note", ""),
                "status": chart.get("status", "ok"),
                "takeaway_warning": rev.get("takeaway_warning"),
            })

    return items


# ============================================================================
# 3. Cleaning Log & Dataset Summaries
# ============================================================================

def get_cleaning_summary(final_state: dict[str, Any]) -> dict[str, Any]:
    """
    Computes before/after rows and column counts of raw and cleaned datasets.
    Safely handles missing files and returns None values if files are unreadable.
    """
    if not isinstance(final_state, dict):
        return {
            "raw_rows": None,
            "raw_cols": None,
            "clean_rows": None,
            "clean_cols": None,
        }

    # Precomputed summary from snapshot (e.g. demo mode)
    if "cleaning_summary" in final_state and isinstance(final_state["cleaning_summary"], dict):
        return dict(final_state["cleaning_summary"])

    raw_path = final_state.get("raw_df_path")
    clean_path = final_state.get("clean_df_path")

    raw_rows, raw_cols = None, None
    if raw_path and os.path.exists(raw_path):
        try:
            df_raw, err = load_csv_safe(raw_path)
            if df_raw is not None and err is None:
                raw_rows, raw_cols = int(len(df_raw)), int(len(df_raw.columns))
        except Exception:
            pass

    clean_rows, clean_cols = None, None
    if clean_path and os.path.exists(clean_path):
        try:
            df_clean, err = load_csv_safe(clean_path)
            if df_clean is not None and err is None:
                clean_rows, clean_cols = int(len(df_clean)), int(len(df_clean.columns))
        except Exception:
            pass

    return {
        "raw_rows": raw_rows,
        "raw_cols": raw_cols,
        "clean_rows": clean_rows,
        "clean_cols": clean_cols,
    }


def build_cleaning_rows(
    final_state: dict[str, Any],
    return_summary: bool = True,
) -> Any:
    """
    Classifies cleaning log entries into applied, skipped, or warning categories.

    Rules:
    - Starts with "Skipped" -> "skipped"
    - Contains "warning" -> "warning"
    - Otherwise -> "applied"

    Args:
        final_state: State dictionary from final pipeline execution.
        return_summary: If True, returns (rows, summary_dict).

    Returns:
        tuple[list[dict[str, Any]], dict[str, Any]] or list[dict[str, Any]].
    """
    if not isinstance(final_state, dict):
        log_entries = []
    else:
        log_entries = final_state.get("cleaning_log", [])
        if not isinstance(log_entries, list):
            log_entries = []

    rows: list[dict[str, Any]] = []
    for idx, entry in enumerate(log_entries, start=1):
        msg = str(entry).strip()
        lowered = msg.lower()
        if lowered.startswith("skipped"):
            kind = "skipped"
        elif "warning" in lowered:
            kind = "warning"
        else:
            kind = "applied"

        rows.append({
            "step": idx,
            "kind": kind,
            "message": msg,
        })

    summary = get_cleaning_summary(final_state)

    if return_summary:
        return rows, summary
    return rows


# ============================================================================
# 4. Reviewer Findings & Audit Cards
# ============================================================================

def _shorten_val(val: Any, max_len: int = 300) -> str:
    """Safely converts an object into a compact string representation."""
    if val is None:
        return ""
    if isinstance(val, pd.DataFrame):
        val_str = f"DataFrame ({len(val)} rows, {len(val.columns)} cols):\n{val.head(5).to_string()}"
    elif isinstance(val, pd.Series):
        val_str = f"Series ({len(val)} items):\n{val.head(5).to_string()}"
    else:
        val_str = str(val)

    if len(val_str) > max_len:
        return val_str[:max_len] + "..."
    return val_str


def prepare_display_value(val: Any) -> dict[str, Any]:
    """
    Builds a display-ready representation of an analyst result or recomputed value.

    Rules:
    - Scalars: formatted number with thousands separators and at most 2 decimals.
    - DataFrames & Series: small dataframe with first 8 rows, and caption
      "showing first 8 of N rows" when total rows exceed 8.
    - Fallback: string representation if object is text or raw value unavailable.
    - Missing / None: gracefully handled without crashing.

    Returns:
        dict[str, Any] with:
            - kind: "scalar" | "dataframe" | "text" | "missing"
            - value: raw or sliced object
            - formatted: formatted string (for scalars and text)
            - caption: optional string (e.g. "showing first 8 of N rows")
            - total_rows: int (for dataframes/series)
            - total_cols: int (for dataframes/series)
    """
    if val is None:
        return {
            "kind": "missing",
            "value": None,
            "formatted": "(None)",
            "caption": None,
            "total_rows": 0,
            "total_cols": 0,
        }

    # 1. DataFrame
    if isinstance(val, pd.DataFrame):
        total_rows = len(val)
        total_cols = len(val.columns)
        df_sub = val.head(8)
        caption = f"showing first 8 of {total_rows:,} rows" if total_rows > 8 else None
        return {
            "kind": "dataframe",
            "value": df_sub,
            "formatted": "",
            "caption": caption,
            "total_rows": total_rows,
            "total_cols": total_cols,
        }

    # 2. Series
    if isinstance(val, pd.Series):
        total_rows = len(val)
        df_converted = val.to_frame().reset_index()
        total_cols = len(df_converted.columns)
        df_sub = df_converted.head(8)
        caption = f"showing first 8 of {total_rows:,} rows" if total_rows > 8 else None
        return {
            "kind": "dataframe",
            "value": df_sub,
            "formatted": "",
            "caption": caption,
            "total_rows": total_rows,
            "total_cols": total_cols,
        }

    # 3. Numeric Scalar (int, float, np.integer, np.floating)
    if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
        return {
            "kind": "scalar",
            "value": val,
            "formatted": f"{int(val):,}",
            "caption": None,
            "total_rows": 1,
            "total_cols": 1,
        }

    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            formatted = str(val)
        else:
            rounded = round(float(val), 2)
            if rounded.is_integer():
                formatted = f"{int(rounded):,}"
            else:
                formatted = f"{rounded:,.2f}".rstrip("0").rstrip(".")
        return {
            "kind": "scalar",
            "value": val,
            "formatted": formatted,
            "caption": None,
            "total_rows": 1,
            "total_cols": 1,
        }

    # 4. String that might represent a number or textual summary fallback
    if isinstance(val, str):
        cleaned_str = val.strip()
        try:
            num = float(cleaned_str)
            if not (np.isnan(num) or np.isinf(num)):
                rounded = round(num, 2)
                if rounded.is_integer() and "." not in cleaned_str:
                    formatted = f"{int(rounded):,}"
                else:
                    formatted = f"{rounded:,.2f}".rstrip("0").rstrip(".")
                return {
                    "kind": "scalar",
                    "value": num,
                    "formatted": formatted,
                    "caption": None,
                    "total_rows": 1,
                    "total_cols": 1,
                }
        except ValueError:
            pass

        return {
            "kind": "text",
            "value": val,
            "formatted": cleaned_str,
            "caption": None,
            "total_rows": 0,
            "total_cols": 0,
        }

    # 5. Generic fallback
    return {
        "kind": "text",
        "value": val,
        "formatted": str(val),
        "caption": None,
        "total_rows": 0,
        "total_cols": 0,
    }


def find_differing_values(orig: Any, recomp: Any) -> list[str]:
    """
    Highlights the first 3 differing rows or cells when an analyst result
    and reviewer recomputation mismatch.
    """
    if orig is None or recomp is None:
        return [f"Analyst: {orig} vs Recomputed: {recomp}"]

    # 1. Both are numeric scalars
    if isinstance(orig, (int, float, np.number)) and isinstance(recomp, (int, float, np.number)):
        return [f"Analyst: {orig} vs Recomputed: {recomp}"]

    # 2. Both are DataFrames or Series
    df_o = orig.to_frame().reset_index() if isinstance(orig, pd.Series) else orig
    df_r = recomp.to_frame().reset_index() if isinstance(recomp, pd.Series) else recomp

    if isinstance(df_o, pd.DataFrame) and isinstance(df_r, pd.DataFrame):
        if len(df_o) != len(df_r):
            return [f"Row count mismatch: Analyst has {len(df_o)} rows vs Recomputed has {len(df_r)} rows"]

        diffs: list[str] = []
        common_cols = [c for c in df_o.columns if c in df_r.columns]
        if not common_cols:
            return [f"Column mismatch: Analyst {list(df_o.columns)} vs Recomputed {list(df_r.columns)}"]

        for idx in range(min(len(df_o), len(df_r))):
            row_o = df_o.iloc[idx]
            row_r = df_r.iloc[idx]
            for col in common_cols:
                vo = row_o[col]
                vr = row_r[col]
                differ = False
                try:
                    if isinstance(vo, (int, float, np.number)) and isinstance(vr, (int, float, np.number)):
                        if not np.isclose(float(vo), float(vr), rtol=1e-3, atol=1e-4):
                            differ = True
                    elif str(vo).strip() != str(vr).strip():
                        differ = True
                except Exception:
                    if str(vo) != str(vr):
                        differ = True

                if differ:
                    diffs.append(f"Row {idx + 1}, column '{col}': Analyst={vo} vs Recomputed={vr}")
                    if len(diffs) >= 3:
                        return diffs

        if diffs:
            return diffs
        if list(df_o.columns) != list(df_r.columns):
            return [f"Columns differ: Analyst {list(df_o.columns)} vs Recomputed {list(df_r.columns)}"]

    # 3. Fallback
    return [f"Analyst: {str(orig)[:80]} vs Recomputed: {str(recomp)[:80]}"]


def build_review_cards(final_state: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Constructs detailed audit review cards for all analysis questions.
    Includes original and recomputed values, reasons, code, and retry history.

    Args:
        final_state: State dictionary from final pipeline execution.

    Returns:
        list[dict[str, Any]]: List of review cards.
    """
    if not isinstance(final_state, dict):
        return []

    analysis_results = final_state.get("analysis_results", [])
    if not isinstance(analysis_results, list):
        analysis_results = []

    review_report = final_state.get("review_report", [])
    if not isinstance(review_report, list):
        review_report = []

    caught_errors = final_state.get("caught_errors", [])
    if not isinstance(caught_errors, list):
        caught_errors = []

    code_by_id = {
        ar["id"]: ar.get("code", "")
        for ar in analysis_results
        if isinstance(ar, dict) and "id" in ar
    }
    question_by_id = {
        ar["id"]: ar.get("question", "")
        for ar in analysis_results
        if isinstance(ar, dict) and "id" in ar
    }

    cards: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    for rev in review_report:
        if not isinstance(rev, dict) or "id" not in rev:
            continue
        iid = rev["id"]
        seen_ids.add(iid)
        q = rev.get("question") or question_by_id.get(iid, f"Question {iid + 1}")
        tag = item_tag(final_state, iid)

        # Retry history for this item
        history = [
            {
                "retry_round": int(e.get("retry_round", 0)),
                "reason": str(e.get("reason", "")),
            }
            for e in caught_errors
            if isinstance(e, dict) and e.get("id") == iid
        ]

        # Extract raw objects for rich UI display (with fallback to string)
        ar_item = next(
            (a for a in analysis_results if isinstance(a, dict) and a.get("id") == iid),
            None,
        )
        orig_raw = ar_item.get("result") if ar_item is not None else None
        if orig_raw is None:
            orig_raw = rev.get("original_value")

        recomp_raw = rev.get("recomputed_result")
        if recomp_raw is None:
            recomp_raw = rev.get("recomputed_value")

        orig_disp = prepare_display_value(orig_raw)
        recomp_disp = prepare_display_value(recomp_raw)

        differing_notes: list[str] = []
        if rev.get("verdict") == "fail":
            differing_notes = find_differing_values(orig_raw, recomp_raw)

        # Extract Check 3 advisory notes
        logic_warns = rev.get("logic_warnings")
        if logic_warns is None:
            raw_warns = rev.get("warnings", []) if isinstance(rev.get("warnings"), list) else []
            logic_warns = [
                w for w in raw_warns
                if not str(w).startswith("takeaway_mismatch:") and not str(w).startswith("Independent recomputation")
            ]

        cards.append({
            "id": iid,
            "question": q,
            "tag": tag,
            "verdict": rev.get("verdict", "unverified"),
            "reason": rev.get("reason", ""),
            "original_value": _shorten_val(orig_raw if orig_raw is not None else rev.get("original_value")),
            "recomputed_value": _shorten_val(recomp_raw if recomp_raw is not None else rev.get("recomputed_value")),
            "original_display": orig_disp,
            "recomputed_display": recomp_disp,
            "differing_notes": differing_notes,
            "logic_warnings": logic_warns,
            "warnings": rev.get("warnings", []) if isinstance(rev.get("warnings"), list) else [],
            "takeaway_warning": rev.get("takeaway_warning"),
            "original_code": code_by_id.get(iid, ""),
            "retry_history": history,
        })

    # Include any analysis_results not present in review_report (e.g. failed analysis)
    for ar in analysis_results:
        if not isinstance(ar, dict) or "id" not in ar:
            continue
        iid = ar["id"]
        if iid not in seen_ids:
            seen_ids.add(iid)
            tag = item_tag(final_state, iid)
            history = [
                {
                    "retry_round": int(e.get("retry_round", 0)),
                    "reason": str(e.get("reason", "")),
                }
                for e in caught_errors
                if isinstance(e, dict) and e.get("id") == iid
            ]
            orig_raw = ar.get("result")
            orig_disp = prepare_display_value(orig_raw)
            recomp_disp = prepare_display_value(None)

            cards.append({
                "id": iid,
                "question": ar.get("question", f"Question {iid + 1}"),
                "tag": tag,
                "verdict": "fail" if ar.get("status") == "failed" else "unverified",
                "reason": ar.get("error") or "Analysis was not reviewed",
                "original_value": _shorten_val(orig_raw),
                "recomputed_value": "",
                "original_display": orig_disp,
                "recomputed_display": recomp_disp,
                "differing_notes": [],
                "logic_warnings": [],
                "warnings": [],
                "takeaway_warning": None,
                "original_code": ar.get("code", ""),
                "retry_history": history,
            })

    return cards


def review_summary(final_state: dict[str, Any]) -> dict[str, Any]:
    """
    Computes key review metrics across all verified, unverified, and unresolved items.

    Args:
        final_state: State dictionary from final pipeline execution.

    Returns:
        dict[str, Any]: Summary metrics containing items_checked, verified,
        unverified, unresolved, caught_first_pass, fixed_by_retry, and
        total_flag_events.
    """
    if not isinstance(final_state, dict):
        return {
            "items_checked": 0,
            "verified": 0,
            "unverified": 0,
            "unresolved": 0,
            "caught_first_pass": 0,
            "fixed_by_retry": 0,
            "total_flag_events": 0,
        }

    review_report = final_state.get("review_report", [])
    if not isinstance(review_report, list):
        review_report = []

    analysis_results = final_state.get("analysis_results", [])
    if not isinstance(analysis_results, list):
        analysis_results = []

    caught_errors = final_state.get("caught_errors", [])
    if not isinstance(caught_errors, list):
        caught_errors = []

    stats = final_state.get("run_stats", {})
    if not isinstance(stats, dict):
        stats = {}

    # Total items evaluated
    all_ids = set()
    for r in review_report:
        if isinstance(r, dict) and "id" in r:
            all_ids.add(r["id"])
    for a in analysis_results:
        if isinstance(a, dict) and "id" in a:
            all_ids.add(a["id"])

    items_checked = len(all_ids) if all_ids else len(review_report)

    verified = sum(
        1 for r in review_report if isinstance(r, dict) and r.get("verdict") == "pass"
    )
    unverified = sum(
        1 for r in review_report if isinstance(r, dict) and r.get("verdict") == "unverified"
    )
    unresolved = sum(
        1 for r in review_report if isinstance(r, dict) and r.get("verdict") == "fail"
    ) + sum(
        1 for a in analysis_results if isinstance(a, dict) and a.get("status") != "ok"
    )

    first_round_ids = {
        e["id"]
        for e in caught_errors
        if isinstance(e, dict) and e.get("retry_round", 0) == 0 and "id" in e
    }
    caught_first_pass = len(first_round_ids)

    final_verdicts = {
        r["id"]: r.get("verdict")
        for r in review_report
        if isinstance(r, dict) and "id" in r
    }
    fixed_by_retry = sum(
        1 for iid in first_round_ids if final_verdicts.get(iid) == "pass"
    )
    if "fixed_by_retry" in stats and fixed_by_retry == 0:
        fixed_by_retry = int(stats["fixed_by_retry"])

    return {
        "items_checked": items_checked,
        "verified": verified,
        "unverified": unverified,
        "unresolved": unresolved,
        "caught_first_pass": caught_first_pass,
        "fixed_by_retry": fixed_by_retry,
        "total_flag_events": len(caught_errors),
    }


# ============================================================================
# 5. File Exports & Serialization Helpers
# ============================================================================

def report_filename(final_state: dict[str, Any]) -> str:
    """
    Generates a sanitized, safe filename for markdown report download.

    Args:
        final_state: State dictionary from final pipeline execution.

    Returns:
        str: E.g., 'analysis_report_sample.md'.
    """
    if not isinstance(final_state, dict):
        return "analysis_report_dataset.md"

    raw_path = final_state.get("raw_df_path", "dataset")
    base_stem = os.path.splitext(os.path.basename(str(raw_path)))[0]
    safe_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", base_stem).strip("._-")
    if not safe_name:
        safe_name = "dataset"
    return f"analysis_report_{safe_name}.md"


def cleaned_csv_bytes(final_state: dict[str, Any]) -> bytes | None:
    """
    Safely reads the cleaned dataset CSV as bytes for Streamlit download.
    Falls back to demo/sample_run_cleaned.csv if present.

    Args:
        final_state: State dictionary from final pipeline execution.

    Returns:
        bytes | None: Raw file bytes or None if missing.
    """
    if not isinstance(final_state, dict):
        return None

    clean_path = final_state.get("clean_df_path")
    if clean_path and os.path.exists(clean_path) and os.path.isfile(clean_path):
        try:
            return Path(clean_path).read_bytes()
        except Exception:
            pass

    # Demo fallback check
    demo_cleaned = os.path.join("demo", "sample_run_cleaned.csv")
    if os.path.exists(demo_cleaned) and os.path.isfile(demo_cleaned):
        try:
            return Path(demo_cleaned).read_bytes()
        except Exception:
            pass

    return None


def _make_json_safe(obj: Any) -> Any:
    """Recursively converts NumPy, Pandas, and custom types into standard JSON-serializable types."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return str(obj)
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return str(obj)
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_make_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return _make_json_safe(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _make_json_safe(obj.to_dict())
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(x) for x in obj]
    return str(obj)


def review_json_bytes(final_state: dict[str, Any]) -> bytes:
    """
    Serializes review_report and caught_errors into clean, JSON-safe bytes.
    Preserves shortened text summaries (<= 300 chars) for JSON storage.

    Args:
        final_state: State dictionary from final pipeline execution.

    Returns:
        bytes: UTF-8 encoded JSON string.
    """
    if not isinstance(final_state, dict):
        final_state = {}

    clean_review_report = []
    for r in final_state.get("review_report", []):
        if isinstance(r, dict):
            item_copy = dict(r)
            # Exclude raw dataframe objects from JSON export to preserve compact text summaries
            item_copy.pop("recomputed_result", None)
            clean_review_report.append(item_copy)

    export_obj = {
        "summary": review_summary(final_state),
        "review_report": _make_json_safe(clean_review_report),
        "caught_errors": _make_json_safe(final_state.get("caught_errors", [])),
    }

    json_str = json.dumps(export_obj, default=str, indent=2)
    return json_str.encode("utf-8")
