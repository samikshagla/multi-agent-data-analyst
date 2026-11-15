"""
tests/test_results_data.py
Unit tests for components/results_data.py.
Validates item_tag logic across all five cases, build_chart_items with skipped/missing items,
build_cleaning_rows classification and missing-file handling, build_review_cards retry history
and value shortening, review_summary counts, review_json_bytes serialization with numpy/pandas,
and report_filename sanitization.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from components.results_data import (
    item_tag,
    build_chart_items,
    build_cleaning_rows,
    get_cleaning_summary,
    build_review_cards,
    prepare_display_value,
    find_differing_values,
    review_summary,
    report_filename,
    cleaned_csv_bytes,
    review_json_bytes,
)


# ============================================================================
# 1. Test item_tag for all five cases
# ============================================================================

def test_item_tag_all_five_cases():
    """item_tag correctly assigns Verified, Fixed after retry, Unverified, Unresolved, or Failed."""
    # Base state with 5 items representing each case
    final_state = {
        "analysis_results": [
            {"id": 0, "question": "Q0", "status": "ok"},
            {"id": 1, "question": "Q1", "status": "ok"},
            {"id": 2, "question": "Q2", "status": "ok"},
            {"id": 3, "question": "Q3", "status": "ok"},
            {"id": 4, "question": "Q4", "status": "failed"},
        ],
        "review_report": [
            {"id": 0, "verdict": "pass", "reason": "Matches"},
            {"id": 1, "verdict": "pass", "reason": "Matches on retry"},
            {"id": 2, "verdict": "fail", "reason": "Math mismatch"},
            {"id": 3, "verdict": "unverified", "reason": "Cannot verify"},
            {"id": 4, "verdict": "fail", "reason": "Not executed"},
        ],
        "caught_errors": [
            # Item 1 was caught on pass 0 and later passed
            {"id": 1, "retry_round": 0, "reason": "First attempt failed"}
        ],
    }

    # Case 1: Verified (pass and never caught)
    assert item_tag(final_state, 0) == "Verified"

    # Case 2: Fixed after retry (pass and was caught)
    assert item_tag(final_state, 1) == "Fixed after retry"

    # Case 3: Unresolved (latest verdict is fail)
    assert item_tag(final_state, 2) == "Unresolved"

    # Case 4: Unverified (verdict is unverified)
    assert item_tag(final_state, 3) == "Unverified"

    # Case 5: Failed (analysis status is failed)
    assert item_tag(final_state, 4) == "Failed"


# ============================================================================
# 2. Test build_chart_items includes skipped items with reason
# ============================================================================

def test_build_chart_items_includes_skipped_items():
    """build_chart_items includes Plotly charts, metric cards, and skipped items with reason."""
    final_state = {
        "analysis_results": [
            {"id": 0, "question": "Revenue by region", "status": "ok"},
            {"id": 1, "question": "Total transactions", "status": "ok"},
            {"id": 2, "question": "Broken query", "status": "failed", "error": "Syntax error in SQL"},
        ],
        "charts": [
            {
                "id": 0,
                "type": "bar",
                "title": "Revenue by Region",
                "takeaway": "North leads with $50k",
                "figure_json": '{"data": [{"type": "bar"}]}',
                "status": "ok",
                "note": "Exported",
            },
            {
                "id": 1,
                "type": "metric",
                "title": "Total transactions",
                "takeaway": "Result: 1,420",
                "figure_json": None,
                "status": "ok",
                "note": "Metric value: 1,420",
            },
        ],
        "review_report": [
            {"id": 0, "verdict": "pass", "takeaway_warning": "Rounded from 49.8k"},
            {"id": 1, "verdict": "pass", "takeaway_warning": None},
        ],
    }

    items = build_chart_items(final_state)
    assert len(items) == 3

    # Item 0: Bar chart
    assert items[0]["id"] == 0
    assert items[0]["type"] == "bar"
    assert items[0]["status"] == "ok"
    assert items[0]["figure_json"] is not None
    assert items[0]["takeaway_warning"] == "Rounded from 49.8k"

    # Item 1: Metric card
    assert items[1]["id"] == 1
    assert items[1]["type"] == "metric"
    assert items[1]["status"] == "ok"
    assert "1,420" in items[1]["takeaway"]

    # Item 2: Skipped / Failed chart
    assert items[2]["id"] == 2
    assert items[2]["status"] == "skipped"
    assert items[2]["figure_json"] is None
    assert "Syntax error" in items[2]["note"]
    assert items[2]["tag"] == "Failed"


# ============================================================================
# 3. Test build_cleaning_rows classification & missing file handling
# ============================================================================

def test_build_cleaning_rows_classification_and_missing_file(tmp_path: Path):
    """build_cleaning_rows classifies applied, skipped, and warning lines and handles missing files."""
    # Create raw CSV
    raw_file = tmp_path / "raw.csv"
    raw_file.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")
    missing_clean_file = tmp_path / "does_not_exist.csv"

    final_state = {
        "raw_df_path": str(raw_file),
        "clean_df_path": str(missing_clean_file),
        "cleaning_log": [
            "Applied median imputation for missing values in column 'a'",
            "Skipped outlier clipping: IQR was 0",
            "Warning: High null ratio detected in column 'c' (45%)",
            "Standardized column names to snake_case",
        ],
    }

    rows, summary = build_cleaning_rows(final_state)

    # Classification assertions
    assert len(rows) == 4
    assert rows[0]["kind"] == "applied"
    assert rows[1]["kind"] == "skipped"
    assert rows[2]["kind"] == "warning"
    assert rows[3]["kind"] == "applied"

    # Before / After Summary assertions
    assert summary["raw_rows"] == 3
    assert summary["raw_cols"] == 3
    assert summary["clean_rows"] is None
    assert summary["clean_cols"] is None

    # Test get_cleaning_summary standalone
    summary_direct = get_cleaning_summary(final_state)
    assert summary_direct["raw_rows"] == 3
    assert summary_direct["clean_rows"] is None


# ============================================================================
# 4. Test build_review_cards retry history & value shortening
# ============================================================================

def test_build_review_cards_retry_history_and_shortens_long_values():
    """build_review_cards includes retry history and truncates values exceeding 300 characters."""
    long_text = "A" * 500

    final_state = {
        "analysis_results": [
            {
                "id": 0,
                "question": "Show long text",
                "code": "result = 'A' * 500",
                "result": long_text,
                "status": "ok",
            }
        ],
        "review_report": [
            {
                "id": 0,
                "question": "Show long text",
                "verdict": "pass",
                "reason": "Recomputed match",
                "original_value": long_text,
                "recomputed_value": long_text,
                "warnings": ["Minor precision note"],
                "takeaway_warning": None,
            }
        ],
        "caught_errors": [
            {"id": 0, "retry_round": 0, "reason": "Initial calculation error"},
            {"id": 0, "retry_round": 1, "reason": "Secondary format error"},
        ],
    }

    cards = build_review_cards(final_state)
    assert len(cards) == 1
    card = cards[0]

    assert card["id"] == 0
    assert card["tag"] == "Fixed after retry"
    assert card["original_code"] == "result = 'A' * 500"

    # Truncation verification
    assert len(card["original_value"]) <= 303
    assert card["original_value"].endswith("...")
    assert len(card["recomputed_value"]) <= 303
    assert card["recomputed_value"].endswith("...")

    # Retry history verification
    assert len(card["retry_history"]) == 2
    assert card["retry_history"][0]["retry_round"] == 0
    assert card["retry_history"][0]["reason"] == "Initial calculation error"
    assert card["retry_history"][1]["retry_round"] == 1
    assert card["retry_history"][1]["reason"] == "Secondary format error"


# ============================================================================
# 5. Test review_summary counts
# ============================================================================

def test_review_summary_counts():
    """review_summary computes items_checked, verified, unverified, unresolved, and retry metrics."""
    final_state = {
        "analysis_results": [
            {"id": 0, "status": "ok"},
            {"id": 1, "status": "ok"},
            {"id": 2, "status": "ok"},
            {"id": 3, "status": "ok"},
        ],
        "review_report": [
            {"id": 0, "verdict": "pass"},
            {"id": 1, "verdict": "pass"},
            {"id": 2, "verdict": "unverified"},
            {"id": 3, "verdict": "fail"},
        ],
        "caught_errors": [
            {"id": 1, "retry_round": 0, "reason": "Calculation error"},
            {"id": 3, "retry_round": 0, "reason": "Type error"},
            {"id": 3, "retry_round": 1, "reason": "Still failing"},
        ],
    }

    summary = review_summary(final_state)
    assert summary["items_checked"] == 4
    assert summary["verified"] == 2
    assert summary["unverified"] == 1
    assert summary["unresolved"] == 1
    assert summary["caught_first_pass"] == 2  # item 1 and item 3
    assert summary["fixed_by_retry"] == 1     # item 1 passed on retry
    assert summary["total_flag_events"] == 3


# ============================================================================
# 6. Test review_json_bytes serialization with numpy & pandas
# ============================================================================

def test_review_json_bytes_serializes_numpy_and_pandas():
    """review_json_bytes safely converts NumPy types and DataFrames into valid JSON bytes."""
    df_sample = pd.DataFrame({"col_a": [1, 2], "col_b": [10.5, 20.5]})

    final_state = {
        "review_report": [
            {
                "id": np.int64(0),
                "question": "What is the total?",
                "verdict": "pass",
                "original_value": np.float64(42.5),
                "recomputed_value": df_sample,
                "timestamp": pd.Timestamp("2026-11-15 12:00:00"),
            }
        ],
        "caught_errors": [
            {
                "id": np.int64(0),
                "retry_round": np.int32(0),
                "reason": "Test reason",
                "matrix": np.array([1, 2, 3]),
            }
        ],
    }

    raw_bytes = review_json_bytes(final_state)
    assert isinstance(raw_bytes, bytes)

    # Parse and verify valid JSON
    parsed = json.loads(raw_bytes.decode("utf-8"))
    assert "summary" in parsed
    assert "review_report" in parsed
    assert "caught_errors" in parsed
    assert parsed["review_report"][0]["id"] == 0
    assert parsed["review_report"][0]["original_value"] == 42.5
    assert len(parsed["review_report"][0]["recomputed_value"]) == 2
    assert parsed["caught_errors"][0]["matrix"] == [1, 2, 3]


# ============================================================================
# 7. Test report_filename sanitization & cleaned_csv_bytes
# ============================================================================

def test_report_filename_sanitization():
    """report_filename sanitizes path traversals and special characters."""
    # 1. Path traversal injection
    s1 = {"raw_df_path": "../../etc/passwd; DROP TABLE.csv"}
    fname1 = report_filename(s1)
    assert fname1.startswith("analysis_report_")
    assert fname1.endswith(".md")
    assert "/" not in fname1
    assert "\\" not in fname1
    assert ".." not in fname1

    # 2. Spaces and punctuation
    s2 = {"raw_df_path": "C:\\data\\My Sales (2026) #1.csv"}
    fname2 = report_filename(s2)
    assert " " not in fname2
    assert "#" not in fname2
    assert fname2.endswith(".md")

    # 3. None or empty state
    assert report_filename({}) == "analysis_report_dataset.md"
    assert report_filename(None) == "analysis_report_dataset.md"


def test_cleaned_csv_bytes(tmp_path: Path):
    """cleaned_csv_bytes returns file bytes when present and None when absent."""
    csv_file = tmp_path / "cleaned.csv"
    csv_file.write_text("x,y\n1,2\n", encoding="utf-8")

    # Present file
    state_present = {"clean_df_path": str(csv_file)}
    content = cleaned_csv_bytes(state_present)
    assert content is not None
    assert b"x,y" in content

    # Missing file
    state_missing = {"clean_df_path": str(tmp_path / "missing.csv")}
    assert cleaned_csv_bytes(state_missing) is None

    # Empty state
    assert cleaned_csv_bytes({}) is None


# ============================================================================
# 8. Test prepare_display_value (scalar, small table, long table, missing)
# ============================================================================

def test_prepare_display_value_scalar():
    """prepare_display_value formats numbers with thousands separators and at most 2 decimals."""
    # 1. Large float with 2 decimals
    res1 = prepare_display_value(1234567.89)
    assert res1["kind"] == "scalar"
    assert res1["formatted"] == "1,234,567.89"
    assert res1["caption"] is None

    # 2. Large integer
    res2 = prepare_display_value(1000000)
    assert res2["kind"] == "scalar"
    assert res2["formatted"] == "1,000,000"

    # 3. Float with 1 decimal
    res3 = prepare_display_value(1234.5)
    assert res3["kind"] == "scalar"
    assert res3["formatted"] == "1,234.5"

    # 4. Small integer
    res4 = prepare_display_value(42)
    assert res4["kind"] == "scalar"
    assert res4["formatted"] == "42"


def test_prepare_display_value_small_table():
    """prepare_display_value shows small table without caption."""
    df_small = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})
    res = prepare_display_value(df_small)

    assert res["kind"] == "dataframe"
    assert len(res["value"]) == 3
    assert res["total_rows"] == 3
    assert res["caption"] is None


def test_prepare_display_value_long_table_with_caption():
    """prepare_display_value slices long table to first 8 rows and adds caption."""
    df_long = pd.DataFrame({"num": range(25)})
    res = prepare_display_value(df_long)

    assert res["kind"] == "dataframe"
    assert len(res["value"]) == 8
    assert res["total_rows"] == 25
    assert res["caption"] == "showing first 8 of 25 rows"

    # Same for pd.Series
    s_long = pd.Series(range(12), name="val")
    res_s = prepare_display_value(s_long)
    assert res_s["kind"] == "dataframe"
    assert len(res_s["value"]) == 8
    assert res_s["caption"] == "showing first 8 of 12 rows"


def test_prepare_display_value_missing_and_fallback():
    """prepare_display_value handles None missing values and text fallback without crashing."""
    # None value
    res_none = prepare_display_value(None)
    assert res_none["kind"] == "missing"
    assert res_none["formatted"] == "(None)"

    # Text value
    res_text = prepare_display_value("Execution error: column missing")
    assert res_text["kind"] == "text"
    assert res_text["formatted"] == "Execution error: column missing"


def test_find_differing_values_mismatch():
    """find_differing_values lists up to 3 differing cells between analyst and reviewer."""
    df_a = pd.DataFrame({"cat": ["A", "B", "C"], "val": [10, 20, 30]})
    df_b = pd.DataFrame({"cat": ["A", "B", "C"], "val": [10, 25, 35]})

    diffs = find_differing_values(df_a, df_b)
    assert len(diffs) == 2
    assert "val" in diffs[0]
    assert "Analyst=20" in diffs[0]
    assert "Recomputed=25" in diffs[0]
    assert "Analyst=30" in diffs[1]
    assert "Recomputed=35" in diffs[1]
