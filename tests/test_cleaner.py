"""
test_cleaner.py
Unit tests for agents/cleaner.py using pytest.
Validates cleaning actions, LLM plan parsing and fallback,
safety constraints (<=20% row removal), and immutability of raw CSV files.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from agents.cleaner import cleaner_node
from state import new_state


def create_temp_csv(tmp_path: Path, df: pd.DataFrame, filename: str = "test_raw.csv") -> str:
    """Helper to save a DataFrame to a temporary CSV file and return its path."""
    csv_path = tmp_path / filename
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_duplicates_removed_and_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 1: Duplicates get removed and the log mentions it."""
    # 10 rows with 1 duplicate row (Row 0 and Row 1 are identical) -> 10% removal
    data = {
        "id": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "name": ["Alice", "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan"],
    }
    raw_path = create_temp_csv(tmp_path, pd.DataFrame(data), "dups.csv")
    state = new_state(raw_path)

    fake_plan = json.dumps([
        {"action": "drop_duplicates", "column": "all", "reason": "Remove exact duplicate rows"}
    ])
    monkeypatch.setattr("agents.cleaner.call_llm", lambda prompt: fake_plan)

    result = cleaner_node(state)

    cleaned_df = pd.read_csv(result["clean_df_path"])
    assert len(cleaned_df) == 9
    assert any("Removed 1 duplicate row(s)" in entry for entry in result["cleaning_log"])
    assert any(e["type"] == "info" for e in result["events"])


def test_median_fill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 2: Median fill works on a numeric column with missing values."""
    # [10.0, 20.0, 30.0, NaN, 40.0] -> median is 25.0
    data = {
        "score": [10.0, 20.0, 30.0, np.nan, 40.0],
        "user": ["u1", "u2", "u3", "u4", "u5"],
    }
    raw_path = create_temp_csv(tmp_path, pd.DataFrame(data), "median.csv")
    state = new_state(raw_path)

    fake_plan = json.dumps([
        {"action": "fill_missing", "column": "score", "method": "median", "reason": "Fill null scores"}
    ])
    monkeypatch.setattr("agents.cleaner.call_llm", lambda prompt: fake_plan)

    result = cleaner_node(state)

    cleaned_df = pd.read_csv(result["clean_df_path"])
    assert cleaned_df["score"].isna().sum() == 0
    assert cleaned_df.loc[3, "score"] == 25.0
    assert any("Filled 1 missing value(s) in 'score' with median" in entry for entry in result["cleaning_log"])


def test_invalid_action_skipped_and_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 3: An invalid action name is skipped and logged."""
    data = {"col_a": [1, 2, 3], "col_b": ["a", "b", "c"]}
    raw_path = create_temp_csv(tmp_path, pd.DataFrame(data), "invalid_action.csv")
    state = new_state(raw_path)

    fake_plan = json.dumps([
        {"action": "drop_entire_database", "column": "all", "reason": "Malicious or unknown action"}
    ])
    monkeypatch.setattr("agents.cleaner.call_llm", lambda prompt: fake_plan)

    result = cleaner_node(state)

    cleaned_df = pd.read_csv(result["clean_df_path"])
    assert len(cleaned_df) == 3
    assert any("Skipped unknown action 'drop_entire_database'" in entry for entry in result["cleaning_log"])
    assert any(e["type"] == "warning" for e in result["events"])


def test_nonexistent_column_skipped_and_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 4: A step on a non-existent column is skipped and logged."""
    data = {"col_a": [1, 2, 3]}
    raw_path = create_temp_csv(tmp_path, pd.DataFrame(data), "missing_col.csv")
    state = new_state(raw_path)

    fake_plan = json.dumps([
        {"action": "fill_missing", "column": "ghost_column", "method": "median", "reason": "Bad column reference"}
    ])
    monkeypatch.setattr("agents.cleaner.call_llm", lambda prompt: fake_plan)

    result = cleaner_node(state)

    assert any("column 'ghost_column' not found in dataset" in entry for entry in result["cleaning_log"])
    assert any(e["type"] == "warning" for e in result["events"])


def test_row_removal_exceeding_20_percent_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 5: A step that would remove more than 20% of rows is skipped with a warning."""
    # 10 rows, 3 of which have missing values in 'val' (30% > 20% threshold)
    data = {
        "id": list(range(10)),
        "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, np.nan, np.nan, np.nan],
    }
    raw_path = create_temp_csv(tmp_path, pd.DataFrame(data), "heavy_nulls.csv")
    state = new_state(raw_path)

    fake_plan = json.dumps([
        {"action": "drop_missing_rows", "column": "val", "reason": "Drop null rows"}
    ])
    monkeypatch.setattr("agents.cleaner.call_llm", lambda prompt: fake_plan)

    result = cleaner_node(state)

    cleaned_df = pd.read_csv(result["clean_df_path"])
    # Rows should NOT be dropped because 30% > 20% threshold
    assert len(cleaned_df) == 10
    assert any("exceeding the 20% safety threshold" in entry for entry in result["cleaning_log"])
    assert any(e["type"] == "warning" and "exceeding the 20% safety threshold" in e["message"] for e in result["events"])


def test_invalid_json_uses_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 6: If the fake LLM returns invalid JSON, the fallback plan is used."""
    data = {
        "id": [1, 1, 2, 3, 4],  # 1 duplicate row
        "text": ["  clean me  ", "  clean me  ", "hello ", " world", "test"],
    }
    raw_path = create_temp_csv(tmp_path, pd.DataFrame(data), "fallback.csv")
    state = new_state(raw_path)

    # Completely invalid non-JSON response from LLM
    monkeypatch.setattr(
        "agents.cleaner.call_llm",
        lambda prompt: "I apologize, but as an AI I cannot generate that JSON: ```error```"
    )

    result = cleaner_node(state)

    cleaned_df = pd.read_csv(result["clean_df_path"])
    # Fallback plan removes duplicates and trims whitespace
    assert len(cleaned_df) == 4
    assert cleaned_df.loc[0, "text"] == "clean me"
    assert any("Applied safe default cleaning plan" in entry for entry in result["cleaning_log"])


def test_raw_csv_unmodified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 7: The raw CSV file is not modified during or after cleaning."""
    data = {
        "val": [10.0, np.nan, 20.0, 30.0, 40.0],
        "str_col": [" a ", " b ", " c ", " d ", " e "],
    }
    raw_path = create_temp_csv(tmp_path, pd.DataFrame(data), "immutable_raw.csv")
    state = new_state(raw_path)

    # Read original bytes before running cleaner
    with open(raw_path, "rb") as f:
        original_bytes = f.read()

    fake_plan = json.dumps([
        {"action": "strip_whitespace", "column": "all", "reason": "Trim spaces"},
        {"action": "fill_missing", "column": "val", "method": "median", "reason": "Fill nulls"},
    ])
    monkeypatch.setattr("agents.cleaner.call_llm", lambda prompt: fake_plan)

    cleaner_node(state)

    # Check raw file bytes after cleaning
    with open(raw_path, "rb") as f:
        post_bytes = f.read()

    assert post_bytes == original_bytes
