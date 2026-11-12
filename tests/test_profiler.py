"""
test_profiler.py
Unit tests for tools/profiler.py using pytest.
Tests the profiler against both an in-memory test dataframe with known properties
and the full data/sample.csv dataset, verifying structure and JSON serializability.
"""

import json
import os
import numpy as np
import pandas as pd
import pytest

from tools.profiler import profile_dataframe, profile_to_text


def test_profile_in_memory_dataframe():
    """
    Tests profile_dataframe on a controlled, in-memory dataset with:
    - 5 total rows (including 1 duplicate row)
    - 4 columns (numeric, categorical, date string, boolean)
    - Known null counts and duplicate counts
    """
    # 5 rows, with Row 0 and Row 3 being exact duplicates
    data = {
        "id": [101, 102, 103, 101, 105],
        "score": [85.5, np.nan, 92.0, 85.5, 78.0],  # 1 null out of 5 = 20%
        "category": ["Alpha", "Beta", "Alpha", "Alpha", None],  # 1 null out of 5 = 20%
        "joined_date": ["2023-01-15", "2023-02-20", "2023-03-25", "2023-01-15", "2023-05-10"],
    }
    df = pd.DataFrame(data)

    profile = profile_dataframe(df)

    # 1. Basic shape & duplicates
    assert profile["n_rows"] == 5
    assert profile["n_cols"] == 4
    assert profile["duplicate_rows"] == 1

    # 2. Columns metadata
    columns = profile["columns"]
    assert "score" in columns
    assert columns["score"]["null_percentage"] == 20.0
    assert columns["score"]["unique_values"] == 3  # 85.5, 92.0, 78.0
    assert columns["category"]["null_percentage"] == 20.0
    assert columns["category"]["unique_values"] == 2  # Alpha, Beta

    # 3. Numeric summary
    numeric_summary = profile["numeric_columns"]
    assert "score" in numeric_summary
    assert numeric_summary["score"]["min"] == 78.0
    assert numeric_summary["score"]["max"] == 92.0
    assert pytest.approx(numeric_summary["score"]["mean"], 0.01) == 85.25

    # 4. Categorical summary
    categorical_summary = profile["categorical_columns"]
    assert "category" in categorical_summary
    assert categorical_summary["category"]["Alpha"] == 3
    assert categorical_summary["category"]["<NA>"] == 1

    # 5. Possible date columns
    assert "joined_date" in profile["possible_date_columns"]

    # 6. Sample rows
    assert len(profile["sample_rows"]) == 5
    assert profile["sample_rows"][0]["id"] == 101

    # 7. JSON serializability
    json_str = json.dumps(profile)
    assert isinstance(json_str, str)
    assert len(json_str) > 0

    # 8. profile_to_text output
    text_summary = profile_to_text(profile)
    assert "Dataset Overview:" in text_summary
    assert "Duplicate Rows: 1" in text_summary
    assert "Possible Date Columns: joined_date" in text_summary


def test_profile_sample_csv():
    """
    Tests profile_dataframe on data/sample.csv, ensuring all fields are populated
    and the entire profile dictionary is strictly JSON-serializable.
    """
    csv_path = os.path.join("data", "sample.csv")
    assert os.path.exists(csv_path), f"File {csv_path} not found"

    df = pd.read_csv(csv_path)
    profile = profile_dataframe(df)

    # Check top-level dimensions
    assert profile["n_rows"] == len(df)
    assert profile["n_cols"] == len(df.columns)
    assert isinstance(profile["duplicate_rows"], int)

    # Check columns
    assert len(profile["columns"]) == len(df.columns)
    for col_name, meta in profile["columns"].items():
        assert "dtype" in meta
        assert "null_percentage" in meta
        assert "unique_values" in meta
        assert isinstance(meta["null_percentage"], float)
        assert isinstance(meta["unique_values"], int)

    # Check numeric columns
    assert len(profile["numeric_columns"]) > 0
    for num_col, stats in profile["numeric_columns"].items():
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats

    # Check sample rows
    assert len(profile["sample_rows"]) == min(5, len(df))

    # CRITICAL: Verify JSON serialization without any TypeError or OverflowError
    json_str = json.dumps(profile)
    assert isinstance(json_str, str)
    assert len(json_str) > 0

    # Verify profile_to_text works and produces informative text
    text_block = profile_to_text(profile)
    assert isinstance(text_block, str)
    assert "Dataset Overview:" in text_block
    assert "Columns Profile:" in text_block


def test_profile_empty_dataframe():
    """
    Tests edge cases such as an empty DataFrame to ensure no divide-by-zero errors.
    """
    df_empty = pd.DataFrame(columns=["col_a", "col_b"])
    profile = profile_dataframe(df_empty)

    assert profile["n_rows"] == 0
    assert profile["n_cols"] == 2
    assert profile["duplicate_rows"] == 0
    assert profile["columns"]["col_a"]["null_percentage"] == 0.0

    json_str = json.dumps(profile)
    assert isinstance(json_str, str)
