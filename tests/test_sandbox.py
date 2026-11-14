"""
test_sandbox.py
Unit tests for tools/sandbox.py using pytest.
Validates security layers, AST static analysis, process timeouts,
error handling, and original dataframe immutability.
"""

import pandas as pd
import pytest

from tools.sandbox import run_code


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture providing a sample DataFrame for sandbox tests."""
    return pd.DataFrame({
        "city": ["New York", "Chicago", "New York", "Chicago", "Houston"],
        "sales": [100, 150, 200, 250, 300],
        "profit": [20, 30, 45, 50, 70],
    })


def test_valid_groupby(sample_df: pd.DataFrame):
    """Test case 1: Valid groupby code returns status 'ok' with correct numbers."""
    code = 'result = df.groupby("city")["sales"].sum()'
    res = run_code(code, sample_df)

    assert res["status"] == "ok"
    assert res["error"] is None
    assert isinstance(res["result"], pd.DataFrame)
    # Check that Series was converted to DataFrame via reset_index
    assert "city" in res["result"].columns
    assert "sales" in res["result"].columns

    # Verify numbers
    ny_sales = res["result"].loc[res["result"]["city"] == "New York", "sales"].values[0]
    chicago_sales = res["result"].loc[res["result"]["city"] == "Chicago", "sales"].values[0]
    assert ny_sales == 300
    assert chicago_sales == 400


def test_valid_scalar_result(sample_df: pd.DataFrame):
    """Test that a scalar numeric result is converted to a plain Python number."""
    code = 'result = df["sales"].sum()'
    res = run_code(code, sample_df)

    assert res["status"] == "ok"
    assert res["error"] is None
    assert res["result"] == 1000
    assert type(res["result"]) is int


def test_blocked_import(sample_df: pd.DataFrame):
    """Test case 2: 'import os' is blocked by AST static inspection."""
    code = 'import os\nresult = os.listdir(".")'
    res = run_code(code, sample_df)

    assert res["status"] == "blocked"
    assert res["result"] is None
    assert "Importing module 'os' is blocked" in res["error"]


def test_blocked_from_import(sample_df: pd.DataFrame):
    """Test that from-imports of disallowed modules are blocked."""
    code = 'from sys import exit\nresult = 1'
    res = run_code(code, sample_df)

    assert res["status"] == "blocked"
    assert res["result"] is None
    assert "Importing from 'sys' is blocked" in res["error"]


def test_blocked_open(sample_df: pd.DataFrame):
    """Test case 3: open('x.txt') is blocked by AST static inspection."""
    code = 'f = open("x.txt")\nresult = f.read()'
    res = run_code(code, sample_df)

    assert res["status"] == "blocked"
    assert res["result"] is None
    assert "Call to 'open' is blocked" in res["error"]


def test_blocked_dunder(sample_df: pd.DataFrame):
    """Test case 4: Code using __class__ is blocked by AST static inspection."""
    code = 'result = df.__class__'
    res = run_code(code, sample_df)

    assert res["status"] == "blocked"
    assert res["result"] is None
    assert "Attribute access starting with '__'" in res["error"]


def test_timeout(sample_df: pd.DataFrame):
    """Test case 5: 'while True: pass' returns 'timeout' with timeout=2."""
    code = 'while True:\n    pass\nresult = 1'
    res = run_code(code, sample_df, timeout=2)

    assert res["status"] == "timeout"
    assert res["result"] is None
    assert "Execution timed out after 2 seconds" in res["error"]


def test_no_result_variable(sample_df: pd.DataFrame):
    """Test case 6: Code with no result variable returns an error."""
    code = 'total = df["sales"].sum()'
    res = run_code(code, sample_df)

    assert res["status"] == "error"
    assert res["result"] is None
    assert res["error"] == "code did not define a variable named result"


def test_missing_column(sample_df: pd.DataFrame):
    """Test case 7: Code with a missing column name returns status 'error' and a readable message."""
    code = 'result = df["nonexistent_column"].mean()'
    res = run_code(code, sample_df)

    assert res["status"] == "error"
    assert res["result"] is None
    assert "KeyError" in res["error"]
    assert "nonexistent_column" in res["error"]


def test_original_df_unchanged(sample_df: pd.DataFrame):
    """Test case 8: Original dataframe is unchanged after code that modifies df in place."""
    # Capture original state
    original_cols = list(sample_df.columns)
    original_len = len(sample_df)
    original_values = sample_df["sales"].tolist()

    # Code that attempts to modify df in-place
    code = '''
df["new_col"] = 999
df.drop(columns=["sales"], inplace=True)
result = len(df)
'''
    res = run_code(code, sample_df)

    assert res["status"] == "ok"
    assert res["result"] == 5

    # Verify original df was not mutated
    assert list(sample_df.columns) == original_cols
    assert len(sample_df) == original_len
    assert sample_df["sales"].tolist() == original_values
    assert "new_col" not in sample_df.columns


def test_tuple_and_list_results_rejected(sample_df: pd.DataFrame):
    """Test 9: Returning a tuple or list triggers an error asking for a single table."""
    code_tuple = 'result = (df["sales"].sum(), df["profit"].sum())'
    res_tuple = run_code(code_tuple, sample_df)
    assert res_tuple["status"] == "error"
    assert "result must be ONE DataFrame, Series, or a single number, not a tuple/list. Combine the answer into a single table." in res_tuple["error"]

    code_list = 'result = [df["sales"].mean(), df["profit"].mean()]'
    res_list = run_code(code_list, sample_df)
    assert res_list["status"] == "error"
    assert "result must be ONE DataFrame, Series, or a single number, not a tuple/list." in res_list["error"]


def test_interval_period_timestamp_converted(sample_df: pd.DataFrame):
    """Test 10: Interval, Period, and Timestamp values in result columns and index are converted to strings."""
    code = '''
df["sales_bin"] = pd.cut(df["sales"], bins=2)
result = df.groupby("sales_bin")["sales"].mean().reset_index()
'''
    res = run_code(code, sample_df)
    assert res["status"] == "ok"
    assert isinstance(res["result"], pd.DataFrame)
    # The sales_bin column should be string, not IntervalDtype
    assert isinstance(res["result"]["sales_bin"].iloc[0], str)
