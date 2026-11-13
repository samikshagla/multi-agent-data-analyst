"""
test_reviewer.py
Unit tests for agents/reviewer.py using pytest.
Validates independent recomputation, results_match tolerance comparison,
rule-based invariant checks, advisory logic review, and feedback isolation.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from agents.reviewer import reviewer_node, results_match, rule_checks
from state import new_state


class MockLLMSequence:
    """Helper mock to return a planned sequence of string responses."""
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return '{"issues": []}'


def setup_test_state(tmp_path: Path, df: pd.DataFrame) -> dict:
    """Helper to save a DataFrame to CSV and return an initialized state."""
    csv_path = tmp_path / "clean_data.csv"
    df.to_csv(csv_path, index=False)
    state = new_state(str(csv_path))
    state["clean_df_path"] = str(csv_path)
    return state


def test_results_match_all_cases():
    """Test 1: results_match helper across scalars, DataFrames, and tolerance rules."""
    # 1. Equal scalars within tolerance (1%)
    assert results_match(100.0, 100.5) is True
    assert results_match(100.0, 101.0) is True
    assert results_match(0, 0) is True

    # 2. Different scalars (> 1% diff)
    assert results_match(100.0, 105.0) is False

    # 3. DataFrames equal after reordering rows
    df1 = pd.DataFrame({"city": ["Chicago", "Atlanta"], "sales": [200.0, 100.0]})
    df2 = pd.DataFrame({"city": ["Atlanta", "Chicago"], "sales": [100.2, 200.1]})
    assert results_match(df1, df2) is True

    # 4. DataFrames with a different numeric value
    df3 = pd.DataFrame({"city": ["Atlanta", "Chicago"], "sales": [100.0, 500.0]})
    assert results_match(df1, df3) is False

    # 5. DataFrames with different row counts
    df4 = pd.DataFrame({"city": ["Atlanta"], "sales": [100.0]})
    assert results_match(df1, df4) is False


def test_recomputation_matches_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 2: An analysis result whose independent recomputation matches gets verdict 'pass'."""
    df = pd.DataFrame({"sales": [100, 200, 300]})
    state = setup_test_state(tmp_path, df)

    state["analysis_results"] = [
        {
            "id": 0,
            "question": "Total sales?",
            "code": "result = df['sales'].sum()",
            "result": 600,
            "status": "ok",
        }
    ]

    mock_llm = MockLLMSequence([
        # 1. Check 1 alternative code: sum using numpy
        'result = int(np.sum(df["sales"]))',
        # 2. Check 3 logic review
        '{"issues": []}',
    ])
    monkeypatch.setattr("agents.reviewer.call_llm", mock_llm)

    res = reviewer_node(state)

    assert len(res["review_feedback"]) == 0
    assert len(res["review_report"]) == 1
    report_item = res["review_report"][0]
    assert report_item["verdict"] == "pass"
    assert "All checks passed" in report_item["reason"]


def test_wrong_result_fails_and_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 3: An analysis result with a wrong original result gets verdict 'fail' and is recorded in feedback & errors."""
    df = pd.DataFrame({"sales": [100, 200, 300]})
    state = setup_test_state(tmp_path, df)

    state["analysis_results"] = [
        {
            "id": 1,
            "question": "Total sales?",
            "code": "result = 99999",
            "result": 99999,  # Deliberately wrong number
            "status": "ok",
        }
    ]

    mock_llm = MockLLMSequence([
        # Check 1 alternative code computes the true value (600)
        'result = df["sales"].sum()',
        # Check 3 logic review
        '{"issues": []}',
    ])
    monkeypatch.setattr("agents.reviewer.call_llm", mock_llm)

    res = reviewer_node(state)

    # Must appear in review_feedback
    assert len(res["review_feedback"]) == 1
    fb = res["review_feedback"][0]
    assert fb["id"] == 1
    assert fb["verdict"] == "fail"
    assert "different result" in fb["reason"]
    assert "Re-check the aggregation" in fb["suggested_fix"]

    # Must appear in caught_errors
    assert len(res["caught_errors"]) == 1
    assert res["caught_errors"][0]["id"] == 1

    # Report has fail verdict
    assert res["review_report"][0]["verdict"] == "fail"


def test_rule_check_flags_empty_result():
    """Test 4: Rule check flags an empty result."""
    empty_df_item = {"id": 2, "result": pd.DataFrame()}
    problems1 = rule_checks(empty_df_item, pd.DataFrame())
    assert any("empty" in p.lower() for p in problems1)

    none_item = {"id": 3, "result": None}
    problems2 = rule_checks(none_item, pd.DataFrame())
    assert any("none" in p.lower() for p in problems2)


def test_rule_check_takeaway_numbers():
    """Test 5: Rule check validates takeaway numbers against result values."""
    res_df = pd.DataFrame({"revenue": [1200000.0], "sales": [1200.0]})
    item = {"id": 4, "result": res_df}

    # Valid chart: mentions $1,200, 1.2M, year 2024 (ignored), percentage 10% (ignored), small int 3 (ignored)
    valid_chart = {
        "id": 4,
        "takeaway": "Top store had $1,200 in sales, 1.2M revenue in 2024 with 3 reps (10% growth).",
    }
    problems_valid = rule_checks(item, pd.DataFrame(), chart=valid_chart)
    assert len(problems_valid) == 0

    # Invalid chart: mentions 88888 which is completely absent from result data
    invalid_chart = {
        "id": 4,
        "takeaway": "Top store had 88888 transactions.",
    }
    problems_invalid = rule_checks(item, pd.DataFrame(), chart=invalid_chart)
    assert len(problems_invalid) == 1
    assert "88888" in problems_invalid[0]


def test_check3_warnings_do_not_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 6: Check 3 issues are recorded as warnings but do NOT cause a failure."""
    df = pd.DataFrame({"sales": [10, 20]})
    state = setup_test_state(tmp_path, df)

    state["analysis_results"] = [
        {
            "id": 5,
            "question": "Sum?",
            "code": "result = df['sales'].sum()",
            "result": 30,
            "status": "ok",
        }
    ]

    mock_llm = MockLLMSequence([
        # Check 1 matches
        'result = df["sales"].sum()',
        # Check 3 flags advisory warning
        '{"issues": ["Potential double-counting in raw dataset"]}',
    ])
    monkeypatch.setattr("agents.reviewer.call_llm", mock_llm)

    res = reviewer_node(state)

    # Still passes
    assert res["review_report"][0]["verdict"] == "pass"
    assert "Potential double-counting" in res["review_report"][0]["warnings"][0]
    # review_feedback must be empty
    assert len(res["review_feedback"]) == 0


def test_recomputation_failure_is_inconclusive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 7: If alternative recomputation code keeps failing, the item is 'pass' with an inconclusive note."""
    df = pd.DataFrame({"sales": [10, 20]})
    state = setup_test_state(tmp_path, df)

    state["analysis_results"] = [
        {
            "id": 6,
            "question": "Sum?",
            "code": "result = df['sales'].sum()",
            "result": 30,
            "status": "ok",
        }
    ]

    mock_llm = MockLLMSequence([
        # Attempt 1 fails
        'result = df["missing_col_1"].sum()',
        # Attempt 2 (retry) fails
        'result = df["missing_col_2"].sum()',
        # Check 3 returns no issues
        '{"issues": []}',
    ])
    monkeypatch.setattr("agents.reviewer.call_llm", mock_llm)

    res = reviewer_node(state)

    # Item still passes
    assert res["review_report"][0]["verdict"] == "pass"
    assert any("Recomputation code failed" in w for w in res["review_report"][0]["warnings"])
    assert len(res["review_feedback"]) == 0


def test_review_feedback_only_failing_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 8: review_feedback only contains failing items."""
    df = pd.DataFrame({"sales": [10, 20]})
    state = setup_test_state(tmp_path, df)

    state["analysis_results"] = [
        # Item 7 is correct
        {"id": 7, "question": "Q7", "code": "result = 30", "result": 30, "status": "ok"},
        # Item 8 has wrong result (0 instead of 30)
        {"id": 8, "question": "Q8", "code": "result = 0", "result": 0, "status": "ok"},
    ]

    mock_llm = MockLLMSequence([
        # Item 7: Check 1 matches
        'result = df["sales"].sum()',
        '{"issues": []}',
        # Item 8: Check 1 recomputes 30 (mismatch with 0)
        'result = df["sales"].sum()',
        '{"issues": []}',
    ])
    monkeypatch.setattr("agents.reviewer.call_llm", mock_llm)

    res = reviewer_node(state)

    # Exactly 1 item in feedback (only item 8)
    assert len(res["review_feedback"]) == 1
    assert res["review_feedback"][0]["id"] == 8
    # But report has both items
    assert len(res["review_report"]) == 2


def test_resilience_when_item_breaks(tmp_path: Path):
    """Test 9: Node doesn't crash when one item breaks; passes with a warning."""
    df = pd.DataFrame({"sales": [10, 20]})
    state = setup_test_state(tmp_path, df)

    # Result contains an object that throws an exception during rule_checks/recomputation
    state["analysis_results"] = [
        {"id": 9, "question": "Broken item", "code": "...", "result": object(), "status": "ok"}
    ]

    res = reviewer_node(state)

    assert len(res["review_report"]) == 1
    report = res["review_report"][0]
    assert report["verdict"] == "pass"
    assert any("could not verify this item" in w for w in report["warnings"])
    assert len(res["review_feedback"]) == 0
