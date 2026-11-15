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

from agents.reviewer import reviewer_node, results_match, results_match_with_trace, rule_checks, filter_logic_issues
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
    err = res["caught_errors"][0]
    assert err["id"] == 1
    assert err["type"] == "recomputation_mismatch"
    assert err["retry_round"] == 0
    assert err["original_code"] == "result = 99999"
    assert 'df["sales"].sum()' in err["recomputed_code"]

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
    """Test 5: Takeaway extraction checks numbers, unit suffixes, and label ranges."""
    from agents.reviewer import _extract_takeaway_numbers

    res_df = pd.DataFrame({
        "tier": ["0-10%", "10-20%"],
        "revenue": [1200000.0, 500000.0],
        "sales": [1200.0, 800.0],
    })

    # Valid: 1.2M, $1,200, label 0-10% ignored, 7 minutes not treated as million
    valid_text = "Top tier 0-10% had $1,200 in sales, 1.2M revenue, and took 7 minutes."
    assert len(_extract_takeaway_numbers(valid_text, res_df)) == 0

    # Invalid: mentions 88888 absent from result data
    invalid_text = "Top store had 88888 transactions."
    problems = _extract_takeaway_numbers(invalid_text, res_df)
    assert len(problems) == 1
    assert "88888" in problems[0]


def test_takeaway_mismatch_warning_does_not_trigger_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Takeaway mismatch is recorded as a warning, verdict is pass, and no review feedback is emitted."""
    df = pd.DataFrame({"sales": [10, 20]})
    state = setup_test_state(tmp_path, df)
    state["analysis_results"] = [
        {"id": 4, "question": "Total?", "code": "result = 30", "result": 30, "status": "ok"}
    ]
    state["charts"] = [
        {"id": 4, "status": "ok", "takeaway": "Total was 99999 sales."}
    ]

    mock_llm = MockLLMSequence([
        'result = df["sales"].sum()',
        '{"issues": []}',
    ])
    monkeypatch.setattr("agents.reviewer.call_llm", mock_llm)

    res = reviewer_node(state)
    report_item = res["review_report"][0]
    assert report_item["verdict"] == "pass"
    assert report_item["takeaway_warning"] is not None
    assert "99999" in report_item["takeaway_warning"]
    assert len(res["review_feedback"]) == 0


def test_results_match_layout_independent():
    """Long vs wide table with same numbers matches; different numbers mismatch."""
    long_df = pd.DataFrame({
        "city": ["NY", "NY", "CHI", "CHI"],
        "category": ["A", "B", "A", "B"],
        "val": [10.0, 20.0, 30.0, 40.0],
    })
    wide_df = pd.DataFrame({
        "city": ["NY", "CHI"],
        "A": [10.0, 30.0],
        "B": [20.0, 40.0],
    })
    # Same numbers in different shape
    assert results_match(long_df, wide_df) is True

    # Same shape with one different number
    diff_df = pd.DataFrame({
        "city": ["NY", "CHI"],
        "A": [10.0, 30.0],
        "B": [20.0, 999.0],
    })
    assert results_match(long_df, diff_df) is False


def test_reviewer_rerun_only_reviews_rerun_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """On rerun, reviewer reuses previous verdicts for unchanged items."""
    df = pd.DataFrame({"sales": [10, 20]})
    state = setup_test_state(tmp_path, df)
    state["rerun_ids"] = [1]
    state["review_report"] = [
        {"id": 0, "verdict": "pass", "reason": "Passed in round 1"},
        {"id": 1, "verdict": "fail", "reason": "Failed in round 1"},
    ]
    state["analysis_results"] = [
        {"id": 0, "question": "Q0", "code": "result = 30", "result": 30, "status": "ok"},
        {"id": 1, "question": "Q1", "code": "result = 30", "result": 30, "status": "ok"},
    ]

    # Only item 1 should be recomputed by LLM!
    mock_llm = MockLLMSequence([
        'result = df["sales"].sum()',
        '{"issues": []}',
    ])
    monkeypatch.setattr("agents.reviewer.call_llm", mock_llm)

    res = reviewer_node(state)
    assert len(res["review_report"]) == 2
    r0 = next(r for r in res["review_report"] if r["id"] == 0)
    assert r0["reason"] == "Passed in round 1"
    r1 = next(r for r in res["review_report"] if r["id"] == 1)
    assert r1["verdict"] == "pass"
    # LLM should have only been invoked for item 1
    assert len(mock_llm.prompts) == 2


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

    # Item is marked as unverified (does not fail and does not trigger retry)
    assert res["review_report"][0]["verdict"] == "unverified"
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
    assert report["verdict"] == "unverified"
    assert any("could not verify this item" in w for w in report["warnings"])
    assert len(res["review_feedback"]) == 0


def test_filter_logic_issues_drops_formatting_only_keeps_real_issues():
    """filter_logic_issues removes Series/DataFrame/formatting complaints while retaining real errors."""
    raw_issues = [
        "The sandbox returned a Series instead of a DataFrame",
        "Pure formatting difference in output index naming",
        "Incorrect dtype: column is object instead of float64",
        "missing filter on date column",
        "wrong aggregation: used sum instead of average",
        "double counting rows due to missing group by filter",
        "DataFrame column names do not match expected format",
    ]

    filtered = filter_logic_issues(raw_issues)

    # Should keep only the 3 real issues:
    assert len(filtered) == 3
    assert "missing filter on date column" in filtered
    assert "wrong aggregation: used sum instead of average" in filtered
    assert "double counting rows due to missing group by filter" in filtered

    # Formatting-only issues dropped:
    assert not any("Series instead of a DataFrame" in x for x in filtered)
    assert not any("formatting difference" in x for x in filtered)
    assert not any("column names do not match" in x for x in filtered)
    assert not any("Incorrect dtype" in x for x in filtered)


def test_logic_review_prompt_includes_markdown_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Check 3 prompt includes a real preview table and Series/DataFrame instruction."""
    df = pd.DataFrame({"sales": [10, 20]})
    state = setup_test_state(tmp_path, df)
    state["analysis_results"] = [
        {"id": 0, "question": "Total sales?", "code": "result = df['sales'].sum()", "result": pd.DataFrame({"sales": [30]}), "status": "ok"}
    ]

    captured_prompts = []

    def mock_call(p: str) -> str:
        captured_prompts.append(p)
        if "recomputed" in p.lower() or "independent" in p.lower():
            return "result = df['sales'].sum()"
        return '{"issues": []}'

    monkeypatch.setattr("agents.reviewer.call_llm", mock_call)
    reviewer_node(state)

    # Find the logic review prompt
    logic_prompts = [p for p in captured_prompts if "never report a Series versus DataFrame difference" in p]
    assert len(logic_prompts) == 1
    prompt = logic_prompts[0]

    # Verify table preview and instruction are in prompt
    assert "| sales |" in prompt
    assert "| 30 |" in prompt
    assert "The sandbox automatically converts a pandas Series into a two-column DataFrame" in prompt
    assert "Only report problems that can change the numbers" in prompt


def test_results_match_bug1_and_bug2_regressions():
    """Regression tests for Bug 1 (layout-independent labels) and Bug 2 (scalar vs single-cell table)."""

    # 1. near-tied values with different labels -> mismatch (and the trace reason mentions the labels)
    df_labels_a = pd.DataFrame({
        "cuisine_category": ["Burgers & Fast Food", "Healthy & Bowls", "Salads"],
        "customer_rating": [4.10, 4.12, 4.14],
    })
    df_labels_b = pd.DataFrame({
        "cuisine_category": ["Indian & Curries", "Japanese & Sushi", "Thai & Noodles"],
        "customer_rating": [4.10, 4.12, 4.14],
    })
    matched, trace = results_match_with_trace(df_labels_a, df_labels_b)
    assert matched is False
    assert "text labels differ" in trace["reason"]
    # Differing labels are mentioned in the trace reason
    assert "burgers & fast food" in trace["reason"] or "Burgers & Fast Food" in trace["reason"]
    assert "indian & curries" in trace["reason"] or "Indian & Curries" in trace["reason"]
    assert results_match(df_labels_a, df_labels_b) is False

    # 2. same labels in different row order -> match
    df_order_a = pd.DataFrame({
        "cuisine_category": ["Sandwiches", "Mexican & Tacos", "Italian & Pasta"],
        "customer_rating": [4.20, 3.85, 4.50],
    })
    df_order_b = pd.DataFrame({
        "cuisine_category": ["Italian & Pasta", "Sandwiches", "Mexican & Tacos"],
        "customer_rating": [4.50, 4.20, 3.85],
    })
    matched, trace = results_match_with_trace(df_order_a, df_order_b)
    assert matched is True
    assert results_match(df_order_a, df_order_b) is True

    # 3. long table vs equivalent wide table -> match
    df_long = pd.DataFrame({
        "cuisine_category": ["Bakery & Desserts", "Coffee & Beverages"],
        "order_value_usd": [1520.50, 890.25],
    })
    df_wide = pd.DataFrame({
        "Bakery & Desserts": [1520.50],
        "Coffee & Beverages": [890.25],
    })
    matched, trace = results_match_with_trace(df_long, df_wide)
    assert matched is True
    assert trace["branch"] == "layout-independent fallback"
    assert results_match(df_long, df_wide) is True

    # Also reverse comparison (wide vs long)
    matched_rev, trace_rev = results_match_with_trace(df_wide, df_long)
    assert matched_rev is True
    assert trace_rev["branch"] == "layout-independent fallback"
    assert results_match(df_wide, df_long) is True

    # 4. scalar vs table with 1 label cell and 1 number: equal -> match, different -> mismatch
    df_single_cell_match = pd.DataFrame({
        "is_peak_hour": [True],
        "total_discount": [456.78],
    })
    matched, trace = results_match_with_trace(456.78, df_single_cell_match)
    assert matched is True
    assert trace["branch"] == "scalar vs table (single numeric cell)"
    assert results_match(456.78, df_single_cell_match) is True

    matched_rev, trace_rev = results_match_with_trace(df_single_cell_match, 456.78)
    assert matched_rev is True
    assert trace_rev["branch"] == "scalar vs table (single numeric cell)"
    assert results_match(df_single_cell_match, 456.78) is True

    df_single_cell_diff = pd.DataFrame({
        "is_peak_hour": [True],
        "total_discount": [999.99],
    })
    matched_diff, trace_diff = results_match_with_trace(456.78, df_single_cell_diff)
    assert matched_diff is False
    assert trace_diff["branch"] == "scalar vs table (single numeric cell)"
    assert results_match(456.78, df_single_cell_diff) is False

    # 1x1 table
    df_1x1 = pd.DataFrame([[456.78]])
    matched_1x1, trace_1x1 = results_match_with_trace(456.78, df_1x1)
    assert matched_1x1 is True
    assert trace_1x1["branch"] == "scalar vs table (single numeric cell)"
    assert results_match(456.78, df_1x1) is True

    # 5. scalar vs table with 2 numeric cells -> not comparable
    df_multi_cell = pd.DataFrame({
        "category": ["Bakery & Desserts", "Coffee & Beverages"],
        "count": [10.0, 20.0],
    })
    matched, trace = results_match_with_trace(10.0, df_multi_cell)
    assert matched is False
    assert trace["branch"] == "scalar vs table"
    assert "not comparable" in trace["reason"]
    assert results_match(10.0, df_multi_cell) is False

    # 6. text labels that differ only in case or whitespace -> match
    df_case_space_a = pd.DataFrame({
        "cuisine_category": ["  Mediterranean  ", "STREET FOOD\t"],
        "avg_time": [28.4, 19.2],
    })
    df_case_space_b = pd.DataFrame({
        "cuisine_category": ["mediterranean", "Street Food"],
        "avg_time": [28.4, 19.2],
    })
    matched, trace = results_match_with_trace(df_case_space_a, df_case_space_b)
    assert matched is True
    assert results_match(df_case_space_a, df_case_space_b) is True
