"""
test_analyst.py
Unit tests for agents/analyst.py using pytest.
Validates business question generation, sandbox code execution with retry,
failure handling, fallback mechanisms, and review feedback re-run workflows.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from agents.analyst import analyst_node
from state import new_state
from tools.profiler import profile_dataframe


class MockLLMSequence:
    """Helper mock to return a planned sequence of string responses."""
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return "result = 0"


def setup_test_state(tmp_path: Path, df: pd.DataFrame) -> dict:
    """Creates a temporary cleaned CSV and an initialized state dictionary."""
    clean_csv_path = tmp_path / "cleaned_data.csv"
    df.to_csv(clean_csv_path, index=False)
    state = new_state(str(clean_csv_path))
    state["clean_df_path"] = str(clean_csv_path)
    state["profile"] = profile_dataframe(df)
    return state


def test_questions_parsed_and_results_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 1: Questions are parsed and each gets a result with status 'ok'."""
    df = pd.DataFrame({
        "category": ["Electronics", "Clothing", "Electronics"],
        "sales": [100, 200, 300],
    })
    state = setup_test_state(tmp_path, df)

    mock_llm = MockLLMSequence([
        # 1. Questions prompt reply
        json.dumps(["What is the total sales?", "What is the average sales by category?"]),
        # 2. Code for Question 1
        'result = df["sales"].sum()',
        # 3. Code for Question 2
        'result = df.groupby("category")["sales"].mean()',
    ])
    monkeypatch.setattr("agents.analyst.call_llm", mock_llm)

    res = analyst_node(state)

    assert len(res["questions"]) == 2
    assert len(res["analysis_results"]) == 2

    # Check question 1 results
    item1 = res["analysis_results"][0]
    assert item1["status"] == "ok"
    assert item1["result"] == 600
    assert item1["attempts"] == 1
    assert item1["error"] is None

    # Check question 2 results
    item2 = res["analysis_results"][1]
    assert item2["status"] == "ok"
    assert isinstance(item2["result"], pd.DataFrame)
    assert item2["attempts"] == 1


def test_code_retry_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 2: Code that fails first and works on retry ends with status 'ok' and attempts == 2."""
    df = pd.DataFrame({"sales": [50, 150, 250]})
    state = setup_test_state(tmp_path, df)

    mock_llm = MockLLMSequence([
        # 1. Questions prompt
        json.dumps(["What is the max sales?"]),
        # 2. First attempt (fails due to nonexistent column)
        'result = df["nonexistent_column"].max()',
        # 3. Second attempt (retry correction succeeds)
        'result = df["sales"].max()',
    ])
    monkeypatch.setattr("agents.analyst.call_llm", mock_llm)

    res = analyst_node(state)

    assert len(res["analysis_results"]) == 1
    item = res["analysis_results"][0]
    assert item["status"] == "ok"
    assert item["attempts"] == 2
    assert item["result"] == 250
    assert item["error"] is None


def test_code_continuous_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 3: Code that keeps failing ends with status 'failed' and does not crash."""
    df = pd.DataFrame({"sales": [10, 20, 30]})
    state = setup_test_state(tmp_path, df)

    mock_llm = MockLLMSequence([
        # 1. Questions prompt
        json.dumps(["Calculate something broken"]),
        # 2. Initial attempt fails
        'result = df["bad_col_1"].sum()',
        # 3. Retry 1 fails
        'result = df["bad_col_2"].sum()',
        # 4. Retry 2 fails
        'result = df["bad_col_3"].sum()',
    ])
    monkeypatch.setattr("agents.analyst.call_llm", mock_llm)

    res = analyst_node(state)

    assert len(res["analysis_results"]) == 1
    item = res["analysis_results"][0]
    assert item["status"] == "failed"
    assert item["attempts"] == 3
    assert item["error"] is not None
    assert "KeyError" in item["error"]
    assert any(e["type"] == "error" for e in res["events"])


def test_invalid_json_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 4: Invalid JSON for questions triggers the default-question fallback."""
    df = pd.DataFrame({
        "city": ["New York", "Chicago"],
        "revenue": [1000, 2000],
    })
    state = setup_test_state(tmp_path, df)

    mock_llm = MockLLMSequence([
        # 1. Invalid first reply
        "Here are 3 questions: 1. Total? 2. Revenue? 3. City?",
        # 2. Invalid retry reply
        "I cannot provide valid JSON right now.",
        # 3, 4, 5. Code replies for the 3 fallback questions
        'result = len(df)',
        'result = df["revenue"].mean()',
        'result = df.groupby("city")["revenue"].mean()',
    ])
    monkeypatch.setattr("agents.analyst.call_llm", mock_llm)

    res = analyst_node(state)

    # 3 default questions must be generated
    assert len(res["questions"]) == 3
    assert len(res["analysis_results"]) == 3
    assert any("Questions JSON parsing failed after retry" in e["message"] for e in res["events"])


def test_rerun_with_review_feedback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 5: On re-run with review_feedback for one id, only that item changes and others stay the same."""
    df = pd.DataFrame({"sales": [100, 200, 300]})
    state = setup_test_state(tmp_path, df)

    # Pre-populate state with two existing results
    state["questions"] = ["Question 0", "Question 1"]
    state["analysis_results"] = [
        {"id": 0, "question": "Question 0", "code": "result = 10", "result": 10, "status": "ok", "attempts": 1, "notes": ""},
        {"id": 1, "question": "Question 1", "code": "result = 20", "result": 20, "status": "ok", "attempts": 1, "notes": ""},
    ]
    # Reviewer critique flagging only item id 1
    state["review_feedback"] = [
        {"id": 1, "verdict": "fail", "reason": "Expected total sum of sales", "suggested_fix": "Use df['sales'].sum()"}
    ]

    mock_llm = MockLLMSequence([
        # Response for the re-run of Question 1
        'result = df["sales"].sum()',
    ])
    monkeypatch.setattr("agents.analyst.call_llm", mock_llm)

    res = analyst_node(state)

    results = res["analysis_results"]
    assert len(results) == 2

    # Item 0 remains identical
    assert results[0]["id"] == 0
    assert results[0]["result"] == 10
    assert results[0]["code"] == "result = 10"

    # Item 1 has been recomputed and updated
    assert results[1]["id"] == 1
    assert results[1]["result"] == 600
    assert 'df["sales"].sum()' in results[1]["code"]


def test_review_feedback_cleared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 6: review_feedback is cleared in the returned dict after a re-run."""
    df = pd.DataFrame({"sales": [100, 200]})
    state = setup_test_state(tmp_path, df)

    state["analysis_results"] = [
        {"id": 0, "question": "Q0", "code": "result = 5", "result": 5, "status": "ok", "attempts": 1, "notes": ""}
    ]
    state["review_feedback"] = [
        {"id": 0, "verdict": "fail", "reason": "Incorrect value", "suggested_fix": "Fix formula"}
    ]

    mock_llm = MockLLMSequence(['result = df["sales"].sum()'])
    monkeypatch.setattr("agents.analyst.call_llm", mock_llm)

    res = analyst_node(state)

    assert "review_feedback" in res
    assert res["review_feedback"] == []


def test_predefined_questions_skips_question_generation_llm_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test 7: When state['questions'] is already provided, skip question generation LLM call."""
    df = pd.DataFrame({"sales": [100, 200, 300]})
    state = setup_test_state(tmp_path, df)
    state["questions"] = ["What is the total sales?"]

    # Only one LLM response needed: code generation for the pre-defined question
    mock_llm = MockLLMSequence(['result = df["sales"].sum()'])
    monkeypatch.setattr("agents.analyst.call_llm", mock_llm)

    res = analyst_node(state)

    # Question generation LLM call was skipped
    assert len(mock_llm.prompts) == 1
    assert "Write pandas code that answers the question" in mock_llm.prompts[0]
    assert "What is the total sales?" in mock_llm.prompts[0]
    assert "propose 3-5 insightful business questions" not in mock_llm.prompts[0]

    # Results match the pre-defined question
    assert res["questions"] == ["What is the total sales?"]
    assert len(res["analysis_results"]) == 1
    assert res["analysis_results"][0]["question"] == "What is the total sales?"
    assert res["analysis_results"][0]["result"] == 600
    assert res["analysis_results"][0]["status"] == "ok"

