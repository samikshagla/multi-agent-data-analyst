"""
tests/test_eval_e2e.py
Unit tests for End-to-End Three-System Comparison Benchmark (eval/run_e2e_eval.py).

Verifies:
1. System A failure handling: sandbox execution errors count as incorrect, no retries.
2. First-pass sharing: System B and System C share the exact same first attempt;
   fake analyst is called only once for the initial attempt.
3. Paired classification scenarios:
   - Scenario where reviewer fixes a wrong answer (fixed = 1)
   - Scenario where reviewer harms a correct answer (harmed = 1)
   - Scenario where retry limit stops at 3 retries (4 rounds total)
4. Label leakage guard: assert raises ValueError if reference_answer, label, etc., is in state.
5. Resumption: existing cases in cases.jsonl are skipped cleanly.
6. Budget and LLM error handling: BudgetExceeded and LLMUnavailableError stop cleanly with resume command.
7. Generic exception handling: unhandled exception in an item records status="error" and run continues.
8. Paired table computation, Wilson intervals on unique items, and flips table on multi-repeat handmade data.
9. Summarize-only mode regenerating summary without LLM calls.
"""

import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pytest

from eval.budget import BudgetExceeded, TokenBudget
from eval.metrics import wilson_ci_from_rate
from eval.run_e2e_eval import (
    _assert_no_label_leakage,
    _to_json_friendly,
    build_analyst_code_prompt,
    classify_paired_ab,
    classify_paired_bc,
    generate_summary,
    run_evaluation,
    run_system_a,
    run_system_b,
    run_system_c,
    summarize_run,
)
from llm import LLMUnavailableError, LLMConfigError


# ============================================================================
# 1. Tests for System A: Single Prompt & Sandbox Execution
# ============================================================================

def test_system_a_sandbox_error_is_incorrect_and_no_retry():
    """
    Verifies that System A makes exactly ONE LLM call, runs code once in the sandbox,
    and if a runtime error occurs (e.g. ZeroDivisionError or SyntaxError), records
    status='failed', error message, and is_correct=False without retrying.
    """
    df = pd.DataFrame({"val": [10, 20, 30]})
    budget = TokenBudget()
    call_count = 0

    def mock_llm_syntax_error(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "result = 10 / 0  # ZeroDivisionError"

    res_a = run_system_a(
        question="What is the average val?",
        df=df,
        budget=budget,
        call_llm_fn=mock_llm_syntax_error,
    )

    assert call_count == 1, "System A must call the LLM exactly once (no retries)"
    assert res_a["status"] in ("failed", "error")
    assert res_a["result"] is None
    assert "division by zero" in str(res_a["error"]).lower() or "zerodivision" in str(res_a["error"]).lower()
    assert res_a["execution_error"] is not None


def test_system_a_success_produces_result():
    """Verifies that valid code generated in System A succeeds and extracts result."""
    df = pd.DataFrame({"val": [10, 20, 30]})
    budget = TokenBudget()

    def mock_llm_success(prompt: str) -> str:
        return "result = int(df['val'].sum())"

    res_a = run_system_a(
        question="What is the sum of val?",
        df=df,
        budget=budget,
        call_llm_fn=mock_llm_success,
    )

    assert res_a["status"] == "ok"
    assert res_a["result"] == 60
    assert res_a["error"] is None


def test_build_analyst_code_prompt_matches_analyst_template():
    """Verifies that build_analyst_code_prompt formats identically to agents/analyst.py."""
    prompt = build_analyst_code_prompt("What is total revenue?", ["order_id", "revenue"])
    assert "Question: What is total revenue?" in prompt
    assert "Columns available:\norder_id, revenue" in prompt
    assert "Store the final answer in a variable called result" in prompt
    assert "Use only pd and np, no imports, no file access. Return ONLY the code, no explanation." in prompt


# ============================================================================
# 2. Tests for First Attempt Sharing (System B and System C)
# ============================================================================

def test_b_and_c_share_same_first_attempt():
    """
    Verifies that System C starts directly from System B's initial output.
    When reviewer approves (verdict: 'pass'), fake analyst is called exactly ONCE total
    (during System B), and System C makes zero additional analyst calls.
    """
    analyst_call_count = 0
    reviewer_call_count = 0

    def fake_analyst_node(state: dict[str, Any]) -> dict[str, Any]:
        nonlocal analyst_call_count
        analyst_call_count += 1
        q = state.get("questions", [""])[0]
        return {
            "questions": [q],
            "analysis_results": [{
                "id": 0,
                "question": q,
                "code": "result = 42",
                "result": 42,
                "status": "ok",
                "attempts": 1,
            }],
            "events": [],
        }

    def fake_reviewer_node(state: dict[str, Any]) -> dict[str, Any]:
        nonlocal reviewer_call_count
        reviewer_call_count += 1
        return {
            "review_feedback": [],
            "review_report": [{
                "id": 0,
                "verdict": "pass",
                "reason": "Recomputed answer matches 42 perfectly.",
            }],
            "caught_errors": [],
            "events": [],
        }

    budget = TokenBudget()
    csv_path = "data/sample.csv"
    profile_data = {"columns": ["order_id"]}

    # Run System B
    res_b, out_b = run_system_b(
        question="What is the answer?",
        csv_path=csv_path,
        profile_data=profile_data,
        budget=budget,
        analyst_fn=fake_analyst_node,
    )
    assert analyst_call_count == 1
    assert res_b["result"] == 42

    # Run System C starting from out_b
    res_c = run_system_c(
        question="What is the answer?",
        csv_path=csv_path,
        profile_data=profile_data,
        out_b=out_b,
        ref_answer=42,
        budget=budget,
        analyst_fn=fake_analyst_node,
        reviewer_fn=fake_reviewer_node,
    )

    # Analyst was NOT called again because reviewer passed on round 1!
    assert analyst_call_count == 1, "Fake analyst must NOT be called for System C when first-pass is approved"
    assert reviewer_call_count == 1
    assert res_c["final_code"] == res_b["code"]
    assert res_c["final_result"] == res_b["result"]
    assert res_c["final_verdict"] == "pass"
    assert res_c["rounds_count"] == 1
    assert res_c["retries_used"] == 0
    assert res_c["is_correct"] is True


# ============================================================================
# 3. Tests for Paired Classifications (FIXED, HARMED, RETRY LIMIT)
# ============================================================================

def test_paired_classification_fixed_by_reviewer():
    """
    Scenario where reviewer fixes a wrong answer:
    - System B first pass produces wrong answer (999).
    - Reviewer flags it as fail.
    - Analyst on retry produces correct answer (42).
    - Reviewer approves on round 2.
    Outcome: B is wrong, C is correct -> paired_bc = 'fixed', fixed_by_reviewer = True.
    """
    analyst_calls = 0

    def fake_analyst(state: dict[str, Any]) -> dict[str, Any]:
        nonlocal analyst_calls
        analyst_calls += 1
        q = state.get("questions", [""])[0]
        fb = state.get("review_feedback", [])
        if not fb:
            # First pass: returns wrong result
            return {
                "questions": [q],
                "analysis_results": [{
                    "id": 0, "question": q, "code": "result = 999", "result": 999, "status": "ok", "attempts": 1,
                }],
                "events": [],
            }
        else:
            # Retry with feedback: fixes code to return 42
            return {
                "questions": [q],
                "analysis_results": [{
                    "id": 0, "question": q, "code": "result = 42", "result": 42, "status": "ok", "attempts": 1,
                }],
                "review_feedback": [],
                "events": [],
            }

    def fake_reviewer(state: dict[str, Any]) -> dict[str, Any]:
        item = state.get("analysis_results", [{}])[0]
        curr_val = item.get("result")
        if curr_val == 42:
            return {
                "review_feedback": [],
                "review_report": [{"id": 0, "verdict": "pass", "reason": "Correct"}],
                "caught_errors": [],
                "events": [],
            }
        else:
            return {
                "review_feedback": [{"id": 0, "verdict": "fail", "reason": "Recomputed value 42 does not match 999"}],
                "review_report": [{"id": 0, "verdict": "fail", "reason": "Recomputation mismatch"}],
                "caught_errors": [{"id": 0, "type": "recomputation_mismatch"}],
                "events": [],
            }

    budget = TokenBudget()
    csv_path = "data/sample.csv"
    profile_data = {}

    res_b, out_b = run_system_b("Question?", csv_path, profile_data, budget, analyst_fn=fake_analyst)
    assert res_b["result"] == 999

    res_c = run_system_c("Question?", csv_path, profile_data, out_b, ref_answer=42, budget=budget, analyst_fn=fake_analyst, reviewer_fn=fake_reviewer)

    assert res_c["final_result"] == 42
    assert res_c["is_correct"] is True
    assert res_c["rounds_count"] == 2
    assert res_c["retries_used"] == 1

    b_correct = (res_b["result"] == 42)
    c_correct = res_c["is_correct"]
    paired = classify_paired_bc(b_correct, c_correct)
    assert paired == "fixed"


def test_paired_classification_harmed_by_reviewer():
    """
    Scenario where reviewer harms a correct answer:
    - System B first pass produces correct answer (42).
    - Reviewer has a false alarm: flags it as fail.
    - Analyst on retry is misled and alters code to wrong answer (999).
    - System C ends with wrong answer.
    Outcome: B is correct, C is wrong -> paired_bc = 'harmed', harmed_by_reviewer = True.
    """
    def fake_analyst(state: dict[str, Any]) -> dict[str, Any]:
        fb = state.get("review_feedback", [])
        q = state.get("questions", [""])[0]
        if not fb:
            return {
                "questions": [q],
                "analysis_results": [{
                    "id": 0, "question": q, "code": "result = 42", "result": 42, "status": "ok", "attempts": 1,
                }],
                "events": [],
            }
        else:
            return {
                "questions": [q],
                "analysis_results": [{
                    "id": 0, "question": q, "code": "result = 999", "result": 999, "status": "ok", "attempts": 1,
                }],
                "review_feedback": [],
                "events": [],
            }

    def fake_reviewer(state: dict[str, Any]) -> dict[str, Any]:
        item = state.get("analysis_results", [{}])[0]
        curr_val = item.get("result")
        if curr_val == 42:
            # False alarm
            return {
                "review_feedback": [{"id": 0, "verdict": "fail", "reason": "False alarm rejection"}],
                "review_report": [{"id": 0, "verdict": "fail", "reason": "False alarm"}],
                "caught_errors": [],
                "events": [],
            }
        else:
            # Accepts the revised (wrong) answer or gives pass
            return {
                "review_feedback": [],
                "review_report": [{"id": 0, "verdict": "pass", "reason": "Approved revised"}],
                "caught_errors": [],
                "events": [],
            }

    budget = TokenBudget()
    res_b, out_b = run_system_b("Question?", "data/sample.csv", {}, budget, analyst_fn=fake_analyst)
    assert res_b["result"] == 42

    res_c = run_system_c("Question?", "data/sample.csv", {}, out_b, ref_answer=42, budget=budget, analyst_fn=fake_analyst, reviewer_fn=fake_reviewer)

    assert res_c["final_result"] == 999
    assert res_c["is_correct"] is False
    assert res_c["rounds_count"] == 2

    paired = classify_paired_bc(True, False)
    assert paired == "harmed"


def test_paired_classification_retry_limit_stops_at_3():
    """
    Scenario where reviewer continuously fails:
    - Loop must stop after 3 retries (4 rounds total).
    - Retries used = 3, rounds_count = 4.
    """
    analyst_call_count = 0
    reviewer_call_count = 0

    def fake_analyst(state: dict[str, Any]) -> dict[str, Any]:
        nonlocal analyst_call_count
        analyst_call_count += 1
        q = state.get("questions", [""])[0]
        return {
            "questions": [q],
            "analysis_results": [{
                "id": 0, "question": q, "code": f"result = {analyst_call_count}", "result": analyst_call_count, "status": "ok", "attempts": 1,
            }],
            "review_feedback": [],
            "events": [],
        }

    def fake_reviewer(state: dict[str, Any]) -> dict[str, Any]:
        nonlocal reviewer_call_count
        reviewer_call_count += 1
        return {
            "review_feedback": [{"id": 0, "verdict": "fail", "reason": "Still wrong"}],
            "review_report": [{"id": 0, "verdict": "fail", "reason": "Still wrong"}],
            "caught_errors": [],
            "events": [],
        }

    budget = TokenBudget()
    res_b, out_b = run_system_b("Question?", "data/sample.csv", {}, budget, analyst_fn=fake_analyst)
    assert analyst_call_count == 1

    res_c = run_system_c("Question?", "data/sample.csv", {}, out_b, ref_answer=999, budget=budget, analyst_fn=fake_analyst, reviewer_fn=fake_reviewer)

    # 1 initial round + 3 retries = 4 reviewer calls, 3 additional analyst calls (4 total analyst calls)
    assert reviewer_call_count == 4
    assert analyst_call_count == 4
    assert res_c["rounds_count"] == 4
    assert res_c["retries_used"] == 3
    assert res_c["final_verdict"] == "fail"


def test_classify_paired_ab_categories():
    """Verifies all four paired classifications between System A and System B."""
    assert classify_paired_ab(True, True) == "both_correct"
    assert classify_paired_ab(False, True) == "b_improved"
    assert classify_paired_ab(True, False) == "b_worsened"
    assert classify_paired_ab(False, False) == "both_wrong"


# ============================================================================
# 4. Tests for Label Leakage Guard
# ============================================================================

def test_label_leakage_guard_raises_on_forbidden_keys():
    """Verifies that _assert_no_label_leakage raises ValueError if benchmark ground-truth keys are in state."""
    forbidden = ["reference_answer", "label", "ground_truth", "kind", "is_wrong", "certified_answer"]

    for k in forbidden:
        leaky_state = {"clean_df_path": "data/sample.csv", k: "secret_label"}
        with pytest.raises(ValueError, match="CRITICAL EVALUATION LEAK"):
            _assert_no_label_leakage(leaky_state)

    # Test nested leakage inside analysis_results
    leaky_nested = {
        "clean_df_path": "data/sample.csv",
        "analysis_results": [{"id": 0, "reference_answer": 42}],
    }
    with pytest.raises(ValueError, match="CRITICAL EVALUATION LEAK"):
        _assert_no_label_leakage(leaky_nested)

    # Clean state should not raise
    clean_state = {
        "clean_df_path": "data/sample.csv",
        "analysis_results": [{"id": 0, "question": "Total?", "code": "result = 1", "result": 1}],
    }
    _assert_no_label_leakage(clean_state)


# ============================================================================
# 5. Tests for Resumption (cases.jsonl)
# ============================================================================

def test_resume_skips_finished_cases(tmp_path: Path):
    """Verifies that existing cases in cases.jsonl are skipped and only remaining items are executed."""
    run_dir = tmp_path / "resume_run"
    run_dir.mkdir(parents=True)
    cases_file = run_dir / "cases.jsonl"

    # Write finished record for item 1, repeat 0
    record_1 = {
        "item_id": 1,
        "repeat_index": 0,
        "case_id": "q01_r1",
        "question": "Total sales?",
        "category": "aggregate",
        "difficulty": "easy",
        "system_a": {"status": "ok", "is_correct": True, "tokens": 100, "llm_calls": 1},
        "system_b": {"status": "ok", "is_correct": True, "tokens": 100, "llm_calls": 1},
        "system_c": {"status": "ok", "is_correct": True, "tokens_used": 100, "rounds_count": 1},
        "paired_bc": "both_correct",
        "paired_ab": "both_correct",
    }
    with open(cases_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record_1) + "\n")

    executed_items: list[int] = []

    def mock_analyst(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "questions": ["Q?"],
            "analysis_results": [{"id": 0, "code": "result = 10", "result": 10, "status": "ok"}],
            "events": [],
        }

    def mock_reviewer(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "review_feedback": [],
            "review_report": [{"id": 0, "verdict": "pass", "reason": "ok"}],
            "caught_errors": [],
            "events": [],
        }

    def mock_call_llm(prompt: str) -> str:
        return "result = 10"

    # Track which items run by wrapping run_system_a or checking cases
    # Run evaluation on items [1, 4]
    run_evaluation(
        items=[1, 4],
        repeats=1,
        run_dir=str(run_dir),
        analyst_fn=mock_analyst,
        reviewer_fn=mock_reviewer,
        call_llm_fn=mock_call_llm,
    )

    # Read back cases.jsonl: should now have 2 items
    records = []
    with open(cases_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 2
    assert records[0]["item_id"] == 1
    assert records[1]["item_id"] == 4


# ============================================================================
# 6. Tests for Budget and AI Service Error Handling
# ============================================================================

def test_budget_exceeded_stops_cleanly(tmp_path: Path):
    """Verifies that BudgetExceeded pauses evaluation cleanly and exits with code 1."""
    run_dir = tmp_path / "budget_run"
    run_dir.mkdir(parents=True)

    def mock_llm_with_budget(prompt: str) -> str:
        # Simulate budget check raising BudgetExceeded
        raise BudgetExceeded(max_tokens=50, used_tokens=100)

    with pytest.raises(SystemExit) as exc_info:
        run_evaluation(
            items=[1],
            repeats=1,
            run_dir=str(run_dir),
            call_llm_fn=mock_llm_with_budget,
        )
    assert exc_info.value.code == 1


def test_llm_unavailable_stops_cleanly(tmp_path: Path):
    """Verifies that LLMUnavailableError pauses evaluation cleanly and exits with code 1."""
    run_dir = tmp_path / "unavailable_run"
    run_dir.mkdir(parents=True)

    def mock_llm_unavailable(prompt: str) -> str:
        raise LLMUnavailableError("API key quota exhausted")

    with pytest.raises(SystemExit) as exc_info:
        run_evaluation(
            items=[1],
            repeats=1,
            run_dir=str(run_dir),
            call_llm_fn=mock_llm_unavailable,
        )
    assert exc_info.value.code == 1


# ============================================================================
# 7. Tests for Generic Exception Handling
# ============================================================================

def test_generic_exception_records_status_error_and_continues(tmp_path: Path):
    """
    Verifies that an unhandled individual exception in an item records status='error',
    is_correct=False, and the run continues with subsequent items.
    """
    run_dir = tmp_path / "generic_err_run"
    run_dir.mkdir(parents=True)

    call_count = 0

    def faulty_call_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First item encounters an unexpected runtime error
            raise RuntimeError("Unexpected simulated internal failure")
        return "result = 42"

    def mock_analyst(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "questions": ["Q?"],
            "analysis_results": [{"id": 0, "code": "result = 42", "result": 42, "status": "ok"}],
            "events": [],
        }

    def mock_reviewer(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "review_feedback": [],
            "review_report": [{"id": 0, "verdict": "pass", "reason": "ok"}],
            "caught_errors": [],
            "events": [],
        }

    # Evaluate 2 items: item 1 fails with RuntimeError, item 4 succeeds
    run_evaluation(
        items=[1, 4],
        repeats=1,
        run_dir=str(run_dir),
        call_llm_fn=faulty_call_llm,
        analyst_fn=mock_analyst,
        reviewer_fn=mock_reviewer,
    )

    cases_file = run_dir / "cases.jsonl"
    records = []
    with open(cases_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 2
    # Item 1 should have status="error" and is_correct=False
    assert records[0]["item_id"] == 1
    assert records[0]["system_a"]["status"] == "error"
    assert records[0]["system_a"]["is_correct"] is False
    assert records[0]["paired_bc"] == "both_wrong"

    # Item 4 should have run normally
    assert records[1]["item_id"] == 4
    assert records[1]["system_a"]["status"] == "ok"


# ============================================================================
# 8. Tests for Paired Table, Wilson Intervals, and Flips on Handmade Data
# ============================================================================

def test_paired_table_intervals_and_flips_on_handmade_data(tmp_path: Path):
    """
    Verifies generate_summary calculation on synthetic data with repeats > 1:
    - Unique items = 2, repeats = 2, total finished = 4.
    - Accuracies averaged over repeats.
    - Wilson CIs computed with unique items n = 2.
    - Paired counts: both_correct, fixed, harmed, both_wrong.
    - Flips table identifies items whose verdicts differed across repeats.
    """
    run_dir = tmp_path / "summary_test"
    run_dir.mkdir(parents=True)

    records = [
        # Repeat 1:
        # Item 1: A correct, B correct, C correct (both_correct)
        {
            "item_id": 1,
            "repeat_index": 0,
            "question": "What is revenue?",
            "category": "aggregate",
            "difficulty": "easy",
            "reference_answer": 100,
            "system_a": {"is_correct": True, "tokens": 50, "llm_calls": 1, "seconds": 1.0},
            "system_b": {"is_correct": True, "tokens": 60, "llm_calls": 1, "seconds": 1.2},
            "system_c": {"is_correct": True, "rounds_count": 1, "retries_used": 0, "additional_tokens": 40, "additional_llm_calls": 1, "additional_seconds": 1.0, "rounds": [{"reviewer_verdict": "pass"}]},
            "paired_bc": "both_correct",
            "paired_ab": "both_correct",
        },
        # Item 2: A wrong, B wrong, C correct (fixed by reviewer)
        {
            "item_id": 2,
            "repeat_index": 0,
            "question": "Top category?",
            "category": "ranking",
            "difficulty": "medium",
            "reference_answer": "Electronics",
            "system_a": {"is_correct": False, "tokens": 50, "llm_calls": 1, "seconds": 1.0},
            "system_b": {"is_correct": False, "tokens": 60, "llm_calls": 1, "seconds": 1.2},
            "system_c": {"is_correct": True, "rounds_count": 2, "retries_used": 1, "additional_tokens": 80, "additional_llm_calls": 2, "additional_seconds": 2.0, "rounds": [{"reviewer_verdict": "fail"}, {"reviewer_verdict": "pass"}]},
            "paired_bc": "fixed",
            "paired_ab": "both_wrong",
        },
        # Repeat 2:
        # Item 1: A wrong, B correct, C wrong (harmed by reviewer) -> FLIP for A and C!
        {
            "item_id": 1,
            "repeat_index": 1,
            "question": "What is revenue?",
            "category": "aggregate",
            "difficulty": "easy",
            "reference_answer": 100,
            "system_a": {"is_correct": False, "tokens": 50, "llm_calls": 1, "seconds": 1.0},
            "system_b": {"is_correct": True, "tokens": 60, "llm_calls": 1, "seconds": 1.2},
            "system_c": {"is_correct": False, "rounds_count": 2, "retries_used": 1, "additional_tokens": 80, "additional_llm_calls": 2, "additional_seconds": 2.0, "rounds": [{"reviewer_verdict": "fail", "reviewer_reason": "False alarm"}, {"reviewer_verdict": "pass"}]},
            "paired_bc": "harmed",
            "paired_ab": "b_improved",
        },
        # Item 2: A wrong, B wrong, C wrong (both_wrong) -> FLIP for C (was correct in r1)!
        {
            "item_id": 2,
            "repeat_index": 1,
            "question": "Top category?",
            "category": "ranking",
            "difficulty": "medium",
            "reference_answer": "Electronics",
            "system_a": {"is_correct": False, "tokens": 50, "llm_calls": 1, "seconds": 1.0},
            "system_b": {"is_correct": False, "tokens": 60, "llm_calls": 1, "seconds": 1.2},
            "system_c": {"is_correct": False, "rounds_count": 1, "retries_used": 0, "additional_tokens": 40, "additional_llm_calls": 1, "additional_seconds": 1.0, "rounds": [{"reviewer_verdict": "pass"}]},
            "paired_bc": "both_wrong",
            "paired_ab": "both_wrong",
        },
    ]

    summary = generate_summary(records, run_dir, repeats=2, expected_total=4)

    assert summary["finished_cases"] == 4
    assert summary["unique_items_count"] == 2
    assert summary["repeats"] == 2

    # Accuracy checks:
    # Repeat 1: A=1/2 (50%), B=1/2 (50%), C=2/2 (100%)
    # Repeat 2: A=0/2 (0%), B=1/2 (50%), C=0/2 (0%)
    # Mean: A=25%, B=50%, C=50%
    assert summary["accuracy"]["system_a"]["accuracy_rate"] == 0.25
    assert summary["accuracy"]["system_b"]["accuracy_rate"] == 0.50
    assert summary["accuracy"]["system_c"]["accuracy_rate"] == 0.50

    # Wilson CIs on n=2 unique items
    expected_ci_b = wilson_ci_from_rate(0.50, 2)
    assert summary["accuracy"]["system_b"]["accuracy_ci"] == list(expected_ci_b)

    # Paired B vs C checks:
    # 4 observations: 1 both_correct, 1 fixed, 1 harmed, 1 both_wrong
    bc_counts = summary["paired_bc"]["counts"]
    assert bc_counts["both_correct"] == 1
    assert bc_counts["fixed"] == 1
    assert bc_counts["harmed"] == 1
    assert bc_counts["both_wrong"] == 1

    # Fixed items detail
    assert len(summary["paired_bc"]["fixed_items"]) == 1
    assert summary["paired_bc"]["fixed_items"][0]["item_id"] == 2

    # Harmed items detail
    assert len(summary["paired_bc"]["harmed_items"]) == 1
    assert summary["paired_bc"]["harmed_items"][0]["item_id"] == 1

    # Flips check: both Item 1 and Item 2 flipped across repeats
    assert len(summary["flips"]) == 2

    # Verify summary.md and summary.json files were created
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "summary.json").exists()

    md_content = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "## 1. System Accuracy Comparison" in md_content
    assert "## 2. Paired Analysis: System B vs System C" in md_content
    assert "FIXED by reviewer" in md_content
    assert "HARMED by reviewer" in md_content
    assert "With n = 12 the paired counts are the main result; differences of one or two questions are not statistically meaningful." in md_content


# ============================================================================
# 9. Tests for Summarize-Only Mode
# ============================================================================

def test_summarize_only_mode(tmp_path: Path):
    """
    Verifies that summarize_run reads cases.jsonl in read-only mode and regenerates
    summary.md and summary.json without modifying cases.jsonl or making LLM calls.
    """
    run_dir = tmp_path / "summarize_only_run"
    run_dir.mkdir(parents=True)
    cases_file = run_dir / "cases.jsonl"

    record = {
        "item_id": 1,
        "repeat_index": 0,
        "question": "Total?",
        "category": "aggregate",
        "difficulty": "easy",
        "system_a": {"is_correct": True, "tokens": 10, "llm_calls": 1},
        "system_b": {"is_correct": True, "tokens": 10, "llm_calls": 1},
        "system_c": {"is_correct": True, "additional_tokens": 10, "additional_llm_calls": 1, "rounds_count": 1},
        "paired_bc": "both_correct",
        "paired_ab": "both_correct",
    }
    with open(cases_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    initial_mtime = cases_file.stat().st_mtime_ns

    summary = summarize_run(run_dir=run_dir, repeats=1)

    assert summary["finished_cases"] == 1
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "summary.json").exists()

    # Verify cases.jsonl was untouched
    final_mtime = cases_file.stat().st_mtime_ns
    assert initial_mtime == final_mtime


# ============================================================================
# 10. Tests for Robust Serialization (_to_json_friendly)
# ============================================================================

def test_to_json_friendly_all_types():
    """
    Verifies that _to_json_friendly safely converts DataFrames, Series, numpy arrays/scalars,
    NaN, Timestamps, Intervals, Periods, and nested structures to JSON-serializable output.
    """
    # 1. DataFrame small (under cap, not truncated)
    df_small = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    res_df = _to_json_friendly(df_small)
    assert res_df["columns"] == ["a", "b"]
    assert res_df["rows"] == [[1, "x"], [2, "y"]]
    assert res_df["total_rows"] == 2
    assert res_df["truncated"] is False
    assert json.dumps(res_df)

    # 2. DataFrame large (> 50 rows -> truncated)
    df_large = pd.DataFrame({"x": range(100)})
    res_large = _to_json_friendly(df_large, max_rows=50)
    assert len(res_large["rows"]) == 50
    assert res_large["total_rows"] == 100
    assert res_large["truncated"] is True
    assert json.dumps(res_large)

    # 3. Series
    s = pd.Series([10, 20, 30], name="metric")
    res_s = _to_json_friendly(s)
    assert "columns" in res_s and "rows" in res_s
    assert json.dumps(res_s)

    # 4. NumPy numbers and arrays
    assert _to_json_friendly(np.int64(42)) == 42
    assert _to_json_friendly(np.float64(3.14)) == 3.14
    assert _to_json_friendly(np.bool_(True)) is True
    assert _to_json_friendly(np.array([1, 2, 3])) == [1, 2, 3]

    # 5. NaN and Inf
    assert _to_json_friendly(np.nan) is None
    assert _to_json_friendly(float("nan")) is None
    assert _to_json_friendly(float("inf")) is None
    assert _to_json_friendly(None) is None

    # 6. Timestamp, Interval, Period
    ts = pd.Timestamp("2026-11-17 12:00:00")
    assert _to_json_friendly(ts) == "2026-11-17 12:00:00"
    iv = pd.Interval(0, 10)
    assert _to_json_friendly(iv) == "(0, 10]"
    p = pd.Period("2026-11", freq="M")
    assert _to_json_friendly(p) == "2026-11"

    # 7. Nested dicts and lists
    nested = {
        "df": df_small,
        "items": [ts, iv, np.nan, np.int32(7)],
        "sub": {"val": np.float32(1.5)},
    }
    res_nested = _to_json_friendly(nested)
    dumped = json.dumps(res_nested)
    assert dumped is not None
    loaded = json.loads(dumped)
    assert loaded["items"][2] is None
    assert loaded["items"][3] == 7
    assert loaded["sub"]["val"] == 1.5

    # 8. Unknown custom class
    class UnknownObj:
        def __str__(self):
            return "CustomUnknown"
    assert _to_json_friendly(UnknownObj()) == "CustomUnknown"


# ============================================================================
# 11. Integration Test with Real Sandbox and Real Ground Truth
# ============================================================================

def test_e2e_integration_items_1_9_17_real_sandbox_and_ground_truth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Integration test for items 1 (scalar), 9 (table), and 17 (top-N table):
    - Uses fake call_llm returning each item's reference_code
    - Uses REAL sandbox (tools/sandbox.py run_code)
    - Uses REAL ground truth (eval/ground_truth.json)
    - Verifies whole e2e flow (systems A, B, C) completes and writes valid JSON lines for all 3 items.
    """
    run_dir = tmp_path / "integration_e2e"
    run_dir.mkdir(parents=True)

    from eval.benchmark import BENCHMARK
    from agents.analyst import analyst_node
    from agents.reviewer import reviewer_node

    bm_by_id = {b["id"]: b for b in BENCHMARK}

    def fake_call_llm(prompt: str) -> str:
        # Reviewer logic review prompt returns clean JSON issues
        if "Return ONLY JSON" in prompt or "potential logical issues" in prompt:
            return '{"issues": []}'

        # Check which benchmark question is in prompt
        for item_id in (1, 9, 17):
            q_text = bm_by_id[item_id]["question"]
            if q_text in prompt:
                return bm_by_id[item_id]["reference_code"]

        # Default fallback
        return bm_by_id[1]["reference_code"]

    monkeypatch.setattr("llm.call_llm", fake_call_llm)
    monkeypatch.setattr("agents.analyst.call_llm", fake_call_llm)
    monkeypatch.setattr("agents.reviewer.call_llm", fake_call_llm)

    run_evaluation(
        items=[1, 9, 17],
        repeats=1,
        run_dir=str(run_dir),
        call_llm_fn=fake_call_llm,
        analyst_fn=analyst_node,
        reviewer_fn=reviewer_node,
    )

    cases_file = run_dir / "cases.jsonl"
    assert cases_file.exists()

    records = []
    with open(cases_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                rec = json.loads(line_str)
                records.append(rec)

    assert len(records) == 3
    for rec in records:
        assert rec["system_a"]["is_correct"] is True
        assert rec["system_b"]["is_correct"] is True
        assert rec["system_c"]["is_correct"] is True
        assert rec["paired_bc"] == "both_correct"
        assert rec["paired_ab"] == "both_correct"

    # Verify item 9 (table answer) is properly structured as a dict with columns and rows
    rec_9 = next(r for r in records if r["item_id"] == 9)
    assert isinstance(rec_9["reference_answer"], dict)
    assert "columns" in rec_9["reference_answer"] and "rows" in rec_9["reference_answer"]
    assert isinstance(rec_9["system_a"]["result"], dict)
    assert isinstance(rec_9["system_b"]["result"], dict)
    assert isinstance(rec_9["system_c"]["final_result"], dict)

    # Verify summary files exist
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "summary.json").exists()


# ============================================================================
# 12. Test Task Serialization Failure & Error Recovery
# ============================================================================

def test_task_serialization_failure_records_error_and_next_runs(tmp_path: Path):
    """
    Verifies that if a task encounters an error during processing/serialization,
    an error record is safely recorded without stopping the run, and subsequent tasks run.
    """
    run_dir = tmp_path / "ser_fail_run"
    run_dir.mkdir(parents=True)

    call_count = 0

    def mock_llm_with_one_ser_failure(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "result = 42"

    def mock_analyst_fails_item_1(state: dict[str, Any]) -> dict[str, Any]:
        q = state.get("questions", [""])[0]
        if "sum of order_value_usd" in q:
            raise TypeError("Simulated object failure in item 1")
        return {
            "questions": [q],
            "analysis_results": [{"id": 0, "code": "result = 42", "result": 42, "status": "ok"}],
            "events": [],
        }

    def mock_reviewer(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "review_feedback": [],
            "review_report": [{"id": 0, "verdict": "pass", "reason": "ok"}],
            "caught_errors": [],
            "events": [],
        }

    run_evaluation(
        items=[1, 4],
        repeats=1,
        run_dir=str(run_dir),
        call_llm_fn=mock_llm_with_one_ser_failure,
        analyst_fn=mock_analyst_fails_item_1,
        reviewer_fn=mock_reviewer,
    )

    cases_file = run_dir / "cases.jsonl"
    assert cases_file.exists()

    records = []
    with open(cases_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                rec = json.loads(line_str)
                records.append(rec)

    assert len(records) == 2
    # Item 1 must be recorded as an error
    assert records[0]["item_id"] == 1
    assert records[0]["system_b"]["status"] == "error"
    assert "Simulated object failure" in records[0]["system_b"]["error"]

    # Item 4 must have run successfully
    assert records[1]["item_id"] == 4
    assert records[1]["system_b"]["status"] == "ok"

