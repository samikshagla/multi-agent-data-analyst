"""
test_eval_reviewer.py
Unit tests for Reviewer agent evaluation harness:
- wilson_ci calculation and zero handling
- magnitude_bucket classification for scalars and tables
- confusion matrix, rates, and subgroup breakdowns
- runner with scripted reviewer_node (metrics, resume, label leak guard, LLM unavailable pause, error handling)
- interleaved case ordering with --limit
"""

import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pytest

from eval.benchmark import BENCHMARK
from eval.ground_truth import load_ground_truth
from eval.metrics import breakdown, confusion, magnitude_bucket, wilson_ci
from eval.run_reviewer_eval import (
    _assert_no_label_leakage,
    build_evaluation_cases,
    run_evaluation,
)
from llm import LLMUnavailableError


# ============================================================================
# 1. Tests for wilson_ci
# ============================================================================

def test_wilson_ci_known_values_and_zero():
    # total = 0 safe return
    low, high = wilson_ci(0, 0)
    assert low == 0.0 and high == 0.0

    # 10 successes out of 10
    low10, high10 = wilson_ci(10, 10)
    assert high10 == 1.0
    assert 0.65 < low10 < 1.0

    # 0 successes out of 10
    low0, high0 = wilson_ci(0, 10)
    assert low0 == 0.0
    assert 0.0 < high0 < 0.35

    # 50 out of 100
    low50, high50 = wilson_ci(50, 100)
    assert 0.39 < low50 < 0.42
    assert 0.58 < high50 < 0.61

    # Verify bounds [0, 1]
    for s, t in [(1, 5), (3, 7), (19, 20), (99, 100)]:
        l, h = wilson_ci(s, t)
        assert 0.0 <= l <= h <= 1.0


# ============================================================================
# 2. Tests for magnitude_bucket
# ============================================================================

def test_magnitude_bucket_scalars():
    # Subtle (< 5%)
    assert magnitude_bucket(100.0, 102.0) == "subtle"
    assert magnitude_bucket(50.0, 51.0) == "subtle"

    # Moderate (5% to 50%)
    assert magnitude_bucket(100.0, 125.0) == "moderate"
    assert magnitude_bucket(100.0, 80.0) == "moderate"

    # Gross (> 50%)
    assert magnitude_bucket(100.0, 200.0) == "gross"
    assert magnitude_bucket(100.0, 10.0) == "gross"

    # Zero handling
    assert magnitude_bucket(0.0, 0.01) == "subtle"
    assert magnitude_bucket(0.0, 0.20) == "moderate"
    assert magnitude_bucket(0.0, 1.5) == "gross"


def test_magnitude_bucket_tables():
    ref = pd.DataFrame({
        "city": ["Berlin", "London"],
        "val": [100.0, 200.0],
    })

    # Subtle table change (2% diff)
    wrong_subtle = pd.DataFrame({
        "city": ["Berlin", "London"],
        "val": [102.0, 200.0],
    })
    assert magnitude_bucket(ref, wrong_subtle) == "subtle"

    # Moderate table change (25% diff)
    wrong_moderate = pd.DataFrame({
        "city": ["Berlin", "London"],
        "val": [125.0, 200.0],
    })
    assert magnitude_bucket(ref, wrong_moderate) == "moderate"

    # Gross table change (80% diff)
    wrong_gross = pd.DataFrame({
        "city": ["Berlin", "London"],
        "val": [180.0, 200.0],
    })
    assert magnitude_bucket(ref, wrong_gross) == "gross"

    # Structural difference: different row counts
    wrong_rows = pd.DataFrame({
        "city": ["Berlin"],
        "val": [100.0],
    })
    assert magnitude_bucket(ref, wrong_rows) == "structural"

    # Structural difference: mismatched text labels
    wrong_labels = pd.DataFrame({
        "city": ["Berlin", "Paris"],
        "val": [100.0, 200.0],
    })
    assert magnitude_bucket(ref, wrong_labels) == "structural"


# ============================================================================
# 3. Tests for confusion and breakdown
# ============================================================================

def test_confusion_and_breakdown():
    test_cases = [
        # Wrong cases
        {"is_wrong": True, "verdict": "fail", "kind": "wrong_aggregation", "category": "cat1"},
        {"is_wrong": True, "verdict": "fail", "kind": "wrong_aggregation", "category": "cat1"},
        {"is_wrong": True, "verdict": "pass", "kind": "missing_filter", "category": "cat2"},
        {"is_wrong": True, "verdict": "unverified", "kind": "wrong_column", "category": "cat2"},
        # Correct cases
        {"is_wrong": False, "verdict": "pass", "kind": "correct", "category": "cat1"},
        {"is_wrong": False, "verdict": "pass", "kind": "correct", "category": "cat1"},
        {"is_wrong": False, "verdict": "fail", "kind": "correct", "category": "cat2"},
        {"is_wrong": False, "verdict": "unverified", "kind": "correct", "category": "cat2"},
    ]

    conf = confusion(test_cases)

    assert conf["detection"] == 2
    assert conf["missed"] == 1
    assert conf["wrong_unverified"] == 1
    assert conf["total_wrong"] == 4

    assert conf["false_alarms"] == 1
    assert conf["correct_passed"] == 2
    assert conf["correct_unverified"] == 1
    assert conf["total_correct"] == 4

    assert conf["total_cases"] == 8

    # Rates
    assert conf["detection_rate"] == 2 / 4
    assert conf["false_alarm_rate"] == 1 / 4
    assert conf["unverified_rate"] == 2 / 8
    assert conf["precision"] == 2 / (2 + 1)

    # CIs within [0, 1]
    assert 0.0 <= conf["detection_rate_ci"][0] <= conf["detection_rate_ci"][1] <= 1.0
    assert 0.0 <= conf["false_alarm_rate_ci"][0] <= conf["false_alarm_rate_ci"][1] <= 1.0
    assert 0.0 <= conf["precision_ci"][0] <= conf["precision_ci"][1] <= 1.0

    # Breakdown by category
    b_cat = breakdown(test_cases, "category")
    assert "cat1" in b_cat and "cat2" in b_cat
    assert b_cat["cat1"]["count"] == 4
    assert b_cat["cat2"]["count"] == 4
    assert b_cat["cat1"]["detection"] == 2


# ============================================================================
# 4. Tests for runner with monkeypatched reviewer_node
# ============================================================================

def test_runner_scripted_verdicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies runner flow, metrics generation, and summary artifacts with scripted reviewer."""
    run_dir = tmp_path / "test_run"

    # Scripted reviewer: fails if query mentions wrong variant or returns pass/fail based on state
    def mock_reviewer_node(state: dict[str, Any]) -> dict[str, Any]:
        res_list = state.get("analysis_results", [])
        code = res_list[0]["code"] if res_list else ""
        q = res_list[0]["question"] if res_list else ""
        # Pass correct reference code, fail wrong variant codes
        if "total number of orders" in q:
            is_ref = (code.strip() == "result = len(df)")
        elif "total sum of order_value_usd" in q:
            is_ref = (code.strip() == "result = df['order_value_usd'].sum()")
        else:
            is_ref = True

        verdict = "pass" if is_ref else "fail"
        failed_check = None if is_ref else "recomputation_mismatch"
        return {
            "review_report": [
                {
                    "id": 0,
                    "question": q,
                    "verdict": verdict,
                    "failed_check": failed_check,
                    "reason": "Scripted reason",
                    "recomputed_value": "100",
                    "recomputed_result": 100,
                }
            ],
            "review_feedback": [],
            "caught_errors": [],
            "events": [],
        }

    monkeypatch.setattr("eval.run_reviewer_eval.reviewer_node", mock_reviewer_node)

    # Run with limit=4
    out_dir = run_evaluation(limit=4, repeats=1, run_dir=str(run_dir), csv_path="data/sample.csv")
    assert Path(out_dir).exists()

    cases_file = run_dir / "cases.jsonl"
    summary_file = run_dir / "summary.json"
    summary_md = run_dir / "summary.md"

    assert cases_file.exists()
    assert summary_file.exists()
    assert summary_md.exists()

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["totals"]["total_cases"] == 4
    # All wrong cases detected, 0 false alarms
    assert summary["overall"]["detection_rate"] == 1.0
    assert summary["overall"]["false_alarm_rate"] == 0.0


def test_runner_resume_skips_finished_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies that re-running with existing cases.jsonl skips already finished cases."""
    run_dir = tmp_path / "resume_run"
    call_counts = {"count": 0}

    def mock_reviewer_node(state: dict[str, Any]) -> dict[str, Any]:
        call_counts["count"] += 1
        return {
            "review_report": [
                {
                    "id": 0,
                    "question": "q",
                    "verdict": "pass",
                    "failed_check": None,
                    "reason": "ok",
                    "recomputed_value": "1",
                    "recomputed_result": 1,
                }
            ],
            "review_feedback": [],
            "caught_errors": [],
            "events": [],
        }

    monkeypatch.setattr("eval.run_reviewer_eval.reviewer_node", mock_reviewer_node)

    # Run first 2 cases
    run_evaluation(limit=2, repeats=1, run_dir=str(run_dir), csv_path="data/sample.csv")
    assert call_counts["count"] == 2

    # Run with limit=4 (should only call reviewer for the 2 new cases)
    run_evaluation(limit=4, repeats=1, run_dir=str(run_dir), csv_path="data/sample.csv")
    assert call_counts["count"] == 4  # 2 original + 2 new


def test_runner_label_leakage_guard():
    """Verifies that _assert_no_label_leakage strictly rejects state containing forbidden keys."""
    # Good state
    clean_state = {
        "analysis_results": [{"id": 0, "question": "q", "code": "c", "result": 1, "status": "ok"}],
        "clean_df_path": "data/sample.csv",
        "charts": [],
        "retry_count": 0,
    }
    _assert_no_label_leakage(clean_state)

    # Leaked state in root
    leaked_root = dict(clean_state)
    leaked_root["is_wrong"] = True
    with pytest.raises(AssertionError, match="Information leakage detected in state: key 'is_wrong'"):
        _assert_no_label_leakage(leaked_root)

    # Leaked state in analysis_results
    leaked_item = {
        "analysis_results": [{"id": 0, "question": "q", "code": "c", "kind": "wrong_aggregation"}],
    }
    with pytest.raises(AssertionError, match="Information leakage detected in analysis_results item: key 'kind'"):
        _assert_no_label_leakage(leaked_item)


def test_runner_llm_unavailable_pauses_and_saves_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies that LLMUnavailableError exits cleanly and preserves finished cases in cases.jsonl."""
    run_dir = tmp_path / "pause_run"
    call_index = [0]

    def mock_reviewer_node(state: dict[str, Any]) -> dict[str, Any]:
        call_index[0] += 1
        if call_index[0] == 2:
            raise LLMUnavailableError("AI service unavailable timeout")
        return {
            "review_report": [{"id": 0, "verdict": "pass", "reason": "ok"}],
            "review_feedback": [],
            "caught_errors": [],
            "events": [],
        }

    monkeypatch.setattr("eval.run_reviewer_eval.reviewer_node", mock_reviewer_node)

    with pytest.raises(SystemExit) as exc_info:
        run_evaluation(limit=3, repeats=1, run_dir=str(run_dir), csv_path="data/sample.csv")

    assert exc_info.value.code == 1

    # Case 1 was recorded before pause
    cases_file = run_dir / "cases.jsonl"
    assert cases_file.exists()
    lines = cases_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["verdict"] == "pass"


def test_runner_generic_exception_becomes_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies that an unexpected exception in an item records verdict 'error' and continues."""
    run_dir = tmp_path / "error_run"

    def mock_reviewer_node(state: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("Unexpected sandbox math failure")

    monkeypatch.setattr("eval.run_reviewer_eval.reviewer_node", mock_reviewer_node)

    run_evaluation(limit=2, repeats=1, run_dir=str(run_dir), csv_path="data/sample.csv")

    summary_file = run_dir / "summary.json"
    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["totals"]["errors"] == 2
    assert summary["totals"]["total_cases"] == 2


def test_limit_produces_both_correct_and_wrong_cases():
    """Verifies that interleaving guarantees both correct and wrong cases in limited subsets."""
    gt_path = "eval/ground_truth.json"
    if not os.path.exists(gt_path):
        pytest.skip("eval/ground_truth.json not found")

    ground_truth = load_ground_truth(gt_path)
    cases = build_evaluation_cases(BENCHMARK, ground_truth, repeats=1)

    for lim in [2, 4, 6]:
        sub = cases[:lim]
        has_correct = any(not c["is_wrong"] for c in sub)
        has_wrong = any(c["is_wrong"] for c in sub)
        assert has_correct, f"Limit {lim} had no correct cases"
        assert has_wrong, f"Limit {lim} had no wrong cases"


# ============================================================================
# 5. Tests for Debug Visibility, Diagnostics CLI, and Enhanced Reporting
# ============================================================================

def test_reviewer_trace_and_debug_fields_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies that review_report preserves recomputed_code, real recomputed_result, and comparison trace."""
    from agents.reviewer import reviewer_node, results_match_with_trace
    from state import new_state

    # 1. Test results_match_with_trace branches directly
    # Branch a: scalar vs scalar
    m_s, t_s = results_match_with_trace(100.0, 100.5)
    assert m_s is True
    assert t_s["branch"] == "scalar vs scalar"
    assert "numeric values match" in t_s["reason"]

    # Branch b: scalar vs table
    df_small = pd.DataFrame({"a": [1, 2]})
    m_st, t_st = results_match_with_trace(100.0, df_small)
    assert m_st is False
    assert t_st["branch"] == "scalar vs table"
    assert "scalar vs table with 2 cells" in t_st["reason"]

    # Branch c: table same layout
    df1 = pd.DataFrame({"cat": ["A", "B"], "val": [10.0, 20.0]})
    df2 = pd.DataFrame({"cat": ["A", "B"], "val": [10.0, 25.0]})
    m_tl, t_tl = results_match_with_trace(df1, df2)
    assert m_tl is False
    assert t_tl["branch"] == "table same layout"
    assert "values differ" in t_tl["reason"]

    # Branch d: layout-independent fallback
    df_long = pd.DataFrame({"item": ["X", "Y"], "v": [1.0, 2.0]})
    df_wide = pd.DataFrame({"x": [1.0], "y": [2.0]})
    m_li, t_li = results_match_with_trace(df_long, df_wide)
    assert m_li is True
    assert t_li["branch"] == "layout-independent fallback"

    # Branch e: shape mismatch
    df_3 = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
    df_2 = pd.DataFrame({"v": [10.0, 20.0]})
    m_sm, t_sm = results_match_with_trace(df_3, df_2)
    assert m_sm is False
    assert t_sm["branch"] == "shape mismatch"
    assert "shape mismatch" in t_sm["reason"]

    # 2. Test reviewer_node saves all debug fields and caps at 20 rows
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("x,y\n1,10\n2,20\n", encoding="utf-8")
    state = new_state(str(csv_file))
    state["clean_df_path"] = str(csv_file)
    state["analysis_results"] = [
        {"id": 0, "question": "What is total y?", "code": "result = df['y'].sum()", "result": 30.0, "status": "ok"}
    ]

    # Mock call_llm to return alternative code generating a 25-row DataFrame
    big_df_code = "result = pd.DataFrame({'a': range(25)})"
    monkeypatch.setattr("agents.reviewer.call_llm", lambda prompt: big_df_code)

    res = reviewer_node(state)
    assert len(res["review_report"]) == 1
    rep = res["review_report"][0]

    # Verify debug fields
    assert "recomputed_code" in rep
    assert "result = pd.DataFrame({'a': range(25)})" in rep["recomputed_code"]
    assert "comparison_trace" in rep
    assert rep["comparison_trace"]["branch"] == "scalar vs table"

    # Verify recomputed_result is capped at 20 rows
    recomp_res = rep["recomputed_result"]
    assert isinstance(recomp_res, pd.DataFrame)
    assert len(recomp_res) == 20


def test_diagnose_cases_prints_expected_sections(capsys, tmp_path: Path):
    """Verifies that diagnose_cases prints all required diagnostic sections with a faked reviewer."""
    from eval.diagnose_cases import diagnose_cases

    def fake_reviewer_node(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "review_report": [
                {
                    "id": 0,
                    "question": state["analysis_results"][0]["question"],
                    "verdict": "fail",
                    "reason": "Recomputation mismatch",
                    "recomputed_code": "result = 999.0",
                    "recomputed_result": 999.0,
                    "comparison_trace": {
                        "branch": "scalar vs scalar",
                        "reason": "numeric values differ (100 vs 999)",
                    },
                }
            ],
            "review_feedback": [],
            "caught_errors": [],
            "events": [],
        }

    records = diagnose_cases(
        case_ids=["q18_v1", "q10_correct", "q11_correct"],
        repeats=2,
        csv_path="data/sample.csv",
        reviewer_fn=fake_reviewer_node,
    )

    assert len(records) == 3
    captured = capsys.readouterr().out

    # Check that all expected section headers are printed
    assert "Question:" in captured
    assert "Analyst Code:" in captured
    assert "Analyst Result:" in captured
    assert "Reference Answer:" in captured
    assert "Reviewer Recomputation Code:" in captured
    assert "Reviewer Recomputed Result:" in captured
    assert "Comparison Trace:" in captured
    assert "Branch: scalar vs scalar" in captured
    assert "Verdict:" in captured
    assert "Reviewer Recompute Correct vs Reference:" in captured
    assert "DIAGNOSTIC SUMMARY TABLE" in captured
    assert "q18_v1" in captured
    assert "q10_correct" in captured
    assert "q11_correct" in captured


def test_recomputation_accuracy_right_reason_by_luck_and_none_row_gone(tmp_path: Path):
    """Verifies right reason vs by luck counts and that 'none' is absent from magnitude table."""
    from eval.run_reviewer_eval import generate_summary

    # Hand-made list of 6 cases:
    # 2 correct cases: 1 recompute matched ref, 1 recompute mismatched ref
    # 4 wrong cases:
    #   - 2 detected (verdict='fail'): 1 right reason (recomp matched ref), 1 by luck (recomp wrong)
    #   - 2 missed (verdict='pass'): 1 recomp matched ref, 1 recomp wrong
    records = [
        # Correct 1: Recompute matched reference (True)
        {
            "case_id": "c1",
            "item_id": 1,
            "is_wrong": False,
            "kind": "correct",
            "category": "simple",
            "difficulty": "easy",
            "magnitude_bucket": "none",
            "verdict": "pass",
            "recomputed_result_correct": True,
            "llm_calls": 1,
            "tokens": 50,
            "seconds": 0.1,
        },
        # Correct 2: False alarm, recompute did not match reference (False)
        {
            "case_id": "c2",
            "item_id": 2,
            "is_wrong": False,
            "kind": "correct",
            "category": "simple",
            "difficulty": "easy",
            "magnitude_bucket": "none",
            "verdict": "fail",
            "recomputed_result_correct": False,
            "llm_calls": 1,
            "tokens": 50,
            "seconds": 0.1,
        },
        # Wrong 1: Detected for the right reason (recomputed_result_correct is True)
        {
            "case_id": "w1",
            "item_id": 3,
            "is_wrong": True,
            "kind": "missing_filter",
            "category": "filter",
            "difficulty": "easy",
            "magnitude_bucket": "subtle",
            "verdict": "fail",
            "recomputed_result_correct": True,
            "llm_calls": 1,
            "tokens": 50,
            "seconds": 0.1,
        },
        # Wrong 2: Detected by luck (recomputed_result_correct is False)
        {
            "case_id": "w2",
            "item_id": 4,
            "is_wrong": True,
            "kind": "wrong_column",
            "category": "filter",
            "difficulty": "easy",
            "magnitude_bucket": "moderate",
            "verdict": "fail",
            "recomputed_result_correct": False,
            "llm_calls": 1,
            "tokens": 50,
            "seconds": 0.1,
        },
        # Wrong 3: Missed, recompute correct
        {
            "case_id": "w3",
            "item_id": 5,
            "is_wrong": True,
            "kind": "wrong_aggregation",
            "category": "groupby",
            "difficulty": "medium",
            "magnitude_bucket": "gross",
            "verdict": "pass",
            "recomputed_result_correct": True,
            "llm_calls": 1,
            "tokens": 50,
            "seconds": 0.1,
        },
        # Wrong 4: Missed, recompute wrong
        {
            "case_id": "w4",
            "item_id": 6,
            "is_wrong": True,
            "kind": "wrong_aggregation",
            "category": "groupby",
            "difficulty": "medium",
            "magnitude_bucket": "gross",
            "verdict": "pass",
            "recomputed_result_correct": False,
            "llm_calls": 1,
            "tokens": 50,
            "seconds": 0.1,
        },
    ]

    summary = generate_summary(records, tmp_path)

    # 1. Verify recomputation_accuracy in summary.json
    ra = summary["recomputation_accuracy"]
    assert ra["correct_cases"]["total"] == 2
    assert ra["correct_cases"]["recomputed_correct"] == 1
    assert ra["correct_cases"]["rate"] == 0.5

    assert ra["wrong_cases"]["total"] == 4
    assert ra["wrong_cases"]["recomputed_correct"] == 2
    assert ra["wrong_cases"]["rate"] == 0.5

    det = ra["detected_wrong_cases"]
    assert det["total_detected"] == 2
    assert det["detected_right_reason"] == 1
    assert det["right_reason_rate"] == 0.5
    assert det["detected_by_luck"] == 1
    assert det["by_luck_rate"] == 0.5

    # 2. Verify 'none' row is gone from magnitude buckets
    assert "none" not in summary["breakdowns"]["by_magnitude_bucket"]

    # 3. Verify small sample caveat is added for buckets with < 10 cases
    subtle_data = summary["breakdowns"]["by_magnitude_bucket"]["subtle"]
    assert "small sample (n=1), interpret with care" in subtle_data.get("caveat", "")

    # 4. Verify summary.md does not have 'none' in magnitude table and has correct cases table
    md_text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## Correct Cases Performance" in md_text
    assert "Was the Reviewer's Own Recomputation Correct?" in md_text
    assert "Detected for the right reason" in md_text
    assert "Detected by luck" in md_text
    assert "small sample (n=1), interpret with care" in md_text
    # 'none' should not appear as a row in the magnitude table
    assert "| `none` |" not in md_text


# ============================================================================
# Step 6: Tests for --summarize-only, partial marking, per-repeat CI, flips
# ============================================================================

def test_summarize_only_partial_marking(tmp_path: Path):
    """Verifies PARTIAL RUN warning is added at the top when finished cases < expected total."""
    from eval.run_reviewer_eval import generate_summary, summarize_run

    cases_file = tmp_path / "cases.jsonl"
    records = [
        {
            "case_id": f"q0{i}_correct",
            "item_id": i,
            "label": "Correct",
            "is_wrong": False,
            "kind": "correct",
            "verdict": "pass",
            "seconds": 0.1,
            "llm_calls": 1,
            "tokens": 50,
        }
        for i in range(5)
    ]
    with open(cases_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Partial run: 5 finished out of 10 expected
    summary = summarize_run(tmp_path, limit=10)
    assert summary["is_partial"] is True
    assert summary["expected_total_cases"] == 10
    assert summary["finished_cases"] == 5

    md_text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert md_text.startswith("PARTIAL RUN: 5 of 10 cases finished")

    # Full run: 5 finished out of 5 expected
    summary_full = summarize_run(tmp_path, limit=5)
    assert summary_full["is_partial"] is False
    md_text_full = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "PARTIAL RUN" not in md_text_full


def test_summarize_only_never_modifies_cases_jsonl(tmp_path: Path):
    """Verifies that summarize-only operates strictly read-only on cases.jsonl."""
    from eval.run_reviewer_eval import summarize_run

    cases_file = tmp_path / "cases.jsonl"
    original_content = (
        json.dumps({"case_id": "q00_correct", "item_id": 0, "is_wrong": False, "verdict": "pass"})
        + "\n"
        + json.dumps({"case_id": "q00_v1", "item_id": 0, "is_wrong": True, "verdict": "fail"})
        + "\n"
    )
    cases_file.write_text(original_content, encoding="utf-8")

    # Execute summarize_run
    summarize_run(tmp_path, limit=2)

    # Content must remain exactly identical byte-for-byte
    after_content = cases_file.read_text(encoding="utf-8")
    assert after_content == original_content


def test_per_repeat_counts_and_ci_on_unique_cases(tmp_path: Path):
    """
    Verifies that:
    - Per-repeat counts and Wilson CIs are computed on the cases in that repeat.
    - Headline detection and false alarm rates are the MEAN of per-repeat rates.
    - Headline Wilson CIs are computed on the count of UNIQUE cases, NOT pooled repeats.
    """
    from eval.metrics import wilson_ci
    from eval.run_reviewer_eval import generate_summary

    records = []
    # 10 unique wrong cases (items 1..10) and 5 unique correct cases (items 11..15)
    # Repeat 1 (r1): 9/10 wrong detected, 0/5 false alarms
    for i in range(1, 11):
        v = "fail" if i <= 9 else "pass"
        records.append({
            "case_id": f"q{i:02d}_v1_r1",
            "item_id": i,
            "repeat_index": 0,
            "is_wrong": True,
            "kind": "wrong_aggregation",
            "verdict": v,
        })
    for i in range(11, 16):
        records.append({
            "case_id": f"q{i:02d}_correct_r1",
            "item_id": i,
            "repeat_index": 0,
            "is_wrong": False,
            "kind": "correct",
            "verdict": "pass",
        })

    # Repeat 2 (r2): 7/10 wrong detected, 1/5 false alarms
    for i in range(1, 11):
        v = "fail" if i <= 7 else "pass"
        records.append({
            "case_id": f"q{i:02d}_v1_r2",
            "item_id": i,
            "repeat_index": 1,
            "is_wrong": True,
            "kind": "wrong_aggregation",
            "verdict": v,
        })
    for i in range(11, 16):
        v = "fail" if i == 11 else "pass"
        records.append({
            "case_id": f"q{i:02d}_correct_r2",
            "item_id": i,
            "repeat_index": 1,
            "is_wrong": False,
            "kind": "correct",
            "verdict": v,
        })

    summary = generate_summary(records, tmp_path, repeats=2, expected_total=30)

    # 1. Per-repeat performance table
    pr = summary["per_repeat"]
    assert len(pr) == 2

    # r1: 9/10 detected, 0/5 false alarms
    assert pr[0]["repeat"] == "r1"
    assert pr[0]["detection_counts"] == "9/10"
    assert pr[0]["detection_rate"] == 0.90
    assert pr[0]["detection_rate_ci"] == wilson_ci(9, 10)
    assert pr[0]["false_alarm_counts"] == "0/5"
    assert pr[0]["false_alarm_rate"] == 0.0
    assert pr[0]["false_alarm_rate_ci"] == wilson_ci(0, 5)

    # r2: 7/10 detected, 1/5 false alarms
    assert pr[1]["repeat"] == "r2"
    assert pr[1]["detection_counts"] == "7/10"
    assert pr[1]["detection_rate"] == 0.70
    assert pr[1]["detection_rate_ci"] == wilson_ci(7, 10)
    assert pr[1]["false_alarm_counts"] == "1/5"
    assert pr[1]["false_alarm_rate"] == 0.20
    assert pr[1]["false_alarm_rate_ci"] == wilson_ci(1, 5)

    # 2. Overall headline numbers: MEAN of per-repeat rates
    ov = summary["overall"]
    assert ov["detection_rate"] == pytest.approx(0.80)  # (0.90 + 0.70) / 2
    assert ov["false_alarm_rate"] == pytest.approx(0.10)  # (0.0 + 0.20) / 2

    # 3. Overall headline CI: computed on n_unique_cases (10 wrong cases), NOT pooled 20
    expected_ci_unique = wilson_ci(round(0.80 * 10), 10)  # wilson_ci(8, 10)
    pooled_ci = wilson_ci(16, 20)  # pooled calculation from old code
    assert ov["detection_rate_ci"] == list(expected_ci_unique)
    assert ov["detection_rate_ci"] != list(pooled_ci)

    # Unique CI interval should be strictly wider than pooled CI interval
    unique_width = expected_ci_unique[1] - expected_ci_unique[0]
    pooled_width = pooled_ci[1] - pooled_ci[0]
    assert unique_width > pooled_width


def test_verdict_flips_table(tmp_path: Path):
    """
    Verifies verdict flips table lists cases whose verdicts differed between repeats,
    and counts cases with identical verdicts.
    """
    from eval.run_reviewer_eval import generate_summary

    records = [
        # Case A: flips (pass in r1, fail in r2)
        {"case_id": "q01_v1_r1", "repeat_index": 0, "label": "Case A", "kind": "wrong_agg", "is_wrong": True, "verdict": "pass"},
        {"case_id": "q01_v1_r2", "repeat_index": 1, "label": "Case A", "kind": "wrong_agg", "is_wrong": True, "verdict": "fail"},

        # Case B: identical verdicts across repeats (fail, fail)
        {"case_id": "q02_v1_r1", "repeat_index": 0, "label": "Case B", "kind": "wrong_filter", "is_wrong": True, "verdict": "fail"},
        {"case_id": "q02_v1_r2", "repeat_index": 1, "label": "Case B", "kind": "wrong_filter", "is_wrong": True, "verdict": "fail"},

        # Case C: identical verdicts across repeats (pass, pass)
        {"case_id": "q03_correct_r1", "repeat_index": 0, "label": "Case C", "kind": "correct", "is_wrong": False, "verdict": "pass"},
        {"case_id": "q03_correct_r2", "repeat_index": 1, "label": "Case C", "kind": "correct", "is_wrong": False, "verdict": "pass"},
    ]

    summary = generate_summary(records, tmp_path, repeats=2, expected_total=6)
    flips_data = summary["verdict_flips"]

    assert flips_data["multi_repeat_cases_count"] == 3
    assert flips_data["identical_verdicts_count"] == 2
    assert flips_data["flipped_verdicts_count"] == 1
    assert len(flips_data["flips"]) == 1

    flip_item = flips_data["flips"][0]
    assert flip_item["case_id"] == "q01_v1"
    assert flip_item["verdicts"] == {"r1": "pass", "r2": "fail"}

    # Verify markdown representation
    md_text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## Verdict Flips Across Repeats" in md_text
    assert "Cases evaluated in 2 or more repeats**: 3" in md_text
    assert "Cases with identical verdicts across all repeats**: 2" in md_text
    assert "Cases with differing verdicts (flips)**: 1" in md_text
    assert "| `q01_v1` | Case A | `wrong_agg` | pass | fail |" in md_text


def test_summarize_only_cli_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies python -m eval.run_reviewer_eval --summarize-only --run-dir <dir> CLI works."""
    import sys
    from eval.run_reviewer_eval import main

    cases_file = tmp_path / "cases.jsonl"
    records = [
        {"case_id": "q00_correct", "item_id": 0, "is_wrong": False, "kind": "correct", "verdict": "pass"},
        {"case_id": "q00_v1", "item_id": 0, "is_wrong": True, "kind": "wrong_agg", "verdict": "fail"},
    ]
    with open(cases_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    monkeypatch.setattr(
        sys,
        "argv",
        ["eval.run_reviewer_eval", "--summarize-only", "--run-dir", str(tmp_path), "--limit", "2"],
    )

    main()

    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "summary.json").exists()

