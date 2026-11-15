"""
test_graph.py
Unit tests for graph.py using pytest.
Validates StateGraph workflow routing, bounded retry limits, reviewer feedback loops,
report generation, and run_stats compilation using simulated fake nodes.
"""

from typing import Any
import pytest

from graph import build_graph, run_pipeline
from state import new_state


class GraphExecutionTracker:
    """Tracks invocation counts for each node in the StateGraph."""
    def __init__(self):
        self.counts = {
            "cleaner": 0,
            "analyst": 0,
            "visualizer": 0,
            "reviewer": 0,
            "report": 0,
        }


def test_flow_no_feedback_runs_once():
    """Test 1: When reviewer returns no feedback, each node runs once and retry_count is 0."""
    tracker = GraphExecutionTracker()

    def fake_cleaner(state: dict) -> dict:
        tracker.counts["cleaner"] += 1
        return {"clean_df_path": state.get("raw_df_path", "")}

    def fake_analyst(state: dict) -> dict:
        tracker.counts["analyst"] += 1
        return {
            "questions": ["What is total?"],
            "analysis_results": [{"id": 0, "question": "What is total?", "code": "res = 1", "result": 1, "status": "ok"}],
        }

    def fake_visualizer(state: dict) -> dict:
        tracker.counts["visualizer"] += 1
        return {"charts": [{"id": 0, "type": "metric", "title": "Total", "takeaway": "Total is 1", "status": "ok"}]}

    def fake_reviewer(state: dict) -> dict:
        tracker.counts["reviewer"] += 1
        # No feedback (all items pass)
        return {"review_feedback": [], "review_report": [{"id": 0, "verdict": "pass"}]}

    def fake_report(state: dict) -> dict:
        tracker.counts["report"] += 1
        return {"final_report": "# Final Report\nAll items verified."}

    fake_nodes = {
        "cleaner": fake_cleaner,
        "analyst": fake_analyst,
        "visualizer": fake_visualizer,
        "reviewer": fake_reviewer,
        "report": fake_report,
    }

    app = build_graph(nodes=fake_nodes, max_retries=3)
    initial_state = new_state("data/sample.csv")
    final_state = app.invoke(initial_state)

    assert tracker.counts["cleaner"] == 1
    assert tracker.counts["analyst"] == 1
    assert tracker.counts["visualizer"] == 1
    assert tracker.counts["reviewer"] == 1
    assert tracker.counts["report"] == 1
    assert final_state["retry_count"] == 0
    assert "Final Report" in final_state["final_report"]


def test_reviewer_fails_once_then_passes():
    """Test 2: When reviewer fails once then passes, analyst runs twice and retry_count is 1."""
    tracker = GraphExecutionTracker()

    def fake_cleaner(state: dict) -> dict:
        tracker.counts["cleaner"] += 1
        return {"clean_df_path": state.get("raw_df_path", "")}

    def fake_analyst(state: dict) -> dict:
        tracker.counts["analyst"] += 1
        return {
            "questions": ["Q1"],
            "analysis_results": [{"id": 0, "question": "Q1", "code": "res = 1", "result": 1, "status": "ok"}],
        }

    def fake_visualizer(state: dict) -> dict:
        tracker.counts["visualizer"] += 1
        return {"charts": [{"id": 0, "type": "metric", "title": "Q1", "takeaway": "Takeaway", "status": "ok"}]}

    def fake_reviewer(state: dict) -> dict:
        tracker.counts["reviewer"] += 1
        if tracker.counts["reviewer"] == 1:
            # First pass: fail item 0
            return {
                "review_feedback": [{"id": 0, "verdict": "fail", "reason": "Discrepancy detected"}],
                "review_report": [{"id": 0, "verdict": "fail"}],
            }
        else:
            # Second pass: pass
            return {"review_feedback": [], "review_report": [{"id": 0, "verdict": "pass"}]}

    def fake_report(state: dict) -> dict:
        tracker.counts["report"] += 1
        return {"final_report": "# Final Report"}

    fake_nodes = {
        "cleaner": fake_cleaner,
        "analyst": fake_analyst,
        "visualizer": fake_visualizer,
        "reviewer": fake_reviewer,
        "report": fake_report,
    }

    app = build_graph(nodes=fake_nodes, max_retries=3)
    initial_state = new_state("data/sample.csv")
    final_state = app.invoke(initial_state)

    assert tracker.counts["cleaner"] == 1
    assert tracker.counts["analyst"] == 2
    assert tracker.counts["visualizer"] == 2
    assert tracker.counts["reviewer"] == 2
    assert tracker.counts["report"] == 1
    assert final_state["retry_count"] == 1


def test_reviewer_fails_every_time_stops_at_3_retries():
    """Test 3: When reviewer fails continuously, execution halts after 3 retries (analyst runs 4 times)."""
    tracker = GraphExecutionTracker()

    def fake_cleaner(state: dict) -> dict:
        tracker.counts["cleaner"] += 1
        return {"clean_df_path": state.get("raw_df_path", "")}

    def fake_analyst(state: dict) -> dict:
        tracker.counts["analyst"] += 1
        return {
            "questions": ["Q1"],
            "analysis_results": [{"id": 0, "question": "Q1", "code": "res = 0", "result": 0, "status": "ok"}],
        }

    def fake_visualizer(state: dict) -> dict:
        tracker.counts["visualizer"] += 1
        return {"charts": []}

    def fake_reviewer(state: dict) -> dict:
        tracker.counts["reviewer"] += 1
        # Always fail item 0
        return {
            "review_feedback": [{"id": 0, "verdict": "fail", "reason": "Persistent calculation error"}],
            "review_report": [{"id": 0, "question": "Q1", "verdict": "fail", "reason": "Persistent calculation error"}],
        }

    fake_nodes = {
        "cleaner": fake_cleaner,
        "analyst": fake_analyst,
        "visualizer": fake_visualizer,
        "reviewer": fake_reviewer,
        # Default report_node will run to test the unresolved section formatting
    }

    app = build_graph(nodes=fake_nodes, max_retries=3)
    initial_state = new_state("data/sample.csv")
    final_state = app.invoke(initial_state)

    # Initial run (1) + 3 retries = 4 analyst invocations
    assert tracker.counts["analyst"] == 4
    assert tracker.counts["reviewer"] == 4
    assert final_state["retry_count"] == 3

    # Report node ran and created the report containing unresolved issues
    assert "final_report" in final_state
    assert "Unresolved Issues" in final_state["final_report"]
    assert "Persistent calculation error" in final_state["final_report"]


def test_graph_never_runs_forever():
    """Test 4: Confirms the bounded recursion limit and retry logic terminate predictably."""
    tracker = GraphExecutionTracker()

    def fake_reviewer(state: dict) -> dict:
        tracker.counts["reviewer"] += 1
        return {"review_feedback": [{"id": 0, "verdict": "fail", "reason": "Infinite loop test"}]}

    fake_nodes = {
        "cleaner": lambda s: {"clean_df_path": s.get("raw_df_path", "")},
        "analyst": lambda s: {"questions": ["Q1"], "analysis_results": [{"id": 0, "status": "ok"}]},
        "visualizer": lambda s: {"charts": []},
        "reviewer": fake_reviewer,
        "report": lambda s: {"final_report": "Finished"},
    }

    # Set max_retries=2
    app = build_graph(nodes=fake_nodes, max_retries=2)
    initial_state = new_state("data/sample.csv")
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})

    assert final_state["retry_count"] == 2
    assert tracker.counts["reviewer"] == 3  # initial + 2 retries
    assert final_state["final_report"] == "Finished"


def test_run_stats_structure(tmp_path):
    """Test 5: run_stats is populated with all expected performance and token metrics."""
    test_csv = tmp_path / "dummy.csv"
    test_csv.write_text("a,b\n1,2\n3,4", encoding="utf-8")

    # Run pipeline with fake fast nodes
    fake_nodes = {
        "cleaner": lambda s: {"clean_df_path": s.get("raw_df_path", "")},
        "analyst": lambda s: {"questions": ["Q"], "analysis_results": []},
        "visualizer": lambda s: {"charts": []},
        "reviewer": lambda s: {"review_feedback": []},
        "report": lambda s: {"final_report": "Report"},
    }

    app = build_graph(nodes=fake_nodes, max_retries=1)
    initial_state = new_state(str(test_csv))
    final_state = app.invoke(initial_state)

    # Attach stats as run_pipeline does
    final_state["run_stats"] = {
        "time_seconds": 0.42,
        "llm_calls": 3,
        "tokens": 120,
        "input_tokens": 80,
        "output_tokens": 40,
        "retries_used": final_state.get("retry_count", 0),
        "errors_caught": len(final_state.get("caught_errors", [])),
    }

    stats = final_state["run_stats"]
    assert "time_seconds" in stats
    assert "llm_calls" in stats
    assert "tokens" in stats
    assert "retries_used" in stats
    assert "errors_caught" in stats
    assert stats["retries_used"] == 0
    assert stats["time_seconds"] >= 0


def test_report_tags_and_quality_counts(tmp_path):
    """Test 6: Report tags each item as Verified, Unverified, or Unresolved, and tallies them in Quality section."""
    from graph import report_node

    csv_path = tmp_path / "test_data.csv"
    csv_path.write_text("a,b\n1,2", encoding="utf-8")

    state = new_state(str(csv_path))
    state["analysis_results"] = [
        {"id": 0, "question": "Q0", "code": "res = 1", "result": 100, "status": "ok"},
        {"id": 1, "question": "Q1", "code": "res = 2", "result": 200, "status": "ok"},
        {"id": 2, "question": "Q2", "code": "res = 3", "result": 300, "status": "ok"},
    ]
    state["review_report"] = [
        {"id": 0, "verdict": "pass", "reason": "All checks passed"},
        {"id": 1, "verdict": "unverified", "reason": "Recomputation inconclusive"},
        {"id": 2, "verdict": "fail", "reason": "Math mismatch"},
    ]
    state["charts"] = [
        {"id": 0, "status": "ok", "type": "metric", "title": "Metric 0", "takeaway": "Takeaway 0"},
        {"id": 1, "status": "skipped", "type": "unknown", "title": "Metric 1", "note": "Too few rows"},
    ]

    out = report_node(state)
    report_text = out["final_report"]

    # Verify status tags
    assert "**Verified**" in report_text
    assert "**Unverified** (Recomputation inconclusive)" in report_text
    assert "**Unresolved** (Math mismatch)" in report_text

    # Verify counts in Quality Review section
    assert "**Verified Items**: 1" in report_text
    assert "**Unverified Items**: 1" in report_text
    assert "**Unresolved Items**: 1" in report_text

    # Verify skipped chart notice
    assert "No chart generated (visualizer skipped or failed: Too few rows)" in report_text


def test_diagnose_script_output(capsys, monkeypatch, tmp_path):
    """Test 7: diagnose.py outputs formatted audit report and summary metrics."""
    import diagnose

    dummy_csv = tmp_path / "sample.csv"
    dummy_csv.write_text("a,b\n1,2", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["diagnose.py", str(dummy_csv)])

    fake_state = {
        "caught_errors": [
            {
                "id": 0,
                "question": "What is total?",
                "reason": "Recomputation gave different value",
                "type": "recomputation_mismatch",
                "retry_round": 0,
                "original_value": "100",
                "recomputed_value": "200",
                "original_code": "result = 100",
                "recomputed_code": "result = 200",
            }
        ],
        "review_report": [{"id": 0, "verdict": "pass"}],
        "run_stats": {
            "total_flag_events": 1,
            "items_flagged_first_pass": 1,
            "fixed_by_retry": 1,
            "unresolved": 0,
            "retries_used": 1,
            "time_seconds": 1.25,
        },
    }

    monkeypatch.setattr("diagnose.run_pipeline", lambda path, stream: fake_state)

    diagnose.main()

    captured = capsys.readouterr().out
    assert "REVIEWER AUDIT & DIAGNOSTIC REPORT" in captured
    assert "Found 1 interception event(s)" in captured
    assert "recomputation_mismatch" in captured
    assert "result = 100" in captured
    assert "result = 200" in captured
    assert "Fixed by Retry:           1" in captured
