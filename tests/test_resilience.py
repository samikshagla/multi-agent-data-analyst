"""
test_resilience.py
Unit tests for AI service resilience, exception propagation, offline handling,
workflow guards, and UI state error reporting.
"""

from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import pytest

from llm import (
    call_llm,
    check_llm_available,
    LLMUnavailableError,
    LLMConfigError,
)
from agents.cleaner import cleaner_node
from agents.analyst import analyst_node
from agents.visualizer import visualizer_node
from agents.reviewer import reviewer_node
from graph import build_graph, report_node
from state import new_state
from components.ui_state import (
    initial_statuses,
    apply_node_update,
    format_local_timestamp,
    get_final_status_message,
)


# ============================================================================
# 1. LLM Error Handling & Retry Tests
# ============================================================================

def test_call_llm_connection_error_retries_and_raises_unavailable(monkeypatch: pytest.MonkeyPatch):
    """When client raises connection error, retries up to 4 attempts and raises LLMUnavailableError."""
    call_count = 0

    def fake_call(prompt: str):
        nonlocal call_count
        call_count += 1
        raise ConnectionError("Network unreachable")

    # Patch provider to return gemini and fake the gemini call
    monkeypatch.setattr("llm._get_provider", lambda: "gemini")
    monkeypatch.setattr("llm._call_gemini", fake_call)
    monkeypatch.setattr("time.sleep", lambda s: None)  # instant retry

    with pytest.raises(LLMUnavailableError) as exc_info:
        call_llm("test prompt")

    assert call_count == 4
    assert "AI service unreachable after 4 attempts" in str(exc_info.value)


def test_call_llm_auth_or_404_raises_config_error_no_retries(monkeypatch: pytest.MonkeyPatch):
    """A 401/404 or auth error raises LLMConfigError immediately without retrying."""
    call_count = 0

    def fake_call(prompt: str):
        nonlocal call_count
        call_count += 1
        raise ValueError("401 Unauthorized: Invalid API key")

    monkeypatch.setattr("llm._get_provider", lambda: "gemini")
    monkeypatch.setattr("llm._call_gemini", fake_call)
    monkeypatch.setattr("time.sleep", lambda s: None)

    with pytest.raises(LLMConfigError) as exc_info:
        call_llm("test prompt")

    assert call_count == 1  # No retries!
    assert "LLM configuration error" in str(exc_info.value)


def test_check_llm_available_returns_false_on_error(monkeypatch: pytest.MonkeyPatch):
    """check_llm_available returns (False, msg) on connection or configuration errors."""
    monkeypatch.setattr("llm._get_provider", lambda: "gemini")

    # 1. Connection error
    monkeypatch.setattr(
        "llm._call_gemini",
        lambda p: (_ for _ in ()).throw(ConnectionError("No internet access")),
    )
    ok, msg = check_llm_available()
    assert ok is False
    assert "Can't reach the AI service" in msg

    # 2. Config error
    monkeypatch.setattr(
        "llm._call_gemini",
        lambda p: (_ for _ in ()).throw(ValueError("403 Forbidden: bad api_key")),
    )
    ok_cfg, msg_cfg = check_llm_available()
    assert ok_cfg is False
    assert "configuration problem" in msg_cfg


# ============================================================================
# 2. Agent Node Exception Propagation vs Swallowing
# ============================================================================

def setup_clean_test_state(tmp_path: Path) -> dict:
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,10\n2,20\n3,30\n", encoding="utf-8")
    state = new_state(str(csv_file))
    state["clean_df_path"] = str(csv_file)
    return state


def test_cleaner_node_reraises_llm_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """cleaner_node re-raises LLMUnavailableError rather than swallowing it."""
    state = setup_clean_test_state(tmp_path)

    def fake_llm(p):
        raise LLMUnavailableError("Service down")

    monkeypatch.setattr("agents.cleaner.call_llm", fake_llm)

    with pytest.raises(LLMUnavailableError):
        cleaner_node(state)


def test_analyst_node_reraises_llm_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """analyst_node re-raises LLMUnavailableError rather than swallowing it."""
    state = setup_clean_test_state(tmp_path)

    def fake_llm(p):
        raise LLMUnavailableError("Service down")

    monkeypatch.setattr("agents.analyst.call_llm", fake_llm)

    with pytest.raises(LLMUnavailableError):
        analyst_node(state)


def test_visualizer_node_reraises_llm_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """visualizer_node re-raises LLMUnavailableError rather than swallowing it."""
    state = setup_clean_test_state(tmp_path)
    state["analysis_results"] = [
        {"id": 0, "question": "Total?", "result": pd.DataFrame({"a": [1, 2]}), "status": "ok"}
    ]

    def fake_llm(p):
        raise LLMUnavailableError("Service down")

    monkeypatch.setattr("agents.visualizer.call_llm", fake_llm)

    with pytest.raises(LLMUnavailableError):
        visualizer_node(state)


def test_reviewer_node_reraises_llm_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """reviewer_node re-raises LLMUnavailableError rather than swallowing it."""
    state = setup_clean_test_state(tmp_path)
    state["analysis_results"] = [
        {"id": 0, "question": "Total?", "code": "result = 1", "result": 1, "status": "ok"}
    ]

    def fake_llm(p):
        raise LLMUnavailableError("Service down")

    monkeypatch.setattr("agents.reviewer.call_llm", fake_llm)

    with pytest.raises(LLMUnavailableError):
        reviewer_node(state)


# ============================================================================
# 3. Graph Routing Guard: Zero OK Items Skips to Report
# ============================================================================

def test_graph_skips_visualizer_and_reviewer_when_zero_ok_items(tmp_path: Path):
    """When analyst produces zero 'ok' results, graph skips visualizer/reviewer and report notes failure."""
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("col1,col2\n1,2\n3,4\n", encoding="utf-8")

    invocations = {"cleaner": 0, "analyst": 0, "visualizer": 0, "reviewer": 0, "report": 0}

    def fake_cleaner(s):
        invocations["cleaner"] += 1
        return {"clean_df_path": s.get("raw_df_path", "")}

    def fake_analyst(s):
        invocations["analyst"] += 1
        # Zero items succeeded!
        return {
            "questions": ["Q1"],
            "analysis_results": [
                {"id": 0, "question": "Q1", "status": "failed", "error": "Execution error"}
            ],
            "events": [{"agent": "analyst", "type": "error", "message": "Failed Q1 execution"}],
        }

    def fake_visualizer(s):
        invocations["visualizer"] += 1
        return {"charts": []}

    def fake_reviewer(s):
        invocations["reviewer"] += 1
        return {"review_feedback": []}

    fake_nodes = {
        "cleaner": fake_cleaner,
        "analyst": fake_analyst,
        "visualizer": fake_visualizer,
        "reviewer": fake_reviewer,
    }

    app = build_graph(nodes=fake_nodes, max_retries=1)
    initial_state = new_state(str(csv_file))
    final_state = app.invoke(initial_state)

    # Invocations verification: visualizer and reviewer were SKIPPED
    assert invocations["cleaner"] == 1
    assert invocations["analyst"] == 1
    assert invocations["visualizer"] == 0
    assert invocations["reviewer"] == 0

    # Report verification
    report_text = final_state.get("final_report", "")
    assert "No analysis could be completed" in report_text
    assert "Failed Q1 execution" in report_text
    assert "Executive Key Insights" not in report_text
    assert "**Verified**" not in report_text


# ============================================================================
# 4. UI State & Timestamp Formatting Tests
# ============================================================================

def test_ui_state_card_notes_completed_vs_completed_with_warnings():
    """apply_node_update sets 'Completed' on clean run and 'Completed with warnings' on issues."""
    init = initial_statuses()

    # 1. Clean run
    clean_update = {
        "clean_df_path": "path.csv",
        "events": [{"agent": "cleaner", "type": "info", "message": "All good"}],
    }
    s1 = apply_node_update(init, "cleaner", clean_update)
    assert s1["cleaner"]["status"] == "done"
    assert s1["cleaner"]["note"] == "Completed"

    # 2. Run with warning
    warn_update = {
        "clean_df_path": "path.csv",
        "events": [{"agent": "cleaner", "type": "warning", "message": "Skipped step"}],
    }
    s2 = apply_node_update(init, "cleaner", warn_update)
    assert s2["cleaner"]["status"] == "flagged"
    assert s2["cleaner"]["note"] == "Completed with warnings"


def test_ui_state_final_message_logic():
    """get_final_status_message decides success vs issues from the final state."""
    # 1. Clean success
    state_ok = {
        "analysis_results": [{"id": 0, "status": "ok"}],
        "review_report": [{"id": 0, "verdict": "pass"}],
    }
    statuses_ok = {
        "cleaner": {"status": "done", "note": "Completed"},
        "analyst": {"status": "done", "note": "Completed"},
        "visualizer": {"status": "done", "note": "Completed"},
        "reviewer": {"status": "done", "note": "Completed"},
    }
    clean_ok, msg_ok = get_final_status_message(state_ok, statuses_ok)
    assert clean_ok is True
    assert msg_ok == "Analysis complete"

    # 2. No items succeeded
    state_fail = {"analysis_results": [{"id": 0, "status": "failed"}]}
    clean_fail, msg_fail = get_final_status_message(state_fail, statuses_ok)
    assert clean_fail is False
    assert "No analysis could be completed" in msg_fail

    # 3. Agent had an error
    statuses_err = dict(statuses_ok)
    statuses_err["analyst"] = {"status": "error", "note": "Failed"}
    clean_e, msg_e = get_final_status_message(state_ok, statuses_err)
    assert clean_e is False
    assert "Analyst failed" in msg_e

    # 4. Interim warning events do not prevent clean success when final state is clean
    statuses_warn = dict(statuses_ok)
    statuses_warn["reviewer"] = {"status": "flagged", "note": "Completed with warnings"}
    clean_w, msg_w = get_final_status_message(state_ok, statuses_warn)
    assert clean_w is True
    assert msg_w == "Analysis complete"


def test_format_local_timestamp_converts_utc_to_local():
    """format_local_timestamp accurately converts UTC ISO timestamps to local time."""
    utc_iso = "2026-11-14T06:30:00Z"
    expected = datetime.fromisoformat("2026-11-14T06:30:00+00:00").astimezone().strftime("%H:%M:%S")

    formatted = format_local_timestamp(utc_iso)
    assert formatted == expected
