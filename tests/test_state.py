"""
test_state.py
Unit tests for state.py ensuring AgentState structure and new_state factory.
"""

from state import new_state, AgentState


def test_new_state_defaults():
    raw_path = "data/sample.csv"
    state: AgentState = new_state(raw_path)

    assert state["raw_df_path"] == raw_path
    assert state["clean_df_path"] == ""
    assert state["profile"] == {}
    assert state["cleaning_log"] == []
    assert state["questions"] == []
    assert state["analysis_results"] == []
    assert state["charts"] == []
    assert state["review_feedback"] == []
    assert state["review_report"] == []
    assert state["caught_errors"] == []
    assert state["retry_count"] == 0
    assert state["final_report"] == ""
    assert state["events"] == []
