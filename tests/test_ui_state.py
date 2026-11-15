"""
test_ui_state.py
Unit tests for components/ui_state.py using pytest.
Validates pure state-management functions: initial statuses, apply_node_update state machine
transitions (forward flow, review feedback retry resets, clean pass report flag),
new_events deduplication, event_to_html XSS escaping & color classification,
and summarize_run resilience against missing keys.
"""

from components.ui_state import (
    initial_statuses,
    apply_node_update,
    new_events,
    event_to_html,
    summarize_run,
    get_final_status_message,
    get_final_agent_statuses,
    format_error_details,
    AGENTS,
)


def test_initial_statuses():
    """Initial statuses marks all agents as waiting with empty notes and report_running=False."""
    statuses = initial_statuses()

    assert statuses["report_running"] is False
    for agent in AGENTS:
        k = agent["key"]
        assert k in statuses
        assert statuses[k]["status"] == "waiting"
        assert statuses[k]["note"] == ""


def test_cleaner_finishing_marks_analyst_running():
    """When cleaner finishes, cleaner is done and analyst is running."""
    init = initial_statuses()
    update = {"clean_df_path": "data/cleaned/sample_cleaned.csv"}

    res = apply_node_update(init, "cleaner", update)
    assert res["cleaner"]["status"] == "done"
    assert res["analyst"]["status"] == "running"
    assert res["visualizer"]["status"] == "waiting"
    assert res["reviewer"]["status"] == "waiting"


def test_analyst_and_visualizer_transitions():
    """Analyst done triggers visualizer running; visualizer done triggers reviewer running."""
    s1 = {
        "cleaner": {"status": "done", "note": ""},
        "analyst": {"status": "running", "note": ""},
        "visualizer": {"status": "waiting", "note": ""},
        "reviewer": {"status": "waiting", "note": ""},
    }

    # Analyst finishes
    s2 = apply_node_update(s1, "analyst", {"analysis_results": []})
    assert s2["analyst"]["status"] == "done"
    assert s2["visualizer"]["status"] == "running"

    # Visualizer finishes
    s3 = apply_node_update(s2, "visualizer", {"charts": []})
    assert s3["visualizer"]["status"] == "done"
    assert s3["reviewer"]["status"] == "running"


def test_reviewer_finishing_with_feedback():
    """Reviewer with feedback marks reviewer flagged, analyst running (retry note), and resets visualizer."""
    s3 = {
        "cleaner": {"status": "done", "note": ""},
        "analyst": {"status": "done", "note": ""},
        "visualizer": {"status": "done", "note": ""},
        "reviewer": {"status": "running", "note": ""},
    }

    # Reviewer flags 2 items on retry 0
    update = {
        "review_feedback": [
            {"id": 1, "reason": "Math mismatch"},
            {"id": 2, "reason": "Empty table"},
        ]
    }
    s4 = apply_node_update(s3, "reviewer", update, retry_count=0)

    assert s4["reviewer"]["status"] == "flagged"
    assert "Flagged 2 items" in s4["reviewer"]["note"]
    assert s4["analyst"]["status"] == "running"
    assert s4["analyst"]["note"] == "Retry 1/3"
    assert s4["visualizer"]["status"] == "waiting"
    assert s4.get("report_running") is False


def test_reviewer_finishing_without_feedback_sets_report_running():
    """Reviewer with no feedback marks reviewer done and sets report_running=True."""
    s3 = {
        "cleaner": {"status": "done", "note": ""},
        "analyst": {"status": "done", "note": ""},
        "visualizer": {"status": "done", "note": ""},
        "reviewer": {"status": "running", "note": ""},
    }

    update = {"review_feedback": []}
    s4 = apply_node_update(s3, "reviewer", update)

    assert s4["reviewer"]["status"] == "done"
    assert s4["report_running"] is True


def test_new_events_never_duplicates_when_node_returns_whole_list():
    """new_events slices only unseen events even when nodes return cumulative events lists."""
    ev1 = {"agent": "cleaner", "type": "info", "message": "Cleaned 10 rows"}
    ev2 = {"agent": "analyst", "type": "info", "message": "Computed query"}
    ev3 = {"agent": "reviewer", "type": "warning", "message": "Flagged item"}

    # Step 1: cleaner produces ev1
    seen = 0
    all_events_1 = [ev1]
    fresh_1 = new_events(all_events_1, seen)
    assert fresh_1 == [ev1]
    seen = len(all_events_1)

    # Step 2: analyst returns [ev1, ev2] (cumulative)
    all_events_2 = [ev1, ev2]
    fresh_2 = new_events(all_events_2, seen)
    assert fresh_2 == [ev2]
    seen = len(all_events_2)

    # Step 3: visualizer returns [ev1, ev2] (no new events)
    fresh_3 = new_events(all_events_2, seen)
    assert fresh_3 == []

    # Step 4: reviewer returns [ev1, ev2, ev3]
    all_events_4 = [ev1, ev2, ev3]
    fresh_4 = new_events(all_events_4, seen)
    assert fresh_4 == [ev3]


def test_event_to_html_escapes_script_and_sets_correct_color():
    """event_to_html escapes dangerous HTML and assigns color classes appropriately."""
    # 1. HTML injection escaping
    ev_xss = {
        "timestamp": "2026-11-14T12:00:00Z",
        "agent": "analyst",
        "type": "info",
        "message": "<script>alert('exploit')</script>",
    }
    html_xss = event_to_html(ev_xss)
    assert "<script>" not in html_xss
    assert "&lt;script&gt;alert(&#x27;exploit&#x27;)&lt;/script&gt;" in html_xss
    assert "event-info" in html_xss

    # 2. Warning styling
    ev_warn = {"timestamp": "12:00:01", "agent": "reviewer", "type": "warning", "message": "Caution note"}
    html_warn = event_to_html(ev_warn)
    assert "event-warning" in html_warn

    # 3. Error styling
    ev_err = {"timestamp": "12:00:02", "agent": "system", "type": "error", "message": "Failed step"}
    html_err = event_to_html(ev_err)
    assert "event-error" in html_err

    # 4. Retry styling takes precedence (blue)
    ev_retry = {
        "timestamp": "12:00:03",
        "agent": "reviewer",
        "type": "warning",
        "message": "Reviewer flagged 1 item(s), sending back for Retry 1/3",
    }
    html_retry = event_to_html(ev_retry)
    assert "event-retry" in html_retry


def test_summarize_run_works_with_missing_keys():
    """summarize_run safely handles missing run_stats keys and None inputs."""
    # 1. Missing run_stats or None
    summary_empty = summarize_run(None)
    assert summary_empty["duration_sec"] == 0.0
    assert summary_empty["llm_calls"] == 0
    assert summary_empty["tokens"] == 0
    assert summary_empty["retries_used"] == 0
    assert summary_empty["items_flagged_first_pass"] == 0
    assert summary_empty["fixed_by_retry"] == 0
    assert summary_empty["unresolved"] == 0
    assert summary_empty["unverified"] == 0
    assert summary_empty["final_status"] == "failed"
    assert summary_empty["report_path"] is None

    # 2. Populated run_stats
    state_populated = {
        "retry_count": 1,
        "run_stats": {
            "time_seconds": 23.45,
            "llm_calls": 5,
            "tokens": 4200,
            "retries_used": 1,
            "items_flagged_first_pass": 2,
            "fixed_by_retry": 2,
            "unresolved": 0,
        },
    }
    summary_pop = summarize_run(state_populated)
    assert summary_pop["duration_sec"] == 23.45
    assert summary_pop["llm_calls"] == 5
    assert summary_pop["tokens"] == 4200
    assert summary_pop["retries_used"] == 1
    assert summary_pop["items_flagged_first_pass"] == 2
    assert summary_pop["fixed_by_retry"] == 2
    assert summary_pop["unresolved"] == 0
    assert summary_pop["unverified"] == 0
    assert summary_pop["final_status"] == "success"


def test_summarize_run_final_status_cases():
    """summarize_run properly classifies final_status into success, issues, or failed."""
    # Case A: Success (ok_count >= 1, unresolved == 0, unverified == 0)
    state_succ = {
        "analysis_results": [{"id": 0, "status": "ok"}],
        "review_report": [{"id": 0, "verdict": "pass"}],
    }
    sum_succ = summarize_run(state_succ)
    assert sum_succ["final_status"] == "success"
    assert sum_succ["unresolved"] == 0
    assert sum_succ["unverified"] == 0

    # Case B1: Issues (unresolved > 0)
    state_unres = {
        "analysis_results": [
            {"id": 0, "status": "ok"},
            {"id": 1, "status": "failed"},
        ],
        "review_report": [{"id": 0, "verdict": "pass"}],
    }
    sum_unres = summarize_run(state_unres)
    assert sum_unres["final_status"] == "issues"
    assert sum_unres["unresolved"] == 1

    # Case B2: Issues (unverified > 0)
    state_unver = {
        "analysis_results": [{"id": 0, "status": "ok"}],
        "review_report": [{"id": 0, "verdict": "unverified", "reason": "No rule"}],
    }
    sum_unver = summarize_run(state_unver)
    assert sum_unver["final_status"] == "issues"
    assert sum_unver["unverified"] == 1

    # Case C: Failed (no ok items)
    state_fail = {
        "analysis_results": [{"id": 0, "status": "failed"}],
    }
    sum_fail = summarize_run(state_fail)
    assert sum_fail["final_status"] == "failed"


def test_format_error_details_no_duplication():
    """format_error_details returns clean, single-sentence messages without exception duplication."""
    # 1. Connection / unreachable error
    conn_err = "ConnectionError: HTTPSConnectionPool(host='api.groq.com', port=443): Max retries exceeded"
    friendly1, tech1 = format_error_details(conn_err)
    assert friendly1 == "Can't reach the AI service. Check your internet connection and try again."
    assert tech1 == conn_err
    assert "HTTPSConnectionPool" not in friendly1
    assert "Max retries exceeded" not in friendly1

    # 2. Config / auth error
    cfg_err = "LLMConfigError: 401 Unauthorized: Invalid API key provided"
    friendly2, tech2 = format_error_details(cfg_err)
    assert friendly2 == "AI service configuration problem (API key or model name). Check your .env file."
    assert tech2 == cfg_err
    assert "401 Unauthorized" not in friendly2

    # 3. Generic failure
    gen_err = "RuntimeError: System crashed"
    friendly3, tech3 = format_error_details(gen_err)
    assert friendly3 == "The AI service is busy or something failed. Please try again in a minute."
    assert tech3 == gen_err
    assert "System crashed" not in friendly3


def test_get_final_status_message_cases():
    """get_final_status_message accurately reflects Case A, Case B, and Case C."""
    statuses = {
        "cleaner": {"status": "done", "note": "Completed"},
        "analyst": {"status": "done", "note": "Completed"},
        "visualizer": {"status": "done", "note": "Completed"},
        "reviewer": {"status": "done", "note": "Completed"},
    }

    # Case A: Success without retries
    state_clean = {
        "analysis_results": [{"id": 0, "status": "ok"}],
        "review_report": [{"id": 0, "verdict": "pass"}],
    }
    ok_a, msg_a = get_final_status_message(state_clean, statuses)
    assert ok_a is True
    assert msg_a == "Analysis complete"

    # Case A: Success with retries used
    state_retry_clean = {
        "retry_count": 1,
        "run_stats": {"fixed_by_retry": 2, "retries_used": 1},
        "analysis_results": [{"id": 0, "status": "ok"}],
        "review_report": [{"id": 0, "verdict": "pass"}],
    }
    ok_ar, msg_ar = get_final_status_message(state_retry_clean, statuses)
    assert ok_ar is True
    assert msg_ar == "Analysis complete (The reviewer caught 2 issue(s) and they were fixed after retry)"

    # Case B: Issues (unresolved items)
    state_unres = {
        "analysis_results": [
            {"id": 0, "status": "ok"},
            {"id": 1, "status": "failed"},
        ],
        "review_report": [{"id": 0, "verdict": "pass"}],
    }
    ok_b, msg_b = get_final_status_message(state_unres, statuses)
    assert ok_b is False
    assert msg_b == "Finished with issues (1 unresolved, 0 unverified)"

    # Case B: Issues (unverified items)
    state_unver = {
        "analysis_results": [{"id": 0, "status": "ok"}],
        "review_report": [{"id": 0, "verdict": "unverified"}],
    }
    ok_bv, msg_bv = get_final_status_message(state_unver, statuses)
    assert ok_bv is False
    assert msg_bv == "Finished with issues (0 unresolved, 1 unverified)"

    # Case C: No ok items
    state_fail = {"analysis_results": [{"id": 0, "status": "failed"}]}
    ok_c, msg_c = get_final_status_message(state_fail, statuses)
    assert ok_c is False
    assert msg_c == "Finished with issues (No analysis could be completed)"

    # Case C: Agent error status
    statuses_err = dict(statuses)
    statuses_err["analyst"] = {"status": "error", "note": "Failed"}
    ok_ce, msg_ce = get_final_status_message(state_clean, statuses_err)
    assert ok_ce is False
    assert "Analyst failed" in msg_ce


def test_reviewer_card_note_and_status_from_final_state():
    """get_final_agent_statuses decides reviewer card from final state, ignoring interim warnings."""
    # Statuses contaminated by interim warning event during retry pass
    statuses_interim = {
        "cleaner": {"status": "done", "note": "Completed"},
        "analyst": {"status": "running", "note": "Retry 1/3"},
        "visualizer": {"status": "waiting", "note": ""},
        "reviewer": {"status": "flagged", "note": "Completed with warnings"},
    }

    # 1. Clean success with retries fixed
    final_state_fixed = {
        "retry_count": 1,
        "run_stats": {"fixed_by_retry": 1, "retries_used": 1},
        "analysis_results": [{"id": 0, "status": "ok"}],
        "review_report": [{"id": 0, "verdict": "pass"}],
    }
    final_s1 = get_final_agent_statuses(final_state_fixed, statuses_interim)
    assert final_s1["reviewer"]["status"] == "done"
    assert final_s1["reviewer"]["note"] == "Caught 1, fixed after retry"
    assert final_s1["analyst"]["status"] == "done"

    # 2. Clean success without retries
    final_state_clean = {
        "analysis_results": [{"id": 0, "status": "ok"}],
        "review_report": [{"id": 0, "verdict": "pass"}],
    }
    final_s2 = get_final_agent_statuses(final_state_clean, statuses_interim)
    assert final_s2["reviewer"]["status"] == "done"
    assert final_s2["reviewer"]["note"] == "Completed"

    # 3. Final state has unresolved issues
    final_state_unres = {
        "analysis_results": [{"id": 0, "status": "ok"}],
        "review_report": [{"id": 0, "verdict": "fail", "reason": "Math mismatch"}],
    }
    final_s3 = get_final_agent_statuses(final_state_unres, statuses_interim)
    assert final_s3["reviewer"]["status"] == "flagged"
    assert final_s3["reviewer"]["note"] == "Completed with warnings"


def test_apply_node_update_final_state():
    """apply_node_update with node_name='__final__' applies final status logic."""
    statuses = {
        "cleaner": {"status": "done", "note": "Completed"},
        "analyst": {"status": "running", "note": "Retry 1/3"},
        "visualizer": {"status": "waiting", "note": ""},
        "reviewer": {"status": "flagged", "note": "Flagged 1 item"},
    }
    final_state = {
        "run_stats": {"fixed_by_retry": 3, "retries_used": 1},
        "analysis_results": [{"id": 0, "status": "ok"}],
        "review_report": [{"id": 0, "verdict": "pass"}],
    }
    res = apply_node_update(statuses, "__final__", final_state)
    assert res["reviewer"]["status"] == "done"
    assert res["reviewer"]["note"] == "Caught 3, fixed after retry"
    assert res["analyst"]["status"] == "done"
    assert res["report_running"] is False
