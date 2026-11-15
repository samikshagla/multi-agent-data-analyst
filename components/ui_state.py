"""
components/ui_state.py
Pure Python state-machine and helper functions for the Streamlit UI.
Zero Streamlit imports to ensure easy, robust unit testing.
Manages agent card status transitions, timeline event filtering, HTML formatting,
and execution summary metrics.
"""

from datetime import datetime, timezone
import html
import os
from typing import Any

# ============================================================================
# Agent Definitions & Status Types
# ============================================================================

AGENTS = [
    {
        "key": "cleaner",
        "name": "Cleaner",
        "emoji": "🧹",
        "role": "Imputes nulls, removes duplicates & standardizes dtypes",
    },
    {
        "key": "analyst",
        "name": "Analyst",
        "emoji": "🔍",
        "role": "Generates questions & computes data findings via pandas",
    },
    {
        "key": "visualizer",
        "name": "Visualizer",
        "emoji": "📈",
        "role": "Selects chart types & synthesizes takeaway observations",
    },
    {
        "key": "reviewer",
        "name": "Reviewer",
        "emoji": "🛡️",
        "role": "Recomputes independently & checks mathematical invariants",
    },
]

STATUS_VALUES = ("waiting", "running", "done", "flagged", "error")

AGENT_KEYS = [a["key"] for a in AGENTS]
AGENT_EMOJIS = {a["key"]: a["emoji"] for a in AGENTS}
AGENT_EMOJIS["system"] = "⚙️"
AGENT_EMOJIS["report"] = "📝"


# ============================================================================
# Status Management
# ============================================================================

def initial_statuses() -> dict[str, Any]:
    """
    Returns initial UI statuses for all agents in 'waiting' state.
    """
    statuses: dict[str, Any] = {
        agent["key"]: {"status": "waiting", "note": ""}
        for agent in AGENTS
    }
    statuses["report_running"] = False
    return statuses


def _agent_has_issues(agent_key: str, update_dict: dict[str, Any]) -> bool:
    """Checks whether the agent emitted any 'warning' or 'error' events."""
    evs = update_dict.get("events", [])
    if not isinstance(evs, list):
        return False
    return any(
        str(e.get("agent", "")).lower() == agent_key and e.get("type") in ("error", "warning")
        for e in evs
        if isinstance(e, dict)
    )


def apply_node_update(
    statuses: dict[str, Any],
    node_name: str,
    update: dict[str, Any],
    retry_count: int = 0,
) -> dict[str, Any]:
    """
    Computes updated agent card statuses based on node completion updates.

    Rules:
    - Cleaner done -> Analyst running.
    - Analyst done -> Visualizer running (or straight to report if 0 items succeeded).
    - Visualizer done -> Reviewer running (even after a retry).
    - Reviewer done:
      * If review_feedback is non-empty:
        - Reviewer -> 'flagged' with 'Flagged N item(s)' note.
        - Analyst -> 'running' with 'Retry X/3' note.
        - Visualizer -> reset to 'waiting'.
      * If no feedback:
        - Reviewer -> 'done' or 'flagged' (Completed with warnings).
        - Sets report_running = True.
    - Report done -> report_running = False.

    Args:
        statuses: Current dictionary of agent statuses.
        node_name: The name of the node that just finished execution.
        update: State dict yielded by the pipeline node.
        retry_count: Current retry iteration count.

    Returns:
        dict[str, Any]: New dictionary with updated agent statuses.
    """
    new_statuses: dict[str, Any] = {}
    for k, v in statuses.items():
        if isinstance(v, dict):
            new_statuses[k] = dict(v)
        else:
            new_statuses[k] = v

    if node_name == "cleaner":
        has_warn = _agent_has_issues("cleaner", update)
        new_statuses["cleaner"] = {
            "status": "flagged" if has_warn else "done",
            "note": "Completed with warnings" if has_warn else "Completed",
        }
        new_statuses["analyst"] = {"status": "running", "note": "Analyzing..."}

    elif node_name == "analyst":
        has_warn = _agent_has_issues("analyst", update)
        new_statuses["analyst"] = {
            "status": "flagged" if has_warn else "done",
            "note": "Completed with warnings" if has_warn else "Completed",
        }
        results = update.get("analysis_results", [])
        if results and sum(1 for r in results if r.get("status") == "ok") == 0:
            new_statuses["visualizer"] = {"status": "waiting", "note": "Skipped"}
            new_statuses["reviewer"] = {"status": "waiting", "note": "Skipped"}
            new_statuses["report_running"] = True
        else:
            new_statuses["visualizer"] = {"status": "running", "note": "Charting..."}

    elif node_name == "visualizer":
        has_warn = _agent_has_issues("visualizer", update)
        new_statuses["visualizer"] = {
            "status": "flagged" if has_warn else "done",
            "note": "Completed with warnings" if has_warn else "Completed",
        }
        new_statuses["reviewer"] = {"status": "running", "note": "Auditing..."}

    elif node_name == "reviewer":
        feedback = update.get("review_feedback", [])
        if feedback:
            count = len(feedback)
            next_retry = retry_count + 1
            item_word = "item" if count == 1 else "items"
            new_statuses["reviewer"] = {
                "status": "flagged",
                "note": f"Flagged {count} {item_word}",
            }
            new_statuses["analyst"] = {
                "status": "running",
                "note": f"Retry {next_retry}/3",
            }
            new_statuses["visualizer"] = {"status": "waiting", "note": ""}
            new_statuses["report_running"] = False
        else:
            has_warn = _agent_has_issues("reviewer", update)
            new_statuses["reviewer"] = {
                "status": "flagged" if has_warn else "done",
                "note": "Completed with warnings" if has_warn else "Completed",
            }
            new_statuses["report_running"] = True

    elif node_name == "report":
        new_statuses["report_running"] = False

    elif node_name in ("complete", "__final__"):
        new_statuses["report_running"] = False
        if node_name == "__final__":
            return get_final_agent_statuses(update, new_statuses)
        for a in AGENTS:
            k = a["key"]
            cur = new_statuses.get(k, {})
            if cur.get("status") == "running":
                cur["status"] = "done"
                if not cur.get("note"):
                    cur["note"] = "Completed"

    return new_statuses


# ============================================================================
# Activity Feed & Event Utilities
# ============================================================================

def new_events(
    all_events: list[dict[str, Any]],
    already_seen_count: int,
) -> list[dict[str, Any]]:
    """
    Extracts only newly added events from a cumulative events list,
    preventing duplicate timeline display.

    Args:
        all_events: Full cumulative list of event dicts.
        already_seen_count: Number of events already processed.

    Returns:
        list[dict[str, Any]]: Sliced list of new event dicts.
    """
    if not all_events:
        return []
    if already_seen_count < 0:
        already_seen_count = 0
    if already_seen_count >= len(all_events):
        return []
    return all_events[already_seen_count:]


def format_local_timestamp(iso_str: str) -> str:
    """
    Converts a UTC or ISO timestamp string into a local time string (HH:MM:SS)
    using astimezone().
    """
    if not iso_str:
        return ""
    try:
        cleaned = str(iso_str).strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local_dt = dt.astimezone()
        return local_dt.strftime("%H:%M:%S")
    except Exception:
        return str(iso_str).split("T")[-1][:8]


def event_to_html(event: dict[str, Any]) -> str:
    """
    Formats a single event dict into a styled timeline HTML snippet.
    Colors by type:
    - Info: Teal
    - Warning: Amber
    - Error: Red
    - Retry messages (message containing 'retry'): Blue
    All message text is safely escaped using html.escape to prevent XSS.
    Timestamps are converted from UTC to the user's local time zone.

    Args:
        event: Dictionary representing an agent or system event.

    Returns:
        str: HTML markup string.
    """
    raw_msg = str(event.get("message", ""))
    escaped_msg = html.escape(raw_msg)

    # Convert UTC ISO string to local time HH:MM:SS
    time_part = format_local_timestamp(event.get("timestamp", ""))

    agent_key = str(event.get("agent", "system")).lower()
    icon = AGENT_EMOJIS.get(agent_key, "•")

    # Determine color style
    # Retry messages take precedence and render blue
    if "retry" in raw_msg.lower():
        color_class = "event-retry"
    elif event.get("type") == "error":
        color_class = "event-error"
    elif event.get("type") == "warning":
        color_class = "event-warning"
    else:
        color_class = "event-info"

    return (
        f'<div class="event-item {color_class}">'
        f'<span class="event-time">{time_part}</span>'
        f'<span class="event-icon">{icon}</span>'
        f'<span class="event-msg">{escaped_msg}</span>'
        f'</div>'
    )


# ============================================================================
# Run Summary & Final Status
# ============================================================================

def format_error_details(err: Any) -> tuple[str, str]:
    """
    Returns (friendly_message, technical_details) for UI error presentation.
    The friendly message is a concise, single-sentence summary without appended
    exception text or technical details. Technical details contains the raw
    exception string.
    """
    tech_str = str(err).strip() if err is not None else ""
    lowered = tech_str.lower()

    if (
        "config" in lowered
        or "api key" in lowered
        or "unauthorized" in lowered
        or "forbidden" in lowered
    ):
        friendly = "AI service configuration problem (API key or model name). Check your .env file."
    elif (
        "reach" in lowered
        or "connection" in lowered
        or "unreachable" in lowered
        or "timeout" in lowered
        or "offline" in lowered
    ):
        friendly = "Can't reach the AI service. Check your internet connection and try again."
    else:
        friendly = "The AI service is busy or something failed. Please try again in a minute."

    return friendly, tech_str


def summarize_run(final_state: dict[str, Any] | None) -> dict[str, Any]:
    """
    Extracts display metrics from final_state and run_stats using safe defaults.

    Args:
        final_state: State dictionary from final pipeline execution.

    Returns:
        dict[str, Any]: Display-ready dictionary with duration_sec, llm_calls,
        tokens, retries_used, items_flagged_first_pass, fixed_by_retry,
        unresolved, unverified, final_status, and report_path.
    """
    if not isinstance(final_state, dict):
        final_state = {}

    stats = final_state.get("run_stats", {})
    if not isinstance(stats, dict):
        stats = {}

    analysis_results = final_state.get("analysis_results", [])
    if not isinstance(analysis_results, list):
        analysis_results = []

    review_report = final_state.get("review_report", [])
    if not isinstance(review_report, list):
        review_report = []

    caught_errors = final_state.get("caught_errors", [])
    if not isinstance(caught_errors, list):
        caught_errors = []

    # 1. Compute ok_count
    if "analysis_results" in final_state:
        ok_count = sum(
            1 for r in analysis_results if isinstance(r, dict) and r.get("status") == "ok"
        )
    else:
        # Fallback for synthetic/mock stats dicts
        ok_count = 1 if stats.get("fixed_by_retry", 0) > 0 or (
            stats and stats.get("unresolved", 0) == 0 and stats.get("llm_calls", 0) > 0
        ) else 0

    # 2. Compute unresolved
    if "unresolved" in stats:
        unresolved = int(stats["unresolved"])
    else:
        unresolved = sum(
            1 for r in review_report if isinstance(r, dict) and r.get("verdict") == "fail"
        ) + sum(
            1 for item in analysis_results if isinstance(item, dict) and item.get("status") != "ok"
        )

    # 3. Compute unverified
    if "unverified" in stats:
        unverified = int(stats["unverified"])
    else:
        unverified = sum(
            1 for r in review_report if isinstance(r, dict) and r.get("verdict") == "unverified"
        )

    # 4. Compute fixed_by_retry
    if "fixed_by_retry" in stats:
        fixed_by_retry = int(stats["fixed_by_retry"])
    else:
        first_round_ids = {
            e["id"]
            for e in caught_errors
            if isinstance(e, dict) and e.get("retry_round", 0) == 0 and "id" in e
        }
        final_verdicts = {
            r["id"]: r.get("verdict")
            for r in review_report
            if isinstance(r, dict) and "id" in r
        }
        fixed_by_retry = sum(
            1 for iid in first_round_ids if final_verdicts.get(iid) == "pass"
        )

    # 5. Determine final_status string ("success", "issues", or "failed")
    if ok_count == 0:
        final_status = "failed"
    elif unresolved > 0 or unverified > 0:
        final_status = "issues"
    else:
        final_status = "success"

    # Extract report path if generated
    report_path: str | None = None
    raw_df_path = final_state.get("raw_df_path")
    if raw_df_path:
        stem = os.path.splitext(os.path.basename(raw_df_path))[0]
        candidate_path = os.path.normpath(os.path.join("outputs", f"report_{stem}.md"))
        if os.path.exists(candidate_path):
            report_path = candidate_path

    return {
        "duration_sec": float(stats.get("time_seconds", 0.0)),
        "llm_calls": int(stats.get("llm_calls", 0)),
        "tokens": int(stats.get("tokens", 0)),
        "retries_used": int(stats.get("retries_used", final_state.get("retry_count", 0))),
        "items_flagged_first_pass": int(stats.get("items_flagged_first_pass", 0)),
        "fixed_by_retry": fixed_by_retry,
        "unresolved": unresolved,
        "unverified": unverified,
        "final_status": final_status,
        "report_path": report_path,
    }


def get_final_status_message(
    final_state: dict[str, Any] | None,
    statuses: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """
    Decides the final completion message based on the FINAL state,
    not from warning events observed during the run.

    Rules:
    - Case A (Clean Success):
      At least 1 item succeeded, 0 unresolved items, and 0 unverified items.
      Message: "Analysis complete"
      If retries were used: add "The reviewer caught N issue(s) and they were fixed after retry"
      Returns: (True, message)
    - Case B (Finished with issues):
      Any item is unresolved (> 0) or unverified (> 0).
      Message: "Finished with issues (X unresolved, Y unverified)"
      Returns: (False, message)
    - Case C (Error / Failure):
      No items succeeded or an agent errored.
      Returns: (False, failure reason)
    """
    if not isinstance(final_state, dict):
        return False, "Finished with issues: No final state recorded"

    # Check if any agent errored during execution (Case C)
    if statuses:
        for agent in AGENTS:
            k = agent["key"]
            st_data = statuses.get(k, {})
            if st_data.get("status") == "error":
                return False, f"Finished with issues ({agent['name']} failed)"

    summary = summarize_run(final_state)

    # Case C: No successful items
    if summary["final_status"] == "failed":
        return False, "Finished with issues (No analysis could be completed)"

    # Case B: Unresolved or unverified items exist
    if summary["final_status"] == "issues":
        unres = summary["unresolved"]
        unver = summary["unverified"]
        return False, f"Finished with issues ({unres} unresolved, {unver} unverified)"

    # Case A: Success
    fixed_count = summary["fixed_by_retry"]
    retries = summary["retries_used"]
    if fixed_count > 0 or retries > 0:
        n = fixed_count if fixed_count > 0 else retries
        return True, f"Analysis complete (The reviewer caught {n} issue(s) and they were fixed after retry)"

    return True, "Analysis complete"


def get_final_agent_statuses(
    final_state: dict[str, Any] | None,
    statuses: dict[str, Any],
) -> dict[str, Any]:
    """
    Derives the finalized agent statuses dict from the FINAL state.

    Rules:
    - Transitions any still 'running' agents to 'done'.
    - Reviewer card note and status are determined strictly from the final state:
      * Case A (Success): status 'done' (green).
        If N > 0 issues were fixed by retry, note is 'Caught N, fixed after retry'.
        Otherwise note is 'Completed'.
      * Case B (Issues): status 'flagged' (amber), note 'Completed with warnings'.
      * Case C (Error): preserves error status or sets error note.
    """
    new_statuses: dict[str, Any] = {}
    for k, v in statuses.items():
        if isinstance(v, dict):
            new_statuses[k] = dict(v)
        else:
            new_statuses[k] = v

    if not isinstance(final_state, dict):
        return new_statuses

    summary = summarize_run(final_state)
    final_status = summary["final_status"]

    # Mark any running agent as done
    for a in AGENTS:
        k = a["key"]
        cur = new_statuses.get(k, {})
        if cur.get("status") == "running":
            cur["status"] = "done"
            if not cur.get("note"):
                cur["note"] = "Completed"

    reviewer_cur = new_statuses.get("reviewer", {})
    if reviewer_cur.get("status") != "error":
        if final_status == "success":
            n = summary["fixed_by_retry"]
            if n == 0 and summary["retries_used"] > 0:
                n = summary["retries_used"]
            reviewer_cur["status"] = "done"
            reviewer_cur["note"] = f"Caught {n}, fixed after retry" if n > 0 else "Completed"
        elif final_status == "issues":
            reviewer_cur["status"] = "flagged"
            reviewer_cur["note"] = "Completed with warnings"

    new_statuses["reviewer"] = reviewer_cur
    new_statuses["report_running"] = False
    return new_statuses
