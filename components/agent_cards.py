"""
components/agent_cards.py
Streamlit component to render live agent execution cards and workflow progress.
Draws 4 agent cards in one row (Cleaner, Analyst, Visualizer, Reviewer) with emojis,
role summaries, status badges (waiting, running, done, flagged, error), and note messages,
along with a progress bar underneath.
"""

from typing import Any
import streamlit as st

from components.ui_state import AGENTS, get_final_status_message, get_final_agent_statuses


def render_agent_cards(
    placeholder: Any,
    statuses: dict[str, Any],
    final_state: dict[str, Any] | None = None,
) -> None:
    """
    Renders 4 live agent cards and an overarching progress bar inside a Streamlit placeholder.

    Args:
        placeholder: Streamlit placeholder (e.g. st.empty()).
        statuses: Dictionary of current agent statuses and notes from ui_state.
        final_state: Optional final state dictionary once pipeline finishes.
    """
    if final_state is not None:
        statuses = get_final_agent_statuses(final_state, statuses)

    with placeholder.container():
        cols = st.columns(len(AGENTS))

        for col, agent in zip(cols, AGENTS):
            key = agent["key"]
            agent_data = statuses.get(key, {"status": "waiting", "note": ""})
            status = agent_data.get("status", "waiting")
            note = agent_data.get("note", "")

            # Format badge label and styling
            if status == "running":
                badge_label = "RUNNING"
            elif status == "error":
                badge_label = "ERROR"
            elif status == "flagged":
                badge_label = "WARNINGS" if "warning" in note.lower() else "FLAGGED"
            elif status == "done":
                badge_label = "COMPLETED"
            else:
                badge_label = "WAITING"

            card_html = f"""
<div class="agent-card status-{status}">
    <div class="agent-card-top">
        <span class="agent-card-emoji">{agent['emoji']}</span>
        <span class="agent-badge badge-{status}">{badge_label}</span>
    </div>
    <div class="agent-card-name">{agent['name']}</div>
    <div class="agent-card-role">{agent['role']}</div>
    <div class="agent-card-note">{note}</div>
</div>
"""
            with col:
                st.markdown(card_html, unsafe_allow_html=True)

        # Calculate workflow progress
        finished_count = sum(
            1 for a in AGENTS if statuses.get(a["key"], {}).get("status") in ("done", "flagged")
        )

        if final_state is not None:
            _, progress_text = get_final_status_message(final_state, statuses)
            progress_val = 1.0
        elif statuses.get("report_running"):
            progress_val = 0.90
            progress_text = "Agents finished &bull; Generating executive report..."
        elif finished_count == 4:
            progress_val = 1.0
            progress_text = "All agents completed &bull; Analysis finished"
        else:
            progress_val = min(1.0, finished_count / 4.0)
            if finished_count == 0:
                progress_text = "Ready to start pipeline"
            else:
                progress_text = f"Agent progress: {finished_count}/4 completed"

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.progress(progress_val, text=progress_text)
