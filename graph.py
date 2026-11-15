"""
graph.py
LangGraph Orchestration: Constructs and compiles the StateGraph connecting the
Cleaner, Analyst, Visualizer, and Reviewer agents. Implements feedback conditional edges,
bounded retry loops (prepare_retry), final report synthesis, and execution pipeline runners.
"""

from datetime import datetime, timezone
import json
import os
import re
import time
from typing import Any, Callable, Generator
import pandas as pd
from langgraph.graph import StateGraph, START, END

from agents.cleaner import cleaner_node
from agents.analyst import analyst_node
from agents.visualizer import visualizer_node
from agents.reviewer import reviewer_node
from llm import call_llm, reset_usage_stats, get_usage_stats, LLMUnavailableError, LLMConfigError
from state import AgentState, new_state


# ============================================================================
# Helpers: Markdown Formatting & Reporting
# ============================================================================

def _df_to_markdown_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Formats a pandas DataFrame as a clean GitHub-flavored markdown table without external dependencies."""
    if df.empty:
        return "*Empty table*"
    df_sub = df.head(max_rows)
    headers = [str(c) for c in df_sub.columns]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"

    rows = []
    for _, row in df_sub.iterrows():
        row_str = "| " + " | ".join(str(val) for val in row.values) + " |"
        rows.append(row_str)

    truncated_note = f"\n*(Showing first {max_rows} rows of {len(df)})*" if len(df) > max_rows else ""
    return "\n".join([header_line, sep_line] + rows) + truncated_note


# ============================================================================
# Graph Nodes: prepare_retry & report
# ============================================================================

def prepare_retry_node(state: AgentState) -> dict[str, Any]:
    """
    Increments retry_count by 1 and records an event before routing back to the Analyst agent.
    """
    events = list(state.get("events", []))
    feedback = state.get("review_feedback", [])
    current_retries = state.get("retry_count", 0)
    new_retries = current_retries + 1

    msg = f"Reviewer flagged {len(feedback)} item(s), sending back to analyst (retry {new_retries}/3)"
    events.append({
        "agent": "reviewer",
        "type": "warning",
        "message": msg,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "retry_count": new_retries,
        "events": events,
    }


def report_node(state: AgentState) -> dict[str, Any]:
    """
    Synthesizes the finalized executive markdown report including:
    - Overview metadata (dataset, date).
    - Executive Key Insights (via LLM or fallback takeaways).
    - Detailed Question-by-Question breakdown (with verified / unresolved status tags).
    - Chart artifact links.
    - Data cleaning audit log.
    - Verification and Quality Review metrics.
    - Unresolved issues list.
    Saves the final report to outputs/report_<dataset name>.md.
    """
    events = list(state.get("events", []))
    raw_path = state.get("raw_df_path", "dataset.csv")
    dataset_name = os.path.splitext(os.path.basename(raw_path))[0]
    date_str = datetime.now().strftime("%Y-%m-%d")

    analysis_results = state.get("analysis_results", [])
    charts = state.get("charts", [])
    review_report = state.get("review_report", [])
    caught_errors = state.get("caught_errors", [])
    cleaning_log = state.get("cleaning_log", [])
    retry_count = state.get("retry_count", 0)

    charts_by_id = {c.get("id"): c for c in charts if isinstance(c, dict)}
    reviews_by_id = {r.get("id"): r for r in review_report if isinstance(r, dict)}

    # Identify unresolved items
    unresolved_items: list[dict[str, Any]] = []
    for r in review_report:
        if r.get("verdict") == "fail":
            unresolved_items.append(r)

    unresolved_ids = {u["id"] for u in unresolved_items} | {
        item.get("id") for item in analysis_results if item.get("status") != "ok"
    }

    # Check if there are zero successful analysis items
    ok_results = [item for item in analysis_results if item.get("status") == "ok"]
    if not ok_results:
        # Zero successful analysis queries: skip LLM and build fallback failure report
        failure_reasons = []
        for ev in events:
            if ev.get("type") in ("error", "warning") and ev.get("message"):
                failure_reasons.append(ev.get("message"))
        if not failure_reasons:
            failure_reasons.append("All proposed analytical questions failed code generation or sandbox execution.")

        md_lines = [
            f"# Multi-Agent Data Analysis Report: {dataset_name.replace('_', ' ').title()}",
            f"\n**Dataset**: `{dataset_name}`  ",
            f"**Generated**: {date_str}  ",
            f"**Pipeline**: Multi-Agent LangGraph System (Cleaner -> Analyst -> Report)  ",
            "\n---\n",
            "## No analysis could be completed\n",
            "None of the analytical queries could be successfully calculated on this dataset.\n",
            "### Failure Details:\n",
        ]
        for r in failure_reasons:
            md_lines.append(f"- {r}")

        if analysis_results:
            md_lines.append("\n### Attempted Queries:\n")
            for item in analysis_results:
                md_lines.append(f"- **{item.get('question')}**: Failed ({item.get('error') or 'Execution failed'})")

        md_lines.append("\n---\n")
        if cleaning_log:
            md_lines.append("## Data Cleaning Summary\n")
            for log_entry in cleaning_log:
                md_lines.append(f"- {log_entry}")
            md_lines.append("\n---\n")

        full_report = "\n".join(md_lines)
        os.makedirs("outputs", exist_ok=True)
        report_path = os.path.normpath(os.path.join("outputs", f"report_{dataset_name}.md"))
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(full_report)

        events.append({
            "agent": "report",
            "type": "warning",
            "message": f"Generated fallback report (no analysis completed) saved to {report_path}.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return {
            "final_report": full_report,
            "events": events,
        }

    # 1. Key Insights generation (strictly verified items only)
    verified_results = [
        item for item in analysis_results
        if item.get("status") == "ok"
        and reviews_by_id.get(item.get("id"), {}).get("verdict") == "pass"
    ]

    verified_takeaways = [
        charts_by_id[item["id"]]["takeaway"]
        for item in verified_results
        if item["id"] in charts_by_id and charts_by_id[item["id"]].get("takeaway")
    ]

    key_insights: list[str] = []
    if verified_results:
        summary_bullets = "\n".join(
            f"- Question: {item.get('question')}\n  Takeaway: {charts_by_id.get(item.get('id'), {}).get('takeaway', 'No takeaway')}\n  Result: {str(item.get('result'))[:150]}"
            for item in verified_results
        )

        insights_prompt = f"""You are a principal data analyst. Write 3-5 concise, high-impact executive insight bullets based ONLY on the verified data below.
Use only actual numbers present in the data given; do not invent or extrapolate any facts.

Verified findings:
{summary_bullets}

Return ONLY a bulleted list of 3-5 lines starting with '- '."""

        try:
            llm_reply = call_llm(insights_prompt)
            lines = [l.strip() for l in llm_reply.strip().split("\n") if l.strip().startswith(("-", "*", "•"))]
            if lines:
                key_insights = [re.sub(r"^[-*•]\s*", "", l) for l in lines[:5]]
        except (LLMUnavailableError, LLMConfigError):
            raise
        except Exception:
            pass

    # Fallback to chart takeaways if LLM synthesis is empty
    if not key_insights:
        key_insights = verified_takeaways[:5] if verified_takeaways else ["Analysis completed successfully."]

    # 2. Build Markdown Sections
    md_lines: list[str] = []
    md_lines.append(f"# Multi-Agent Data Analysis Report: {dataset_name.replace('_', ' ').title()}")
    md_lines.append(f"\n**Dataset**: `{dataset_name}`  ")
    md_lines.append(f"**Generated**: {date_str}  ")
    md_lines.append(f"**Pipeline**: Multi-Agent LangGraph System (Cleaner -> Analyst -> Visualizer -> Reviewer)  ")
    md_lines.append("\n---\n")

    # Section 1: Key Insights
    md_lines.append("## 1. Executive Key Insights\n")
    for insight in key_insights:
        md_lines.append(f"- {insight}")
    md_lines.append("\n---\n")

    # Section 2: Analysis Details
    count_verified = 0
    count_unverified = 0
    count_unresolved = 0

    md_lines.append("## 2. Analysis Details\n")
    for item in analysis_results:
        item_id = item.get("id")
        q = item.get("question", "")
        res_val = item.get("result")
        chart_info = charts_by_id.get(item_id, {})
        takeaway = chart_info.get("takeaway", "No takeaway available.")
        rev = reviews_by_id.get(item_id, {})
        verdict = rev.get("verdict")

        # Status tag determination
        if item.get("status") != "ok" or item_id in unresolved_ids or verdict == "fail":
            count_unresolved += 1
            reason = rev.get("reason") or item.get("error") or "Failed analysis or failed review after maximum retries"
            status_tag = f"**Unresolved** ({reason})"
        elif verdict == "unverified":
            count_unverified += 1
            reason = rev.get("reason", "Reviewer could not check it")
            status_tag = f"**Unverified** ({reason})"
        elif verdict == "pass":
            count_verified += 1
            status_tag = "**Verified**"
        else:
            count_unverified += 1
            status_tag = "**Unverified** (Not reviewed)"

        md_lines.append(f"### Q{item_id + 1}: {q}")
        md_lines.append(f"- **Verification Status**: {status_tag}")
        md_lines.append(f"- **Takeaway**: {takeaway}")
        if not chart_info or chart_info.get("status") != "ok":
            skip_note = chart_info.get("note", "chart generation was skipped or failed") if chart_info else "no chart generated"
            md_lines.append(f"- **Visualization**: *No chart generated (visualizer skipped or failed: {skip_note})*")

        # Render table or scalar
        md_lines.append("\n**Result Preview**:\n")
        if isinstance(res_val, pd.DataFrame):
            md_lines.append(_df_to_markdown_table(res_val, max_rows=10))
        elif res_val is not None:
            md_lines.append(f"`{res_val}`")
        else:
            md_lines.append("*No result data generated*")

        md_lines.append("\n")

    md_lines.append("---\n")

    # Section 3: Charts & Visualizations
    md_lines.append("## 3. Visualizations\n")
    if charts:
        for c in charts:
            cid = c.get("id")
            title = c.get("title", f"Chart {cid}")
            ctype = c.get("type", "chart")
            if ctype == "metric":
                md_lines.append(f"- **Metric Card**: {title} — *{c.get('takeaway')}*")
            elif c.get("status") == "ok":
                html_path = os.path.normpath(os.path.join("outputs", "charts", f"chart_{cid}.html"))
                md_lines.append(f"- **{title}** ({ctype.title()} Chart): [`outputs/charts/chart_{cid}.html`]({html_path})")
            else:
                md_lines.append(f"- *{title}* (Visualization skipped: {c.get('note')})")
    else:
        md_lines.append("*No charts generated.*")
    md_lines.append("\n---\n")

    # Section 4: Data Cleaning Summary
    md_lines.append("## 4. Data Cleaning Summary\n")
    if cleaning_log:
        for log_entry in cleaning_log:
            md_lines.append(f"- {log_entry}")
    else:
        md_lines.append("*No data transformations were required.*")
    md_lines.append("\n---\n")

    # Section 5: Quality Review
    first_round_flagged_ids = {
        e["id"] for e in caught_errors if e.get("retry_round", 0) == 0 and "id" in e
    }
    items_flagged_first_pass = len(first_round_flagged_ids)
    final_verdicts = {
        r["id"]: r.get("verdict") for r in review_report if isinstance(r, dict) and "id" in r
    }
    fixed_by_retry = sum(
        1 for iid in first_round_flagged_ids if final_verdicts.get(iid) == "pass"
    )

    md_lines.append("## 5. Quality & Verification Review\n")
    md_lines.append(f"- **Items Evaluated**: {len(analysis_results)}")
    md_lines.append(f"- **Verified Items**: {count_verified}")
    md_lines.append(f"- **Unverified Items**: {count_unverified}")
    md_lines.append(f"- **Unresolved Items**: {count_unresolved}")
    md_lines.append(f"- **Items Flagged on First Pass**: {items_flagged_first_pass}")
    md_lines.append(f"- **Issues Fixed by Retry**: {fixed_by_retry}")
    md_lines.append(f"- **Total Flag Events**: {len(caught_errors)}")
    md_lines.append(f"- **Review Retries Executed**: {retry_count}")

    if caught_errors:
        md_lines.append("\n### Intercepted Errors Log:")
        for err in caught_errors:
            md_lines.append(f"- Retry Round {err.get('retry_round', 0)} | Item {err.get('id')}: {err.get('reason')}")
    md_lines.append("\n---\n")

    # Section 6: Unresolved Issues
    md_lines.append("## 6. Unresolved Issues\n")
    if unresolved_items:
        for u in unresolved_items:
            md_lines.append(f"- **Item {u['id']}** ('{u.get('question')}'): {u.get('reason')}")
    else:
        md_lines.append("None")

    final_report_md = "\n".join(md_lines)

    # Save report to outputs/report_<dataset name>.md
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    report_path = os.path.normpath(os.path.join(outputs_dir, f"report_{dataset_name}.md"))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_report_md)

    events.append({
        "agent": "report",
        "type": "info",
        "message": f"Final report generated and saved to '{report_path}'.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "final_report": final_report_md,
        "run_stats": dict(state.get("run_stats", {})),
        "events": events,
    }


# ============================================================================
# LangGraph Workflow Construction
# ============================================================================

def build_graph(nodes: dict[str, Callable] | None = None, max_retries: int = 3) -> Any:
    """
    Constructs and compiles the multi-agent StateGraph:
    cleaner -> analyst -> visualizer -> reviewer -> [conditional: prepare_retry or report]
    """
    node_map = {
        "cleaner": cleaner_node,
        "analyst": analyst_node,
        "visualizer": visualizer_node,
        "reviewer": reviewer_node,
        "report": report_node,
    }
    if nodes:
        node_map.update(nodes)

    workflow = StateGraph(AgentState)

    # Add workflow nodes
    workflow.add_node("cleaner", node_map["cleaner"])
    workflow.add_node("analyst", node_map["analyst"])
    workflow.add_node("visualizer", node_map["visualizer"])
    workflow.add_node("reviewer", node_map["reviewer"])
    workflow.add_node("prepare_retry", prepare_retry_node)
    workflow.add_node("report", node_map["report"])

    # Define edges
    workflow.add_edge(START, "cleaner")
    workflow.add_edge("cleaner", "analyst")

    # Routing conditional edge after analyst: if zero items succeeded, skip straight to report
    def analyst_routing(state: AgentState) -> str:
        results = state.get("analysis_results", [])
        ok_count = sum(1 for r in results if r.get("status") == "ok")
        if ok_count == 0:
            return "report"
        return "visualizer"

    workflow.add_conditional_edges(
        "analyst",
        analyst_routing,
        {
            "visualizer": "visualizer",
            "report": "report",
        },
    )
    workflow.add_edge("visualizer", "reviewer")

    # Routing conditional edge after reviewer
    def routing_condition(state: AgentState) -> str:
        feedback = state.get("review_feedback", [])
        retries = state.get("retry_count", 0)
        if feedback and retries < max_retries:
            return "prepare_retry"
        return "report"

    workflow.add_conditional_edges(
        "reviewer",
        routing_condition,
        {
            "prepare_retry": "prepare_retry",
            "report": "report",
        },
    )

    # Retry loop: prepare_retry directs back into analyst
    workflow.add_edge("prepare_retry", "analyst")
    workflow.add_edge("report", END)

    return workflow.compile()


# ============================================================================
# Pipeline Runner: Synchronous & Streaming
# ============================================================================

def _compute_run_stats(state: dict[str, Any], elapsed: float, usage: dict[str, Any]) -> dict[str, Any]:
    """Computes comprehensive execution performance and error recovery metrics."""
    caught_errors = state.get("caught_errors", [])
    review_report = state.get("review_report", [])
    analysis_results = state.get("analysis_results", [])

    first_round_ids = {
        e["id"] for e in caught_errors if e.get("retry_round", 0) == 0 and "id" in e
    }
    items_flagged_first_pass = len(first_round_ids)

    final_verdicts = {
        r["id"]: r.get("verdict") for r in review_report if isinstance(r, dict) and "id" in r
    }
    fixed_by_retry = sum(
        1 for iid in first_round_ids if final_verdicts.get(iid) == "pass"
    )

    unresolved_count = sum(
        1 for r in review_report if isinstance(r, dict) and r.get("verdict") == "fail"
    ) + sum(1 for item in analysis_results if item.get("status") != "ok")

    by_model = usage.get("by_model", {})
    models_used = list(by_model.keys())

    return {
        "time_seconds": elapsed,
        "llm_calls": usage.get("calls", 0),
        "tokens": usage.get("total_tokens", 0),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "by_model": by_model,
        "models_used": models_used,
        "multiple_models_used": len(models_used) > 1,
        "retries_used": state.get("retry_count", 0),
        "items_flagged_first_pass": items_flagged_first_pass,
        "fixed_by_retry": fixed_by_retry,
        "unresolved": unresolved_count,
        "total_flag_events": len(caught_errors),
        "errors_caught": len(caught_errors),
    }


def run_pipeline(
    raw_df_path: str,
    stream: bool = False,
    max_retries: int = 3,
) -> Any:
    """
    Executes the Multi-Agent Data Analyst pipeline.

    Args:
        raw_df_path (str): File path to the raw dataset CSV.
        stream (bool): If True, returns a generator yielding (node_name, state_update).
                       If False, executes synchronously and returns the final state dict.
        max_retries (int): Maximum review/retry correction cycles (default: 3).

    Returns:
        AgentState (if stream=False) or Generator[tuple[str, dict], None, None] (if stream=True).
    """
    reset_usage_stats()
    start_time = time.time()
    initial_state = new_state(raw_df_path)
    app = build_graph(max_retries=max_retries)

    if not stream:
        # Synchronous execution
        final_state = app.invoke(initial_state, config={"recursion_limit": 50})
        elapsed = round(time.time() - start_time, 2)
        usage = get_usage_stats()

        final_stats = _compute_run_stats(final_state, elapsed, usage)
        final_state["run_stats"] = final_stats
        return final_state

    # Streaming execution generator
    def _stream_generator() -> Generator[tuple[str, dict[str, Any]], None, None]:
        accumulated_state: dict[str, Any] = dict(initial_state)

        for event in app.stream(initial_state, config={"recursion_limit": 50}):
            for node_name, state_update in event.items():
                accumulated_state.update(state_update)
                yield node_name, state_update

        elapsed = round(time.time() - start_time, 2)
        usage = get_usage_stats()

        final_stats = _compute_run_stats(accumulated_state, elapsed, usage)
        accumulated_state["run_stats"] = final_stats
        yield "complete", {"run_stats": final_stats}
        yield "__final__", accumulated_state

    return _stream_generator()
