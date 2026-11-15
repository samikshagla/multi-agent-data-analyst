"""
visualizer.py
Visualizer Agent: Translates analytical outputs and pandas DataFrames into
interactive Plotly visualizations or metric cards. Handles spec generation via LLM,
strict column validation, category caps, fallback chart generation, and HTML exports.
"""

from datetime import datetime, timezone
import json
import os
import re
from typing import Any, Callable
import numpy as np
import pandas as pd
import plotly.express as px

from llm import call_llm, LLMUnavailableError, LLMConfigError
from state import AgentState

# Visualization design constants
MAIN_COLOR = "#14B8A6"  # Teal
COLOR_PALETTE = ["#14B8A6", "#06B6D4", "#0EA5E9", "#3B82F6", "#6366F1", "#8B5CF6"]
ALLOWED_CHART_TYPES = {"bar", "line", "scatter", "histogram", "pie"}


# ============================================================================
# Helpers: Parsing & Validation
# ============================================================================

def _clean_code_fence(text: str) -> str:
    """Strips markdown code fences from LLM text."""
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_spec_json(text: str) -> dict[str, Any] | None:
    """Extracts and parses chart specification JSON dictionary."""
    cleaned = _clean_code_fence(text)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Regex search for JSON object
    match = re.search(r"\{\s*\"chart_type\".*?\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


def _generate_default_spec(df_res: pd.DataFrame, question: str) -> dict[str, Any] | None:
    """
    Generates a deterministic fallback chart specification when LLM output is unavailable:
    - Line chart if the first column represents dates/time.
    - Bar chart if first column is text and second column is numeric.
    - Histogram if only one numeric column exists.
    """
    cols = list(df_res.columns)
    if not cols:
        return None

    if len(cols) == 1:
        if pd.api.types.is_numeric_dtype(df_res[cols[0]]):
            return {
                "chart_type": "histogram",
                "x": cols[0],
                "y": None,
                "title": f"Distribution of {cols[0]}",
                "x_label": cols[0],
                "y_label": "Frequency",
                "takeaway": f"Distribution overview for {cols[0]}.",
            }
        return None

    col0, col1 = cols[0], cols[1]

    # Check if first column looks like a date/time series
    is_date = (
        pd.api.types.is_datetime64_any_dtype(df_res[col0])
        or any(k in str(col0).lower() for k in ("date", "time", "year", "month", "day"))
    )

    if is_date and pd.api.types.is_numeric_dtype(df_res[col1]):
        return {
            "chart_type": "line",
            "x": col0,
            "y": col1,
            "title": f"{col1} over {col0}",
            "x_label": col0,
            "y_label": col1,
            "takeaway": f"Trend of {col1} across {col0}.",
        }

    # Bar chart if second column is numeric
    if pd.api.types.is_numeric_dtype(df_res[col1]):
        return {
            "chart_type": "bar",
            "x": col0,
            "y": col1,
            "title": f"{col1} by {col0}",
            "x_label": col0,
            "y_label": col1,
            "takeaway": f"Comparison of {col1} grouped by {col0}.",
        }

    # Scatter plot if both are numeric
    if pd.api.types.is_numeric_dtype(df_res[col0]) and pd.api.types.is_numeric_dtype(df_res[col1]):
        return {
            "chart_type": "scatter",
            "x": col0,
            "y": col1,
            "title": f"{col1} vs {col0}",
            "x_label": col0,
            "y_label": col1,
            "takeaway": f"Scatter relationship between {col0} and {col1}.",
        }

    return None


def _validate_and_sanitize_spec(
    spec: dict[str, Any],
    df_res: pd.DataFrame,
    question: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Validates and cleans the chart spec:
    1. Ensures chart_type is supported.
    2. Matches x and y columns (case-insensitive fallback).
    3. Verifies y is numeric for bar, line, scatter, and pie.
    4. Converts pie to bar if categories exceed 6.
    Returns (validated_spec, error_note).
    """
    chart_type = str(spec.get("chart_type", "bar")).strip().lower()
    if chart_type not in ALLOWED_CHART_TYPES:
        chart_type = "bar"

    x = spec.get("x")
    y = spec.get("y")

    col_map = {str(c).lower(): c for c in df_res.columns}

    # Validate x column
    if x not in df_res.columns:
        if isinstance(x, str) and x.lower() in col_map:
            x = col_map[x.lower()]
        else:
            # Fall back to default spec
            default_spec = _generate_default_spec(df_res, question)
            if not default_spec:
                return None, f"Column '{x}' not found and no valid fallback spec could be generated."
            return default_spec, None

    # Validate y column
    if y is not None:
        if y not in df_res.columns:
            if isinstance(y, str) and y.lower() in col_map:
                y = col_map[y.lower()]
            else:
                default_spec = _generate_default_spec(df_res, question)
                if not default_spec:
                    return None, f"Column '{y}' not found and no valid fallback spec could be generated."
                return default_spec, None

    # Enforce numeric y for charts requiring numeric measurements
    if chart_type in ("bar", "line", "scatter", "pie"):
        if y is None or not pd.api.types.is_numeric_dtype(df_res[y]):
            return None, f"Column '{y}' is not numeric, which is required for a {chart_type} chart."

    # Enforce pie chart constraints (max 6 categories, else switch to bar)
    if chart_type == "pie":
        unique_cats = df_res[x].nunique(dropna=True)
        if unique_cats > 6:
            chart_type = "bar"

    validated = {
        "chart_type": chart_type,
        "x": x,
        "y": y,
        "title": str(spec.get("title", question)),
        "x_label": str(spec.get("x_label", x if x else "")),
        "y_label": str(spec.get("y_label", y if y else "")),
        "takeaway": str(spec.get("takeaway", "Insight generated from data analysis.")),
    }
    return validated, None


def _build_plotly_figure(spec: dict[str, Any], df_res: pd.DataFrame) -> Any:
    """Constructs the Plotly Express figure using the dark template and teal accent."""
    chart_type = spec["chart_type"]
    x = spec["x"]
    y = spec["y"]
    title = spec["title"]
    x_label = spec["x_label"]
    y_label = spec["y_label"]

    df_plot = df_res.copy()

    # Rule: For bar charts with > 15 categories, keep top 15 by y
    if chart_type == "bar" and y is not None:
        if len(df_plot) > 15:
            df_plot = df_plot.sort_values(by=y, ascending=False).head(15)

    if chart_type == "bar":
        fig = px.bar(
            df_plot,
            x=x,
            y=y,
            title=title,
            labels={x: x_label, y: y_label},
            template="plotly_dark",
            color_discrete_sequence=[MAIN_COLOR],
        )
    elif chart_type == "line":
        fig = px.line(
            df_plot,
            x=x,
            y=y,
            title=title,
            labels={x: x_label, y: y_label},
            template="plotly_dark",
            color_discrete_sequence=[MAIN_COLOR],
        )
    elif chart_type == "scatter":
        fig = px.scatter(
            df_plot,
            x=x,
            y=y,
            title=title,
            labels={x: x_label, y: y_label},
            template="plotly_dark",
            color_discrete_sequence=[MAIN_COLOR],
        )
    elif chart_type == "histogram":
        fig = px.histogram(
            df_plot,
            x=x,
            y=y if y else None,
            title=title,
            labels={x: x_label, y: y_label or "Count"},
            template="plotly_dark",
            color_discrete_sequence=[MAIN_COLOR],
        )
    elif chart_type == "pie":
        fig = px.pie(
            df_plot,
            names=x,
            values=y,
            title=title,
            template="plotly_dark",
            color_discrete_sequence=COLOR_PALETTE,
        )
    else:
        fig = px.bar(
            df_plot,
            x=x,
            y=y,
            title=title,
            template="plotly_dark",
            color_discrete_sequence=[MAIN_COLOR],
        )

    # Clean layout styling
    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=12),
        title_font=dict(size=14),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def _sanitize_dataframe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """Converts Interval, Period, Timestamp, Categorical, and numpy types in DataFrame to strings or plain numbers."""
    cleaned = df.copy()

    # Clean index
    if isinstance(cleaned.index, pd.MultiIndex):
        cleaned.index = pd.Index([
            tuple(str(v) if isinstance(v, (pd.Interval, pd.Period, pd.Timestamp)) else v for v in row)
            for row in cleaned.index
        ])
    elif (
        isinstance(cleaned.index.dtype, pd.CategoricalDtype)
        or isinstance(cleaned.index.dtype, pd.IntervalDtype)
        or isinstance(cleaned.index.dtype, pd.PeriodDtype)
        or pd.api.types.is_datetime64_any_dtype(cleaned.index.dtype)
    ):
        cleaned.index = cleaned.index.astype(str)
    else:
        cleaned.index = cleaned.index.map(
            lambda x: str(x) if isinstance(x, (pd.Interval, pd.Period, pd.Timestamp)) else x
        )

    # Clean columns
    for col in cleaned.columns:
        s = cleaned[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            cleaned[col] = s.astype(str)
        elif (
            isinstance(s.dtype, pd.IntervalDtype)
            or isinstance(s.dtype, pd.PeriodDtype)
            or pd.api.types.is_datetime64_any_dtype(s.dtype)
        ):
            cleaned[col] = s.astype(str)
        elif pd.api.types.is_object_dtype(s.dtype):
            cleaned[col] = s.map(
                lambda x: str(x) if isinstance(x, (pd.Interval, pd.Period, pd.Timestamp)) else x
            )
    return cleaned


# ============================================================================
# LangGraph Node: visualizer_node
# ============================================================================

def visualizer_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph Visualizer Node:
    - Scans analysis_results with status 'ok'.
    - Formats scalar outputs as 'metric' cards.
    - Generates Plotly charts for tabular outputs via validated LLM specs.
    - Saves HTML charts in outputs/charts/chart_<id>.html.
    - Stores figure_json strings in state['charts'] (no live figure objects).
    - Preserves existing charts for unchanged items during review re-runs.
    """
    events = list(state.get("events", []))
    analysis_results = state.get("analysis_results", [])
    rerun_ids = state.get("rerun_ids", [])
    existing_charts = state.get("charts", [])
    existing_charts_by_id = {
        c["id"]: c for c in existing_charts if isinstance(c, dict) and "id" in c
    }

    charts: list[dict[str, Any]] = []

    # Ensure output directory exists for HTML charts
    charts_dir = os.path.join("outputs", "charts")
    os.makedirs(charts_dir, exist_ok=True)

    for item in analysis_results:
        # Ignore failed or incomplete analysis items
        if item.get("status") != "ok":
            continue

        item_id = item.get("id")
        question = item.get("question", "")
        raw_res = item.get("result")

        # On re-run, keep existing chart if this item was not re-run by the analyst
        if rerun_ids and item_id not in rerun_ids:
            if item_id in existing_charts_by_id:
                charts.append(existing_charts_by_id[item_id])
                continue

        try:
            # ----------------------------------------------------------------
            # Case 1: Scalar results -> Metric Card
            # ----------------------------------------------------------------
            is_scalar = (
                raw_res is None
                or isinstance(raw_res, (int, float, str, bool, np.number, np.bool_))
            )

            if is_scalar:
                if isinstance(raw_res, int):
                    formatted_val = f"{raw_res:,}"
                elif isinstance(raw_res, float):
                    formatted_val = f"{raw_res:,.2f}"
                else:
                    formatted_val = str(raw_res)

                takeaway = f"Result: {formatted_val}"
                charts.append({
                    "id": item_id,
                    "type": "metric",
                    "title": question,
                    "takeaway": takeaway,
                    "figure_json": None,
                    "spec": None,
                    "status": "ok",
                    "note": f"Metric value: {formatted_val}",
                })
                events.append({
                    "agent": "visualizer",
                    "type": "info",
                    "message": f"Created metric card for analysis id {item_id}.",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                continue

            # ----------------------------------------------------------------
            # Case 2: Tabular results -> Plotly Chart
            # ----------------------------------------------------------------
            if isinstance(raw_res, pd.DataFrame):
                df_res = raw_res.copy()
            elif isinstance(raw_res, (list, dict)):
                df_res = pd.DataFrame(raw_res)
            else:
                df_res = pd.DataFrame([{"value": raw_res}])

            # Convert any non-JSON-serializable values (Interval, Period, Timestamp, Categorical)
            df_res = _sanitize_dataframe_for_json(df_res)

            if df_res.empty:
                charts.append({
                    "id": item_id,
                    "type": "unknown",
                    "title": question,
                    "takeaway": "Empty dataset",
                    "figure_json": None,
                    "spec": None,
                    "status": "skipped",
                    "note": "Result DataFrame is empty",
                })
                continue

            # Prompt LLM for chart spec
            dtypes_info = {col: str(df_res[col].dtype) for col in df_res.columns}
            head_sample = df_res.head(5).to_dict(orient="records")

            spec_prompt = f"""You are a data visualization specialist.
Analyze this query and data sample, and choose the most effective chart.

Question: {question}
Columns: {list(df_res.columns)}
Data Types: {dtypes_info}
Sample Rows (First 5):
{head_sample}

Return ONLY JSON with keys: chart_type (one of bar, line, scatter, histogram, pie), x (column name), y (column name or null for histogram), title, x_label, y_label, takeaway (one sentence stating the main insight using actual numbers from the data). Do not return code. In the takeaway, use numbers exactly as they appear in the table, with the same units. Do not abbreviate units (write 'minutes', not 'm'). Do not quote bracket labels as if they were values."""

            raw_spec_reply = call_llm(spec_prompt)
            spec_dict = _parse_spec_json(raw_spec_reply)

            # Retry once if parsing failed
            if not spec_dict:
                events.append({
                    "agent": "visualizer",
                    "type": "warning",
                    "message": f"Initial spec JSON parsing failed for analysis id {item_id}. Retrying...",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                retry_prompt = spec_prompt + "\n\nCRITICAL: Return ONLY valid JSON format."
                spec_dict = _parse_spec_json(call_llm(retry_prompt))

            # Fallback to default spec if parsing still failed
            if not spec_dict:
                events.append({
                    "agent": "visualizer",
                    "type": "warning",
                    "message": f"Spec JSON parsing failed for id {item_id}. Using default spec fallback.",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                spec_dict = _generate_default_spec(df_res, question)

            if not spec_dict:
                charts.append({
                    "id": item_id,
                    "type": "unknown",
                    "title": question,
                    "takeaway": "No suitable chart spec could be determined.",
                    "figure_json": None,
                    "spec": None,
                    "status": "skipped",
                    "note": "Could not generate or validate chart specification.",
                })
                continue

            # Validate and sanitize specification
            validated_spec, error_note = _validate_and_sanitize_spec(spec_dict, df_res, question)

            if not validated_spec or error_note:
                skip_msg = error_note or "Spec validation failed."
                charts.append({
                    "id": item_id,
                    "type": spec_dict.get("chart_type", "unknown"),
                    "title": question,
                    "takeaway": spec_dict.get("takeaway", ""),
                    "figure_json": None,
                    "spec": spec_dict,
                    "status": "skipped",
                    "note": skip_msg,
                })
                events.append({
                    "agent": "visualizer",
                    "type": "warning",
                    "message": f"Skipped chart for id {item_id}: {skip_msg}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                continue

            # Cheap check: verify takeaway numbers match table
            takeaway = validated_spec.get("takeaway", "")
            if takeaway:
                from agents.reviewer import _extract_takeaway_numbers
                mismatches = _extract_takeaway_numbers(takeaway, df_res)
                if mismatches:
                    regen_prompt = f"""Question: {question}
Table values:
{df_res.head(10).to_string()}

The previous chart takeaway was flagged because it used numbers not matching the table:
Previous takeaway: {takeaway}
Flagged numbers: {'; '.join(mismatches)}

Rewrite ONLY the one-sentence takeaway using numbers exactly as they appear in the table above. Use the exact same units and do not abbreviate.
Return ONLY the one-sentence takeaway string."""
                    try:
                        new_takeaway = call_llm(regen_prompt)
                        cleaned_new_takeaway = _clean_code_fence(new_takeaway).strip().strip('"').strip("'")
                        if cleaned_new_takeaway:
                            validated_spec["takeaway"] = cleaned_new_takeaway
                    except (LLMUnavailableError, LLMConfigError):
                        raise
                    except Exception:
                        pass

            # Build figure
            fig = _build_plotly_figure(validated_spec, df_res)
            fig_json = fig.to_json()

            # Save HTML file
            html_path = os.path.normpath(os.path.join(charts_dir, f"chart_{item_id}.html"))
            fig.write_html(html_path, include_plotlyjs="cdn")

            charts.append({
                "id": item_id,
                "type": validated_spec["chart_type"],
                "title": validated_spec["title"],
                "takeaway": validated_spec["takeaway"],
                "figure_json": fig_json,
                "spec": validated_spec,
                "status": "ok",
                "note": f"Exported to {html_path}",
            })
            events.append({
                "agent": "visualizer",
                "type": "info",
                "message": f"Created {validated_spec['chart_type']} chart for analysis id {item_id}.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        except (LLMUnavailableError, LLMConfigError):
            raise
        except Exception as e:
            # Failure isolation: log error and mark item skipped without failing node
            err_msg = f"Unexpected visualization failure on item {item_id}: {e}"
            charts.append({
                "id": item_id,
                "type": "unknown",
                "title": question,
                "takeaway": "",
                "figure_json": None,
                "spec": None,
                "status": "skipped",
                "note": err_msg,
            })
            events.append({
                "agent": "visualizer",
                "type": "error",
                "message": err_msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    return {
        "charts": charts,
        "events": events,
    }
