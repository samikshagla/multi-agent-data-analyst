"""
test_visualizer.py
Unit tests for agents/visualizer.py using pytest.
Validates Plotly figure generation, JSON serializability, metric card conversion,
column validation, category truncation, error isolation, and HTML output creation.
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from agents.visualizer import visualizer_node
from state import new_state


def test_table_result_chart_ok(monkeypatch: pytest.MonkeyPatch):
    """Test 1: A table result gives a chart with status 'ok' and valid figure_json."""
    df_result = pd.DataFrame({
        "city": ["New York", "Chicago", "Houston"],
        "sales": [1200, 850, 950],
    })

    state = new_state("data/sample.csv")
    state["analysis_results"] = [
        {
            "id": 1,
            "question": "What are sales by city?",
            "code": "result = ...",
            "result": df_result,
            "status": "ok",
            "attempts": 1,
            "notes": "",
        }
    ]

    fake_spec = json.dumps({
        "chart_type": "bar",
        "x": "city",
        "y": "sales",
        "title": "Sales by City",
        "x_label": "City",
        "y_label": "Sales (USD)",
        "takeaway": "New York leads with 1,200 USD in sales.",
    })
    monkeypatch.setattr("agents.visualizer.call_llm", lambda prompt: fake_spec)

    res = visualizer_node(state)

    assert len(res["charts"]) == 1
    chart = res["charts"][0]
    assert chart["id"] == 1
    assert chart["status"] == "ok"
    assert chart["type"] == "bar"
    assert chart["figure_json"] is not None

    # Verify figure_json parses correctly as JSON
    parsed_fig = json.loads(chart["figure_json"])
    assert "data" in parsed_fig
    assert "layout" in parsed_fig

    # Verify HTML file was saved
    html_file = os.path.join("outputs", "charts", "chart_1.html")
    assert os.path.exists(html_file)


def test_scalar_result_gives_metric():
    """Test 2: A scalar result gives a 'metric' item with no figure."""
    state = new_state("data/sample.csv")
    state["analysis_results"] = [
        {
            "id": 2,
            "question": "What is total revenue?",
            "code": "result = ...",
            "result": 98500.75,
            "status": "ok",
            "attempts": 1,
            "notes": "",
        }
    ]

    res = visualizer_node(state)

    assert len(res["charts"]) == 1
    chart = res["charts"][0]
    assert chart["id"] == 2
    assert chart["type"] == "metric"
    assert chart["status"] == "ok"
    assert chart["figure_json"] is None
    assert "98,500.75" in chart["takeaway"]


def test_failed_analysis_not_charted():
    """Test 3: A failed analysis result is not charted."""
    state = new_state("data/sample.csv")
    state["analysis_results"] = [
        {
            "id": 3,
            "question": "Broken query",
            "code": "bad code",
            "result": None,
            "status": "failed",
            "attempts": 3,
            "notes": "",
        }
    ]

    res = visualizer_node(state)
    assert len(res["charts"]) == 0


def test_case_insensitive_column_match(monkeypatch: pytest.MonkeyPatch):
    """Test 4a: A spec with mismatched column casing is fixed by case-insensitive matching."""
    df_result = pd.DataFrame({
        "Metro_City": ["Seattle", "Austin"],
        "Order_Count": [500, 420],
    })

    state = new_state("data/sample.csv")
    state["analysis_results"] = [
        {
            "id": 4,
            "question": "Orders by city?",
            "code": "...",
            "result": df_result,
            "status": "ok",
            "attempts": 1,
            "notes": "",
        }
    ]

    # Spec uses lowercase 'metro_city' and 'order_count'
    fake_spec = json.dumps({
        "chart_type": "bar",
        "x": "metro_city",
        "y": "order_count",
        "title": "Orders",
        "x_label": "City",
        "y_label": "Orders",
        "takeaway": "Insight",
    })
    monkeypatch.setattr("agents.visualizer.call_llm", lambda prompt: fake_spec)

    res = visualizer_node(state)
    chart = res["charts"][0]
    assert chart["status"] == "ok"
    assert chart["spec"]["x"] == "Metro_City"
    assert chart["spec"]["y"] == "Order_Count"


def test_missing_column_triggers_default_spec(monkeypatch: pytest.MonkeyPatch):
    """Test 4b: A spec referencing a non-existent column falls back to default spec."""
    df_result = pd.DataFrame({
        "category": ["Food", "Retail"],
        "total": [300, 450],
    })

    state = new_state("data/sample.csv")
    state["analysis_results"] = [
        {
            "id": 5,
            "question": "Total by category?",
            "code": "...",
            "result": df_result,
            "status": "ok",
            "attempts": 1,
            "notes": "",
        }
    ]

    # Spec specifies a hallucinated column 'ghost_col'
    fake_spec = json.dumps({
        "chart_type": "bar",
        "x": "ghost_col",
        "y": "total",
        "title": "Total",
        "x_label": "Category",
        "y_label": "Total",
        "takeaway": "Insight",
    })
    monkeypatch.setattr("agents.visualizer.call_llm", lambda prompt: fake_spec)

    res = visualizer_node(state)
    chart = res["charts"][0]
    assert chart["status"] == "ok"
    # Fallback to category as x
    assert chart["spec"]["x"] == "category"


def test_non_numeric_y_skipped(monkeypatch: pytest.MonkeyPatch):
    """Test 5: A non-numeric y column leads to 'skipped' with an informative note."""
    df_result = pd.DataFrame({
        "city": ["New York", "Boston"],
        "region": ["East", "Northeast"],  # Non-numeric
    })

    state = new_state("data/sample.csv")
    state["analysis_results"] = [
        {
            "id": 6,
            "question": "Regions by city?",
            "code": "...",
            "result": df_result,
            "status": "ok",
            "attempts": 1,
            "notes": "",
        }
    ]

    fake_spec = json.dumps({
        "chart_type": "bar",
        "x": "city",
        "y": "region",
        "title": "Regions by City",
        "x_label": "City",
        "y_label": "Region",
        "takeaway": "Region overview",
    })
    monkeypatch.setattr("agents.visualizer.call_llm", lambda prompt: fake_spec)

    res = visualizer_node(state)
    chart = res["charts"][0]
    assert chart["status"] == "skipped"
    assert "not numeric" in chart["note"]
    assert chart["figure_json"] is None


def test_bar_chart_category_cap(monkeypatch: pytest.MonkeyPatch):
    """Test 6: A bar chart with 30 categories is limited to the top 15."""
    df_result = pd.DataFrame({
        "item": [f"Item_{i}" for i in range(30)],
        "value": list(range(30)),
    })

    state = new_state("data/sample.csv")
    state["analysis_results"] = [
        {
            "id": 7,
            "question": "Top items?",
            "code": "...",
            "result": df_result,
            "status": "ok",
            "attempts": 1,
            "notes": "",
        }
    ]

    fake_spec = json.dumps({
        "chart_type": "bar",
        "x": "item",
        "y": "value",
        "title": "Item Values",
        "x_label": "Item",
        "y_label": "Value",
        "takeaway": "Top items listed",
    })
    monkeypatch.setattr("agents.visualizer.call_llm", lambda prompt: fake_spec)

    res = visualizer_node(state)
    chart = res["charts"][0]
    assert chart["status"] == "ok"

    parsed_fig = json.loads(chart["figure_json"])
    # In Plotly bar charts, data[0]['x'] contains the categories
    x_data = parsed_fig["data"][0]["x"]
    assert len(x_data) == 15


def test_invalid_json_fallback(monkeypatch: pytest.MonkeyPatch):
    """Test 7: Invalid JSON from the LLM triggers the default spec fallback."""
    df_result = pd.DataFrame({
        "product": ["Gadget", "Widget"],
        "count": [10, 25],
    })

    state = new_state("data/sample.csv")
    state["analysis_results"] = [
        {
            "id": 8,
            "question": "Product counts?",
            "code": "...",
            "result": df_result,
            "status": "ok",
            "attempts": 1,
            "notes": "",
        }
    ]

    # Malformed text instead of JSON
    monkeypatch.setattr("agents.visualizer.call_llm", lambda prompt: "I recommend a bar chart.")

    res = visualizer_node(state)
    chart = res["charts"][0]
    assert chart["status"] == "ok"
    assert chart["type"] == "bar"
    assert chart["figure_json"] is not None


def test_resilience_on_broken_item():
    """Test 8: The node never crashes when one item breaks; it logs and continues."""
    state = new_state("data/sample.csv")
    state["analysis_results"] = [
        {
            "id": 9,
            "question": "Malformed entry",
            "code": "...",
            "result": object(),  # arbitrary object that fails DataFrame conversion
            "status": "ok",
            "attempts": 1,
            "notes": "",
        },
        {
            "id": 10,
            "question": "Valid scalar metric",
            "code": "...",
            "result": 100,
            "status": "ok",
            "attempts": 1,
            "notes": "",
        },
    ]

    res = visualizer_node(state)
    assert len(res["charts"]) == 2
    # Item 9 skipped gracefully
    assert res["charts"][0]["status"] == "skipped"
    # Item 10 succeeded
    assert res["charts"][1]["status"] == "ok"
    assert res["charts"][1]["type"] == "metric"


def test_charts_rebuilt_only_for_rerun_ids(monkeypatch: pytest.MonkeyPatch):
    """Test 9: On re-run, visualizer only rebuilds charts for ids in rerun_ids and preserves others."""
    state = new_state("data/sample.csv")
    state["rerun_ids"] = [2]
    # Cached chart for id 1
    state["charts"] = [
        {"id": 1, "status": "ok", "type": "metric", "title": "Cached Chart 1", "takeaway": "Cached takeaway"},
        {"id": 2, "status": "ok", "type": "metric", "title": "Old Chart 2", "takeaway": "Old takeaway"},
    ]
    state["analysis_results"] = [
        {"id": 1, "question": "Q1", "code": "...", "result": 100, "status": "ok"},
        {"id": 2, "question": "Q2", "code": "...", "result": 200, "status": "ok"},
    ]

    res = visualizer_node(state)
    assert len(res["charts"]) == 2
    # Chart 1 is the exact cached chart object
    c1 = next(c for c in res["charts"] if c["id"] == 1)
    assert c1["title"] == "Cached Chart 1"
    # Chart 2 was rebuilt
    c2 = next(c for c in res["charts"] if c["id"] == 2)
    assert c2["title"] == "Q2"


def test_non_json_values_converted_in_visualizer(monkeypatch: pytest.MonkeyPatch):
    """Test 10: Non-JSON values (Interval, Categorical) in DataFrame are sanitized before plotting."""
    df_raw = pd.DataFrame({
        "bin": pd.Series([pd.Interval(0, 10), pd.Interval(10, 20)], dtype="interval"),
        "cat": pd.Series(["A", "B"], dtype="category"),
        "sales": [100, 200],
    })
    state = new_state("data/sample.csv")
    state["analysis_results"] = [
        {"id": 1, "question": "Sales by bin?", "code": "...", "result": df_raw, "status": "ok"}
    ]

    fake_spec = json.dumps({
        "chart_type": "bar",
        "x": "bin",
        "y": "sales",
        "title": "Binned Sales",
        "x_label": "Bin",
        "y_label": "Sales",
        "takeaway": "Sales in bin.",
    })
    monkeypatch.setattr("agents.visualizer.call_llm", lambda prompt: fake_spec)

    res = visualizer_node(state)
    assert len(res["charts"]) == 1
    chart = res["charts"][0]
    assert chart["status"] == "ok"
    assert chart["figure_json"] is not None
    # Ensure figure_json is valid JSON
    parsed = json.loads(chart["figure_json"])
    assert "data" in parsed
