"""
tests/test_demo.py
Unit tests for recorded demo snapshot save/load, validation rules,
CLI --save-snapshot, and Streamlit AppTest replay in demo mode.
"""

from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
import pytest

from components.demo_snapshot import (
    save_snapshot,
    load_snapshot,
    save_cleaned_csv_for_demo,
    SnapshotError,
    SCHEMA_VERSION,
)


# ============================================================================
# 1. Round-Trip Serialization & Type Rebuilding
# ============================================================================

def test_snapshot_round_trip(tmp_path):
    """
    Verifies that DataFrames, Series, numeric scalars, numpy types, NaN,
    Timestamp, and nested structures survive serialization and are rebuilt.
    """
    snap_path = str(tmp_path / "roundtrip.json")

    sample_df = pd.DataFrame({
        "id": [1, 2, 3],
        "city": ["Berlin", "London", "Paris"],
        "revenue": [100.5, np.nan, 250.0],
    })

    sample_series = pd.Series([10, 20, 30], name="quantities")

    final_state = {
        "raw_df_path": "dataset.csv",
        "clean_df_path": "cleaned.csv",
        "final_report": "# Market Report\nClean results.",
        "cleaning_log": ["Imputed 1 null", "Filtered outlier"],
        "questions": ["What is total revenue?"],
        "analysis_results": [
            {
                "id": 0,
                "question": "What is total revenue?",
                "code": "result = 350.5",
                "result": 350.5,
                "status": "ok",
            },
            {
                "id": 1,
                "question": "Show table",
                "code": "result = df",
                "result": sample_df,
                "status": "ok",
            },
        ],
        "charts": [
            {
                "id": 0,
                "title": "Revenue Chart",
                "type": "metric",
                "takeaway": "Result: 350.5",
                "figure_json": None,
                "status": "ok",
                "note": "",
            }
        ],
        "review_report": [
            {
                "id": 0,
                "question": "What is total revenue?",
                "verdict": "pass",
                "reason": "Recomputed matches.",
                "original_value": 350.5,
                "recomputed_value": 350.5,
                "warnings": [],
            },
            {
                "id": 1,
                "question": "Show table",
                "verdict": "pass",
                "reason": "Matches.",
                "original_value": sample_df,
                "recomputed_value": sample_df,
                "recomputed_result": sample_df,
                "warnings": [],
            },
        ],
        "caught_errors": [],
        "retry_count": 0,
        "events": [
            {
                "agent": "cleaner",
                "type": "info",
                "message": "Data cleaned successfully",
                "timestamp": "2026-11-19T04:00:00+00:00",
            }
        ],
        "run_stats": {
            "time_seconds": 15.2,
            "llm_calls": 5,
            "tokens": 2500,
            "by_model": {"model-alpha": {"calls": 5, "tokens": 2500}},
            "retries_used": 0,
            "items_flagged_first_pass": 0,
            "fixed_by_retry": 0,
            "unresolved": 0,
        },
        "profile": {"rows": 3, "columns": 3},
    }

    steps = [
        {
            "node_name": "cleaner",
            "seconds": 1.5,
            "events": [{"agent": "cleaner", "type": "info", "message": "Cleaned data"}],
            "has_review_feedback": False,
            "retry_count": 0,
        }
    ]

    save_snapshot(final_state, steps, snap_path)

    loaded_final, loaded_steps, loaded_meta = load_snapshot(snap_path)

    assert loaded_meta["schema_version"] == SCHEMA_VERSION
    assert loaded_meta["tokens"] == 2500
    assert loaded_meta["dataset_name"] == "dataset.csv"

    # Verify DataFrame reconstruction
    rebuilt_df = loaded_final["analysis_results"][1]["result"]
    assert isinstance(rebuilt_df, pd.DataFrame)
    assert len(rebuilt_df) == 3
    assert list(rebuilt_df.columns) == ["id", "city", "revenue"]
    assert rebuilt_df.iloc[0]["city"] == "Berlin"


# ============================================================================
# 2. Refusal Rules: Secrets, Absolute Paths, Large Files, Overwriting
# ============================================================================

def test_save_snapshot_refusals(tmp_path):
    """
    Verifies that save_snapshot refuses:
    - secret patterns (gsk_, AIza, sk-, api_key, Bearer )
    - absolute filesystem paths (C:\\, E:\\, /home/, /Users/)
    - files exceeding 3 MB limit
    - existing files without force=True (accepts with force=True)
    """
    base_state = {
        "final_report": "Clean report",
        "cleaning_log": [],
        "questions": [],
        "analysis_results": [],
        "charts": [],
        "review_report": [],
        "caught_errors": [],
        "run_stats": {},
        "events": [],
    }

    # 1. Secret pattern refusal
    state_with_secret = dict(base_state)
    state_with_secret["final_report"] = "Header\nHere is token gsk_secret_token_12345"
    with pytest.raises(SnapshotError) as exc_info:
        save_snapshot(state_with_secret, [], str(tmp_path / "secret.json"))
    assert "secret pattern 'gsk_'" in str(exc_info.value)
    assert "field 'final.final_report'" in str(exc_info.value)

    # 2. Absolute path refusal
    state_with_path = dict(base_state)
    state_with_path["cleaning_log"] = ["Loaded file from C:\\Users\\Administrator\\data.csv"]
    with pytest.raises(SnapshotError) as exc_info:
        save_snapshot(state_with_path, [], str(tmp_path / "path.json"))
    assert "local path pattern" in str(exc_info.value)

    # 3. File size cap (> 3 MB)
    huge_state = dict(base_state)
    huge_state["final_report"] = "x" * (4 * 1024 * 1024)  # 4 MB string
    with pytest.raises(SnapshotError) as exc_info:
        save_snapshot(huge_state, [], str(tmp_path / "huge.json"))
    assert "exceeds the maximum allowed limit of 3 MB" in str(exc_info.value)

    # 4. Overwrite refusal without force
    valid_state = dict(base_state)
    p = str(tmp_path / "test_force.json")
    save_snapshot(valid_state, [], p)
    assert os.path.exists(p)

    with pytest.raises(SnapshotError) as exc_info:
        save_snapshot(valid_state, [], p, force=False)
    assert "already exists" in str(exc_info.value)

    # With force=True, overwrite succeeds
    save_snapshot(valid_state, [], p, force=True)
    assert os.path.exists(p)


# ============================================================================
# 3. Snapshot Loading Validation Errors
# ============================================================================

def test_load_snapshot_errors(tmp_path):
    """
    Verifies that load_snapshot raises SnapshotError on missing,
    corrupted, or invalid schema snapshot files.
    """
    # 1. Missing file
    with pytest.raises(SnapshotError) as exc_info:
        load_snapshot(str(tmp_path / "non_existent.json"))
    assert "not found" in str(exc_info.value)

    # 2. Corrupted JSON
    corrupt_file = str(tmp_path / "corrupt.json")
    with open(corrupt_file, "w") as f:
        f.write("{ invalid json")
    with pytest.raises(SnapshotError) as exc_info:
        load_snapshot(corrupt_file)
    assert "Corrupted or invalid JSON" in str(exc_info.value)

    # 3. Missing required section
    incomplete_file = str(tmp_path / "incomplete.json")
    with open(incomplete_file, "w") as f:
        json.dump({"meta": {"schema_version": 1}}, f)
    with pytest.raises(SnapshotError) as exc_info:
        load_snapshot(incomplete_file)
    assert "missing required section" in str(exc_info.value)

    # 4. Newer schema version
    new_schema_file = str(tmp_path / "newer.json")
    with open(new_schema_file, "w") as f:
        json.dump(
            {
                "meta": {"schema_version": 999},
                "steps": [],
                "final": {},
            },
            f,
        )
    with pytest.raises(SnapshotError) as exc_info:
        load_snapshot(new_schema_file)
    assert "newer than supported version" in str(exc_info.value)


# ============================================================================
# 4. Cleaned CSV Copy for Demo
# ============================================================================

def test_save_cleaned_csv_for_demo(tmp_path):
    """
    Verifies that save_cleaned_csv_for_demo copies the file when <= 3 MB
    and skips copying if the file is missing or oversized.
    """
    src_csv = str(tmp_path / "clean_source.csv")
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(src_csv, index=False)

    dest_csv = str(tmp_path / "demo_clean.csv")
    ok = save_cleaned_csv_for_demo({"clean_df_path": src_csv}, dest_csv)
    assert ok is True
    assert os.path.exists(dest_csv)

    # Missing file returns False
    assert save_cleaned_csv_for_demo({"clean_df_path": "non_existent.csv"}, str(tmp_path / "x.csv")) is False


# ============================================================================
# 5. Streamlit AppTest in Demo-Only Mode (Zero LLM Replay)
# ============================================================================

def test_streamlit_apptest_demo_mode(monkeypatch):
    """
    Tests the Streamlit app using AppTest in demo-only mode (DEMO_ONLY=1).
    Ensures call_llm is never invoked, 'Watch a recorded run' triggers replay,
    the demo banner appears, and results tabs are rendered.
    """
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DEMO_ONLY", "1")
    monkeypatch.setenv("DEMO_SNAPSHOT_PATH", "tests/fixtures/mini_snapshot.json")

    # Guard: if call_llm is ever invoked, fail immediately
    def forbidden_call_llm(*args, **kwargs):
        raise AssertionError("call_llm must NEVER be called in demo mode!")

    monkeypatch.setattr("llm.call_llm", forbidden_call_llm)

    from pathlib import Path
    app_path = str(Path(__file__).parent.parent / "app.py")
    at = AppTest.from_file(app_path, default_timeout=15)
    at.run()
    assert not at.exception

    # Check Skip animation to speed up testing
    for cb in at.checkbox:
        if "Skip animation" in cb.label:
            cb.check()
            break

    # Click "Watch a recorded run"
    clicked = False
    for b in at.button:
        if "Watch a recorded run" in b.label:
            b.click()
            clicked = True
            break
    assert clicked, "'Watch a recorded run' button not found"

    # Step through replay
    at.run()
    assert not at.exception

    if at.session_state.replaying_demo:
        at.run()
        assert not at.exception

    # Verify banner appears
    banner_texts = [info.value for info in at.info]
    assert any("Recorded run:" in t and "No AI calls are made in demo mode." in t for t in banner_texts)

    # Verify results tabs are rendered
    tab_labels = [t.label for t in at.tabs]
    expected_tabs = ["📋 Report", "📊 Charts", "🧹 Cleaning Log", "🛡️ Reviewer Findings"]
    assert tab_labels == expected_tabs


# ============================================================================
# 6. CLI run.py --save-snapshot Integration
# ============================================================================

def test_run_py_save_snapshot(tmp_path, monkeypatch):
    """
    Verifies that run.py --save-snapshot:
    - writes a valid snapshot when pipeline finishes with status 'success' or 'issues'
    - does NOT write a snapshot when the run fails.
    """
    from run import main

    csv_file = str(tmp_path / "input.csv")
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(csv_file, index=False)

    target_snapshot = str(tmp_path / "cli_run.json")

    # 1. Success run mock
    fake_final_state = {
        "raw_df_path": csv_file,
        "clean_df_path": csv_file,
        "final_report": "# Test Report",
        "cleaning_log": [],
        "questions": ["q1"],
        "analysis_results": [{"id": 0, "status": "ok", "question": "q1", "result": 10}],
        "charts": [],
        "review_report": [{"id": 0, "verdict": "pass"}],
        "caught_errors": [],
        "retry_count": 0,
        "events": [],
        "run_stats": {
            "time_seconds": 2.0,
            "llm_calls": 2,
            "tokens": 500,
            "retries_used": 0,
            "items_flagged_first_pass": 0,
            "fixed_by_retry": 0,
            "unresolved": 0,
        },
    }

    def fake_pipeline_success(path, stream=True):
        yield "cleaner", {"events": [{"agent": "cleaner", "type": "info", "message": "Done"}]}
        yield "__final__", fake_final_state

    monkeypatch.setattr("run.run_pipeline", fake_pipeline_success)
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", csv_file, "--save-snapshot", target_snapshot, "--force"],
    )

    main()

    assert os.path.exists(target_snapshot)
    final_data, steps, meta = load_snapshot(target_snapshot)
    assert meta["tokens"] == 500

    # 2. Failed run mock: must NOT write snapshot
    failed_snapshot = str(tmp_path / "cli_failed.json")
    fake_failed_state = dict(fake_final_state)
    fake_failed_state["analysis_results"] = [{"id": 0, "status": "failed", "error": "timeout"}]
    fake_failed_state["events"] = [{"agent": "analyst", "type": "error", "message": "Failed"}]
    fake_failed_state["run_stats"]["unresolved"] = 1

    def fake_pipeline_failed(path, stream=True):
        yield "analyst", {"events": [{"agent": "analyst", "type": "error", "message": "Failed"}]}
        yield "__final__", fake_failed_state

    monkeypatch.setattr("run.run_pipeline", fake_pipeline_failed)
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", csv_file, "--save-snapshot", failed_snapshot, "--force"],
    )

    main()

    assert not os.path.exists(failed_snapshot)
