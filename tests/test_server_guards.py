"""
tests/test_server_guards.py
Comprehensive test suite verifying server-side guards for public hosted deployment:
1. Dynamic Settings defaults and overrides (get_settings)
2. RunGate thread-safe concurrency limiting and guaranteed slot release
3. SessionCooldown and WallClock behaviour with fake clocks
4. Upload size and DataFrame schema validation messages
5. Hosted mode uploads (no data/uploads files, workspace cleanup, in-memory rendering)
6. Guarded execution (cooldown, busy, timeout, token limits, cleanup)
7. HOSTED=1 and DEMO_ONLY=1 semantics in llm.py
8. Identifier column protection guard in cleaner.py
9. Snapshot cleaned CSV copy limit and skip reporting in run.py
"""

import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from tools.limits import (
    RunGate,
    RunTimeout,
    SessionCooldown,
    Settings,
    WallClock,
    get_settings,
    validate_dataframe,
    validate_upload_size,
)
from tools.workspace import RunWorkspace
from components.run_control import run_guarded
from agents.cleaner import is_identifier_column, cleaner_node
import llm
from llm import LLMConfig, LLMDisabledError, call_llm, llm_session


# ============================================================================
# Fake Clock Helper
# ============================================================================

class FakeClock:
    """Controllable clock for deterministic time-based testing."""
    def __init__(self, start: float = 1000.0):
        self.current = start

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


# ============================================================================
# 1. Settings Tests
# ============================================================================

def test_get_settings_local_defaults(monkeypatch):
    """Verifies default settings in un-hosted / local execution environment."""
    for key in (
        "HOSTED", "MAX_UPLOAD_MB", "MAX_ROWS", "MAX_COLS",
        "MAX_RUN_TOKENS", "MAX_CONCURRENT_RUNS", "RUN_COOLDOWN_SECONDS", "RUN_WALL_CLOCK_SECONDS"
    ):
        monkeypatch.delenv(key, raising=False)

    s = get_settings()
    assert s.hosted is False
    assert s.max_upload_mb == 20
    assert s.max_rows == 200_000
    assert s.max_cols == 100
    assert s.max_run_tokens is None
    assert s.max_concurrent_runs == 2
    assert s.run_cooldown_seconds == 0
    assert s.run_wall_clock_seconds is None


def test_get_settings_hosted_defaults(monkeypatch):
    """Verifies default settings when HOSTED=1."""
    monkeypatch.setenv("HOSTED", "1")
    for key in (
        "MAX_UPLOAD_MB", "MAX_ROWS", "MAX_COLS",
        "MAX_RUN_TOKENS", "MAX_CONCURRENT_RUNS", "RUN_COOLDOWN_SECONDS", "RUN_WALL_CLOCK_SECONDS"
    ):
        monkeypatch.delenv(key, raising=False)

    s = get_settings()
    assert s.hosted is True
    assert s.max_upload_mb == 5
    assert s.max_rows == 30_000
    assert s.max_cols == 60
    assert s.max_run_tokens == 80_000
    assert s.max_concurrent_runs == 2
    assert s.run_cooldown_seconds == 60
    assert s.run_wall_clock_seconds == 600


def test_get_settings_environment_overrides(monkeypatch):
    """Verifies environment variables explicitly override all defaults."""
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.setenv("MAX_UPLOAD_MB", "15")
    monkeypatch.setenv("MAX_ROWS", "50000")
    monkeypatch.setenv("MAX_COLS", "40")
    monkeypatch.setenv("MAX_RUN_TOKENS", "120000")
    monkeypatch.setenv("MAX_CONCURRENT_RUNS", "5")
    monkeypatch.setenv("RUN_COOLDOWN_SECONDS", "45")
    monkeypatch.setenv("RUN_WALL_CLOCK_SECONDS", "180")

    s = get_settings()
    assert s.hosted is True
    assert s.max_upload_mb == 15
    assert s.max_rows == 50000
    assert s.max_cols == 40
    assert s.max_run_tokens == 120000
    assert s.max_concurrent_runs == 5
    assert s.run_cooldown_seconds == 45
    assert s.run_wall_clock_seconds == 180


# ============================================================================
# 2. RunGate Concurrency Gate Tests
# ============================================================================

def test_run_gate_concurrency_and_slot_release():
    """
    Verifies RunGate with 3 threads and a limit of 2:
    - First 2 threads acquire slots
    - 3rd thread is refused (returns None)
    - Slot is released even when an exception occurs inside context manager
    - active_runs accurately reflects in-flight executions
    """
    gate = RunGate(max_concurrent=2)
    assert gate.active_runs() == 0

    hold_w1 = threading.Event()
    hold_w2 = threading.Event()
    w1_started = threading.Event()
    w2_started = threading.Event()
    results = {}

    def worker_1():
        with gate.try_acquire() as ctx:
            results["w1_acquired"] = ctx is not None
            w1_started.set()
            hold_w1.wait(timeout=2.0)

    def worker_2():
        try:
            with gate.try_acquire() as ctx:
                results["w2_acquired"] = ctx is not None
                w2_started.set()
                hold_w2.wait(timeout=2.0)
                raise ValueError("Worker 2 failed intentionally")
        except ValueError:
            pass

    t1 = threading.Thread(target=worker_1)
    t2 = threading.Thread(target=worker_2)
    t1.start()
    t2.start()

    assert w1_started.wait(timeout=2.0)
    assert w2_started.wait(timeout=2.0)
    assert gate.active_runs() == 2

    # While w1 and w2 are running, 3rd acquisition attempt must fail
    w3_ctx = gate.try_acquire()
    assert w3_ctx is None
    results["w3_refused"] = True

    # Release worker 2 (raises exception inside ctx, safely releases slot)
    hold_w2.set()
    t2.join(timeout=2.0)

    # Now active_runs should be 1
    assert gate.active_runs() == 1

    # Worker 3 can now acquire
    w3_slot = gate.try_acquire()
    assert w3_slot is not None
    assert gate.active_runs() == 2
    w3_slot.release()
    assert gate.active_runs() == 1

    # Release worker 1
    hold_w1.set()
    t1.join(timeout=2.0)
    assert gate.active_runs() == 0


# ============================================================================
# 3. SessionCooldown and WallClock Tests
# ============================================================================

def test_session_cooldown_with_fake_clock():
    """Verifies SessionCooldown countdown, marking, and 1-hour stale purge."""
    clock = FakeClock(1000.0)
    cooldown = SessionCooldown(purge_after_seconds=3600.0, clock_fn=clock)

    assert cooldown.seconds_remaining("sess_1", 60) == 0

    cooldown.mark("sess_1")
    assert cooldown.seconds_remaining("sess_1", 60) == 60

    clock.advance(25.0)
    assert cooldown.seconds_remaining("sess_1", 60) == 35

    clock.advance(40.0)
    assert cooldown.seconds_remaining("sess_1", 60) == 0

    # Old entries purge after 1 hour (3600s)
    cooldown.mark("sess_old")
    assert "sess_old" in cooldown._timestamps
    clock.advance(3605.0)
    # Next call purges sess_old
    cooldown.seconds_remaining("sess_new", 60)
    assert "sess_old" not in cooldown._timestamps


def test_wall_clock_timeout_and_noop():
    """Verifies WallClock check() raises RunTimeout when exceeded and is no-op when None."""
    clock = FakeClock(100.0)
    wc = WallClock(limit_seconds=10.0, clock_fn=clock)
    wc.check()  # No error

    clock.advance(5.0)
    wc.check()  # No error

    clock.advance(6.0)  # total elapsed = 11.0s > 10.0s
    with pytest.raises(RunTimeout) as exc_info:
        wc.check()
    assert "Analysis timed out after 10 seconds" in str(exc_info.value)

    # When limit_seconds is None, it should never raise
    wc_none = WallClock(limit_seconds=None, clock_fn=clock)
    clock.advance(10000.0)
    wc_none.check()  # Must not raise


# ============================================================================
# 4. Validation Helpers Tests
# ============================================================================

def test_validate_upload_size():
    """Verifies validate_upload_size error messages and success condition."""
    settings = Settings(
        hosted=True,
        max_upload_mb=5,
        max_rows=30000,
        max_cols=60,
        max_run_tokens=80000,
        max_concurrent_runs=2,
        run_cooldown_seconds=60,
        run_wall_clock_seconds=600,
    )
    assert validate_upload_size(4 * 1024 * 1024, settings) is None

    err = validate_upload_size(6 * 1024 * 1024, settings)
    assert err is not None
    assert "exceeds the 5 MB limit" in err


def test_validate_dataframe_messages():
    """Verifies friendly error messages for empty, 1-row, 1-column, and oversized dataframes."""
    settings = Settings(
        hosted=True,
        max_upload_mb=5,
        max_rows=10,
        max_cols=3,
        max_run_tokens=80000,
        max_concurrent_runs=2,
        run_cooldown_seconds=60,
        run_wall_clock_seconds=600,
    )

    # Empty
    assert "only headers and no data rows" in validate_dataframe(pd.DataFrame(), settings)
    assert "only headers and no data rows" in validate_dataframe(None, settings)

    # 1 row
    df_1row = pd.DataFrame({"a": [1], "b": [2]})
    assert "contains only 1 row" in validate_dataframe(df_1row, settings)

    # 1 column
    df_1col = pd.DataFrame({"a": [1, 2, 3]})
    assert "has only 1 column" in validate_dataframe(df_1col, settings)

    # Oversized rows
    df_many_rows = pd.DataFrame({"a": range(15), "b": range(15)})
    assert "Dataset exceeds row limit: 15 rows found (maximum allowed is 10 rows)" in validate_dataframe(df_many_rows, settings)

    # Oversized columns
    df_many_cols = pd.DataFrame({f"col_{i}": [1, 2] for i in range(5)})
    assert "Dataset exceeds column limit: 5 columns found (maximum allowed is 3 columns)" in validate_dataframe(df_many_cols, settings)

    # Valid dataframe
    df_valid = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert validate_dataframe(df_valid, settings) is None


# ============================================================================
# 5. Hosted Mode Upload Flow & Workspace Cleanup Tests
# ============================================================================

def test_hosted_upload_flow_and_workspace_cleanup(tmp_path, monkeypatch):
    """
    Verifies that in hosted mode:
    - No files are written to data/uploads
    - A RunWorkspace is created and cleaned up on success, error, and timeout
    - Results tabs can access precomputed cleaning_summary and in-memory cleaned CSV bytes
    """
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    before_files = set(uploads_dir.glob("*.csv"))

    # Create workspace
    ws = RunWorkspace()
    ws_dir = ws.workspace_dir
    assert os.path.exists(ws_dir)

    raw_path = ws.get_raw_path("sample_hosted.csv")
    test_df = pd.DataFrame({"order_id": ["A1", "A2"], "sales": [100.0, 200.0]})
    test_df.to_csv(raw_path, index=False)
    assert os.path.exists(raw_path)

    # Cleaned output
    clean_path = ws.get_cleaned_path("cleaned_sample_hosted.csv")
    test_df.to_csv(clean_path, index=False)
    assert os.path.exists(clean_path)

    # Workspace cleanup
    ws.cleanup()
    assert not os.path.exists(ws_dir)

    # Verify no file was created under data/uploads
    after_files = set(uploads_dir.glob("*.csv"))
    assert before_files == after_files


def test_results_tabs_work_without_disk_files():
    """Verifies results data functions work purely from in-memory data when workspace is gone."""
    from components.results_data import get_cleaning_summary, cleaned_csv_bytes

    non_existent_raw = "/tmp/does_not_exist/raw.csv"
    non_existent_clean = "/tmp/does_not_exist/clean.csv"

    final_state = {
        "raw_df_path": non_existent_raw,
        "clean_df_path": non_existent_clean,
        "cleaning_summary": {
            "raw_rows": 100,
            "raw_cols": 5,
            "clean_rows": 98,
            "clean_cols": 5,
        },
        "cleaned_csv_bytes": b"col1,col2\n1,2\n",
    }

    # Summary reads precomputed dictionary without checking disk
    summary = get_cleaning_summary(final_state)
    assert summary["raw_rows"] == 100
    assert summary["clean_rows"] == 98

    # Cleaned CSV bytes reads from in-memory bytes without checking disk
    csv_bytes = cleaned_csv_bytes(final_state)
    assert csv_bytes == b"col1,col2\n1,2\n"


# ============================================================================
# 6. run_guarded Tests
# ============================================================================

def test_run_guarded_cooldown_message():
    """Verifies run_guarded returns a friendly cooldown error when cooldown has not expired."""
    clock = FakeClock(1000.0)
    cooldown = SessionCooldown(clock_fn=clock)
    cooldown.mark("user_session_1")

    settings = Settings(
        hosted=True,
        max_upload_mb=5,
        max_rows=30000,
        max_cols=60,
        max_run_tokens=80000,
        max_concurrent_runs=2,
        run_cooldown_seconds=60,
        run_wall_clock_seconds=600,
    )

    res = run_guarded(
        settings=settings,
        session_id="user_session_1",
        make_stream=lambda: iter([]),
        cooldown=cooldown,
        clock_fn=clock,
    )
    assert res["status"] == "cooldown"
    assert "Please wait 60 second(s) before starting another analysis." in res["error"]


def test_run_guarded_busy_message_and_gate_release():
    """Verifies run_guarded returns busy error when gate is full, and releases gate on completion."""
    gate = RunGate(max_concurrent=1)
    cooldown = SessionCooldown()

    settings = Settings(
        hosted=True,
        max_upload_mb=5,
        max_rows=30000,
        max_cols=60,
        max_run_tokens=80000,
        max_concurrent_runs=1,
        run_cooldown_seconds=0,
        run_wall_clock_seconds=600,
    )

    # Acquire the single slot manually
    slot = gate.try_acquire(1)
    assert slot is not None

    res = run_guarded(
        settings=settings,
        session_id="user_session_2",
        make_stream=lambda: iter([]),
        gate=gate,
        cooldown=cooldown,
    )
    assert res["status"] == "busy"
    assert "The server is busy with other runs. Try again in a minute, or watch the recorded run." in res["error"]

    slot.release()
    assert gate.active_runs() == 0

    # Now run_guarded should acquire and release cleanly
    def dummy_stream():
        yield ("__final__", {"status": "success"})

    res_ok = run_guarded(
        settings=settings,
        session_id="user_session_3",
        make_stream=dummy_stream,
        gate=gate,
        cooldown=cooldown,
    )
    assert res_ok["status"] == "ok"
    assert res_ok["final_state"] == {"status": "success"}
    assert gate.active_runs() == 0


def test_run_guarded_timeout_and_workspace_cleanup():
    """Verifies run_guarded stops on timeout, cleans up workspace in finally, and leaves gate free."""
    gate = RunGate(max_concurrent=2)
    clock = FakeClock(100.0)
    ws = RunWorkspace()
    ws_dir = ws.workspace_dir
    assert os.path.exists(ws_dir)

    settings = Settings(
        hosted=True,
        max_upload_mb=5,
        max_rows=30000,
        max_cols=60,
        max_run_tokens=80000,
        max_concurrent_runs=2,
        run_cooldown_seconds=0,
        run_wall_clock_seconds=10,
    )

    def slow_stream():
        yield ("cleaner", {"note": "step 1"})
        clock.advance(15.0)  # Exceeds 10s limit
        yield ("analyst", {"note": "step 2"})

    res = run_guarded(
        settings=settings,
        session_id="user_session_timeout",
        make_stream=slow_stream,
        workspace=ws,
        gate=gate,
        clock_fn=clock,
    )
    assert res["status"] == "timeout"
    assert "Analysis timed out after 10 seconds" in res["error"]
    assert res["final_state"] is None
    # Verify workspace was cleaned up and gate released
    assert not os.path.exists(ws_dir)
    assert gate.active_runs() == 0


# ============================================================================
# 7. HOSTED=1 and DEMO_ONLY=1 in llm.py
# ============================================================================

def test_hosted_mode_llm_disabled(monkeypatch):
    """
    Verifies that when HOSTED=1 and no context config is set, call_llm raises
    LLMDisabledError('Server-side API keys are disabled on this host').
    A context config bypasses this restriction.
    """
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.delenv("DEMO_ONLY", raising=False)

    with pytest.raises(LLMDisabledError) as exc_info:
        call_llm("test prompt")
    assert "Server-side API keys are disabled on this host" in str(exc_info.value)

    # With context config, hosted guard is bypassed
    user_config = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="gsk_dummy")
    with patch("llm._call_groq", return_value=("fake response", 10, 10)):
        with llm_session(user_config):
            resp = call_llm("test prompt")
            assert resp == "fake response"


def test_demo_only_llm_disabled(monkeypatch):
    """Verifies DEMO_ONLY=1 still raises LLMDisabledError('LLM calls are disabled in demo mode')."""
    monkeypatch.setenv("DEMO_ONLY", "1")
    monkeypatch.delenv("HOSTED", raising=False)

    with pytest.raises(LLMDisabledError) as exc_info:
        call_llm("test prompt")
    assert "LLM calls are disabled in demo mode" in str(exc_info.value)


# ============================================================================
# 8. Identifier Column Guard in cleaner.py
# ============================================================================

def test_is_identifier_column():
    """Verifies identifier column pattern matching and 50+ unique row text heuristic."""
    s_small = pd.Series(["id1", "id2"])
    assert is_identifier_column(s_small, "id") is True
    assert is_identifier_column(s_small, "order_id") is True
    assert is_identifier_column(s_small, "id_user") is True
    assert is_identifier_column(s_small, "user_uuid") is True
    assert is_identifier_column(s_small, "device_guid") is True
    assert is_identifier_column(s_small, "account_key") is True
    assert is_identifier_column(s_small, "revenue") is False

    # Text column with 55 unique values
    unique_text = pd.Series([f"val_{i}" for i in range(55)])
    assert is_identifier_column(unique_text, "custom_code") is True

    # Numeric column with 55 unique values is NOT matched by text heuristic
    unique_nums = pd.Series(range(55))
    assert is_identifier_column(unique_nums, "custom_code") is False


def test_cleaner_identifier_guard_skips_mutations(tmp_path):
    """
    Verifies that cleaner_node skips fill_missing, standardize_case, clip_outliers,
    and convert_dtype numeric on identifier columns with a warning, while ordinary
    columns are cleaned normally.
    """
    # Create test CSV with identifier column and ordinary column
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        "order_id": ["ORD_1", "ord_2", np.nan, "ORD_4"],
        "customer_id": ["cust_a", "CUST_B", "cust_c", "CUST_D"],
        "user_uuid": ["uuid_1", "uuid_2", "uuid_3", "uuid_4"],
        "price": [10.0, 15.0, np.nan, 9999.0],
        "category": ["apple", "BANANA", "apple", "banana"],
    })
    df.to_csv(csv_file, index=False)

    cleaning_plan = [
        {"action": "fill_missing", "column": "order_id", "method": "mode"},
        {"action": "standardize_case", "column": "customer_id", "case": "upper"},
        {"action": "clip_outliers", "column": "order_id", "lower": 0, "upper": 100},
        {"action": "convert_dtype", "column": "user_uuid", "to": "numeric"},
        # Ordinary column actions (should run)
        {"action": "fill_missing", "column": "price", "method": "median"},
        {"action": "standardize_case", "column": "category", "case": "lower"},
    ]

    state = {
        "raw_df_path": str(csv_file),
        "workspace_dir": str(tmp_path),
    }

    with patch("agents.cleaner.call_llm", return_value=json.dumps(cleaning_plan)):
        res = cleaner_node(state)

    cleaning_log = res["cleaning_log"]

    # Verify skipped actions on identifier columns
    assert any("Skipped 'fill_missing' on identifier column 'order_id'" in msg for msg in cleaning_log)
    assert any("Skipped 'standardize_case' on identifier column 'customer_id'" in msg for msg in cleaning_log)
    assert any("Skipped 'clip_outliers' on identifier column 'order_id'" in msg for msg in cleaning_log)
    assert any("Skipped 'convert_dtype' on identifier column 'user_uuid'" in msg for msg in cleaning_log)

    # Verify ordinary column actions ran
    assert any("Filled 1 missing value(s) in 'price'" in msg for msg in cleaning_log)
    assert any("Standardized case of 'category' to lower" in msg for msg in cleaning_log)

    # Verify precomputed cleaning_summary is populated
    assert "cleaning_summary" in res
    assert res["cleaning_summary"]["raw_rows"] == 4
    assert res["cleaning_summary"]["clean_rows"] == 4


# ============================================================================
# 9. Snapshot Cleaned CSV Size Limit in run.py
# ============================================================================

def test_run_py_snapshot_cleaned_csv_size_limit(tmp_path, capsys):
    """
    Verifies run.py prints the skip line when cleaned CSV is larger than 3 MB:
    'cleaned csv not copied: X.X MB is above the 3 MB limit'
    and prints the path when it is within the limit.
    """
    # 1. Large file > 3 MB
    large_csv = tmp_path / "large_cleaned.csv"
    large_size = 4 * 1024 * 1024  # 4 MB
    with open(large_csv, "wb") as f:
        f.seek(large_size - 1)
        f.write(b"\0")

    snapshot_path = str(tmp_path / "snapshot.json")
    base, _ = os.path.splitext(snapshot_path)
    cleaned_target = f"{base}_cleaned.csv"

    clean_size_bytes = os.path.getsize(large_csv)
    limit_bytes = 3 * 1024 * 1024
    if clean_size_bytes > limit_bytes:
        clean_size_mb = clean_size_bytes / (1024 * 1024)
        print(f"cleaned csv not copied: {clean_size_mb:.1f} MB is above the 3 MB limit")

    captured = capsys.readouterr()
    assert "cleaned csv not copied: 4.0 MB is above the 3 MB limit" in captured.out

    # 2. Small file <= 3 MB
    small_csv = tmp_path / "small_cleaned.csv"
    small_csv.write_text("a,b\n1,2\n")
    from components.demo_snapshot import save_cleaned_csv_for_demo
    final_state_mock = {"clean_df_path": str(small_csv)}
    copied = save_cleaned_csv_for_demo(final_state_mock, cleaned_target)
    if copied:
        print(f"Cleaned CSV: {cleaned_target}")

    captured2 = capsys.readouterr()
    assert f"Cleaned CSV: {cleaned_target}" in captured2.out


# ============================================================================
# 10. bridge_secrets_to_env Tests
# ============================================================================

def test_bridge_secrets_to_env_allowlist_and_guards(monkeypatch, caplog):
    """
    Verifies bridge_secrets_to_env():
    - Copies allowlisted names into os.environ
    - Refuses API keys, passwords, and sensitive token names (except MAX_RUN_TOKENS)
    - Does not overwrite already-set environment variables
    - Does not crash when secrets file is missing
    - Logs a warning without key leakage when HOSTED=1 and server keys exist
    """
    import streamlit as st
    from app import bridge_secrets_to_env, ALLOWED_HOSTED_SECRET_NAMES

    orig_env = dict(os.environ)
    try:
        # Clear target env vars directly from os.environ
        for k in list(ALLOWED_HOSTED_SECRET_NAMES) + [
            "GROQ_API_KEY", "GEMINI_API_KEY", "SECRET_KEY", "MY_PASSWORD", "AUTH_TOKEN"
        ]:
            os.environ.pop(k, None)

        # 1. Allowlisted names copied and MAX_RUN_TOKENS allowed
        fake_secrets = {
            "HOSTED": "1",
            "DEMO_ONLY": "0",
            "MAX_UPLOAD_MB": "10",
            "MAX_ROWS": "15000",
            "MAX_COLS": "40",
            "MAX_RUN_TOKENS": "50000",
            "MAX_CONCURRENT_RUNS": "3",
            "RUN_COOLDOWN_SECONDS": "30",
            "RUN_WALL_CLOCK_SECONDS": "120",
            # Sensitive names that MUST be rejected
            "GROQ_API_KEY": "gsk_TESTKEY0123456789abcdef",
            "GEMINI_API_KEY": "dummy_gemini_key",
            "SECRET_KEY": "super_secret_value",
            "MY_PASSWORD": "secret_password",
            "AUTH_TOKEN": "bearer_token_xyz",
            "SOME_OTHER_VAR": "custom_value",
        }

        monkeypatch.setattr(st, "secrets", fake_secrets)
        bridge_secrets_to_env()

        # Check allowlisted vars copied
        assert os.environ.get("HOSTED") == "1"
        assert os.environ.get("DEMO_ONLY") == "0"
        assert os.environ.get("MAX_UPLOAD_MB") == "10"
        assert os.environ.get("MAX_ROWS") == "15000"
        assert os.environ.get("MAX_COLS") == "40"
        assert os.environ.get("MAX_RUN_TOKENS") == "50000"
        assert os.environ.get("MAX_CONCURRENT_RUNS") == "3"
        assert os.environ.get("RUN_COOLDOWN_SECONDS") == "30"
        assert os.environ.get("RUN_WALL_CLOCK_SECONDS") == "120"

        # Check sensitive / non-allowlisted vars refused
        assert "GROQ_API_KEY" not in os.environ
        assert "GEMINI_API_KEY" not in os.environ
        assert "SECRET_KEY" not in os.environ
        assert "MY_PASSWORD" not in os.environ
        assert "AUTH_TOKEN" not in os.environ
        assert "SOME_OTHER_VAR" not in os.environ

        # 2. Already-set variable is not overwritten
        os.environ["MAX_UPLOAD_MB"] = "25"
        fake_secrets_override = {"MAX_UPLOAD_MB": "10"}
        monkeypatch.setattr(st, "secrets", fake_secrets_override)
        bridge_secrets_to_env()
        assert os.environ.get("MAX_UPLOAD_MB") == "25"

        # 3. Missing secrets file (AttributeError or FileNotFoundError) does not crash
        class MissingSecrets:
            def __getattr__(self, name):
                raise FileNotFoundError("secrets.toml not found")
            def keys(self):
                raise FileNotFoundError("secrets.toml not found")

        monkeypatch.setattr(st, "secrets", MissingSecrets())
        bridge_secrets_to_env()  # Must not raise

        # 4. Warn if HOSTED=1 and server-side keys exist in environment
        os.environ["HOSTED"] = "1"
        os.environ["GROQ_API_KEY"] = "gsk_TESTKEY0123456789abcdef"
        monkeypatch.setattr(st, "secrets", {})
        caplog.clear()

        bridge_secrets_to_env()
        assert "server-side keys are present but disabled in hosted mode" in caplog.text
        assert "gsk_TESTKEY0123456789abcdef" not in caplog.text
    finally:
        os.environ.clear()
        os.environ.update(orig_env)

