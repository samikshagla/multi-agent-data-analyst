"""
tests/test_multi_user.py
Unit tests verifying multi-user isolation, session-scoped LLM configuration,
thread-safe usage tracking, key redaction, DEMO_ONLY guard, and per-run workspaces.
"""

import concurrent.futures
import os
import shutil
import tempfile
import threading
import traceback
import pandas as pd
import pytest

from graph import build_graph, run_pipeline
from llm import (
    LLMConfig,
    UsageTracker,
    llm_session,
    call_llm,
    get_usage_stats,
    reset_usage_stats,
    redact,
    mask_key,
    LLMUnavailableError,
    LLMConfigError,
    LLMBudgetError,
    LLMDisabledError,
    _current_config,
    _current_tracker,
)
from state import new_state, AgentState
from tools.workspace import RunWorkspace, cleanup_stale_workspaces


# ============================================================================
# 1. Real Compiled Graph in 2 Threads with Isolated Keys & Usage Counters
# ============================================================================

def test_two_threads_real_compiled_graph_isolated_keys(monkeypatch):
    """
    Two threads run the REAL compiled graph (via build_graph with custom test nodes)
    with different keys. Each call sees only its own key, usage counters stay isolated,
    and os.environ is never touched or polluted.
    """
    key_a = "gsk_thread_alpha_11111111_xyz"
    key_b = "gsk_thread_beta_22222222_xyz"

    observed_keys_a = []
    observed_keys_b = []
    lock = threading.Lock()

    def fake_call_groq(prompt: str, api_key: str | None = None, model_name: str | None = None):
        with lock:
            if api_key == key_a:
                observed_keys_a.append(api_key)
            elif api_key == key_b:
                observed_keys_b.append(api_key)
        return '{"result": "ok"}', 10, 5

    monkeypatch.setattr("llm._call_groq", fake_call_groq)

    # Custom nodes calling call_llm inside the real graph
    def fake_cleaner(state: AgentState):
        res = call_llm("clean data")
        return {"clean_df_path": "cleaned.csv", "profile": {"rows": 10}}

    def fake_analyst(state: AgentState):
        res = call_llm("analyze data")
        return {
            "analysis_results": [
                {"id": 1, "question": "q1", "code": "x=1", "result": 1, "status": "ok"}
            ]
        }

    def fake_visualizer(state: AgentState):
        return {"charts": []}

    def fake_reviewer(state: AgentState):
        res = call_llm("review data")
        return {
            "review_report": [{"id": 1, "verdict": "pass", "reason": "matches"}],
            "review_feedback": [],
        }

    def fake_report(state: AgentState):
        return {"final_report": "# Test Report"}

    custom_nodes = {
        "cleaner": fake_cleaner,
        "analyst": fake_analyst,
        "visualizer": fake_visualizer,
        "reviewer": fake_reviewer,
        "report": fake_report,
    }
    app = build_graph(nodes=custom_nodes, max_retries=1)

    initial_env_groq = os.environ.get("GROQ_API_KEY")

    results = {}

    def run_graph_thread(thread_id: str, key: str):
        cfg = LLMConfig(provider="groq", model="test-model", api_key=key)
        with llm_session(cfg):
            state = new_state("test.csv")
            final_state = app.invoke(state)
            stats = get_usage_stats()
            results[thread_id] = {
                "final_state": final_state,
                "stats": stats,
            }

    t1 = threading.Thread(target=run_graph_thread, args=("A", key_a))
    t2 = threading.Thread(target=run_graph_thread, args=("B", key_b))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Verify each thread saw exactly 3 calls with its own key
    assert len(observed_keys_a) == 3
    assert all(k == key_a for k in observed_keys_a)
    assert len(observed_keys_b) == 3
    assert all(k == key_b for k in observed_keys_b)

    # Verify each thread's tracker counted only its own calls
    assert results["A"]["stats"]["calls"] == 3
    assert results["B"]["stats"]["calls"] == 3
    assert results["A"]["stats"]["input_tokens"] == 30
    assert results["B"]["stats"]["input_tokens"] == 30

    # Verify os.environ was not polluted
    assert os.environ.get("GROQ_API_KEY") == initial_env_groq
    assert key_a not in os.environ.values()
    assert key_b not in os.environ.values()


# ============================================================================
# 2. 8 Parallel Sessions in ThreadPoolExecutor: Zero Cross-Talk
# ============================================================================

def test_eight_parallel_sessions_zero_crosstalk(monkeypatch):
    """
    8 parallel worker sessions in ThreadPoolExecutor, each with a unique API key
    and wrapped in llm_session. Verifies zero key or metric leakage across sessions.
    """
    session_calls: dict[str, list[str]] = {}
    lock = threading.Lock()

    def fake_call_groq(prompt: str, api_key: str | None = None, model_name: str | None = None):
        with lock:
            session_calls.setdefault(api_key, []).append(prompt)
        return "reply", 20, 10

    monkeypatch.setattr("llm._call_groq", fake_call_groq)

    def worker(session_id: int):
        key = f"key_session_{session_id}_xxxxxxxxxx"
        cfg = LLMConfig(provider="groq", model="test-m", api_key=key, max_tokens=1000)
        with llm_session(cfg):
            for i in range(5):
                call_llm(f"session_{session_id}_msg_{i}")
            stats = get_usage_stats()
            return session_id, key, stats

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(8)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    for session_id, key, stats in results:
        # Each session made exactly 5 calls
        assert stats["calls"] == 5
        assert stats["total_tokens"] == 5 * (20 + 10)
        # All 5 calls for this key contained prompts matching this session_id
        recorded = session_calls[key]
        assert len(recorded) == 5
        assert all(f"session_{session_id}_" in p for p in recorded)


# ============================================================================
# 3. Key Redaction & Masking
# ============================================================================

def test_key_redaction_in_exceptions_and_tracebacks(monkeypatch):
    """
    A fake provider raising an error containing the secret key produces an
    LLMUnavailableError whose message, repr, and traceback text contain neither
    the key nor any 8+ character substring of it.
    Also verifies LLMConfig repr and str are masked.
    """
    secret_key = "gsk_prod_secret_9988776655443322_alpha"
    monkeypatch.setattr("time.sleep", lambda s: None)

    def faulty_groq(prompt: str, api_key: str | None = None, model_name: str | None = None):
        # Network error containing key and substring (not auth keyword so it retries and raises LLMUnavailableError)
        raise ConnectionError(f"Connection timed out contacting cluster with key {secret_key} and chunk {secret_key[9:24]}")

    monkeypatch.setattr("llm._call_groq", faulty_groq)

    cfg = LLMConfig(provider="groq", model="gpt-oss", api_key=secret_key)

    # 1. Verify LLMConfig repr and str are masked
    assert secret_key not in repr(cfg)
    assert secret_key not in str(cfg)
    assert "gsk_***" in repr(cfg)
    assert secret_key[-4:] in repr(cfg)

    # 2. Verify call_llm exception redaction produces LLMUnavailableError
    with llm_session(cfg):
        with pytest.raises(LLMUnavailableError) as exc_info:
            call_llm("test prompt")

        exc = exc_info.value
        msg = str(exc)
        r = repr(exc)
        tb_text = "".join(traceback.format_exception(exc))

        # Check full key
        assert secret_key not in msg
        assert secret_key not in r
        assert secret_key not in tb_text

        # Check any 8+ character substring of secret_key
        n = len(secret_key)
        for length in range(8, n):
            for i in range(n - length + 1):
                sub = secret_key[i:i + length]
                assert sub not in msg, f"Substring {sub} found in exception message"
                assert sub not in r, f"Substring {sub} found in exception repr"
                assert sub not in tb_text, f"Substring {sub} found in exception traceback"


# ============================================================================
# 4. LLMBudgetError Raised Before Over-Budget Call
# ============================================================================

def test_llm_budget_error_raised_before_exceeding_call(monkeypatch):
    """
    LLMBudgetError is raised BEFORE the call that would exceed or meet max_tokens.
    """
    call_count = 0

    def fake_groq(prompt: str, api_key: str | None = None, model_name: str | None = None):
        nonlocal call_count
        call_count += 1
        return "response", 30, 20  # 50 tokens each call

    monkeypatch.setattr("llm._call_groq", fake_groq)

    # Budget is 100 tokens
    cfg = LLMConfig(provider="groq", model="m", api_key="dummy_key_12345678", max_tokens=100)
    with llm_session(cfg):
        # Call 1: uses 50 tokens (tokens used so far = 50 < 100) -> succeeds
        call_llm("call 1")
        assert call_count == 1

        # Call 2: uses 50 tokens (tokens used so far = 100 >= 100) -> succeeds
        call_llm("call 2")
        assert call_count == 2

        # Call 3: check_budget() triggers BEFORE the call because total_tokens (100) >= max_tokens (100)
        with pytest.raises(LLMBudgetError) as exc_info:
            call_llm("call 3")

        assert "reached its token limit" in str(exc_info.value)
        # Call 3 never reached the provider
        assert call_count == 2


# ============================================================================
# 5. Nesting & Exception Safety of llm_session
# ============================================================================

def test_llm_session_nesting_and_exception_safety():
    """
    Verifies that nested llm_session blocks properly push and pop context,
    and that exceptions inside a session restore the prior context.
    """
    cfg_outer = LLMConfig("groq", "outer-m", "key_outer_12345678")
    cfg_inner = LLMConfig("gemini", "inner-m", "key_inner_87654321")

    assert _current_config.get() is None
    assert _current_tracker.get() is None

    with llm_session(cfg_outer) as trk_outer:
        assert _current_config.get() == cfg_outer
        assert _current_tracker.get() == trk_outer

        # Nested session
        with llm_session(cfg_inner) as trk_inner:
            assert _current_config.get() == cfg_inner
            assert _current_tracker.get() == trk_inner
            assert trk_inner is not trk_outer

        # Restored to outer
        assert _current_config.get() == cfg_outer
        assert _current_tracker.get() == trk_outer

        # Exception safety
        try:
            with llm_session(cfg_inner) as trk_inner2:
                assert _current_config.get() == cfg_inner
                raise ValueError("Inner failed")
        except ValueError:
            pass

        # Restored to outer even after exception
        assert _current_config.get() == cfg_outer
        assert _current_tracker.get() == trk_outer

    # Restored to None after outer block exits
    assert _current_config.get() is None
    assert _current_tracker.get() is None


# ============================================================================
# 6. CLI Path Works Unchanged (Environment Fallback & Output Keys)
# ============================================================================

def test_cli_path_works_unchanged(monkeypatch):
    """
    Outside llm_session, call_llm reads from environment and updates global tracker.
    get_usage_stats() retains the exact same top-level keys.
    """
    monkeypatch.setenv("GROQ_API_KEY", "cli_test_groq_key")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_MODEL", "cli-model")

    def fake_groq(prompt: str, api_key: str | None = None, model_name: str | None = None):
        key = api_key or os.getenv("GROQ_API_KEY")
        model = model_name or os.getenv("GROQ_MODEL")
        assert key == "cli_test_groq_key"
        assert model == "cli-model"
        return "cli answer", 15, 10

    monkeypatch.setattr("llm._call_groq", fake_groq)

    reset_usage_stats()
    res = call_llm("test cli prompt")
    assert res == "cli answer"

    stats = get_usage_stats()
    expected_keys = {
        "calls",
        "prompt_chars",
        "response_chars",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "by_model",
    }
    assert expected_keys.issubset(stats.keys())
    assert stats["calls"] == 1
    assert stats["total_tokens"] == 25
    assert "cli-model" in stats["by_model"]


# ============================================================================
# 7. DEMO_ONLY Guard
# ============================================================================

def test_demo_only_guard(monkeypatch):
    """
    When DEMO_ONLY=1 and no context config is set, call_llm raises LLMDisabledError.
    When a context config is set, it bypasses the guard.
    """
    monkeypatch.setenv("DEMO_ONLY", "1")
    monkeypatch.setattr("llm._call_groq", lambda p, **kw: ("ok", 1, 1))

    # Without context config -> raises LLMDisabledError
    with pytest.raises(LLMDisabledError) as exc_info:
        call_llm("hello")
    assert "disabled in demo mode" in str(exc_info.value).lower()

    # With context config -> succeeds
    cfg = LLMConfig("groq", "test-m", "gsk_visitor_key_12345678")
    with llm_session(cfg):
        res = call_llm("hello")
        assert res == "ok"


# ============================================================================
# 8. Concurrent Runs with Isolated Workspaces vs Default Path
# ============================================================================

def test_concurrent_runs_with_isolated_workspaces_and_cleanup(tmp_path):
    """
    Two concurrent pipeline runs with fake nodes write cleaned CSV, charts,
    and report into DIFFERENT workspace directories, never into data/cleaned or outputs.
    cleanup() removes all workspace files. Default (workspace_dir=None) writes to old paths.
    """
    raw_csv = str(tmp_path / "orders.csv")
    pd.DataFrame({"order_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]}).to_csv(raw_csv, index=False)

    ws1 = RunWorkspace()
    ws2 = RunWorkspace()

    try:
        # Check directories are separate
        assert ws1.workspace_dir != ws2.workspace_dir
        assert os.path.exists(ws1.cleaned_dir)
        assert os.path.exists(ws2.cleaned_dir)

        # Custom nodes simulating pipeline writes
        def mock_cleaner(state: AgentState):
            ws_dir = state.get("workspace_dir")
            target_dir = os.path.join(ws_dir, "cleaned") if ws_dir else os.path.join("data", "cleaned")
            os.makedirs(target_dir, exist_ok=True)
            clean_path = os.path.normpath(os.path.join(target_dir, "cleaned_orders.csv"))
            pd.DataFrame({"order_id": [1, 2], "amount": [10.0, 20.0]}).to_csv(clean_path, index=False)
            return {"clean_df_path": clean_path, "profile": {"rows": 2}}

        def mock_analyst(state: AgentState):
            return {"analysis_results": [{"id": 1, "question": "q1", "result": 100, "status": "ok"}]}

        def mock_visualizer(state: AgentState):
            ws_dir = state.get("workspace_dir")
            charts_dir = os.path.join(ws_dir, "charts") if ws_dir else os.path.join("outputs", "charts")
            os.makedirs(charts_dir, exist_ok=True)
            chart_path = os.path.normpath(os.path.join(charts_dir, "chart_1.html"))
            with open(chart_path, "w") as f:
                f.write("<html>Chart</html>")
            return {"charts": [{"id": 1, "title": "Chart 1", "type": "bar", "status": "ok"}]}

        def mock_reviewer(state: AgentState):
            return {"review_report": [{"id": 1, "verdict": "pass"}], "review_feedback": []}

        def mock_report(state: AgentState):
            ws_dir = state.get("workspace_dir")
            reports_dir = os.path.join(ws_dir, "reports") if ws_dir else "outputs"
            os.makedirs(reports_dir, exist_ok=True)
            report_path = os.path.normpath(os.path.join(reports_dir, "report_orders.md"))
            with open(report_path, "w") as f:
                f.write("# Report")
            return {"final_report": "# Report"}

        custom_nodes = {
            "cleaner": mock_cleaner,
            "analyst": mock_analyst,
            "visualizer": mock_visualizer,
            "reviewer": mock_reviewer,
            "report": mock_report,
        }
        app = build_graph(nodes=custom_nodes)

        # Run 1
        state1 = new_state(raw_csv, workspace_dir=ws1.workspace_dir)
        res1 = app.invoke(state1)

        # Run 2
        state2 = new_state(raw_csv, workspace_dir=ws2.workspace_dir)
        res2 = app.invoke(state2)

        # Verify Run 1 wrote to ws1
        assert os.path.exists(os.path.join(ws1.cleaned_dir, "cleaned_orders.csv"))
        assert os.path.exists(os.path.join(ws1.charts_dir, "chart_1.html"))
        assert os.path.exists(os.path.join(ws1.reports_dir, "report_orders.md"))

        # Verify Run 2 wrote to ws2
        assert os.path.exists(os.path.join(ws2.cleaned_dir, "cleaned_orders.csv"))
        assert os.path.exists(os.path.join(ws2.charts_dir, "chart_1.html"))
        assert os.path.exists(os.path.join(ws2.reports_dir, "report_orders.md"))

        # Verify neither contaminated each other
        assert res1["clean_df_path"].startswith(ws1.workspace_dir)
        assert res2["clean_df_path"].startswith(ws2.workspace_dir)

    finally:
        # Verify cleanup removes the workspace
        ws1_path = ws1.workspace_dir
        ws2_path = ws2.workspace_dir
        ws1.cleanup()
        ws2.cleanup()
        assert not os.path.exists(ws1_path)
        assert not os.path.exists(ws2_path)
