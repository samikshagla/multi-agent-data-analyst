"""
test_sandbox_posix.py
POSIX-specific sandbox resource limits and environment isolation tests.
These tests run under POSIX operating systems (Linux CI / Streamlit Cloud)
where resource limits (RLIMIT_AS, RLIMIT_CPU, RLIMIT_NOFILE, RLIMIT_FSIZE)
and strict child environment isolation are enforced.
Skipped on Windows platforms.
"""

import multiprocessing
import os
import sys
from pathlib import Path
import pandas as pd
import pytest

from tools.sandbox import _sandbox_worker, run_code

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-only resource limit and environment isolation tests",
)


@pytest.fixture
def sample_8k_df() -> pd.DataFrame:
    """Loads the real 8,000-row sample dataset for heavy snippet validation."""
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample.csv"
    if sample_path.exists():
        return pd.read_csv(sample_path)
    # Fallback synthetic DataFrame matching columns if file is not found
    return pd.DataFrame({
        "metro_city": ["London", "Paris", "Berlin", "New York"] * 2000,
        "cuisine_category": ["Italian", "Burger", "Asian", "Mexican"] * 2000,
        "order_value_usd": [25.5, 30.0, 15.2, 45.0] * 2000,
        "customer_rating": [4.5, 4.2, 4.8, 3.9] * 2000,
    })


def test_posix_legal_heavy_snippet(sample_8k_df: pd.DataFrame):
    """
    Verifies that a legal, computationally intensive analyst snippet
    (groupby, pivot_table, merge on the 8,000-row dataset) executes
    successfully with status 'ok' under POSIX resource limits (~1.5GB RLIMIT_AS).
    """
    code = """
g = df.groupby(["metro_city", "cuisine_category"])["order_value_usd"].agg(["sum", "mean", "count"]).reset_index()
p = df.pivot_table(index="metro_city", columns="cuisine_category", values="customer_rating", aggfunc="mean").reset_index()
m = pd.merge(g, p, on="metro_city")
result = m
"""
    res = run_code(code, sample_8k_df, timeout=10)
    assert res["status"] == "ok", f"Legal heavy snippet failed: {res.get('error')}"
    assert isinstance(res["result"], pd.DataFrame)
    assert len(res["result"]) > 0


def test_posix_import_pandas_numpy_no_thread_or_blas_error(sample_8k_df: pd.DataFrame):
    """
    Verifies that importing pandas and numpy in the child process does not fail
    with pthread_create, OpenBLAS, or threading allocation errors under POSIX constraints.
    """
    code = """
import pandas as pd
import numpy as np
vals = np.array(df["order_value_usd"].dropna())
result = float(np.mean(vals))
"""
    res = run_code(code, sample_8k_df, timeout=10)
    err = str(res.get("error", ""))
    assert "pthread_create" not in err, f"pthread_create thread exhaustion error detected: {err}"
    assert "OpenBLAS" not in err, f"OpenBLAS initialization error detected: {err}"
    assert res["status"] == "ok", f"Import snippet failed: {err}"
    assert isinstance(res["result"], float)


def test_posix_memory_bomb_stopped_and_parent_recovers(sample_8k_df: pd.DataFrame):
    """
    Verifies that an attempted memory bomb is stopped quickly with status 'error',
    'blocked', or 'timeout', the parent process stays completely healthy, and subsequent
    legal snippets execute without issue.
    """
    code_bomb = "result = [x for x in range(10**9)]"
    res_bomb = run_code(code_bomb, sample_8k_df, timeout=2)
    assert res_bomb["status"] in ("error", "blocked", "timeout")
    assert res_bomb["status"] != "ok"

    # Parent process health check: subsequent legal code must run immediately and succeed
    code_legal = "result = float(df['order_value_usd'].mean())"
    res_legal = run_code(code_legal, sample_8k_df, timeout=10)
    assert res_legal["status"] == "ok", f"Subsequent legal run failed after bomb: {res_legal.get('error')}"
    assert isinstance(res_legal["result"], float)


def test_posix_child_empty_env_and_temp_cwd(sample_8k_df: pd.DataFrame):
    """
    Verifies that the child process worker runs with an empty environment
    (no PATH, no API keys like GROQ_API_KEY / GEMINI_API_KEY) and an isolated
    temporary working directory.
    """
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    # Pass inspection code directly to _sandbox_worker to inspect the child's runtime state
    inspect_code = """
import os
env_snapshot = dict(os.environ)
current_dir = os.getcwd()
result = pd.DataFrame([{
    "env_size": len(env_snapshot),
    "has_path": "PATH" in env_snapshot,
    "has_groq": "GROQ_API_KEY" in env_snapshot,
    "has_gemini": "GEMINI_API_KEY" in env_snapshot,
    "cwd": current_dir,
}])
"""
    p = ctx.Process(
        target=_sandbox_worker,
        args=(child_conn, inspect_code, sample_8k_df.head(5), 10),
    )
    p.start()

    assert parent_conn.poll(10), "Worker process timed out inspecting environment."
    output = parent_conn.recv()
    p.join(timeout=2)
    parent_conn.close()

    assert output["status"] == "ok", f"Worker environment inspection failed: {output.get('error')}"
    res_df = output["result"]
    assert int(res_df.loc[0, "env_size"]) == 0, f"Environment was not empty: {res_df.loc[0, 'env_size']} vars"
    assert bool(res_df.loc[0, "has_path"]) is False, "PATH was leaked into child environment"
    assert bool(res_df.loc[0, "has_groq"]) is False, "GROQ_API_KEY was leaked into child environment"
    assert bool(res_df.loc[0, "has_gemini"]) is False, "GEMINI_API_KEY was leaked into child environment"
    assert "sandbox_" in str(res_df.loc[0, "cwd"]), f"Working directory is not sandbox temp dir: {res_df.loc[0, 'cwd']}"
