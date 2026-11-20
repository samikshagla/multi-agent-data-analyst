"""
tools/limits.py
Server-side limits, concurrency gate, session cooldown, and upload validation
for public hosted deployment. Pure Python with zero Streamlit dependencies.
"""

from dataclasses import dataclass
import os
import threading
import time
from typing import Any, Callable
import pandas as pd


class RunTimeout(Exception):
    """Raised when an execution run exceeds the configured wall-clock time limit."""
    pass


@dataclass(frozen=True)
class Settings:
    """Frozen dataclass containing runtime configuration and resource limits."""
    hosted: bool
    max_upload_mb: int
    max_rows: int
    max_cols: int
    max_run_tokens: int | None
    max_concurrent_runs: int
    run_cooldown_seconds: int
    run_wall_clock_seconds: int | None


def get_settings() -> Settings:
    """
    Reads configuration settings dynamically from environment variables on every call,
    allowing tests and deployment environments to adjust limits via monkeypatching.
    """
    hosted = os.environ.get("HOSTED") == "1"

    # MAX_UPLOAD_MB: default 5 when hosted, 20 locally
    if "MAX_UPLOAD_MB" in os.environ:
        max_upload_mb = int(os.environ["MAX_UPLOAD_MB"])
    else:
        max_upload_mb = 5 if hosted else 20

    # MAX_ROWS: default 30,000 when hosted, 200,000 locally
    if "MAX_ROWS" in os.environ:
        max_rows = int(os.environ["MAX_ROWS"])
    else:
        max_rows = 30_000 if hosted else 200_000

    # MAX_COLS: default 60 when hosted, 100 locally
    if "MAX_COLS" in os.environ:
        max_cols = int(os.environ["MAX_COLS"])
    else:
        max_cols = 60 if hosted else 100

    # MAX_RUN_TOKENS: default 80,000 when hosted, None locally
    if "MAX_RUN_TOKENS" in os.environ:
        val = os.environ["MAX_RUN_TOKENS"].strip()
        max_run_tokens = int(val) if val and val.lower() != "none" else None
    else:
        max_run_tokens = 80_000 if hosted else None

    # MAX_CONCURRENT_RUNS: default 2
    if "MAX_CONCURRENT_RUNS" in os.environ:
        max_concurrent_runs = int(os.environ["MAX_CONCURRENT_RUNS"])
    else:
        max_concurrent_runs = 2

    # RUN_COOLDOWN_SECONDS: default 60 when hosted, 0 locally
    if "RUN_COOLDOWN_SECONDS" in os.environ:
        run_cooldown_seconds = int(os.environ["RUN_COOLDOWN_SECONDS"])
    else:
        run_cooldown_seconds = 60 if hosted else 0

    # RUN_WALL_CLOCK_SECONDS: default 600 when hosted, None locally
    if "RUN_WALL_CLOCK_SECONDS" in os.environ:
        val = os.environ["RUN_WALL_CLOCK_SECONDS"].strip()
        run_wall_clock_seconds = int(val) if val and val.lower() != "none" else None
    else:
        run_wall_clock_seconds = 600 if hosted else None

    return Settings(
        hosted=hosted,
        max_upload_mb=max_upload_mb,
        max_rows=max_rows,
        max_cols=max_cols,
        max_run_tokens=max_run_tokens,
        max_concurrent_runs=max_concurrent_runs,
        run_cooldown_seconds=run_cooldown_seconds,
        run_wall_clock_seconds=run_wall_clock_seconds,
    )


# ============================================================================
# Concurrency Gate
# ============================================================================

class _GateContext:
    """Context manager for acquiring and safely releasing a RunGate slot."""

    def __init__(self, gate: "RunGate"):
        self._gate = gate
        self._released = False

    def release(self) -> None:
        """Idempotently releases the acquired gate slot."""
        if not self._released:
            self._released = True
            self._gate._release()

    def __enter__(self) -> "_GateContext":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class RunGate:
    """
    Process-wide, thread-safe, non-blocking concurrency limiter.
    Ensures that at most MAX_CONCURRENT_RUNS execute simultaneously.
    Slots are ALWAYS released on normal completion and exceptions.
    """

    def __init__(self, max_concurrent: int | None = None):
        self._lock = threading.Lock()
        self._count = 0
        self._max_concurrent = max_concurrent

    def active_runs(self) -> int:
        """Returns the number of currently active run slots."""
        with self._lock:
            return self._count

    def try_acquire(self, max_concurrent: int | None = None) -> _GateContext | None:
        """
        Attempts to non-blockingly acquire a run slot.
        Returns a context manager if acquired, or None if at capacity.
        """
        limit = (
            max_concurrent
            if max_concurrent is not None
            else (self._max_concurrent if self._max_concurrent is not None else get_settings().max_concurrent_runs)
        )
        with self._lock:
            if self._count < limit:
                self._count += 1
                return _GateContext(self)
            return None

    def _release(self) -> None:
        """Internal callback to release an active slot."""
        with self._lock:
            if self._count > 0:
                self._count -= 1


# Process-wide default RunGate instance
global_run_gate = RunGate()


# ============================================================================
# Session Cooldown
# ============================================================================

class SessionCooldown:
    """
    Thread-safe per-session cooldown tracker.
    Prevents a single session/visitor from launching back-to-back runs.
    Automatically purges entries older than one hour.
    """

    def __init__(self, purge_after_seconds: float = 3600.0, clock_fn: Callable[[], float] = time.time):
        self._lock = threading.Lock()
        self._timestamps: dict[str, float] = {}
        self._purge_after = purge_after_seconds
        self._clock = clock_fn

    def _purge_old_entries(self, now: float) -> None:
        """Purges records older than purge_after_seconds (must be called with _lock)."""
        cutoff = now - self._purge_after
        stale_keys = [k for k, ts in self._timestamps.items() if ts < cutoff]
        for k in stale_keys:
            del self._timestamps[k]

    def seconds_remaining(self, session_id: str, cooldown_seconds: int | None = None) -> int:
        """
        Returns the number of seconds remaining in the session's cooldown period,
        or 0 if the cooldown has elapsed or session has not run yet.
        """
        if not session_id:
            return 0

        target_cooldown = (
            cooldown_seconds
            if cooldown_seconds is not None
            else get_settings().run_cooldown_seconds
        )
        if target_cooldown <= 0:
            return 0

        now = self._clock()
        with self._lock:
            self._purge_old_entries(now)
            last_run = self._timestamps.get(session_id)
            if last_run is None:
                return 0
            elapsed = now - last_run
            remaining = int(target_cooldown - elapsed)
            return max(0, remaining)

    def mark(self, session_id: str) -> None:
        """Records a run completion timestamp for the given session_id."""
        if not session_id:
            return
        now = self._clock()
        with self._lock:
            self._purge_old_entries(now)
            self._timestamps[session_id] = now


# Process-wide default SessionCooldown instance
global_session_cooldown = SessionCooldown()


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_upload_size(n_bytes: int, settings: Settings | None = None) -> str | None:
    """
    Validates that uploaded file byte size is within configured MAX_UPLOAD_MB limits.
    Returns a friendly error message or None if valid.
    """
    s = settings or get_settings()
    max_bytes = s.max_upload_mb * 1024 * 1024
    if n_bytes > max_bytes:
        size_mb = n_bytes / (1024 * 1024)
        return (
            f"File size ({size_mb:.1f} MB) exceeds the {s.max_upload_mb} MB limit. "
            "Please upload a smaller dataset."
        )
    return None


def validate_dataframe(df: pd.DataFrame | None, settings: Settings | None = None) -> str | None:
    """
    Validates that a parsed DataFrame meets row, column, and non-empty criteria.
    Returns a friendly error message or None if valid.
    """
    s = settings or get_settings()

    if df is None or len(df) == 0:
        return "The dataset contains only headers and no data rows."

    if len(df) == 1:
        return (
            "The dataset contains only 1 row. Multi-agent analysis requires at least "
            "2 rows to analyze trends."
        )

    if len(df.columns) <= 1:
        return (
            "The dataset has only 1 column. Multi-agent analysis requires at least "
            "2 columns to uncover relationships."
        )

    if len(df) > s.max_rows:
        return (
            f"Dataset exceeds row limit: {len(df):,} rows found "
            f"(maximum allowed is {s.max_rows:,} rows)."
        )

    if len(df.columns) > s.max_cols:
        return (
            f"Dataset exceeds column limit: {len(df.columns)} columns found "
            f"(maximum allowed is {s.max_cols} columns)."
        )

    return None


# ============================================================================
# WallClock Timeout Guard
# ============================================================================

class WallClock:
    """
    Wall-clock execution timer.
    check() raises RunTimeout when elapsed time exceeds limit_seconds.
    No-op when limit_seconds is None.
    """

    def __init__(
        self,
        limit_seconds: float | int | None,
        clock_fn: Callable[[], float] = time.time,
    ):
        self.limit_seconds = float(limit_seconds) if limit_seconds is not None else None
        self._clock = clock_fn
        self.start_time = self._clock()

    def check(self) -> None:
        """Raises RunTimeout if elapsed wall-clock seconds exceed the limit."""
        if self.limit_seconds is not None:
            elapsed = self._clock() - self.start_time
            if elapsed > self.limit_seconds:
                limit_int = int(self.limit_seconds)
                raise RunTimeout(
                    f"Analysis timed out after {limit_int} seconds. "
                    "Please try with a smaller dataset or fewer questions."
                )
