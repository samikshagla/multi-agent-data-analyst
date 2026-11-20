"""
components/run_control.py
Guarded pipeline execution controller for hosted multi-tenant and local runs.
Enforces session cooldown, concurrency limits via RunGate, wall-clock timeout
via WallClock, and guarantees cleanup of gate tokens and temporary workspaces.
"""

from dataclasses import replace
import os
from pathlib import Path
import time
from typing import Any, Callable

from tools.limits import (
    RunGate,
    RunTimeout,
    SessionCooldown,
    Settings,
    WallClock,
    get_settings,
    global_run_gate,
    global_session_cooldown,
)


def run_guarded(
    settings: Settings,
    session_id: str,
    make_stream: Callable[[], Any],
    on_update: Callable[[Any], None] | None = None,
    workspace: Any | None = None,
    session_config: Any | None = None,
    gate: RunGate | None = None,
    cooldown: SessionCooldown | None = None,
    clock_fn: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """
    Executes a pipeline streaming generator protected by concurrency, cooldown,
    and wall-clock execution guards.

    Args:
        settings: Active Settings instance with resource and timing limits.
        session_id: Unique string identifier for the calling session/visitor.
        make_stream: Callable that returns the pipeline streaming generator.
        on_update: Optional callback invoked for each yielded streamed update.
        workspace: Optional RunWorkspace instance to clean up in finally.
        session_config: Optional LLMConfig for visitor-specific API keys and token budget.
        gate: RunGate instance (defaults to global_run_gate).
        cooldown: SessionCooldown instance (defaults to global_session_cooldown).
        clock_fn: Clock function returning seconds (defaults to time.time).

    Returns:
        dict[str, Any]: Execution result dict:
            {"status": "ok"|"cooldown"|"busy"|"timeout"|"error",
             "error": str | None,
             "final_state": dict | None}
    """
    cd = cooldown if cooldown is not None else global_session_cooldown
    g = gate if gate is not None else global_run_gate

    # 1. Cooldown Guard
    rem = cd.seconds_remaining(session_id, settings.run_cooldown_seconds)
    if rem > 0:
        return {
            "status": "cooldown",
            "error": f"Please wait {rem} second(s) before starting another analysis.",
            "final_state": None,
        }

    # 2. Concurrency Gate
    token = g.try_acquire(settings.max_concurrent_runs)
    if token is None:
        return {
            "status": "busy",
            "error": "The server is busy with other runs. Try again in a minute, or watch the recorded run.",
            "final_state": None,
        }

    # 3. Apply MAX_RUN_TOKENS to session config if present
    effective_config = session_config
    if effective_config is not None and settings.max_run_tokens is not None:
        curr_max = getattr(effective_config, "max_tokens", None)
        if curr_max is None or curr_max > settings.max_run_tokens:
            try:
                effective_config = replace(effective_config, max_tokens=settings.max_run_tokens)
            except Exception:
                pass

    # 4. Protected Execution Loop
    try:
        cd.mark(session_id)
        clock = WallClock(settings.run_wall_clock_seconds, clock_fn=clock_fn)

        def _execute_loop() -> dict[str, Any] | None:
            final_obj = None
            stream = make_stream()
            for item in stream:
                clock.check()
                if on_update:
                    on_update(item)

                if isinstance(item, tuple) and len(item) == 2:
                    node_name, state_update = item
                    if node_name == "__final__":
                        final_obj = state_update
                elif isinstance(item, dict) and "__final__" in item:
                    final_obj = item["__final__"]
            return final_obj

        if effective_config is not None:
            from llm import llm_session
            with llm_session(effective_config):
                final_state = _execute_loop()
        else:
            final_state = _execute_loop()

        # Capture cleaned CSV bytes into memory before workspace cleanup
        if final_state and isinstance(final_state, dict):
            clean_path = final_state.get("clean_df_path")
            if clean_path and os.path.exists(clean_path):
                try:
                    size_bytes = os.path.getsize(clean_path)
                    if size_bytes <= 6 * 1024 * 1024:
                        final_state["cleaned_csv_bytes"] = Path(clean_path).read_bytes()
                        final_state["cleaned_csv_oversized"] = False
                    else:
                        final_state["cleaned_csv_bytes"] = None
                        final_state["cleaned_csv_oversized"] = True
                except Exception:
                    pass

        return {
            "status": "ok",
            "error": None,
            "final_state": final_state,
        }

    except RunTimeout as ex:
        return {
            "status": "timeout",
            "error": str(ex),
            "final_state": None,
        }
    except Exception as ex:
        return {
            "status": "error",
            "error": str(ex),
            "final_state": None,
            "exception": ex,
        }
    finally:
        token.release()
        if workspace is not None:
            try:
                workspace.cleanup()
            except Exception:
                pass
