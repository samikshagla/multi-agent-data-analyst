"""
workspace.py
Per-run isolated workspace helper for multi-tenant and parallel executions.
Creates a unique temporary directory per execution run, manages subdirectories
for raw data, cleaned data, charts, and reports, and ensures automatic cleanup.
"""

import atexit
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path


class RunWorkspace:
    """
    Manages an isolated temporary workspace directory for a single execution run.
    Ensures that concurrent pipeline runs do not overwrite each other's datasets,
    charts, or generated reports.
    """

    def __init__(self, prefix: str = "mda_workspace_"):
        self.workspace_id = uuid.uuid4().hex
        self.dir_name = f"{prefix}{self.workspace_id}"
        self.workspace_dir = os.path.normpath(
            os.path.join(tempfile.gettempdir(), self.dir_name)
        )
        self.path = Path(self.workspace_dir)

        # Subdirectories
        self.raw_dir = os.path.normpath(os.path.join(self.workspace_dir, "raw"))
        self.cleaned_dir = os.path.normpath(os.path.join(self.workspace_dir, "cleaned"))
        self.charts_dir = os.path.normpath(os.path.join(self.workspace_dir, "charts"))
        self.reports_dir = os.path.normpath(os.path.join(self.workspace_dir, "reports"))

        # Create subdirectories
        for d in (self.raw_dir, self.cleaned_dir, self.charts_dir, self.reports_dir):
            os.makedirs(d, exist_ok=True)

        self._cleaned_up = False
        atexit.register(self.cleanup)

    def get_raw_path(self, filename: str) -> str:
        """Returns path for storing raw dataset copy."""
        return os.path.normpath(os.path.join(self.raw_dir, os.path.basename(filename)))

    def get_cleaned_path(self, filename: str) -> str:
        """Returns path for storing cleaned dataset output."""
        return os.path.normpath(os.path.join(self.cleaned_dir, os.path.basename(filename)))

    def get_chart_path(self, chart_id: int | str) -> str:
        """Returns path for storing an interactive HTML chart."""
        return os.path.normpath(os.path.join(self.charts_dir, f"chart_{chart_id}.html"))

    def get_report_path(self, dataset_name: str) -> str:
        """Returns path for storing the final markdown report."""
        return os.path.normpath(os.path.join(self.reports_dir, f"report_{dataset_name}.md"))

    def cleanup(self) -> None:
        """Deletes the entire workspace directory and its contents."""
        if not self._cleaned_up:
            self._cleaned_up = True
            try:
                atexit.unregister(self.cleanup)
            except Exception:
                pass
            if os.path.exists(self.workspace_dir):
                shutil.rmtree(self.workspace_dir, ignore_errors=True)

    def __enter__(self) -> "RunWorkspace":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def __repr__(self) -> str:
        return f"RunWorkspace(workspace_dir={self.workspace_dir!r}, active={not self._cleaned_up})"


def cleanup_stale_workspaces(max_age_seconds: int = 7200, prefix: str = "mda_workspace_") -> int:
    """
    Deletes orphaned workspace directories older than max_age_seconds (default 2 hours).
    Safe to call at application startup or on a periodic maintenance schedule.

    Returns:
        int: Number of stale workspaces removed.
    """
    temp_root = tempfile.gettempdir()
    now = time.time()
    removed_count = 0

    try:
        for entry in os.listdir(temp_root):
            if entry.startswith(prefix):
                full_path = os.path.join(temp_root, entry)
                if os.path.isdir(full_path):
                    try:
                        mtime = os.path.getmtime(full_path)
                        if now - mtime > max_age_seconds:
                            shutil.rmtree(full_path, ignore_errors=True)
                            removed_count += 1
                    except OSError:
                        pass
    except Exception:
        pass

    return removed_count
