"""
run.py
CLI Pipeline Runner: Serves as the command-line entrypoint to execute the multi-agent
data analysis workflow end-to-end on local CSV files using live streaming event outputs.

Usage:
    python run.py data/sample.csv
"""

import os
import sys

# Configure stdout and stderr to handle UTF-8 cleanly on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from graph import run_pipeline


def main() -> None:
    """CLI entrypoint for running the Multi-Agent Data Analyst pipeline."""
    # Validate argument presence
    if len(sys.argv) < 2:
        print("Error: Missing dataset path.")
        print("Usage: python run.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1].strip()

    # Validate file existence
    if not os.path.exists(csv_path):
        print(f"Error: Dataset file not found at '{csv_path}'.")
        sys.exit(1)

    print("=" * 70)
    print(" MULTI-AGENT DATA ANALYST PIPELINE")
    print(f" Target Dataset: {csv_path}")
    print("=" * 70)

    seen_events_count = 0
    final_stats: dict = {}
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    report_file_path = os.path.normpath(os.path.join("outputs", f"report_{dataset_name}.md"))

    try:
        # Stream pipeline execution
        for node_name, state_update in run_pipeline(csv_path, stream=True):
            # Print new events live as they are registered
            if "events" in state_update:
                events = state_update["events"]
                new_events = events[seen_events_count:]
                for event in new_events:
                    ts = event.get("timestamp", "").split("T")[-1][:8]  # HH:MM:SS
                    agent = event.get("agent", "system").upper()
                    etype = event.get("type", "info").upper()
                    msg = event.get("message", "")
                    print(f"[{ts}] [{agent:<10}] [{etype:<7}] {msg}")
                seen_events_count = len(events)

            # Capture stats
            if "run_stats" in state_update and state_update["run_stats"]:
                final_stats = state_update["run_stats"]

        # Final Summary Card
        print("\n" + "=" * 70)
        print(" EXECUTION SUMMARY")
        print("=" * 70)
        print(f"  Duration:                 {final_stats.get('time_seconds', 0.0):.2f} seconds")
        print(f"  LLM Invocations:          {final_stats.get('llm_calls', 0)}")
        print(f"  Token Usage:              {final_stats.get('tokens', 0):,} tokens")
        print(f"  Review Retries:           {final_stats.get('retries_used', 0)}")
        print(f"  Items Flagged (Round 1):  {final_stats.get('items_flagged_first_pass', 0)}")
        print(f"  Fixed by Retry:           {final_stats.get('fixed_by_retry', 0)}")
        print(f"  Unresolved Items:         {final_stats.get('unresolved', 0)}")
        print(f"  Total Flag Events:        {final_stats.get('total_flag_events', 0)}")

        if os.path.exists(report_file_path):
            print(f"  Saved Report:             {report_file_path}")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline execution halted: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
