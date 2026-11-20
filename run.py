"""
run.py
CLI Pipeline Runner: Serves as the command-line entrypoint to execute the multi-agent
data analysis workflow end-to-end on local CSV files using live streaming event outputs.
Supports recording pipeline execution snapshots via --save-snapshot.

Usage:
    python run.py data/sample.csv
    python run.py data/sample.csv --save-snapshot demo/sample_run.json --force
"""

import argparse
import os
import sys
import time

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
from llm import LLMUnavailableError, LLMConfigError
from components.ui_state import format_local_timestamp, summarize_run


def main() -> None:
    """CLI entrypoint for running the Multi-Agent Data Analyst pipeline."""
    parser = argparse.ArgumentParser(description="Multi-Agent Data Analyst Pipeline Runner")
    parser.add_argument("csv_path", help="Path to the dataset CSV file")
    parser.add_argument(
        "--save-snapshot",
        dest="snapshot_path",
        default=None,
        help="Save execution snapshot to JSON file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing snapshot file",
    )

    args = parser.parse_args()
    csv_path = args.csv_path.strip()

    # Validate file existence
    if not os.path.exists(csv_path):
        print(f"Error: Dataset file not found at '{csv_path}'.")
        sys.exit(1)

    print("=" * 70)
    print(" MULTI-AGENT DATA ANALYST PIPELINE")
    print(f" Target Dataset: {csv_path}")
    if args.snapshot_path:
        print(f" Snapshot Target: {args.snapshot_path}")
    print("=" * 70)

    seen_events_count = 0
    final_stats: dict = {}
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    report_file_path = os.path.normpath(os.path.join("outputs", f"report_{dataset_name}.md"))

    start_time = time.time()
    recorded_steps: list[dict] = []
    final_state_obj: dict | None = None

    try:
        # Stream pipeline execution
        for node_name, state_update in run_pipeline(csv_path, stream=True):
            time_offset = time.time() - start_time

            # Capture final state when complete
            if node_name == "__final__":
                final_state_obj = state_update
                continue

            # Record step for potential snapshot
            if args.snapshot_path:
                step_copy = dict(state_update)
                step_copy["node_name"] = node_name
                step_copy["seconds"] = round(time_offset, 2)
                recorded_steps.append(step_copy)

            # Print new events live as they are registered
            if "events" in state_update:
                events = state_update["events"]
                new_events = events[seen_events_count:]
                for event in new_events:
                    ts = format_local_timestamp(event.get("timestamp", ""))
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
        by_model = final_stats.get("by_model", {})
        if by_model:
            for m, m_stats in by_model.items():
                print(f"  Model ({m}): {m_stats.get('calls', 0)} calls, {m_stats.get('tokens', m_stats.get('total_tokens', 0)):,} tokens")
            if len(by_model) > 1:
                print("Multiple models were used in this run")
        print(f"  Review Retries:           {final_stats.get('retries_used', 0)}")
        print(f"  Items Flagged (Round 1):  {final_stats.get('items_flagged_first_pass', 0)}")
        print(f"  Fixed by Retry:           {final_stats.get('fixed_by_retry', 0)}")
        print(f"  Unresolved Items:         {final_stats.get('unresolved', 0)}")
        print(f"  Total Flag Events:        {final_stats.get('total_flag_events', 0)}")

        if os.path.exists(report_file_path):
            print(f"  Saved Report:             {report_file_path}")
        print("=" * 70 + "\n")

        # Snapshot saving (only on 'success' or 'issues')
        if args.snapshot_path and final_state_obj:
            summary = summarize_run(final_state_obj)
            final_status = summary.get("final_status", "failed")

            if final_status in ("success", "issues"):
                from components.demo_snapshot import save_snapshot, save_cleaned_csv_for_demo
                save_snapshot(
                    final_state_obj,
                    recorded_steps,
                    args.snapshot_path,
                    force=args.force,
                )
                base, _ = os.path.splitext(args.snapshot_path)
                cleaned_csv_target = f"{base}_cleaned.csv"
                clean_path = final_state_obj.get("clean_df_path")
                if clean_path and os.path.exists(clean_path):
                    clean_size_bytes = os.path.getsize(clean_path)
                    limit_bytes = 3 * 1024 * 1024
                    if clean_size_bytes > limit_bytes:
                        clean_size_mb = clean_size_bytes / (1024 * 1024)
                        print(f"cleaned csv not copied: {clean_size_mb:.1f} MB is above the 3 MB limit")
                    else:
                        copied = save_cleaned_csv_for_demo(final_state_obj, cleaned_csv_target)
                        if copied:
                            print(f"Cleaned CSV: {cleaned_csv_target}")

                size_kb = os.path.getsize(args.snapshot_path) / 1024.0
                items_count = len(final_state_obj.get("analysis_results", []))
                retries_count = final_stats.get("retries_used", 0)
                tokens_count = final_stats.get("tokens", 0)
                print(f"Snapshot saved: {args.snapshot_path} ({size_kb:.1f} KB)")
                print(f"Snapshot summary: {items_count} items evaluated, {retries_count} retries, {tokens_count:,} tokens\n")
            else:
                print(f"Snapshot not saved: pipeline finished with status '{final_status}'.\n")

    except (LLMUnavailableError, LLMConfigError) as e:
        print(f"\n[AI SERVICE ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline execution halted: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
