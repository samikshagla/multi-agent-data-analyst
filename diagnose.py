"""
diagnose.py
Diagnostics Tool: Runs the Multi-Agent Data Analyst pipeline on a dataset (or data/sample.csv)
and inspects all items intercepted by the reviewer across all rounds.
Prints: question, reason, error type, original_value, recomputed_value,
original pandas code, and reviewer alternative code snippet.
"""

import os
import sys

# Windows UTF-8 console output setup
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
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "sample.csv")

    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at '{csv_path}'")
        sys.exit(1)

    print("=" * 70)
    print(f" DIAGNOSTICS: RUNNING PIPELINE ON {csv_path}")
    print("=" * 70)

    final_state = run_pipeline(csv_path, stream=False)
    caught_errors = final_state.get("caught_errors", [])
    review_report = final_state.get("review_report", [])
    run_stats = final_state.get("run_stats", {})

    print("\n" + "=" * 70)
    print(" REVIEWER AUDIT & DIAGNOSTIC REPORT")
    print("=" * 70)

    if not caught_errors:
        print("\n [OK] No items were flagged by the reviewer during this run!")
        print(" All analysis items verified cleanly on the first pass.")
    else:
        print(f"\n Found {len(caught_errors)} interception event(s):\n")

        for idx, err in enumerate(caught_errors, 1):
            item_id = err.get("id")
            q = err.get("question", "N/A")
            reason = err.get("reason", "N/A")
            err_type = err.get("type", "unknown")
            round_num = err.get("retry_round", 0)
            orig_val = err.get("original_value", "N/A")
            recomp_val = err.get("recomputed_value", "N/A")
            orig_code = err.get("original_code", "# None").strip()
            recomp_code = err.get("recomputed_code", "# None").strip()

            print(f"--- [Flag #{idx}] Round {round_num} | Item ID: {item_id} | Type: {err_type} ---")
            print(f"Question:         {q}")
            print(f"Flag Reason:      {reason}")
            print(f"Original Value:   {orig_val}")
            print(f"Recomputed Value: {recomp_val}\n")

            print("Original Analyst Code:")
            print("----------------------")
            print(orig_code)
            print()

            print("Reviewer Alternative Code:")
            print("--------------------------")
            print(recomp_code)
            print("=" * 70 + "\n")

    print("=" * 70)
    print(" SUMMARY METRICS")
    print("=" * 70)
    print(f"  Total Flag Events:        {run_stats.get('total_flag_events', len(caught_errors))}")
    print(f"  Flagged (First Pass):     {run_stats.get('items_flagged_first_pass', 0)}")
    print(f"  Fixed by Retry:           {run_stats.get('fixed_by_retry', 0)}")
    print(f"  Unresolved Remaining:     {run_stats.get('unresolved', 0)}")
    print(f"  Total Retries Used:       {run_stats.get('retries_used', 0)}")
    print(f"  Execution Time:           {run_stats.get('time_seconds', 0.0):.2f}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
