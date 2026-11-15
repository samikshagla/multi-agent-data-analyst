"""
diagnose_cases.py
CLI tool to diagnose specific evaluation cases (e.g. missed wrong answers, false alarms).
Runs the reviewer in isolation on specified cases across repeats and prints comprehensive
diagnostic details:
- Question
- Analyst's Code and Result
- Certified Reference Answer
- Reviewer's Recomputation Code and Result
- Comparison Trace (branch and reason)
- Verdict
- Reviewer Recomputation Correctness vs Reference
Prints a summary table across all repeats at the end.

Usage:
    python -m eval.diagnose_cases q18_v1 q10_correct q11_correct [--repeats 3]
"""

import argparse
import os
import sys
from typing import Any, Callable
import pandas as pd

from agents.reviewer import reviewer_node
from eval.benchmark import BENCHMARK
from eval.ground_truth import compute_ground_truth, load_ground_truth, save_ground_truth
from eval.run_reviewer_eval import _assert_no_label_leakage, build_evaluation_cases
from eval.scoring import answer_correct
from tools.sandbox import run_code


def _format_display_val(val: Any, max_rows: int = 10) -> str:
    """Formats an analytical result for legible console display."""
    if val is None:
        return "None"
    if isinstance(val, (int, float, str, bool)):
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)
    if isinstance(val, pd.Series):
        df_sub = val.reset_index().head(max_rows)
        return df_sub.to_string(index=False)
    if isinstance(val, pd.DataFrame):
        df_sub = val.head(max_rows)
        trunc = f"\n  (showing first {max_rows} of {len(val)} rows)" if len(val) > max_rows else ""
        return df_sub.to_string(index=False) + trunc
    return str(val)


def _match_case(target: str, cases: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Matches a user-provided case ID to a benchmark case dict."""
    target_clean = target.strip().lower()

    # Exact base_case_id match
    for c in cases:
        if c.get("base_case_id", "").lower() == target_clean:
            return c

    # Normalize integer padding (e.g. q18_v1 vs q18_v01 or q1_v1 vs q01_v1)
    for c in cases:
        base_id = c.get("base_case_id", "").lower()
        if base_id.replace("0", "") == target_clean.replace("0", ""):
            return c

    return None


def diagnose_cases(
    case_ids: list[str],
    repeats: int = 3,
    csv_path: str = "data/sample.csv",
    reviewer_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Executes diagnosis on specified case IDs across repeats.
    Returns structured results for each repeat and prints full diagnostics.
    """
    if reviewer_fn is None:
        reviewer_fn = reviewer_node

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at '{csv_path}'")

    df = pd.read_csv(csv_path)

    # 1. Load ground truth
    gt_path = "eval/ground_truth.json"
    if os.path.exists(gt_path):
        ground_truth = load_ground_truth(gt_path)
    else:
        ground_truth = compute_ground_truth(BENCHMARK, csv_path)
        save_ground_truth(ground_truth, gt_path)

    # 2. Build base evaluation cases
    base_cases = build_evaluation_cases(BENCHMARK, ground_truth, repeats=1)

    matched_cases: list[dict[str, Any]] = []
    for cid in case_ids:
        c = _match_case(cid, base_cases)
        if c is not None:
            matched_cases.append(c)
        else:
            print(f"Warning: Case ID '{cid}' not found in benchmark.", file=sys.stderr)

    if not matched_cases:
        print("No valid cases found to diagnose.", file=sys.stderr)
        return []

    print("\n" + "=" * 80)
    print("REVIEWER DIAGNOSTICS: CASE ANALYSIS")
    print(f"Dataset:  {csv_path}")
    print(f"Repeats:  {repeats}")
    print(f"Cases:    {', '.join(c['base_case_id'] for c in matched_cases)}")
    print("=" * 80)

    summary_records: list[dict[str, Any]] = []

    for case in matched_cases:
        case_id = case["base_case_id"]
        q = case["question"]
        ref_ans = case["reference_answer"]

        # Run analyst code in sandbox
        analyst_res = run_code(case["code"], df.copy(deep=True))
        if analyst_res.get("status") != "ok":
            analyst_result = None
            analyst_status = "failed"
        else:
            analyst_result = analyst_res.get("result")
            analyst_status = "ok"

        # Build clean state without label leakage
        state = {
            "analysis_results": [
                {
                    "id": 0,
                    "question": q,
                    "code": case["code"],
                    "result": analyst_result,
                    "status": analyst_status,
                }
            ],
            "clean_df_path": csv_path,
            "charts": [],
            "retry_count": 0,
            "events": [],
        }
        _assert_no_label_leakage(state)

        case_repeat_verdicts: list[str] = []
        case_recomp_correct_count = 0

        for r in range(1, repeats + 1):
            print("\n" + "-" * 80)
            print(f"CASE: {case_id} (Repeat {r}/{repeats})")
            print("-" * 80)
            print(f"Question:\n  {q}\n")
            print(f"Analyst Code:\n  {case['code']}\n")
            print(f"Analyst Result:\n{_format_display_val(analyst_result)}\n")
            print(f"Reference Answer:\n{_format_display_val(ref_ans)}\n")

            rev_out = reviewer_fn(dict(state))
            report_list = rev_out.get("review_report", [])
            rep = report_list[0] if report_list else {}

            verdict = rep.get("verdict", "unverified")
            recomp_code = rep.get("recomputed_code") or rep.get("recomputation_code") or ""
            recomp_result = rep.get("recomputed_result")
            trace = rep.get("comparison_trace") or rep.get("trace") or {}

            # Check if reviewer's own recomputed answer is correct against reference
            ok_recomp = False
            recomp_reason = "No recomputed result"
            if recomp_result is not None:
                try:
                    ok_recomp, recomp_reason = answer_correct(ref_ans, recomp_result)
                except Exception as eval_err:
                    ok_recomp = False
                    recomp_reason = f"Scoring error: {eval_err}"

            case_repeat_verdicts.append(verdict)
            if ok_recomp:
                case_recomp_correct_count += 1

            trace_branch = trace.get("branch", "unknown") if isinstance(trace, dict) else str(trace)
            trace_reason = trace.get("reason", "") if isinstance(trace, dict) else ""

            print(f"Reviewer Recomputation Code:\n  {recomp_code}\n")
            print(f"Reviewer Recomputed Result:\n{_format_display_val(recomp_result)}\n")
            print(f"Comparison Trace:\n  Branch: {trace_branch}\n  Reason: {trace_reason}\n")
            print(f"Verdict:\n  {verdict}\n")
            print(f"Reviewer Recompute Correct vs Reference:\n  {ok_recomp} ({recomp_reason})\n")

        summary_records.append({
            "case_id": case_id,
            "verdicts": case_repeat_verdicts,
            "recomp_correct_count": case_recomp_correct_count,
            "total_repeats": repeats,
        })

    # Print summary table
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY TABLE")
    print("=" * 80)
    print(f"| {'Case ID':<16} | {'Verdicts Across Repeats':<28} | {'Recompute Correct':<20} |")
    print(f"| {'-' * 16} | {'-' * 28} | {'-' * 20} |")
    for s in summary_records:
        verdicts_str = ", ".join(s["verdicts"])
        correct_str = f"{s['recomp_correct_count']}/{s['total_repeats']}"
        print(f"| {s['case_id']:<16} | {verdicts_str:<28} | {correct_str:<20} |")
    print("=" * 80 + "\n")

    return summary_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose specific evaluation cases for reviewer agent.")
    parser.add_argument("cases", nargs="+", help="Case IDs to diagnose, e.g. q18_v1 q10_correct q11_correct")
    parser.add_argument("--repeats", type=int, default=3, help="Number of times to repeat each case (default 3)")
    parser.add_argument("--csv-path", type=str, default="data/sample.csv", help="Path to sample dataset")

    args = parser.parse_args()
    diagnose_cases(args.cases, repeats=args.repeats, csv_path=args.csv_path)


if __name__ == "__main__":
    main()
