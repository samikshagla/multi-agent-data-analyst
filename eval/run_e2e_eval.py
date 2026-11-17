"""
eval/run_e2e_eval.py
End-to-End Three-System Comparison Benchmark with Paired Analysis.

Compares three systems on a standardized benchmark subset of data/sample.csv:
- System A (single prompt): One LLM call with analyst's code prompt, runs once in sandbox, NO retries, NO reviewer.
- System B (analyst): analyst_node with self-repair retries on code errors.
- System C (analyst + reviewer loop): Starting from B's first-pass result, runs reviewer_node and
  iterates with analyst_node on review_feedback until pass, unverified, or max 3 retries.

Features:
- Paired analysis: B vs C (both correct, fixed by reviewer, harmed by reviewer, both wrong) and A vs B.
- Wilson score confidence intervals computed on UNIQUE benchmark items (wilson_ci_from_rate).
- Resumable per (item, repeat) with progress saved immediately to cases.jsonl.
- Clean pause on BudgetExceeded, LLMUnavailableError, and LLMConfigError with exact resume command.
- Summarize-only mode (--summarize-only) to regenerate summary.md and summary.json with zero LLM calls.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable
import numpy as np
import pandas as pd

from agents.analyst import _clean_code_fence, analyst_node
from agents.reviewer import reviewer_node
from eval.benchmark import BENCHMARK
from eval.budget import BudgetExceeded, TokenBudget
from eval.ground_truth import load_ground_truth
from eval.metrics import wilson_ci_from_rate
from eval.scoring import answer_correct
from llm import get_usage_stats, call_llm, LLMConfigError, LLMUnavailableError
from tools.profiler import profile_dataframe
from tools.sandbox import run_code


# Default evaluation subset: 12 benchmark question IDs
DEFAULT_ITEMS = [1, 4, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]

# Forbidden keys in AgentState to guarantee zero ground-truth leakage
FORBIDDEN_LEAK_KEYS = {
    "label",
    "kind",
    "is_wrong",
    "variant",
    "wrong_variant",
    "wrong_variants",
    "reference_answer",
    "reference_code",
    "certified_answer",
    "ground_truth",
}


# ============================================================================
# Helpers: Sanitization, Label Leak Guard & Prompt Formatting
# ============================================================================

def _assert_no_label_leakage(state: dict[str, Any]) -> None:
    """Verifies that no ground-truth label or reference answer is present in state."""
    for key in FORBIDDEN_LEAK_KEYS:
        if key in state:
            raise ValueError(f"CRITICAL EVALUATION LEAK: State contains forbidden benchmark ground-truth key: '{key}'")
    for r in state.get("analysis_results", []):
        if isinstance(r, dict):
            for key in FORBIDDEN_LEAK_KEYS:
                if key in r:
                    raise ValueError(f"CRITICAL EVALUATION LEAK: analysis_results contains forbidden key: '{key}'")


def _to_json_friendly(val: Any, max_rows: int = 50) -> Any:
    """
    Safely converts arbitrary values (DataFrames, Series, numpy arrays/scalars,
    pandas Timestamps/Intervals/Periods, dicts, lists) into JSON-serializable structures.
    DataFrames are stored in the same JSON format as eval/ground_truth.py:
        {"columns": [...], "rows": [[...]]}
    capped at max_rows (default 50) with a 'truncated' boolean flag.
    Anything unknown becomes str(val). It must never raise.
    """
    try:
        if val is None:
            return None

        # 1. Pandas DataFrame
        if isinstance(val, pd.DataFrame):
            total_rows = len(val)
            is_truncated = bool(total_rows > max_rows)
            df_sub = val.iloc[:max_rows]
            columns = [str(c) for c in df_sub.columns]
            rows: list[list[Any]] = []
            for row in df_sub.itertuples(index=False):
                rows.append([_to_json_friendly(cell, max_rows=max_rows) for cell in row])
            return {
                "columns": columns,
                "rows": rows,
                "total_rows": total_rows,
                "truncated": is_truncated,
            }

        # 2. Pandas Series
        if isinstance(val, pd.Series):
            try:
                df_sub = val.reset_index()
            except Exception:
                df_sub = pd.DataFrame({"index": list(val.index), "value": list(val.values)})
            return _to_json_friendly(df_sub, max_rows=max_rows)

        # 3. NumPy ndarray
        if isinstance(val, np.ndarray):
            return [_to_json_friendly(x, max_rows=max_rows) for x in val.tolist()]

        # 4. NumPy scalars & native numeric/bool types
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, (float, np.floating)):
            if np.isnan(val) or not np.isfinite(val):
                return None
            return float(val)

        # 5. Pandas Timestamp, Interval, Period, datetime, date
        if isinstance(val, (pd.Timestamp, pd.Period, pd.Interval, datetime)):
            return str(val)

        # 6. Strings
        if isinstance(val, str):
            return val

        # 7. Dicts (recursively)
        if isinstance(val, dict):
            return {str(k): _to_json_friendly(v, max_rows=max_rows) for k, v in val.items()}

        # 8. Lists, tuples, sets (recursively)
        if isinstance(val, (list, tuple, set)):
            return [_to_json_friendly(x, max_rows=max_rows) for x in val]

        # 9. Scalar NaN check (safe because we already excluded DataFrame, Series, ndarray)
        if pd.api.types.is_scalar(val) and pd.isna(val):
            return None

        # 10. Fallback for anything unknown
        return str(val)
    except Exception:
        try:
            return str(val)
        except Exception:
            return repr(val)


def _format_markdown_value(val: Any, max_rows: int = 6) -> str:
    """Formats a scalar, DataFrame, Series, or serialized result dictionary into clean markdown."""
    if val is None:
        return "*None*"
    if isinstance(val, (pd.DataFrame, pd.Series)):
        val = _to_json_friendly(val, max_rows=max_rows)
    if isinstance(val, dict) and "columns" in val and "rows" in val:
        cols = val["columns"]
        rows = val["rows"]
        if not rows:
            return "*Empty table*"
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        row_strs = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows[:max_rows]]
        tbl = "\n" + "\n".join([header, sep] + row_strs)
        if len(rows) > max_rows or val.get("truncated"):
            tbl += f"\n*(Showing first {min(len(rows), max_rows)} rows of {val.get('total_rows', len(rows))})*"
        return tbl
    if isinstance(val, (bool, np.bool_)):
        return f"`{bool(val)}`"
    if isinstance(val, float):
        if np.isnan(val):
            return "*NaN*"
        return f"`{val:.4f}`"
    return f"`{val}`"


def build_analyst_code_prompt(question: str, columns: list[str]) -> str:
    """
    Builds the exact code generation prompt used by agents/analyst.py.
    Used by System A so it receives the identical prompt text.
    """
    return f"""Write pandas code that answers the question using the dataframe called df.
Question: {question}

Columns available:
{', '.join(columns)}

Store the final answer in a variable called result (a DataFrame, Series, or single number).
Use only pd and np, no imports, no file access. Return ONLY the code, no explanation."""


# ============================================================================
# System Runners: A (Single Prompt), B (Analyst), C (Analyst + Reviewer Loop)
# ============================================================================

def run_system_a(
    question: str,
    df: pd.DataFrame,
    budget: TokenBudget,
    call_llm_fn: Callable[[str], str] = call_llm,
) -> dict[str, Any]:
    """
    System A (single prompt):
    ONE LLM call with the analyst's exact code prompt. Runs once in sandbox.
    NO retries on errors, NO reviewer. Sandbox error counts as incorrect.
    """
    budget.check()
    columns_list = list(df.columns)
    code_prompt = build_analyst_code_prompt(question, columns_list)

    t0 = time.time()
    u0 = get_usage_stats()

    raw_reply = call_llm_fn(code_prompt)
    code = _clean_code_fence(raw_reply)

    sandbox_res = run_code(code, df.copy(deep=True))
    elapsed = time.time() - t0

    u1 = get_usage_stats()
    calls = u1.get("calls", 0) - u0.get("calls", 0)
    tokens = u1.get("total_tokens", 0) - u0.get("total_tokens", 0)

    # Calculate model delta
    by_model: dict[str, dict[str, Any]] = {}
    u0_bm = u0.get("by_model", {})
    u1_bm = u1.get("by_model", {})
    for m, curr_stat in u1_bm.items():
        prev_stat = u0_bm.get(m, {})
        d_calls = curr_stat.get("calls", 0) - prev_stat.get("calls", 0)
        d_toks = curr_stat.get("tokens", curr_stat.get("total_tokens", 0)) - prev_stat.get("tokens", prev_stat.get("total_tokens", 0))
        if d_calls > 0 or d_toks > 0:
            by_model[m] = {
                "provider": curr_stat.get("provider", ""),
                "calls": d_calls,
                "tokens": d_toks,
            }

    status = sandbox_res.get("status", "failed")
    res_val = sandbox_res.get("result") if status == "ok" else None
    err = sandbox_res.get("error") if status != "ok" else None

    return {
        "code": code,
        "result": res_val,
        "status": status,
        "error": err,
        "execution_error": err,
        "llm_calls": calls,
        "tokens": tokens,
        "tokens_used": tokens,
        "seconds": round(elapsed, 2),
        "by_model": by_model,
    }


def run_system_b(
    question: str,
    csv_path: str,
    profile_data: dict[str, Any],
    budget: TokenBudget,
    analyst_fn: Callable[[dict[str, Any]], dict[str, Any]] = analyst_node,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    System B (analyst):
    Calls analyst_node with the question preset. Keeps self-repair retries on code errors.
    Returns: (system_b_result_dict, full_analyst_output_state)
    """
    budget.check()
    state_b = {
        "clean_df_path": csv_path,
        "profile": profile_data,
        "questions": [question],
        "analysis_results": [],
        "charts": [],
        "retry_count": 0,
        "events": [],
        "review_feedback": [],
        "rerun_ids": [],
    }
    _assert_no_label_leakage(state_b)

    t0 = time.time()
    u0 = get_usage_stats()

    out_b = analyst_fn(state_b)
    elapsed = time.time() - t0

    u1 = get_usage_stats()
    calls = u1.get("calls", 0) - u0.get("calls", 0)
    tokens = u1.get("total_tokens", 0) - u0.get("total_tokens", 0)

    by_model: dict[str, dict[str, Any]] = {}
    u0_bm = u0.get("by_model", {})
    u1_bm = u1.get("by_model", {})
    for m, curr_stat in u1_bm.items():
        prev_stat = u0_bm.get(m, {})
        d_calls = curr_stat.get("calls", 0) - prev_stat.get("calls", 0)
        d_toks = curr_stat.get("tokens", curr_stat.get("total_tokens", 0)) - prev_stat.get("tokens", prev_stat.get("total_tokens", 0))
        if d_calls > 0 or d_toks > 0:
            by_model[m] = {
                "provider": curr_stat.get("provider", ""),
                "calls": d_calls,
                "tokens": d_toks,
            }

    analysis_results = out_b.get("analysis_results", [])
    item_res = analysis_results[0] if analysis_results else {}

    res_record = {
        "code": item_res.get("code", ""),
        "result": item_res.get("result"),
        "status": item_res.get("status", "failed"),
        "error": item_res.get("error"),
        "attempts": item_res.get("attempts", 1),
        "attempts_used": item_res.get("attempts", 1),
        "notes": item_res.get("notes", ""),
        "llm_calls": calls,
        "tokens": tokens,
        "tokens_used": tokens,
        "seconds": round(elapsed, 2),
        "by_model": by_model,
    }

    return res_record, out_b


def run_system_c(
    question: str,
    csv_path: str,
    profile_data: dict[str, Any],
    out_b: dict[str, Any],
    ref_answer: Any,
    budget: TokenBudget,
    analyst_fn: Callable[[dict[str, Any]], dict[str, Any]] = analyst_node,
    reviewer_fn: Callable[[dict[str, Any]], dict[str, Any]] = reviewer_node,
) -> dict[str, Any]:
    """
    System C (analyst + reviewer loop):
    Starts from B's first attempt (zero duplicate LLM calls for initial pass).
    Runs reviewer_node; if failed and retry_count < 3, calls analyst_node with feedback,
    and loops until pass, unverified, or retries run out.
    """
    state_c = {
        "clean_df_path": csv_path,
        "profile": profile_data,
        "questions": [question],
        "analysis_results": [dict(out_b.get("analysis_results", [{}])[0])],
        "charts": [],
        "retry_count": 0,
        "events": list(out_b.get("events", [])),
        "review_feedback": [],
        "review_report": [],
        "caught_errors": [],
        "rerun_ids": [],
    }
    _assert_no_label_leakage(state_c)

    t0_c = time.time()
    u0_c = get_usage_stats()

    rounds_data: list[dict[str, Any]] = []
    round_idx = 1
    max_retries = 3

    while True:
        budget.check()
        curr_item = state_c["analysis_results"][0]
        curr_res = curr_item.get("result")
        curr_status = curr_item.get("status", "failed")

        # Evaluate correctness of the current round's analyst result
        is_corr, corr_reason = answer_correct(ref_answer, curr_res) if curr_status == "ok" else (False, "status failed")

        # 1. Run reviewer_node
        _assert_no_label_leakage(state_c)
        rev_out = reviewer_fn(state_c)

        review_report = rev_out.get("review_report", [])
        rep = review_report[0] if review_report else {}
        verdict = rep.get("verdict", "unverified")
        reason = rep.get("reason", "")
        failed_check = rep.get("failed_check")

        rounds_data.append({
            "round": round_idx,
            "analyst_code": curr_item.get("code", ""),
            "analyst_result": _to_json_friendly(curr_res),
            "status": curr_status,
            "reviewer_verdict": verdict,
            "reviewer_reason": reason,
            "failed_check": failed_check,
            "is_correct": is_corr,
        })

        feedback = rev_out.get("review_feedback", [])
        retries_used = state_c.get("retry_count", 0)

        # Check if reviewer flagged failure and retry budget remains
        has_failure = any(str(fb.get("verdict", "")).strip().lower() == "fail" for fb in feedback)
        if has_failure and retries_used < max_retries:
            # 2. Advance retry count and call analyst_node with feedback
            state_c["retry_count"] = retries_used + 1
            state_c["review_feedback"] = feedback
            state_c["review_report"] = review_report
            state_c["caught_errors"] = rev_out.get("caught_errors", [])
            state_c["events"] = rev_out.get("events", [])

            _assert_no_label_leakage(state_c)
            budget.check()
            analyst_out = analyst_fn(state_c)

            state_c["analysis_results"] = analyst_out.get("analysis_results", state_c["analysis_results"])
            state_c["review_feedback"] = []  # cleared by analyst_node
            state_c["events"] = analyst_out.get("events", state_c["events"])

            round_idx += 1
        else:
            # Reached pass, unverified, or exhausted max_retries
            break

    elapsed_c = time.time() - t0_c
    u1_c = get_usage_stats()
    additional_calls = u1_c.get("calls", 0) - u0_c.get("calls", 0)
    additional_tokens = u1_c.get("total_tokens", 0) - u0_c.get("total_tokens", 0)

    # Final outcome
    final_item = state_c["analysis_results"][0]
    final_res = final_item.get("result")
    final_status = final_item.get("status", "failed")
    final_is_corr, final_corr_reason = answer_correct(ref_answer, final_res) if final_status == "ok" else (False, "status failed")
    final_verdict = rounds_data[-1]["reviewer_verdict"] if rounds_data else "unverified"

    return {
        "final_code": final_item.get("code", ""),
        "final_result": final_res,
        "status": final_status,
        "is_correct": final_is_corr,
        "correctness_reason": final_corr_reason,
        "final_verdict": final_verdict,
        "reviewer_verdict": final_verdict,
        "reviewer_feedback": [r.get("reviewer_reason") for r in rounds_data if r.get("reviewer_reason")],
        "rounds_count": len(rounds_data),
        "rounds_used": len(rounds_data),
        "retries_used": state_c.get("retry_count", 0),
        "rounds": rounds_data,
        "additional_llm_calls": additional_calls,
        "additional_tokens": additional_tokens,
        "additional_seconds": round(elapsed_c, 2),
    }


# ============================================================================
# Paired Classifications
# ============================================================================

def classify_paired_bc(b_correct: bool, c_correct: bool) -> str:
    """Classifies B vs C relationship: both_correct, fixed, harmed, both_wrong."""
    if b_correct and c_correct:
        return "both_correct"
    if (not b_correct) and c_correct:
        return "fixed"
    if b_correct and (not c_correct):
        return "harmed"
    return "both_wrong"


def classify_paired_ab(a_correct: bool, b_correct: bool) -> str:
    """Classifies A vs B relationship: both_correct, b_improved, b_worsened, both_wrong."""
    if a_correct and b_correct:
        return "both_correct"
    if (not a_correct) and b_correct:
        return "b_improved"
    if a_correct and (not b_correct):
        return "b_worsened"
    return "both_wrong"


# ============================================================================
# Summary Generation & Statistics (summary.md and summary.json)
# ============================================================================

def generate_summary(
    records: list[dict[str, Any]],
    run_path: Path,
    repeats: int | None = None,
    expected_total: int | None = None,
    items: list[int] | None = None,
) -> dict[str, Any]:
    """Generates comprehensive paired comparison summary metrics and writes summary.json and summary.md."""
    if not records:
        print("No evaluation records provided to generate_summary.")
        return {}

    # Detect repeats & items from records
    detected_repeats = max(r.get("repeat_index", 0) for r in records) + 1 if records else 1
    repeats = repeats or detected_repeats
    unique_item_ids = sorted(set(r["item_id"] for r in records))
    n_unique_items = len(unique_item_ids)

    if expected_total is None:
        expected_total = len(items or unique_item_ids) * repeats

    n_finished = len(records)
    is_partial = n_finished < expected_total

    # Group by repeat index
    records_by_repeat: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        r_idx = int(r.get("repeat_index", 0))
        records_by_repeat.setdefault(r_idx, []).append(r)

    sorted_r_indices = sorted(records_by_repeat.keys())

    # 1. Per-system accuracy across repeats
    per_repeat_acc: list[dict[str, Any]] = []
    for r_idx in sorted_r_indices:
        r_recs = records_by_repeat[r_idx]
        n_r = len(r_recs)
        a_corr = sum(1 for r in r_recs if r.get("system_a", {}).get("is_correct"))
        b_corr = sum(1 for r in r_recs if r.get("system_b", {}).get("is_correct"))
        c_corr = sum(1 for r in r_recs if r.get("system_c", {}).get("is_correct"))

        per_repeat_acc.append({
            "repeat": f"r{r_idx + 1}",
            "finished": n_r,
            "a_correct": a_corr,
            "a_rate": (a_corr / n_r) if n_r > 0 else 0.0,
            "b_correct": b_corr,
            "b_rate": (b_corr / n_r) if n_r > 0 else 0.0,
            "c_correct": c_corr,
            "c_rate": (c_corr / n_r) if n_r > 0 else 0.0,
        })

    # Headline rates: mean over repeats
    a_rates = [p["a_rate"] for p in per_repeat_acc]
    b_rates = [p["b_rate"] for p in per_repeat_acc]
    c_rates = [p["c_rate"] for p in per_repeat_acc]

    mean_a_rate = float(np.mean(a_rates)) if a_rates else 0.0
    mean_b_rate = float(np.mean(b_rates)) if b_rates else 0.0
    mean_c_rate = float(np.mean(c_rates)) if c_rates else 0.0

    a_ci = wilson_ci_from_rate(mean_a_rate, n_unique_items)
    b_ci = wilson_ci_from_rate(mean_b_rate, n_unique_items)
    c_ci = wilson_ci_from_rate(mean_c_rate, n_unique_items)

    mean_a_corr = float(np.mean([p["a_correct"] for p in per_repeat_acc])) if per_repeat_acc else 0.0
    mean_b_corr = float(np.mean([p["b_correct"] for p in per_repeat_acc])) if per_repeat_acc else 0.0
    mean_c_corr = float(np.mean([p["c_correct"] for p in per_repeat_acc])) if per_repeat_acc else 0.0

    tot_a_corr = sum(p["a_correct"] for p in per_repeat_acc)
    tot_b_corr = sum(p["b_correct"] for p in per_repeat_acc)
    tot_c_corr = sum(p["c_correct"] for p in per_repeat_acc)

    # 2. Paired analysis B vs C
    paired_bc_counts = {"both_correct": 0, "fixed": 0, "harmed": 0, "both_wrong": 0}
    fixed_items: list[dict[str, Any]] = []
    harmed_items: list[dict[str, Any]] = []

    for r in records:
        cat = r.get("paired_bc") or classify_paired_bc(
            bool(r.get("system_b", {}).get("is_correct")),
            bool(r.get("system_c", {}).get("is_correct")),
        )
        paired_bc_counts[cat] = paired_bc_counts.get(cat, 0) + 1

        if cat == "fixed":
            fixed_items.append({
                "item_id": r["item_id"],
                "repeat_index": r.get("repeat_index", 0),
                "question": r.get("question", ""),
                "b_result": r.get("system_b", {}).get("result"),
                "c_result": r.get("system_c", {}).get("final_result"),
                "reference_answer": r.get("reference_answer"),
                "b_code": r.get("system_b", {}).get("code"),
                "c_code": r.get("system_c", {}).get("final_code"),
                "rounds": r.get("system_c", {}).get("rounds_count", 1),
            })
        elif cat == "harmed":
            harmed_items.append({
                "item_id": r["item_id"],
                "repeat_index": r.get("repeat_index", 0),
                "question": r.get("question", ""),
                "b_result": r.get("system_b", {}).get("result"),
                "c_result": r.get("system_c", {}).get("final_result"),
                "reference_answer": r.get("reference_answer"),
                "b_code": r.get("system_b", {}).get("code"),
                "c_code": r.get("system_c", {}).get("final_code"),
                "rounds": r.get("system_c", {}).get("rounds_count", 1),
            })

    # Paired analysis A vs B
    paired_ab_counts = {"both_correct": 0, "b_improved": 0, "b_worsened": 0, "both_wrong": 0}
    for r in records:
        cat_ab = r.get("paired_ab") or classify_paired_ab(
            bool(r.get("system_a", {}).get("is_correct")),
            bool(r.get("system_b", {}).get("is_correct")),
        )
        paired_ab_counts[cat_ab] = paired_ab_counts.get(cat_ab, 0) + 1

    # 3. Reviewer behaviour analytics
    total_first_pass = len(records)
    flagged_first_pass = 0
    true_flags = 0
    false_alarms = 0
    false_alarm_details: list[dict[str, Any]] = []

    total_retries = 0
    total_rounds = 0

    for r in records:
        sc = r.get("system_c", {})
        sb = r.get("system_b", {})
        rounds = sc.get("rounds", [])
        total_retries += sc.get("retries_used", 0)
        total_rounds += sc.get("rounds_count", 1)

        first_verdict = rounds[0].get("reviewer_verdict") if rounds else "unverified"
        if first_verdict == "fail":
            flagged_first_pass += 1
            if not sb.get("is_correct"):
                true_flags += 1
            else:
                false_alarms += 1
                # Check what analyst did after retry on false alarm
                c_is_corr = sc.get("is_correct")
                outcome = "kept_correct" if c_is_corr else "changed_to_wrong"
                false_alarm_details.append({
                    "item_id": r["item_id"],
                    "repeat_index": r.get("repeat_index", 0),
                    "question": r.get("question", ""),
                    "b_result": sb.get("result"),
                    "c_result": sc.get("final_result"),
                    "outcome": outcome,
                    "reviewer_reason": rounds[0].get("reviewer_reason", ""),
                })

    avg_rounds = (total_rounds / total_first_pass) if total_first_pass > 0 else 1.0

    # 4. Resource costs & Reviewer overhead
    tot_a_calls = sum(r.get("system_a", {}).get("llm_calls", 0) for r in records)
    tot_a_tokens = sum(r.get("system_a", {}).get("tokens", 0) for r in records)
    tot_a_seconds = sum(r.get("system_a", {}).get("seconds", 0.0) for r in records)

    tot_b_calls = sum(r.get("system_b", {}).get("llm_calls", 0) for r in records)
    tot_b_tokens = sum(r.get("system_b", {}).get("tokens", 0) for r in records)
    tot_b_seconds = sum(r.get("system_b", {}).get("seconds", 0.0) for r in records)

    tot_c_additional_calls = sum(r.get("system_c", {}).get("additional_llm_calls", 0) for r in records)
    tot_c_additional_tokens = sum(r.get("system_c", {}).get("additional_tokens", 0) for r in records)
    tot_c_additional_seconds = sum(r.get("system_c", {}).get("additional_seconds", 0.0) for r in records)

    tot_c_calls = tot_b_calls + tot_c_additional_calls
    tot_c_tokens = tot_b_tokens + tot_c_additional_tokens
    tot_c_seconds = tot_b_seconds + tot_c_additional_seconds

    a_token_mult = (tot_a_tokens / tot_b_tokens) if tot_b_tokens > 0 else 1.0
    c_token_mult = (tot_c_tokens / tot_b_tokens) if tot_b_tokens > 0 else 1.0
    c_call_mult = (tot_c_calls / tot_b_calls) if tot_b_calls > 0 else 1.0
    c_time_mult = (tot_c_seconds / tot_b_seconds) if tot_b_seconds > 0 else 1.0

    # 5. Verdict flips across repeats
    item_groups: dict[int, dict[str, Any]] = {}
    for r in records:
        i_id = r["item_id"]
        r_idx = r.get("repeat_index", 0)
        r_label = f"r{r_idx + 1}"
        if i_id not in item_groups:
            item_groups[i_id] = {
                "item_id": i_id,
                "question": r.get("question", ""),
                "category": r.get("category", ""),
                "difficulty": r.get("difficulty", ""),
                "a_corr": {},
                "b_corr": {},
                "c_corr": {},
            }
        item_groups[i_id]["a_corr"][r_label] = bool(r.get("system_a", {}).get("is_correct"))
        item_groups[i_id]["b_corr"][r_label] = bool(r.get("system_b", {}).get("is_correct"))
        item_groups[i_id]["c_corr"][r_label] = bool(r.get("system_c", {}).get("is_correct"))

    multi_repeat_items = [ig for ig in item_groups.values() if len(ig["a_corr"]) >= 2]
    flipped_items = [
        ig for ig in multi_repeat_items
        if len(set(ig["a_corr"].values())) > 1 or len(set(ig["b_corr"].values())) > 1 or len(set(ig["c_corr"].values())) > 1
    ]

    # Models used
    aggregated_by_model: dict[str, dict[str, Any]] = {}
    for r in records:
        for sys_key in ("system_a", "system_b"):
            bm = r.get(sys_key, {}).get("by_model", {})
            for m, m_data in bm.items():
                if m not in aggregated_by_model:
                    aggregated_by_model[m] = {"calls": 0, "tokens": 0}
                aggregated_by_model[m]["calls"] += m_data.get("calls", 0)
                aggregated_by_model[m]["tokens"] += m_data.get("tokens", m_data.get("total_tokens", 0))

    if not aggregated_by_model:
        current_bm = get_usage_stats().get("by_model", {})
        for m, m_data in current_bm.items():
            aggregated_by_model[m] = {
                "calls": m_data.get("calls", 0),
                "tokens": m_data.get("tokens", m_data.get("total_tokens", 0)),
            }

    summary_data: dict[str, Any] = {
        "is_partial": is_partial,
        "expected_total_cases": expected_total,
        "finished_cases": n_finished,
        "unique_items_count": n_unique_items,
        "repeats": repeats,
        "models_used": list(aggregated_by_model.keys()),
        "accuracy": {
            "system_a": {
                "correct_total": tot_a_corr,
                "accuracy_rate": mean_a_rate,
                "accuracy_ci": list(a_ci),
                "mean_correct_per_repeat": mean_a_corr,
            },
            "system_b": {
                "correct_total": tot_b_corr,
                "accuracy_rate": mean_b_rate,
                "accuracy_ci": list(b_ci),
                "mean_correct_per_repeat": mean_b_corr,
            },
            "system_c": {
                "correct_total": tot_c_corr,
                "accuracy_rate": mean_c_rate,
                "accuracy_ci": list(c_ci),
                "mean_correct_per_repeat": mean_c_corr,
            },
        },
        "paired_bc": {
            "counts": paired_bc_counts,
            "rates": {k: round(v / n_finished, 4) if n_finished > 0 else 0.0 for k, v in paired_bc_counts.items()},
            "fixed_items": fixed_items,
            "harmed_items": harmed_items,
        },
        "paired_ab": {
            "counts": paired_ab_counts,
            "rates": {k: round(v / n_finished, 4) if n_finished > 0 else 0.0 for k, v in paired_ab_counts.items()},
        },
        "reviewer_behavior": {
            "total_first_pass": total_first_pass,
            "flagged_first_pass": flagged_first_pass,
            "flag_rate": round(flagged_first_pass / total_first_pass, 4) if total_first_pass > 0 else 0.0,
            "true_flags": true_flags,
            "false_alarms": false_alarms,
            "total_retries_used": total_retries,
            "avg_rounds_per_item": round(avg_rounds, 2),
            "false_alarm_details": false_alarm_details,
        },
        "costs": {
            "system_a": {"llm_calls": tot_a_calls, "tokens": tot_a_tokens, "seconds": round(tot_a_seconds, 2)},
            "system_b": {"llm_calls": tot_b_calls, "tokens": tot_b_tokens, "seconds": round(tot_b_seconds, 2)},
            "system_c": {
                "llm_calls": tot_c_calls,
                "tokens": tot_c_tokens,
                "seconds": round(tot_c_seconds, 2),
                "additional_calls": tot_c_additional_calls,
                "additional_tokens": tot_c_additional_tokens,
                "additional_seconds": round(tot_c_additional_seconds, 2),
                "token_multiplier_vs_b": round(c_token_mult, 2),
                "call_multiplier_vs_b": round(c_call_mult, 2),
                "time_multiplier_vs_b": round(c_time_mult, 2),
            },
        },
        "flips": [
            {
                "item_id": ig["item_id"],
                "question": ig["question"],
                "a_corr": ig["a_corr"],
                "b_corr": ig["b_corr"],
                "c_corr": ig["c_corr"],
            }
            for ig in flipped_items
        ],
    }

    # Save summary.json
    with open(run_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Build summary.md
    models_display = ", ".join(f"`{m}` ({d['calls']} calls, {d['tokens']:,} tokens)" for m, d in aggregated_by_model.items()) if aggregated_by_model else "none"

    md_lines: list[str] = []
    if is_partial:
        md_lines.append(f"PARTIAL RUN: {n_finished} of {expected_total} cases finished")
        md_lines.append("")

    md_lines.extend([
        "# End-to-End System Comparison: A vs B vs C",
        "",
        f"**Run Path**: `{run_path.resolve()}`  ",
        f"**Cases Finished**: {n_finished} of {expected_total} expected (Unique Items: {n_unique_items}, Repeats: {repeats})  ",
        f"**Resources**: {tot_a_calls + tot_c_calls} LLM calls, {tot_a_tokens + tot_c_tokens:,} tokens  ",
        f"**Models Used**: {models_display}  ",
        "",
        "## 1. System Accuracy Comparison",
        "",
    ])

    if repeats > 1:
        md_lines.extend([
            "| System | Description | Unique items | Observations | Correct (mean per repeat) | Accuracy rate (mean over repeats) | 95% Wilson CI |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| **System A** | Single prompt (no retry, no review) | {n_unique_items} | {n_finished} | {mean_a_corr:.1f} | {mean_a_rate*100:.1f}% | [{a_ci[0]*100:.1f}%, {a_ci[1]*100:.1f}%] |",
            f"| **System B** | Analyst agent (self-repair on errors) | {n_unique_items} | {n_finished} | {mean_b_corr:.1f} | {mean_b_rate*100:.1f}% | [{b_ci[0]*100:.1f}%, {b_ci[1]*100:.1f}%] |",
            f"| **System C** | Analyst + Reviewer feedback loop | {n_unique_items} | {n_finished} | {mean_c_corr:.1f} | {mean_c_rate*100:.1f}% | [{c_ci[0]*100:.1f}%, {c_ci[1]*100:.1f}%] |",
        ])
    else:
        md_lines.extend([
            "| System | Description | Correct / Total | Accuracy Rate | 95% Wilson CI |",
            "| :--- | :--- | :---: | :---: | :---: |",
            f"| **System A** | Single prompt (no retry, no review) | {tot_a_corr}/{n_finished} | {mean_a_rate*100:.1f}% | [{a_ci[0]*100:.1f}%, {a_ci[1]*100:.1f}%] |",
            f"| **System B** | Analyst agent (self-repair on errors) | {tot_b_corr}/{n_finished} | {mean_b_rate*100:.1f}% | [{b_ci[0]*100:.1f}%, {b_ci[1]*100:.1f}%] |",
            f"| **System C** | Analyst + Reviewer feedback loop | {tot_c_corr}/{n_finished} | {mean_c_rate*100:.1f}% | [{c_ci[0]*100:.1f}%, {c_ci[1]*100:.1f}%] |",
        ])

    md_lines.extend([
        "",
        "Intervals treat the unique benchmark questions (not the repeats) as the sample; repeats average out LLM variability.",
        "",
        "## 2. Paired Analysis: System B vs System C",
        "",
        "System C starts directly from System B's first-pass code and result, allowing exact paired diagnosis of reviewer impact:",
        "",
        "| Paired Outcome | Count | Percentage | Description |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Both correct** | {paired_bc_counts['both_correct']} | {paired_bc_counts['both_correct']/n_finished*100:.1f}% | B was correct, reviewer did not harm |",
        f"| **FIXED by reviewer** | **{paired_bc_counts['fixed']}** | **{paired_bc_counts['fixed']/n_finished*100:.1f}%** | **B was wrong -> reviewer feedback led to correct answer** |",
        f"| **HARMED by reviewer** | **{paired_bc_counts['harmed']}** | **{paired_bc_counts['harmed']/n_finished*100:.1f}%** | **B was correct -> reviewer falsely failed and analyst corrupted answer** |",
        f"| **Both wrong** | {paired_bc_counts['both_wrong']} | {paired_bc_counts['both_wrong']/n_finished*100:.1f}% | B was wrong and reviewer could not fix it |",
        "",
    ])

    if fixed_items:
        md_lines.append("### Items FIXED by Reviewer Loop:")
        for it in fixed_items:
            md_lines.extend([
                f"- **Item {it['item_id']}** (Repeat {it['repeat_index'] + 1}): *{it['question']}*",
                f"  - **Analyst Initial (B)**: {_format_markdown_value(it['b_result'])} (Wrong)",
                f"  - **Final After Review (C)**: {_format_markdown_value(it['c_result'])} (Correct)",
                f"  - **Reference Answer**: {_format_markdown_value(it['reference_answer'])}",
                f"  - **Rounds Taken**: {it['rounds']}",
            ])
        md_lines.append("")
    else:
        md_lines.append("*No items were fixed by the reviewer loop in this run.*  \n")

    if harmed_items:
        md_lines.append("### Items HARMED by Reviewer Loop:")
        for it in harmed_items:
            md_lines.extend([
                f"- **Item {it['item_id']}** (Repeat {it['repeat_index'] + 1}): *{it['question']}*",
                f"  - **Analyst Initial (B)**: {_format_markdown_value(it['b_result'])} (Correct)",
                f"  - **Final After Review (C)**: {_format_markdown_value(it['c_result'])} (Wrong)",
                f"  - **Reference Answer**: {_format_markdown_value(it['reference_answer'])}",
                f"  - **Rounds Taken**: {it['rounds']}",
            ])
        md_lines.append("")
    else:
        md_lines.append("*No items were harmed by the reviewer loop in this run.*  \n")

    # Paired A vs B
    md_lines.extend([
        "### Paired Analysis: System A vs System B (Impact of Self-Repair)",
        "",
        "| Paired Outcome | Count | Percentage | Description |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Both correct** | {paired_ab_counts['both_correct']} | {paired_ab_counts['both_correct']/n_finished*100:.1f}% | Both single-prompt and analyst agent correct |",
        f"| **B improved** | {paired_ab_counts['b_improved']} | {paired_ab_counts['b_improved']/n_finished*100:.1f}% | A failed syntax/execution, B self-repaired |",
        f"| **B worsened** | {paired_ab_counts['b_worsened']} | {paired_ab_counts['b_worsened']/n_finished*100:.1f}% | A succeeded, B failed |",
        f"| **Both wrong** | {paired_ab_counts['both_wrong']} | {paired_ab_counts['both_wrong']/n_finished*100:.1f}% | Both systems gave wrong answers |",
        "",
        "## 3. Reviewer Behavior on Benchmark Subset",
        "",
        f"- **First-pass answers flagged**: {flagged_first_pass} of {total_first_pass} ({flagged_first_pass/total_first_pass*100:.1f}%)",
        f"- **True flags** (flagged an actually wrong answer): **{true_flags}**",
        f"- **False alarms** (flagged an actually correct answer): **{false_alarms}**",
        f"- **Total reviewer retries used**: {total_retries}",
        f"- **Average reviewer rounds per item**: {avg_rounds:.2f}",
        "",
    ])

    if false_alarm_details:
        md_lines.append("### False Alarm Diagnoses:")
        for fa in false_alarm_details:
            act_text = "analyst kept correct answer" if fa["outcome"] == "kept_correct" else "analyst was misled into wrong answer"
            md_lines.append(f"- **Item {fa['item_id']}** (r{fa['repeat_index']+1}): Flagged because: `{fa['reviewer_reason']}` -> Outcome: **{act_text}**")
        md_lines.append("")

    # Cost table
    md_lines.extend([
        "## 4. Cost and Resource Overhead",
        "",
        "| System | LLM Calls | Total Tokens | Execution Time | Overhead vs System B |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **System A** (Single prompt) | {tot_a_calls} | {tot_a_tokens:,} | {tot_a_seconds:.1f}s | {a_token_mult:.2f}x tokens |",
        f"| **System B** (Analyst agent) | {tot_b_calls} | {tot_b_tokens:,} | {tot_b_seconds:.1f}s | 1.00x (baseline) |",
        f"| **System C** (Analyst + Reviewer) | {tot_c_calls} | {tot_c_tokens:,} | {tot_c_seconds:.1f}s | **{c_token_mult:.2f}x tokens** ({c_call_mult:.2f}x calls, {c_time_mult:.2f}x time) |",
        "",
        f"*The reviewer loop added {tot_c_additional_calls} LLM calls and {tot_c_additional_tokens:,} tokens beyond the initial analyst generation.*",
        "",
        "## 5. Per-Item Results Table",
        "",
        "| Item ID | Category | Difficulty | System A | System B | System C | Reviewer Rounds | Paired (B->C) |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
    ])

    for r in records:
        i_id = r["item_id"]
        cat = r.get("category", "")
        diff = r.get("difficulty", "")
        a_ok = "PASS" if r.get("system_a", {}).get("is_correct") else "FAIL"
        b_ok = "PASS" if r.get("system_b", {}).get("is_correct") else "FAIL"
        c_ok = "PASS" if r.get("system_c", {}).get("is_correct") else "FAIL"
        rounds_cnt = r.get("system_c", {}).get("rounds_count", 1)
        paired_tag = r.get("paired_bc", "")
        r_suffix = f" (r{r.get('repeat_index', 0) + 1})" if repeats > 1 else ""
        md_lines.append(f"| `{i_id}`{r_suffix} | {cat} | {diff} | {a_ok} | {b_ok} | {c_ok} | {rounds_cnt} | {paired_tag} |")

    md_lines.append("")

    # Flips table across repeats
    if repeats > 1:
        md_lines.extend([
            "## 6. Correctness Flips Across Repeats",
            "",
            f"- **Items evaluated in 2 or more repeats**: {len(multi_repeat_items)}",
            f"- **Items with differing correctness across repeats**: {len(flipped_items)}",
            "",
        ])
        if flipped_items:
            rep_headers = " | ".join(f"**{r_label}**" for r_label in [f"r{i+1}" for i in range(repeats)])
            rep_sep = " | ".join([":---:"] * repeats)
            md_lines.extend([
                f"| Item ID | System | {rep_headers} |",
                f"| :---: | :--- | {rep_sep} |",
            ])
            for fi in flipped_items:
                for sys_name, corr_dict in [("A", fi["a_corr"]), ("B", fi["b_corr"]), ("C", fi["c_corr"])]:
                    cells = " | ".join("PASS" if corr_dict.get(f"r{i+1}") else "FAIL" for i in range(repeats))
                    md_lines.append(f"| `{fi['item_id']}` | System {sys_name} | {cells} |")
            md_lines.append("")
        else:
            md_lines.append("*None. All items had identical correctness verdicts across repeats.*  \n")

    # Honest Limitations Block
    md_lines.extend([
        "---",
        "> [!NOTE] Evaluation Limitations",
        "> - **Small Subset**: Evaluated on 12 questions on a single dataset (`data/sample.csv`).",
        "> - **Model Homogeneity**: The same LLM plays both the analyst and reviewer roles; common failure modes and reasoning biases can be correlated.",
        "> - **Component Isolation**: The data cleaner and visualizer are not included in this evaluation.",
        "> - **Stochasticity**: LLM outputs vary between runs; small sample size (n = 12) leads to wide confidence intervals.",
        "> - **Tuning Overlap**: The benchmark questions were previously inspected during development to find and fix two reviewer comparison bugs.",
        "> ",
        "> With n = 12 the paired counts are the main result; differences of one or two questions are not statistically meaningful.",
    ])

    with open(run_path / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Print summary to console
    print("\n" + "=" * 80)
    print("END-TO-END COMPARISON EVALUATION SUMMARY")
    if is_partial:
        print(f"** PARTIAL RUN: {n_finished} of {expected_total} cases finished **")
    print("=" * 80)
    print(f"Items:           {n_unique_items} unique items x {repeats} repeats ({n_finished} finished)")
    print(f"System A Acc:    {mean_a_rate*100:.1f}% [95% CI: {a_ci[0]*100:.1f}% - {a_ci[1]*100:.1f}%]")
    print(f"System B Acc:    {mean_b_rate*100:.1f}% [95% CI: {b_ci[0]*100:.1f}% - {b_ci[1]*100:.1f}%]")
    print(f"System C Acc:    {mean_c_rate*100:.1f}% [95% CI: {c_ci[0]*100:.1f}% - {c_ci[1]*100:.1f}%]")
    print(f"Paired (B -> C): Both correct: {paired_bc_counts['both_correct']} | FIXED: {paired_bc_counts['fixed']} | HARMED: {paired_bc_counts['harmed']} | Both wrong: {paired_bc_counts['both_wrong']}")
    print(f"Reviewer:        Flagged: {flagged_first_pass}/{total_first_pass} | True flags: {true_flags} | False alarms: {false_alarms} | Avg rounds: {avg_rounds:.2f}")
    print(f"Tokens:          A: {tot_a_tokens:,} | B: {tot_b_tokens:,} | C: {tot_c_tokens:,} ({c_token_mult:.2f}x B)")
    print(f"Summary saved:   {run_path / 'summary.md'}")
    print("=" * 80 + "\n")

    return summary_data


# ============================================================================
# Summarize-Only Mode & Main Evaluation Runner
# ============================================================================

def summarize_run(
    run_dir: str | Path,
    repeats: int | None = None,
    limit: int | None = None,
    items: list[int] | None = None,
) -> dict[str, Any]:
    """
    Reads <run-dir>/cases.jsonl in read-only mode, and generates summary.md
    and summary.json without modifying cases.jsonl or making any LLM calls.
    """
    run_path = Path(run_dir)
    cases_file = run_path / "cases.jsonl"
    if not cases_file.exists():
        raise FileNotFoundError(f"cases.jsonl not found in {run_path}")

    records: list[dict[str, Any]] = []
    with open(cases_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    records.append(json.loads(line_str))
                except Exception:
                    pass

    return generate_summary(
        records=records,
        run_path=run_path,
        repeats=repeats,
        expected_total=limit,
        items=items,
    )


def run_evaluation(
    items: list[int] | None = None,
    repeats: int = 1,
    run_dir: str | None = None,
    max_tokens: int | None = None,
    analyst_fn: Callable = analyst_node,
    reviewer_fn: Callable = reviewer_node,
    call_llm_fn: Callable = call_llm,
) -> str:
    """
    Executes the end-to-end evaluation pipeline for the specified items and repeats.
    """
    items_to_eval = items if items is not None else DEFAULT_ITEMS

    # 1. Initialize run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = Path(run_dir) if run_dir else Path(f"eval/runs/{timestamp}_e2e")
    run_path.mkdir(parents=True, exist_ok=True)
    cases_file = run_path / "cases.jsonl"
    print(f"Run directory: {run_path.resolve()}", flush=True)

    # 2. Load dataset and ground truth
    csv_path = "data/sample.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    profile_data = profile_dataframe(df)

    ground_truth = load_ground_truth("eval/ground_truth.json")
    benchmark_by_id = {b["id"]: b for b in BENCHMARK}

    # 3. Check for existing cases in cases.jsonl for resumption
    completed_keys: set[tuple[int, int]] = set()
    completed_records: list[dict[str, Any]] = []

    if cases_file.exists():
        with open(cases_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        rec = json.loads(line_str)
                        completed_records.append(rec)
                        completed_keys.add((rec["item_id"], rec.get("repeat_index", 0)))
                    except Exception:
                        pass
        if completed_keys:
            print(f"Resuming run: found {len(completed_keys)} already completed case(s).")

    # Build sequence of tasks to run: (item_id, repeat_index)
    all_tasks: list[tuple[int, int]] = []
    for r_idx in range(repeats):
        for item_id in items_to_eval:
            all_tasks.append((item_id, r_idx))

    tasks_to_run = [t for t in all_tasks if t not in completed_keys]
    total_tasks = len(all_tasks)
    done_so_far = len(completed_keys)

    print(f"E2E Evaluation starting: {len(tasks_to_run)} tasks remaining of {total_tasks} total ({len(items_to_eval)} items x {repeats} repeats).")
    budget = TokenBudget(max_tokens=max_tokens)

    # 4. Open cases.jsonl in append mode
    with open(cases_file, "a", encoding="utf-8") as out_f:
        for idx, (item_id, r_idx) in enumerate(tasks_to_run):
            current_num = done_so_far + idx + 1
            case_id = f"q{item_id:02d}_r{r_idx + 1}"
            bm_item = benchmark_by_id.get(item_id, {})
            question = bm_item.get("question", "")
            category = bm_item.get("category", "")
            difficulty = bm_item.get("difficulty", "")
            ref_ans = ground_truth.get(item_id, {}).get("answer")

            items_arg_str = " ".join(str(i) for i in items_to_eval)
            resume_cmd = f"python -m eval.run_e2e_eval --run-dir {run_path} --items {items_arg_str} --repeats {repeats}"
            if max_tokens is not None:
                resume_cmd += f" --max-tokens {max_tokens}"

            try:
                # System A: Single Prompt
                res_a = run_system_a(
                    question=question,
                    df=df,
                    budget=budget,
                    call_llm_fn=call_llm_fn,
                )
                ok_a, reason_a = answer_correct(ref_ans, res_a["result"]) if res_a["status"] == "ok" else (False, res_a.get("error") or "Execution failed")
                res_a["is_correct"] = ok_a
                res_a["reason"] = reason_a

                # System B: Analyst Agent
                res_b, out_b = run_system_b(
                    question=question,
                    csv_path=csv_path,
                    profile_data=profile_data,
                    budget=budget,
                    analyst_fn=analyst_fn,
                )
                ok_b, reason_b = answer_correct(ref_ans, res_b["result"]) if res_b["status"] == "ok" else (False, res_b.get("error") or "Execution failed")
                res_b["is_correct"] = ok_b
                res_b["reason"] = reason_b

                # System C: Analyst + Reviewer Loop (starts from B)
                res_c = run_system_c(
                    question=question,
                    csv_path=csv_path,
                    profile_data=profile_data,
                    out_b=out_b,
                    ref_answer=ref_ans,
                    budget=budget,
                    analyst_fn=analyst_fn,
                    reviewer_fn=reviewer_fn,
                )

                # Classify paired outcomes
                paired_bc = classify_paired_bc(ok_b, res_c["is_correct"])
                paired_ab = classify_paired_ab(ok_a, ok_b)

                # Serialize full record
                record = {
                    "item_id": item_id,
                    "repeat_index": r_idx,
                    "case_id": case_id,
                    "question": question,
                    "category": category,
                    "difficulty": difficulty,
                    "reference_answer": _to_json_friendly(ref_ans),
                    "system_a": {
                        "code": res_a["code"],
                        "result": _to_json_friendly(res_a["result"]),
                        "status": res_a["status"],
                        "error": res_a["error"],
                        "execution_error": res_a["error"],
                        "is_correct": res_a["is_correct"],
                        "reason": res_a["reason"],
                        "llm_calls": res_a["llm_calls"],
                        "tokens": res_a["tokens"],
                        "tokens_used": res_a["tokens"],
                        "seconds": res_a["seconds"],
                        "by_model": res_a["by_model"],
                    },
                    "system_b": {
                        "code": res_b["code"],
                        "result": _to_json_friendly(res_b["result"]),
                        "status": res_b["status"],
                        "error": res_b["error"],
                        "attempts": res_b["attempts"],
                        "attempts_used": res_b["attempts"],
                        "is_correct": res_b["is_correct"],
                        "reason": res_b["reason"],
                        "llm_calls": res_b["llm_calls"],
                        "tokens": res_b["tokens"],
                        "tokens_used": res_b["tokens"],
                        "seconds": res_b["seconds"],
                        "by_model": res_b["by_model"],
                    },
                    "system_c": {
                        "final_code": res_c["final_code"],
                        "final_result": _to_json_friendly(res_c["final_result"]),
                        "status": res_c["status"],
                        "is_correct": res_c["is_correct"],
                        "correctness_reason": res_c["correctness_reason"],
                        "final_verdict": res_c["final_verdict"],
                        "reviewer_verdict": res_c["final_verdict"],
                        "reviewer_feedback": res_c["reviewer_feedback"],
                        "rounds_count": res_c["rounds_count"],
                        "rounds_used": res_c["rounds_count"],
                        "retries_used": res_c["retries_used"],
                        "rounds": res_c["rounds"],
                        "additional_llm_calls": res_c["additional_llm_calls"],
                        "additional_tokens": res_c["additional_tokens"],
                        "additional_seconds": res_c["additional_seconds"],
                        "total_llm_calls": res_b["llm_calls"] + res_c["additional_llm_calls"],
                        "total_tokens": res_b["tokens"] + res_c["additional_tokens"],
                        "tokens_used": res_b["tokens"] + res_c["additional_tokens"],
                        "total_seconds": round(res_b["seconds"] + res_c["additional_seconds"], 2),
                        "first_pass_correct": ok_b,
                        "fixed_by_reviewer": (not ok_b) and res_c["is_correct"],
                        "harmed_by_reviewer": ok_b and (not res_c["is_correct"]),
                    },
                    "paired_bc": paired_bc,
                    "paired_ab": paired_ab,
                }

                line_to_write = ""
                try:
                    line_to_write = json.dumps(record)
                except Exception as ser_err:
                    print(f"[ERROR] Serialization of completed task {case_id} failed: {ser_err}. Serializing safe fallback.", file=sys.stderr)
                    record["reference_answer"] = repr(ref_ans)
                    record["system_a"]["result"] = repr(res_a.get("result"))
                    record["system_b"]["result"] = repr(res_b.get("result"))
                    record["system_c"]["final_result"] = repr(res_c.get("final_result"))
                    line_to_write = json.dumps({k: str(v) if not isinstance(v, (int, float, bool, dict, list)) else v for k, v in record.items()})

                try:
                    out_f.write(line_to_write + "\n")
                    out_f.flush()
                except Exception as write_err:
                    print(f"[ERROR] Failed to write record to disk for {case_id}: {write_err}. Continuing.", file=sys.stderr)
                completed_records.append(record)

                a_tag = "PASS" if ok_a else "FAIL"
                b_tag = "PASS" if ok_b else "FAIL"
                c_tag = "PASS" if res_c["is_correct"] else "FAIL"
                print(
                    f"[{current_num:02d}/{total_tasks:02d}] {case_id:10s} "
                    f"A:{a_tag} B:{b_tag} C:{c_tag} | Paired(B->C): {paired_bc:12s} ({res_c['rounds_count']} rounds)",
                    flush=True,
                )

            except BudgetExceeded as be:
                print(f"\n[TOKEN BUDGET EXCEEDED] Execution paused cleanly: {be}", file=sys.stderr)
                print(f"Finished {done_so_far + idx} of {total_tasks} tasks (used {be.used_tokens:,} tokens, limit {be.max_tokens:,}).", file=sys.stderr)
                print(f"To resume this evaluation, run:\n  {resume_cmd}", file=sys.stderr)
                sys.exit(1)
            except (LLMUnavailableError, LLMConfigError) as llm_err:
                print(f"\n[AI SERVICE UNAVAILABLE] Execution paused cleanly: {llm_err}", file=sys.stderr)
                print(f"Finished {done_so_far + idx} of {total_tasks} tasks.", file=sys.stderr)
                print(f"To resume this evaluation, run:\n  {resume_cmd}", file=sys.stderr)
                sys.exit(1)
            except KeyboardInterrupt:
                print(f"\n[INTERRUPTED] Evaluation interrupted by user.", file=sys.stderr)
                print(f"Finished {done_so_far + idx} of {total_tasks} tasks.", file=sys.stderr)
                print(f"To resume this evaluation, run:\n  {resume_cmd}", file=sys.stderr)
                sys.exit(130)
            except Exception as e:
                # Other individual exceptions are recorded as errors and run continues
                print(f"\n[ERROR] Task {case_id} failed with exception: {type(e).__name__}: {e}. Recording error and continuing.", file=sys.stderr)
                err_record: dict[str, Any] = {}
                line_to_write = ""
                try:
                    safe_ref = _to_json_friendly(ref_ans)
                except Exception:
                    safe_ref = repr(ref_ans)

                try:
                    err_record = {
                        "item_id": item_id,
                        "repeat_index": r_idx,
                        "case_id": case_id,
                        "question": str(question),
                        "category": str(category),
                        "difficulty": str(difficulty),
                        "reference_answer": safe_ref,
                        "system_a": {"status": "error", "error": str(e), "execution_error": str(e), "is_correct": False, "llm_calls": 0, "tokens": 0, "tokens_used": 0, "seconds": 0.0},
                        "system_b": {"status": "error", "error": str(e), "is_correct": False, "attempts": 1, "attempts_used": 1, "llm_calls": 0, "tokens": 0, "tokens_used": 0, "seconds": 0.0},
                        "system_c": {"status": "error", "error": str(e), "is_correct": False, "rounds_count": 0, "rounds_used": 0, "retries_used": 0, "reviewer_verdict": "error", "reviewer_feedback": [str(e)], "additional_llm_calls": 0, "additional_tokens": 0, "tokens_used": 0, "additional_seconds": 0.0, "first_pass_correct": False, "fixed_by_reviewer": False, "harmed_by_reviewer": False},
                        "paired_bc": "both_wrong",
                        "paired_ab": "both_wrong",
                    }
                    line_to_write = json.dumps(err_record)
                except Exception as inner_e:
                    # Emergency fallback using repr text
                    print(f"[ERROR] Structured error record serialization failed: {inner_e}. Writing text fallback record.", file=sys.stderr)
                    err_record = {
                        "item_id": item_id,
                        "repeat_index": r_idx,
                        "case_id": case_id,
                        "question": str(question),
                        "category": str(category),
                        "difficulty": str(difficulty),
                        "reference_answer": repr(ref_ans),
                        "system_a": {"status": "error", "error": repr(e), "execution_error": repr(e), "is_correct": False, "llm_calls": 0, "tokens": 0, "tokens_used": 0, "seconds": 0.0},
                        "system_b": {"status": "error", "error": repr(e), "is_correct": False, "attempts": 1, "attempts_used": 1, "llm_calls": 0, "tokens": 0, "tokens_used": 0, "seconds": 0.0},
                        "system_c": {"status": "error", "error": repr(e), "is_correct": False, "rounds_count": 0, "rounds_used": 0, "retries_used": 0, "reviewer_verdict": "error", "reviewer_feedback": [repr(e)], "additional_llm_calls": 0, "additional_tokens": 0, "tokens_used": 0, "additional_seconds": 0.0, "first_pass_correct": False, "fixed_by_reviewer": False, "harmed_by_reviewer": False},
                        "paired_bc": "both_wrong",
                        "paired_ab": "both_wrong",
                    }
                    line_to_write = json.dumps({k: str(v) if not isinstance(v, (int, float, bool, dict, list)) else v for k, v in err_record.items()})

                try:
                    out_f.write(line_to_write + "\n")
                    out_f.flush()
                except Exception as write_err:
                    print(f"[ERROR] Failed to write error record to disk: {write_err}. Continuing.", file=sys.stderr)
                completed_records.append(err_record)

    # 5. Build and save summary
    if len(completed_records) < total_tasks:
        print(f"\n[INCOMPLETE RUN] Finished {len(completed_records)} of {total_tasks} tasks.", file=sys.stderr)
        print(f"To resume this evaluation, run:\n  {resume_cmd}\n", file=sys.stderr)

    generate_summary(
        records=completed_records,
        run_path=run_path,
        repeats=repeats,
        expected_total=total_tasks,
        items=items_to_eval,
    )
    return str(run_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-End comparison of Systems A, B, and C on benchmark.")
    parser.add_argument("--items", type=int, nargs="+", default=DEFAULT_ITEMS, help="Item IDs to evaluate (default 12 ids)")
    parser.add_argument("--repeats", type=int, default=1, help="Number of times to repeat each item (default 1)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens allowed before stopping cleanly")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to store evaluation artifacts")
    parser.add_argument("--summarize-only", action="store_true", help="Generate summary from cases.jsonl without making LLM calls")

    args = parser.parse_args()

    if args.summarize_only:
        if not args.run_dir:
            print("Error: --run-dir must be specified when using --summarize-only.", file=sys.stderr)
            sys.exit(1)
        summarize_run(
            run_dir=args.run_dir,
            repeats=args.repeats,
            items=args.items,
        )
        return

    run_evaluation(
        items=args.items,
        repeats=args.repeats,
        run_dir=args.run_dir,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
