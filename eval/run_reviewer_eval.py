"""
run_reviewer_eval.py
Reviewer Agent Evaluation CLI: Evaluates the Reviewer agent in isolation against
the benchmark questions, certified reference answers, and discriminating wrong variants.
Calculates detection rate, false alarm rate, unverified rate, precision with Wilson score CIs,
and subgroup breakdowns by error magnitude, error kind, category, and difficulty.

Usage:
    python -m eval.run_reviewer_eval [--limit N] [--repeats R] [--run-dir PATH]
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
import pandas as pd

from agents.reviewer import reviewer_node
from eval.benchmark import BENCHMARK
from eval.budget import BudgetExceeded, TokenBudget
from eval.ground_truth import compute_ground_truth, load_ground_truth, save_ground_truth
from eval.metrics import breakdown, confusion, magnitude_bucket, wilson_ci
from eval.scoring import answer_correct
from llm import get_usage_stats, LLMConfigError, LLMUnavailableError
from tools.sandbox import run_code


FORBIDDEN_LEAK_KEYS = {
    "label",
    "kind",
    "is_wrong",
    "variant",
    "wrong_variant",
    "wrong_variants",
    "is_correct",
    "ground_truth",
    "reference",
    "magnitude_bucket",
}


def _assert_no_label_leakage(state: dict[str, Any]) -> None:
    """
    Guard: verifies that the AgentState dictionary does NOT leak any evaluation labels,
    variant classifications, or ground truth info to the Reviewer agent.
    """
    for k in state.keys():
        if k.lower() in FORBIDDEN_LEAK_KEYS:
            raise AssertionError(f"Information leakage detected in state: key '{k}' is forbidden")

    for res in state.get("analysis_results", []):
        for k in res.keys():
            if k.lower() in FORBIDDEN_LEAK_KEYS:
                raise AssertionError(f"Information leakage detected in analysis_results item: key '{k}' is forbidden")


def build_evaluation_cases(
    benchmark: list[dict[str, Any]],
    ground_truth: dict[int, dict[str, Any]],
    repeats: int = 1,
) -> list[dict[str, Any]]:
    """
    Builds the interleaved list of correct and wrong evaluation cases.
    For each benchmark item:
    - 1 correct case (reference code)
    - 1 case per wrong variant (variant code)
    """
    correct_cases: list[dict[str, Any]] = []
    wrong_cases: list[dict[str, Any]] = []

    for item in benchmark:
        item_id = item["id"]
        q = item["question"]
        ref_code = item["reference_code"]
        cat = item["category"]
        diff = item["difficulty"]
        ref_ans = ground_truth[item_id]["answer"]

        # Correct case
        correct_cases.append({
            "base_case_id": f"q{item_id:02d}_correct",
            "item_id": item_id,
            "question": q,
            "code": ref_code,
            "label": "Correct reference code",
            "is_wrong": False,
            "kind": "correct",
            "category": cat,
            "difficulty": diff,
            "magnitude_bucket": "none",
            "reference_answer": ref_ans,
        })

        # Wrong cases
        variants = item.get("wrong_variants", [])
        for v_idx, v in enumerate(variants, start=1):
            v_code = v["code"]
            v_label = v["label"]
            v_kind = v["kind"]

            # Compute magnitude bucket using ground truth variant answer if available
            gt_variants = ground_truth[item_id].get("wrong_variants", [])
            v_ans = None
            if len(gt_variants) >= v_idx:
                v_ans = gt_variants[v_idx - 1].get("answer")

            if v_ans is not None:
                bucket = magnitude_bucket(ref_ans, v_ans)
            else:
                bucket = "unknown"

            wrong_cases.append({
                "base_case_id": f"q{item_id:02d}_v{v_idx}",
                "item_id": item_id,
                "question": q,
                "code": v_code,
                "label": v_label,
                "is_wrong": True,
                "kind": v_kind,
                "category": cat,
                "difficulty": diff,
                "magnitude_bucket": bucket,
                "reference_answer": ref_ans,
            })

    # Interleave correct and wrong cases so that even small subsets have a balanced mix
    # Format: [C0, W0_v1, C1, W1_v1, ..., W0_v2, W1_v2, ...]
    interleaved_base: list[dict[str, Any]] = []
    max_len = max(len(correct_cases), len(wrong_cases))
    c_idx = 0
    w_idx = 0

    while c_idx < len(correct_cases) or w_idx < len(wrong_cases):
        if c_idx < len(correct_cases):
            interleaved_base.append(correct_cases[c_idx])
            c_idx += 1
        if w_idx < len(wrong_cases):
            interleaved_base.append(wrong_cases[w_idx])
            w_idx += 1

    # Expand across repeats
    final_cases: list[dict[str, Any]] = []
    for r in range(repeats):
        for c in interleaved_base:
            case_copy = dict(c)
            if repeats > 1:
                case_copy["case_id"] = f"{c['base_case_id']}_r{r + 1}"
            else:
                case_copy["case_id"] = c["base_case_id"]
            case_copy["repeat_index"] = r
            final_cases.append(case_copy)

    return final_cases


def _to_json_friendly(val: Any, max_rows: int = 8) -> Any:
    """Converts a value, Series, or DataFrame into a JSON-friendly representation capped at max_rows."""
    if val is None:
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return round(float(val), 4)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, pd.Series):
        df = val.reset_index().head(max_rows)
        return df.to_dict(orient="records")
    if isinstance(val, pd.DataFrame):
        df = val.head(max_rows)
        return df.to_dict(orient="records")
    if isinstance(val, list):
        return val[:max_rows]
    if isinstance(val, dict):
        return val
    return str(val)


def _format_markdown_value(val: Any, max_rows: int = 8) -> str:
    """Formats an analytical result or serializable list-of-dicts into clean markdown."""
    if val is None:
        return "*None*"
    if isinstance(val, (int, float, str, bool)):
        if isinstance(val, float):
            return f"`{val:.4f}`"
        return f"`{val}`"
    if isinstance(val, list) and val and isinstance(val[0], dict):
        headers = list(val[0].keys())
        h_line = "| " + " | ".join(str(h) for h in headers) + " |"
        s_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        r_lines = []
        for row in val[:max_rows]:
            r_str = "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |"
            r_lines.append(r_str)
        return "\n" + "\n".join([h_line, s_line] + r_lines)
    if isinstance(val, pd.DataFrame):
        df_sub = val.head(max_rows)
        headers = [str(c) for c in df_sub.columns]
        h_line = "| " + " | ".join(headers) + " |"
        s_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        r_lines = []
        for _, r in df_sub.iterrows():
            r_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
        return "\n" + "\n".join([h_line, s_line] + r_lines)
    if isinstance(val, pd.Series):
        df_sub = val.head(max_rows).reset_index()
        headers = [str(c) for c in df_sub.columns]
        h_line = "| " + " | ".join(headers) + " |"
        s_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        r_lines = []
        for _, r in df_sub.iterrows():
            r_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
        return "\n" + "\n".join([h_line, s_line] + r_lines)
    return f"`{val}`"


def _safe_serialize_case(case_record: dict[str, Any]) -> dict[str, Any]:
    """Ensures case record is JSON serializable."""
    clean = {}
    for k, v in case_record.items():
        if isinstance(v, (pd.DataFrame, pd.Series)):
            clean[k] = str(v)
        else:
            clean[k] = v
    return clean


def run_evaluation(
    limit: int | None = None,
    repeats: int = 1,
    run_dir: str | None = None,
    csv_path: str = "data/sample.csv",
    max_tokens: int | None = None,
) -> str:
    """
    Executes reviewer evaluation harness. Returns path to run directory.
    """
    # 0. Resolve run directory
    if not run_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"eval/runs/{timestamp}_reviewer"

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    cases_file = run_path / "cases.jsonl"
    budget = TokenBudget(max_tokens=max_tokens)

    print("=" * 80)
    print("REVIEWER AGENT EVALUATION RUNNER")
    print(f"Run directory: {run_path.resolve()}")
    print(f"Dataset:       {csv_path}")
    print(f"Repeats:       {repeats}")
    if limit:
        print(f"Limit:         {limit} cases")
    print("=" * 80)

    # 1. Load dataset, benchmark, and ground truth
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    gt_path = "eval/ground_truth.json"
    if os.path.exists(gt_path):
        ground_truth = load_ground_truth(gt_path)
    else:
        print("Computing certified ground truth...")
        ground_truth = compute_ground_truth(BENCHMARK, csv_path)
        save_ground_truth(ground_truth, gt_path)

    # 2. Build full case list
    all_cases = build_evaluation_cases(BENCHMARK, ground_truth, repeats=repeats)
    if limit is not None and limit > 0:
        all_cases = all_cases[:limit]

    # 3. Handle resumability: read already completed cases
    completed_case_ids: set[str] = set()
    completed_records: list[dict[str, Any]] = []

    if cases_file.exists():
        with open(cases_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        record = json.loads(line_str)
                        c_id = record.get("case_id")
                        if c_id:
                            completed_case_ids.add(c_id)
                            completed_records.append(record)
                    except Exception:
                        pass
        if completed_case_ids:
            print(f"Resuming run: found {len(completed_case_ids)} already completed case(s).")

    cases_to_run = [c for c in all_cases if c["case_id"] not in completed_case_ids]
    total_to_run = len(all_cases)
    done_so_far = len(completed_case_ids)

    print(f"Cases to evaluate: {len(cases_to_run)} remaining of {total_to_run} total.\n")

    # 4. Open cases.jsonl in append mode
    with open(cases_file, "a", encoding="utf-8") as out_f:
        for idx, case in enumerate(cases_to_run):
            try:
                budget.check()
            except BudgetExceeded as be:
                print(f"\n[TOKEN BUDGET EXCEEDED] Execution paused cleanly: {be}", file=sys.stderr)
                print(f"Finished {done_so_far + idx} of {total_to_run} cases (used {be.used_tokens:,} tokens, limit {be.max_tokens:,}).", file=sys.stderr)
                print(f"To resume this evaluation, run:\n  python -m eval.run_reviewer_eval --run-dir {run_dir}", file=sys.stderr)
                sys.exit(1)

            current_num = done_so_far + idx + 1
            case_id = case["case_id"]
            is_wrong = case["is_wrong"]
            kind = case["kind"]
            wrong_str = "wrong" if is_wrong else "correct"

            # Execute analyst code in sandbox
            analyst_res = run_code(case["code"], df.copy(deep=True))
            if analyst_res.get("status") != "ok":
                analyst_result = None
                analyst_status = "failed"
            else:
                analyst_result = analyst_res.get("result")
                analyst_status = "ok"

            # Construct clean, minimal AgentState
            state = {
                "analysis_results": [
                    {
                        "id": 0,
                        "question": case["question"],
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

            # Guard: ensure no evaluation leak
            _assert_no_label_leakage(state)

            t0 = time.time()
            u0 = get_usage_stats()

            verdict = "unverified"
            failed_check = None
            reason = ""
            recomp_correct = None
            recomp_val_short = None
            recomp_res = None
            case_by_model = {}

            try:
                rev_out = reviewer_node(state)
                elapsed = time.time() - t0
                u1 = get_usage_stats()
                llm_calls = u1.get("calls", 0) - u0.get("calls", 0)
                tokens = u1.get("total_tokens", 0) - u0.get("total_tokens", 0)

                report_list = rev_out.get("review_report", [])
                if report_list:
                    rep = report_list[0]
                    verdict = rep.get("verdict", "unverified")
                    failed_check = rep.get("failed_check")
                    reason = rep.get("reason", "")
                    recomp_val_short = rep.get("recomputed_value")
                    recomp_res = rep.get("recomputed_result")

                    # Check reviewer's own recomputation accuracy against certified reference
                    if recomp_res is not None:
                        try:
                            ok_recomp, _ = answer_correct(case["reference_answer"], recomp_res)
                            recomp_correct = bool(ok_recomp)
                        except Exception:
                            recomp_correct = None
                else:
                    verdict = "unverified"
                    reason = "Reviewer returned empty report"

                # Track per-case by_model usage delta
                case_by_model: dict[str, dict[str, Any]] = {}
                u0_bm = u0.get("by_model", {})
                u1_bm = u1.get("by_model", {})
                for m, curr_stat in u1_bm.items():
                    prev_stat = u0_bm.get(m, {})
                    d_calls = curr_stat.get("calls", 0) - prev_stat.get("calls", 0)
                    d_toks = curr_stat.get("tokens", curr_stat.get("total_tokens", 0)) - prev_stat.get("tokens", prev_stat.get("total_tokens", 0))
                    if d_calls > 0 or d_toks > 0:
                        case_by_model[m] = {
                            "provider": curr_stat.get("provider", ""),
                            "calls": d_calls,
                            "tokens": d_toks,
                        }

            except (LLMUnavailableError, LLMConfigError) as llm_err:
                print(f"\n[AI SERVICE UNAVAILABLE] Execution paused cleanly: {llm_err}", file=sys.stderr)
                print(f"Finished {done_so_far + idx} of {total_to_run} cases.", file=sys.stderr)
                print(f"To resume this evaluation, run:\n  python -m eval.run_reviewer_eval --run-dir {run_dir}", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                elapsed = time.time() - t0
                llm_calls = 0
                tokens = 0
                verdict = "error"
                reason = f"Execution exception: {type(e).__name__}: {e}"

            # Format record
            record = {
                "case_id": case_id,
                "item_id": case["item_id"],
                "question": case["question"],
                "code": case["code"],
                "label": case["label"],
                "is_wrong": is_wrong,
                "kind": kind,
                "category": case["category"],
                "difficulty": case["difficulty"],
                "magnitude_bucket": case["magnitude_bucket"],
                "repeat_index": case["repeat_index"],
                "verdict": verdict,
                "failed_check": failed_check,
                "reason": reason,
                "recomputed_result_correct": recomp_correct,
                "recomputed_value": recomp_val_short,
                "analyst_result": _to_json_friendly(analyst_result, max_rows=8),
                "recomputed_result": _to_json_friendly(recomp_res, max_rows=8),
                "reference_answer": _to_json_friendly(case["reference_answer"], max_rows=8),
                "llm_calls": llm_calls,
                "tokens": tokens,
                "by_model": case_by_model if 'case_by_model' in locals() else {},
                "seconds": round(elapsed, 2),
            }

            out_f.write(json.dumps(_safe_serialize_case(record)) + "\n")
            out_f.flush()
            completed_records.append(record)

            print(
                f"[{current_num:02d}/{total_to_run:02d}] {case_id:16s} "
                f"{wrong_str:7s}/{kind:20s} -> {verdict:10s} ({elapsed:.1f}s)",
                flush=True,
            )

    # 5. Build and write summary
    generate_summary(completed_records, run_path, repeats=repeats)
    return str(run_path)


def generate_summary(
    records: list[dict[str, Any]],
    run_path: Path,
    repeats: int = 1,
) -> dict[str, Any]:
    """Generates summary metrics and writes summary.json and summary.md."""
    if not records:
        return {}

    conf = confusion(records)
    by_bucket = breakdown(records, "magnitude_bucket")
    by_kind = breakdown(records, "kind")
    by_category = breakdown(records, "category")
    by_difficulty = breakdown(records, "difficulty")

    total_time = sum(r.get("seconds", 0.0) for r in records)
    total_calls = sum(r.get("llm_calls", 0) for r in records)
    total_tokens = sum(r.get("tokens", 0) for r in records)
    errors_count = sum(1 for r in records if r.get("verdict") == "error")

    # Check distribution among detected cases
    failed_checks_counter: dict[str, int] = {}
    for r in records:
        if r.get("is_wrong") and r.get("verdict") == "fail":
            fc = r.get("failed_check") or "unspecified"
            failed_checks_counter[fc] = failed_checks_counter.get(fc, 0) + 1

    # Missed wrong cases (is_wrong=True, verdict='pass')
    missed_cases = [
        {
            "case_id": r.get("case_id", ""),
            "item_id": r.get("item_id", 0),
            "label": r.get("label", ""),
            "kind": r.get("kind", ""),
            "magnitude_bucket": r.get("magnitude_bucket", ""),
            "question": r.get("question", ""),
            "reason": r.get("reason", ""),
            "analyst_result": r.get("analyst_result"),
            "recomputed_result": r.get("recomputed_result"),
            "reference_answer": r.get("reference_answer"),
            "recomputed_value": r.get("recomputed_value"),
        }
        for r in records
        if r.get("is_wrong") and r.get("verdict") == "pass"
    ]

    # False alarms (is_wrong=False, verdict='fail')
    false_alarm_cases = [
        {
            "case_id": r.get("case_id", ""),
            "item_id": r.get("item_id", 0),
            "question": r.get("question", ""),
            "reason": r.get("reason", ""),
            "failed_check": r.get("failed_check"),
            "analyst_result": r.get("analyst_result"),
            "recomputed_result": r.get("recomputed_result"),
            "reference_answer": r.get("reference_answer"),
            "recomputed_value": r.get("recomputed_value"),
        }
        for r in records
        if (not r.get("is_wrong")) and r.get("verdict") == "fail"
    ]

    # Recomputation correctness analytics (Step 3b)
    correct_cases_list = [r for r in records if not r.get("is_wrong")]
    wrong_cases_list = [r for r in records if r.get("is_wrong")]
    detected_wrong_list = [r for r in wrong_cases_list if r.get("verdict") == "fail"]

    c_count = len(correct_cases_list)
    c_recomp_ok = sum(1 for r in correct_cases_list if r.get("recomputed_result_correct") is True)
    c_rate = round(c_recomp_ok / c_count, 4) if c_count > 0 else 0.0

    w_count = len(wrong_cases_list)
    w_recomp_ok = sum(1 for r in wrong_cases_list if r.get("recomputed_result_correct") is True)
    w_rate = round(w_recomp_ok / w_count, 4) if w_count > 0 else 0.0

    det_count = len(detected_wrong_list)
    detected_right_reason = sum(1 for r in detected_wrong_list if r.get("recomputed_result_correct") is True)
    detected_by_luck = sum(1 for r in detected_wrong_list if r.get("recomputed_result_correct") is not True)
    right_reason_rate = round(detected_right_reason / det_count, 4) if det_count > 0 else 0.0
    by_luck_rate = round(detected_by_luck / det_count, 4) if det_count > 0 else 0.0

    recomputation_accuracy = {
        "correct_cases": {
            "total": c_count,
            "recomputed_correct": c_recomp_ok,
            "rate": c_rate,
        },
        "wrong_cases": {
            "total": w_count,
            "recomputed_correct": w_recomp_ok,
            "rate": w_rate,
        },
        "detected_wrong_cases": {
            "total_detected": det_count,
            "detected_right_reason": detected_right_reason,
            "right_reason_rate": right_reason_rate,
            "detected_by_luck": detected_by_luck,
            "by_luck_rate": by_luck_rate,
        },
    }

    # Filter out 'none' from magnitude buckets (Step 3c) and annotate small sample caveats (Step 3d)
    magnitude_buckets: dict[str, Any] = {}
    small_sample_caveats: list[str] = []
    for b_name in ["subtle", "moderate", "gross", "structural"]:
        if b_name in by_bucket:
            b_data = dict(by_bucket[b_name])
            cnt = b_data.get("count", 0)
            if cnt < 10:
                caveat_str = f"small sample (n={cnt}), interpret with care"
                b_data["caveat"] = caveat_str
                small_sample_caveats.append(f"- `{b_name}`: {caveat_str}")
            magnitude_buckets[b_name] = b_data

    correct_passed = conf["total_correct"] - conf["false_alarms"] - conf["correct_unverified"]
    correct_cases_performance = {
        "count": conf["total_correct"],
        "passed": correct_passed,
        "false_alarms": conf["false_alarms"],
        "unverified": conf["correct_unverified"],
    }

    # Repeats analysis (flips)
    flip_count = 0
    if repeats > 1:
        base_groups: dict[str, list[str]] = {}
        for r in records:
            base_id = r["case_id"].split("_r")[0]
            base_groups.setdefault(base_id, []).append(r.get("verdict", ""))
        for b_id, v_list in base_groups.items():
            if len(set(v_list)) > 1:
                flip_count += 1

    # Aggregate by_model across all records
    aggregated_by_model: dict[str, dict[str, Any]] = {}
    for r in records:
        r_bm = r.get("by_model", {})
        for m, m_data in r_bm.items():
            if m not in aggregated_by_model:
                aggregated_by_model[m] = {
                    "provider": m_data.get("provider", ""),
                    "calls": 0,
                    "tokens": 0,
                }
            aggregated_by_model[m]["calls"] += m_data.get("calls", 0)
            aggregated_by_model[m]["tokens"] += m_data.get("tokens", m_data.get("total_tokens", 0))

    if not aggregated_by_model:
        current_bm = get_usage_stats().get("by_model", {})
        for m, m_data in current_bm.items():
            aggregated_by_model[m] = {
                "provider": m_data.get("provider", ""),
                "calls": m_data.get("calls", 0),
                "tokens": m_data.get("tokens", m_data.get("total_tokens", 0)),
            }

    summary_data = {
        "totals": {
            "total_cases": len(records),
            "correct_cases": conf["total_correct"],
            "wrong_cases": conf["total_wrong"],
            "errors": errors_count,
            "llm_calls": total_calls,
            "total_tokens": total_tokens,
            "total_seconds": round(total_time, 2),
            "by_model": aggregated_by_model,
            "models_used": list(aggregated_by_model.keys()),
            "repeats": repeats,
            "flip_cases_count": flip_count if repeats > 1 else None,
        },
        "overall": {
            "detection_rate": conf["detection_rate"],
            "detection_rate_ci": conf["detection_rate_ci"],
            "detection_counts": f"{conf['detection']}/{conf['total_wrong']}",
            "false_alarm_rate": conf["false_alarm_rate"],
            "false_alarm_rate_ci": conf["false_alarm_rate_ci"],
            "false_alarm_counts": f"{conf['false_alarms']}/{conf['total_correct']}",
            "unverified_rate": conf["unverified_rate"],
            "unverified_rate_ci": conf["unverified_rate_ci"],
            "unverified_counts": f"{conf['unverified_rate'] * len(records):.0f}/{len(records)}",
            "precision": conf["precision"],
            "precision_ci": conf["precision_ci"],
            "precision_counts": f"{conf['detection']}/{conf['detection'] + conf['false_alarms']}",
        },
        "confusion_matrix": conf,
        "recomputation_accuracy": recomputation_accuracy,
        "correct_cases_performance": correct_cases_performance,
        "failed_checks_distribution": failed_checks_counter,
        "breakdowns": {
            "by_magnitude_bucket": magnitude_buckets,
            "by_kind": by_kind,
            "by_category": by_category,
            "by_difficulty": by_difficulty,
        },
        "missed_cases": missed_cases,
        "false_alarm_cases": false_alarm_cases,
        "limitations": (
            "Evaluated on 20 questions against data/sample.csv. Takeaway verification is not "
            "exercised in this isolated agent test setup. LLM outputs may vary across calls."
        ),
    }

    # Save summary.json
    with open(run_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save summary.md
    models_display = ", ".join(f"`{m}` ({d['calls']} calls, {d['tokens']:,} tokens)" for m, d in aggregated_by_model.items()) if aggregated_by_model else "none"

    md_lines = [
        "# Reviewer Agent Evaluation Summary",
        "",
        f"**Run Path**: `{run_path.resolve()}`  ",
        f"**Total Cases**: {len(records)} (Correct: {conf['total_correct']}, Wrong: {conf['total_wrong']}, Errors: {errors_count})  ",
        f"**Resources**: {total_calls} LLM calls, {total_tokens:,} tokens, {total_time:.1f}s execution time  ",
        f"**Models Used**: {models_display}  ",
    ]
    if len(aggregated_by_model) > 1:
        md_lines.append("> Multiple models were used in this run  \n")
    md_lines.extend([
        "",
        "## Overall Metrics",
        "",
        "| Metric | Rate | 95% Wilson CI | Counts |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Detection Rate** (Sensitivity) | {conf['detection_rate']*100:.1f}% | [{conf['detection_rate_ci'][0]*100:.1f}%, {conf['detection_rate_ci'][1]*100:.1f}%] | {conf['detection']}/{conf['total_wrong']} |",
        f"| **False Alarm Rate** (1 - Specificity) | {conf['false_alarm_rate']*100:.1f}% | [{conf['false_alarm_rate_ci'][0]*100:.1f}%, {conf['false_alarm_rate_ci'][1]*100:.1f}%] | {conf['false_alarms']}/{conf['total_correct']} |",
        f"| **Unverified Rate** | {conf['unverified_rate']*100:.1f}% | [{conf['unverified_rate_ci'][0]*100:.1f}%, {conf['unverified_rate_ci'][1]*100:.1f}%] | {conf['wrong_unverified'] + conf['correct_unverified']}/{len(records)} |",
        f"| **Precision** | {conf['precision']*100:.1f}% | [{conf['precision_ci'][0]*100:.1f}%, {conf['precision_ci'][1]*100:.1f}%] | {conf['detection']}/{conf['detection'] + conf['false_alarms']} |",
        "",
        "## Correct Cases Performance",
        "",
        "| Cases | Passed | False Alarms | Unverified |",
        "| :--- | :--- | :--- | :--- |",
        f"| {conf['total_correct']} | {correct_passed} | {conf['false_alarms']} | {conf['correct_unverified']} |",
        "",
        "## Was the Reviewer's Own Recomputation Correct?",
        "",
        "| Cohort | Total Cases | Recompute Matched Reference | Rate |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Correct Cases** | {c_count} | {c_recomp_ok} | {c_rate*100:.1f}% |",
        f"| **Wrong Cases** | {w_count} | {w_recomp_ok} | {w_rate*100:.1f}% |",
        "",
        "### Detected Wrong Cases Breakdown",
        f"Among the **{det_count}** detected wrong cases:",
        f"- **Detected for the right reason** (recompute matched reference): **{detected_right_reason}** ({right_reason_rate*100:.1f}%)",
        f"- **Detected by luck** (recompute was also wrong): **{detected_by_luck}** ({by_luck_rate*100:.1f}%)",
        "",
    ])

    if repeats > 1:
        md_lines.extend([
            f"**Verdict Flips across repeats**: {flip_count} / {len(records) // repeats} cases had differing verdicts across repeats.",
            "",
        ])

    # Breakdown by Magnitude Bucket
    md_lines.extend([
        "## Breakdown by Error Magnitude Bucket",
        "",
        "| Bucket | Cases | Detected | Missed | Unverified | Detection Rate (95% CI) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for b_name in ["subtle", "moderate", "gross", "structural"]:
        if b_name in magnitude_buckets:
            b_stats = magnitude_buckets[b_name]
            md_lines.append(
                f"| `{b_name}` | {b_stats['count']} | {b_stats['detection']} | "
                f"{b_stats['missed']} | {b_stats['wrong_unverified']} | "
                f"{b_stats['detection_rate']*100:.1f}% [{b_stats['detection_rate_ci'][0]*100:.1f}%, {b_stats['detection_rate_ci'][1]*100:.1f}%] |"
            )
    if small_sample_caveats:
        md_lines.append("")
        md_lines.extend(small_sample_caveats)
    md_lines.append("")

    # Breakdown by Error Kind
    md_lines.extend([
        "## Breakdown by Error Kind",
        "",
        "| Error Kind | Cases | Detected | Missed | Detection Rate |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])
    for k_name, k_stats in by_kind.items():
        if k_name != "correct":
            md_lines.append(
                f"| `{k_name}` | {k_stats['count']} | {k_stats['detection']} | "
                f"{k_stats['missed']} | {k_stats['detection_rate']*100:.1f}% |"
            )
    md_lines.append("")

    # Checks caught
    md_lines.extend([
        "## Detection Mechanism Distribution",
        "",
        "| Failed Check Type | Detected Cases Count |",
        "| :--- | :--- |",
    ])
    for check_type, c_count in sorted(failed_checks_counter.items()):
        md_lines.append(f"| `{check_type}` | {c_count} |")
    md_lines.append("")

    # Missed wrong cases
    md_lines.extend([
        "## Missed Wrong Cases (Verdicts = Pass)",
        "",
    ])
    if missed_cases:
        for m in missed_cases:
            md_lines.extend([
                f"- **Case**: `{m['case_id']}` [{m['kind']}] ({m['magnitude_bucket']})",
                f"  - **Label**: {m['label']}",
                f"  - **Question**: *{m['question']}*",
                f"  - **Analyst Result**: {_format_markdown_value(m.get('analyst_result'))}",
                f"  - **Reviewer Recomputed**: {_format_markdown_value(m.get('recomputed_result'))}",
                f"  - **Reference Answer**: {_format_markdown_value(m.get('reference_answer'))}",
                f"  - **Reason**: {m.get('reason', '')}",
            ])
    else:
        md_lines.append("*None. All wrong cases were successfully flagged.*")
    md_lines.append("")

    # False alarm cases
    md_lines.extend([
        "## False Alarms (Verdicts = Fail on Correct Cases)",
        "",
    ])
    if false_alarm_cases:
        for fa in false_alarm_cases:
            md_lines.extend([
                f"- **Case**: `{fa['case_id']}`",
                f"  - **Question**: *{fa['question']}*",
                f"  - **Reason**: {fa['reason']}",
                f"  - **Failed Check**: `{fa.get('failed_check', '')}`",
                f"  - **Analyst Result**: {_format_markdown_value(fa.get('analyst_result'))}",
                f"  - **Reviewer Recomputed**: {_format_markdown_value(fa.get('recomputed_result'))}",
                f"  - **Reference Answer**: {_format_markdown_value(fa.get('reference_answer'))}",
            ])
    else:
        md_lines.append("*None. No false alarms encountered on correct cases.*")
    md_lines.append("")

    # Limitations
    md_lines.extend([
        "---",
        "> [!NOTE] Evaluation Limitations",
        "> Evaluated on 20 standardized questions on `data/sample.csv`. The chart takeaway verification check is not exercised in this isolated reviewer evaluation setup. Results may vary slightly between LLM invocations.",
    ])

    with open(run_path / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Print human-readable summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY RESULTS")
    print("=" * 80)
    print(f"Total Cases:     {len(records)} (Correct: {conf['total_correct']}, Wrong: {conf['total_wrong']}, Errors: {errors_count})")
    print(f"Detection Rate:  {conf['detection_rate']*100:.1f}% ({conf['detection']}/{conf['total_wrong']}) [95% CI: {conf['detection_rate_ci'][0]*100:.1f}% - {conf['detection_rate_ci'][1]*100:.1f}%]")
    print(f"False Alarm Rate:{conf['false_alarm_rate']*100:.1f}% ({conf['false_alarms']}/{conf['total_correct']}) [95% CI: {conf['false_alarm_rate_ci'][0]*100:.1f}% - {conf['false_alarm_rate_ci'][1]*100:.1f}%]")
    print(f"Unverified Rate: {conf['unverified_rate']*100:.1f}% [95% CI: {conf['unverified_rate_ci'][0]*100:.1f}% - {conf['unverified_rate_ci'][1]*100:.1f}%]")
    print(f"Precision:       {conf['precision']*100:.1f}% [95% CI: {conf['precision_ci'][0]*100:.1f}% - {conf['precision_ci'][1]*100:.1f}%]")
    print(f"LLM Calls:       {total_calls} | Tokens: {total_tokens:,} | Seconds: {total_time:.1f}s")
    if aggregated_by_model:
        models_print = ", ".join(f"{m} ({d['calls']} calls, {d['tokens']:,} tokens)" for m, d in aggregated_by_model.items())
        print(f"Models Used:     {models_print}")
    if len(aggregated_by_model) > 1:
        print("Multiple models were used in this run")
    print(f"Summary saved:   {run_path / 'summary.md'}")
    print("=" * 80 + "\n")

    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Reviewer agent in isolation against benchmark.")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N cases (interleaved correct & wrong)")
    parser.add_argument("--repeats", type=int, default=1, help="Number of times to repeat each case (default 1)")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to store evaluation artifacts")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens allowed before stopping cleanly")

    args = parser.parse_args()
    run_evaluation(
        limit=args.limit,
        repeats=args.repeats,
        run_dir=args.run_dir,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
