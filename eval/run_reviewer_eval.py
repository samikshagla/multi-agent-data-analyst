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
import re
import sys
import time
from typing import Any
import numpy as np
import pandas as pd

from agents.reviewer import reviewer_node
from eval.benchmark import BENCHMARK
from eval.budget import BudgetExceeded, TokenBudget
from eval.ground_truth import compute_ground_truth, load_ground_truth, save_ground_truth
from eval.metrics import breakdown, confusion, magnitude_bucket, wilson_ci, wilson_ci_from_rate
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
    generate_summary(completed_records, run_path, repeats=repeats, expected_total=total_to_run)
    return str(run_path)


def _get_base_case_id(case_id: str) -> str:
    """Strips repeat suffix e.g. '_r1' from case_id."""
    return re.sub(r"_r\d+$", "", str(case_id))


def _get_repeat_idx(r: dict[str, Any]) -> int:
    """Extracts 0-based repeat index from record."""
    if "repeat_index" in r and r["repeat_index"] is not None:
        return int(r["repeat_index"])
    m = re.search(r"_r(\d+)$", str(r.get("case_id", "")))
    return int(m.group(1)) - 1 if m else 0


def _calc_subgroup(
    sub_records: list[dict[str, Any]],
    sorted_r_indices: list[int],
    is_correct: bool = False,
) -> dict[str, Any]:
    """Computes subgroup summary statistics across repeats."""
    if not sub_records:
        return {
            "unique_cases": 0,
            "observations": 0,
            "mean_detected": 0.0,
            "detection_rate": 0.0,
            "ci": (0.0, 0.0),
        }
    u_ids = set(_get_base_case_id(r["case_id"]) for r in sub_records)
    n_u = len(u_ids)
    obs = len(sub_records)

    r_counts: list[int] = []
    r_rates: list[float] = []
    for r_idx in sorted_r_indices:
        r_recs = [r for r in sub_records if _get_repeat_idx(r) == r_idx]
        n_r = len(r_recs)
        d_r = sum(1 for r in r_recs if r.get("verdict") == "fail")
        rate_r = (d_r / n_r) if n_r > 0 else 0.0
        r_counts.append(d_r)
        if n_r > 0:
            r_rates.append(rate_r)

    m_det = float(np.mean(r_counts)) if r_counts else 0.0
    m_rate = float(np.mean(r_rates)) if r_rates else 0.0
    ci = wilson_ci_from_rate(m_rate, n_u)
    return {
        "unique_cases": n_u,
        "observations": obs,
        "mean_detected": m_det,
        "detection_rate": m_rate,
        "ci": ci,
    }


def summarize_run(
    run_dir: str | Path,
    repeats: int | None = None,
    limit: int | None = None,
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
        records,
        run_path,
        repeats=repeats,
        expected_total=limit,
    )


def generate_summary(
    records: list[dict[str, Any]],
    run_path: Path,
    repeats: int | None = None,
    expected_total: int | None = None,
) -> dict[str, Any]:
    """Generates summary metrics and writes summary.json and summary.md."""
    if not records:
        print("No evaluation records provided to generate_summary.")
        return {}

    # 1. Detect repeats and expected_total
    detected_r_indices: set[int] = set()
    for r in records:
        if "repeat_index" in r and r["repeat_index"] is not None:
            detected_r_indices.add(int(r["repeat_index"]))
        else:
            m = re.search(r"_r(\d+)$", str(r.get("case_id", "")))
            if m:
                detected_r_indices.add(int(m.group(1)) - 1)
            else:
                detected_r_indices.add(0)

    max_detected_repeats = max(detected_r_indices) + 1 if detected_r_indices else 1
    if repeats is None:
        repeats = max_detected_repeats
    else:
        repeats = max(repeats, max_detected_repeats)

    if expected_total is None:
        expected_total = 60 * repeats

    n_finished = len(records)
    is_partial = n_finished < expected_total

    # 2. Per-repeat statistics & metrics
    records_by_repeat: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        r_idx = _get_repeat_idx(r)
        records_by_repeat.setdefault(r_idx, []).append(r)

    sorted_r_indices = sorted(records_by_repeat.keys())
    per_repeat_stats: list[dict[str, Any]] = []

    for r_idx in sorted_r_indices:
        r_recs = records_by_repeat[r_idx]
        r_label = f"r{r_idx + 1}"
        r_wrong = [r for r in r_recs if r.get("is_wrong")]
        r_correct = [r for r in r_recs if not r.get("is_wrong")]

        n_fin_r = len(r_recs)
        n_wrong_r = len(r_wrong)
        n_correct_r = len(r_correct)

        det_r = sum(1 for r in r_wrong if r.get("verdict") == "fail")
        det_rate_r = (det_r / n_wrong_r) if n_wrong_r > 0 else 0.0
        det_ci_r = wilson_ci(det_r, n_wrong_r)

        right_r = sum(1 for r in r_wrong if r.get("verdict") == "fail" and r.get("recomputed_result_correct") is True)
        right_rate_r = (right_r / n_wrong_r) if n_wrong_r > 0 else 0.0
        right_ci_r = wilson_ci(right_r, n_wrong_r)

        luck_r = sum(1 for r in r_wrong if r.get("verdict") == "fail" and r.get("recomputed_result_correct") is not True)
        luck_rate_r = (luck_r / n_wrong_r) if n_wrong_r > 0 else 0.0

        fa_r = sum(1 for r in r_correct if r.get("verdict") == "fail")
        fa_rate_r = (fa_r / n_correct_r) if n_correct_r > 0 else 0.0
        fa_ci_r = wilson_ci(fa_r, n_correct_r)

        unver_r = sum(1 for r in r_recs if r.get("verdict") == "unverified")
        unver_rate_r = (unver_r / n_fin_r) if n_fin_r > 0 else 0.0
        unver_ci_r = wilson_ci(unver_r, n_fin_r)

        total_flags_r = det_r + fa_r
        prec_r = (det_r / total_flags_r) if total_flags_r > 0 else 0.0
        prec_ci_r = wilson_ci(det_r, total_flags_r)

        per_repeat_stats.append({
            "repeat": r_label,
            "repeat_index": r_idx,
            "finished_cases": n_fin_r,
            "total_wrong": n_wrong_r,
            "detection": det_r,
            "detection_rate": det_rate_r,
            "detection_rate_ci": det_ci_r,
            "detection_counts": f"{det_r}/{n_wrong_r}",
            "detected_right_reason": right_r,
            "right_reason_rate": right_rate_r,
            "right_reason_rate_ci": right_ci_r,
            "right_reason_counts": f"{right_r}/{n_wrong_r}",
            "detected_by_luck": luck_r,
            "by_luck_rate": luck_rate_r,
            "total_correct": n_correct_r,
            "false_alarms": fa_r,
            "false_alarm_rate": fa_rate_r,
            "false_alarm_rate_ci": fa_ci_r,
            "false_alarm_counts": f"{fa_r}/{n_correct_r}",
            "unverified": unver_r,
            "unverified_rate": unver_rate_r,
            "unverified_rate_ci": unver_ci_r,
            "precision": prec_r,
            "precision_ci": prec_ci_r,
        })

    # 3. Overall headline metrics: mean of per-repeat rates, CI on unique cases
    unique_wrong_ids = set(_get_base_case_id(r["case_id"]) for r in records if r.get("is_wrong"))
    n_unique_wrong = len(unique_wrong_ids)

    unique_correct_ids = set(_get_base_case_id(r["case_id"]) for r in records if not r.get("is_wrong"))
    n_unique_correct = len(unique_correct_ids)

    unique_all_ids = set(_get_base_case_id(r["case_id"]) for r in records)
    n_unique_all = len(unique_all_ids)

    det_rates = [s["detection_rate"] for s in per_repeat_stats if s["total_wrong"] > 0]
    headline_det_rate = float(np.mean(det_rates)) if det_rates else 0.0
    headline_det_ci = wilson_ci_from_rate(headline_det_rate, n_unique_wrong)

    right_rates = [s["right_reason_rate"] for s in per_repeat_stats if s["total_wrong"] > 0]
    headline_right_reason_rate = float(np.mean(right_rates)) if right_rates else 0.0
    headline_right_reason_ci = wilson_ci_from_rate(headline_right_reason_rate, n_unique_wrong)

    fa_rates = [s["false_alarm_rate"] for s in per_repeat_stats if s["total_correct"] > 0]
    headline_fa_rate = float(np.mean(fa_rates)) if fa_rates else 0.0
    headline_fa_ci = wilson_ci_from_rate(headline_fa_rate, n_unique_correct)

    unver_rates = [s["unverified_rate"] for s in per_repeat_stats if s["finished_cases"] > 0]
    headline_unver_rate = float(np.mean(unver_rates)) if unver_rates else 0.0
    headline_unver_ci = wilson_ci_from_rate(headline_unver_rate, n_unique_all)

    prec_rates = [s["precision"] for s in per_repeat_stats if (s["detection"] + s["false_alarms"]) > 0]
    headline_precision = float(np.mean(prec_rates)) if prec_rates else 0.0
    mean_fail_verdicts = float(np.mean([s["detection"] + s["false_alarms"] for s in per_repeat_stats])) if per_repeat_stats else 0.0
    headline_prec_ci = wilson_ci_from_rate(headline_precision, mean_fail_verdicts)

    total_det = sum(s["detection"] for s in per_repeat_stats)
    total_w = sum(s["total_wrong"] for s in per_repeat_stats)
    total_right_reason = sum(s["detected_right_reason"] for s in per_repeat_stats)
    total_luck = sum(s["detected_by_luck"] for s in per_repeat_stats)
    total_fa = sum(s["false_alarms"] for s in per_repeat_stats)
    total_c = sum(s["total_correct"] for s in per_repeat_stats)
    total_unver = sum(s["unverified"] for s in per_repeat_stats)
    total_flags = total_det + total_fa

    mean_det = float(np.mean([s["detection"] for s in per_repeat_stats])) if per_repeat_stats else 0.0
    mean_right_reason = float(np.mean([s["detected_right_reason"] for s in per_repeat_stats])) if per_repeat_stats else 0.0
    mean_fa = float(np.mean([s["false_alarms"] for s in per_repeat_stats])) if per_repeat_stats else 0.0
    mean_unver = float(np.mean([s["unverified"] for s in per_repeat_stats])) if per_repeat_stats else 0.0

    if repeats > 1:
        det_counts_str = f"{total_det}/{total_w} ({mean_det:.1f}/{n_unique_wrong} avg)"
        right_reason_counts_str = f"{total_right_reason}/{total_w} ({mean_right_reason:.1f}/{n_unique_wrong} avg)"
        fa_counts_str = f"{total_fa}/{total_c} ({mean_fa:.1f}/{n_unique_correct} avg)"
        unver_counts_str = f"{total_unver}/{n_finished} ({mean_unver:.1f}/{n_unique_all} avg)"
        prec_counts_str = f"{total_det}/{total_flags} ({mean_det:.1f}/{mean_fail_verdicts:.1f} avg)" if total_flags > 0 else "0/0"
    else:
        det_counts_str = f"{total_det}/{total_w}"
        right_reason_counts_str = f"{total_right_reason}/{total_w}"
        fa_counts_str = f"{total_fa}/{total_c}"
        unver_counts_str = f"{total_unver}/{n_finished}"
        prec_counts_str = f"{total_det}/{total_flags}" if total_flags > 0 else "0/0"

    # 4. Verdict flips across repeats
    base_groups: dict[str, dict[str, Any]] = {}
    for r in records:
        base_id = _get_base_case_id(r["case_id"])
        r_idx = r.get("repeat_index")
        if r_idx is None:
            m = re.search(r"_r(\d+)$", str(r.get("case_id", "")))
            r_idx = int(m.group(1)) - 1 if m else 0
        r_label = f"r{r_idx + 1}"

        if base_id not in base_groups:
            base_groups[base_id] = {
                "base_id": base_id,
                "label": r.get("label", ""),
                "kind": r.get("kind", ""),
                "is_wrong": r.get("is_wrong", False),
                "verdicts": {},
            }
        base_groups[base_id]["verdicts"][r_label] = r.get("verdict", "")

    multi_repeat_cases = [info for info in base_groups.values() if len(info["verdicts"]) >= 2]
    identical_cases = [
        info for info in multi_repeat_cases
        if len(set(info["verdicts"].values())) == 1
    ]
    flipped_cases = [
        info for info in multi_repeat_cases
        if len(set(info["verdicts"].values())) > 1
    ]
    identical_count = len(identical_cases)
    flip_count = len(flipped_cases)

    # 5. Cohort calculations & breakdowns
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

    # Recomputation correctness analytics
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

    magnitude_buckets: dict[str, Any] = {}
    small_sample_caveats: list[str] = []
    for b_name in ["subtle", "moderate", "gross", "structural"]:
        b_records = [r for r in records if r.get("magnitude_bucket") == b_name and r.get("is_wrong")]
        if b_records:
            sg = _calc_subgroup(b_records, sorted_r_indices, is_correct=False)
            b_data: dict[str, Any] = {
                "count": sg["observations"],
                "unique_cases": sg["unique_cases"],
                "observations": sg["observations"],
                "detection": sum(1 for r in b_records if r.get("verdict") == "fail"),
                "missed": sum(1 for r in b_records if r.get("verdict") == "pass"),
                "wrong_unverified": sum(1 for r in b_records if r.get("verdict") not in ("fail", "pass")),
                "total_wrong": sg["observations"],
                "detected_mean": sg["mean_detected"],
                "detection_rate": sg["detection_rate"],
                "detection_rate_ci": list(sg["ci"]),
            }
            if sg["unique_cases"] < 10:
                caveat_str = f"small sample (n={sg['unique_cases']}), interpret with care"
                b_data["caveat"] = caveat_str
                small_sample_caveats.append(f"- `{b_name}`: {caveat_str}")
            magnitude_buckets[b_name] = b_data

    by_kind_computed: dict[str, Any] = {}
    all_kinds = sorted(set(str(r.get("kind", "")) for r in records if r.get("kind")))
    for k_name in all_kinds:
        is_corr = (k_name == "correct")
        if is_corr:
            k_records = [r for r in records if r.get("kind") == k_name and not r.get("is_wrong")]
        else:
            k_records = [r for r in records if r.get("kind") == k_name and r.get("is_wrong")]
        if not k_records:
            continue
        sg = _calc_subgroup(k_records, sorted_r_indices, is_correct=is_corr)
        k_data: dict[str, Any] = {
            "count": sg["observations"],
            "unique_cases": sg["unique_cases"],
            "observations": sg["observations"],
            "detection": sum(1 for r in k_records if r.get("verdict") == "fail"),
            "missed": sum(1 for r in k_records if r.get("verdict") == "pass"),
            "detected_mean": sg["mean_detected"],
            "detection_rate": sg["detection_rate"],
            "detection_rate_ci": list(sg["ci"]),
            "is_correct": is_corr,
        }
        if is_corr:
            k_data["false_alarms"] = k_data["detection"]
            k_data["false_alarm_rate"] = sg["detection_rate"]
            k_data["false_alarm_rate_ci"] = list(sg["ci"])
        by_kind_computed[k_name] = k_data

    by_category_computed: dict[str, Any] = {}
    all_cats = sorted(set(str(r.get("category", "")) for r in records if r.get("category")))
    for cat in all_cats:
        cat_wrong = [r for r in records if r.get("category") == cat and r.get("is_wrong")]
        if not cat_wrong:
            continue
        sg = _calc_subgroup(cat_wrong, sorted_r_indices, is_correct=False)
        cat_data: dict[str, Any] = {
            "count": sg["observations"],
            "unique_cases": sg["unique_cases"],
            "observations": sg["observations"],
            "detection": sum(1 for r in cat_wrong if r.get("verdict") == "fail"),
            "missed": sum(1 for r in cat_wrong if r.get("verdict") == "pass"),
            "detected_mean": sg["mean_detected"],
            "detection_rate": sg["detection_rate"],
            "detection_rate_ci": list(sg["ci"]),
        }
        cat_correct = [r for r in records if r.get("category") == cat and not r.get("is_wrong")]
        if cat_correct:
            sg_c = _calc_subgroup(cat_correct, sorted_r_indices, is_correct=True)
            cat_data["unique_correct_cases"] = sg_c["unique_cases"]
            cat_data["total_correct"] = sg_c["observations"]
            cat_data["false_alarms"] = sum(1 for r in cat_correct if r.get("verdict") == "fail")
            cat_data["false_alarm_rate"] = sg_c["detection_rate"]
            cat_data["false_alarm_rate_ci"] = list(sg_c["ci"])
        by_category_computed[cat] = cat_data

    by_difficulty_computed: dict[str, Any] = {}
    all_diffs = sorted(set(str(r.get("difficulty", "")) for r in records if r.get("difficulty")))
    for diff in all_diffs:
        diff_wrong = [r for r in records if r.get("difficulty") == diff and r.get("is_wrong")]
        if not diff_wrong:
            continue
        sg = _calc_subgroup(diff_wrong, sorted_r_indices, is_correct=False)
        diff_data: dict[str, Any] = {
            "count": sg["observations"],
            "unique_cases": sg["unique_cases"],
            "observations": sg["observations"],
            "detection": sum(1 for r in diff_wrong if r.get("verdict") == "fail"),
            "missed": sum(1 for r in diff_wrong if r.get("verdict") == "pass"),
            "detected_mean": sg["mean_detected"],
            "detection_rate": sg["detection_rate"],
            "detection_rate_ci": list(sg["ci"]),
        }
        diff_correct = [r for r in records if r.get("difficulty") == diff and not r.get("is_wrong")]
        if diff_correct:
            sg_c = _calc_subgroup(diff_correct, sorted_r_indices, is_correct=True)
            diff_data["unique_correct_cases"] = sg_c["unique_cases"]
            diff_data["total_correct"] = sg_c["observations"]
            diff_data["false_alarms"] = sum(1 for r in diff_correct if r.get("verdict") == "fail")
            diff_data["false_alarm_rate"] = sg_c["detection_rate"]
            diff_data["false_alarm_rate_ci"] = list(sg_c["ci"])
        by_difficulty_computed[diff] = diff_data

    correct_passed = total_c - total_fa - conf.get("correct_unverified", 0)
    correct_cases_performance = {
        "count": total_c,
        "passed": correct_passed,
        "false_alarms": total_fa,
        "unverified": conf.get("correct_unverified", 0),
    }

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
        "is_partial": is_partial,
        "expected_total_cases": expected_total,
        "finished_cases": n_finished,
        "totals": {
            "total_cases": n_finished,
            "expected_total_cases": expected_total,
            "unique_cases": n_unique_all,
            "unique_wrong_cases": n_unique_wrong,
            "unique_correct_cases": n_unique_correct,
            "correct_cases": total_c,
            "wrong_cases": total_w,
            "errors": errors_count,
            "llm_calls": total_calls,
            "total_tokens": total_tokens,
            "total_seconds": round(total_time, 2),
            "by_model": aggregated_by_model,
            "models_used": list(aggregated_by_model.keys()),
            "repeats": repeats,
            "flip_cases_count": flip_count if multi_repeat_cases else None,
            "identical_verdicts_count": identical_count if multi_repeat_cases else None,
        },
        "overall": {
            "detection_rate": headline_det_rate,
            "detection_rate_ci": list(headline_det_ci),
            "detection_counts": det_counts_str,
            "right_reason_rate": headline_right_reason_rate,
            "right_reason_rate_ci": list(headline_right_reason_ci),
            "right_reason_counts": right_reason_counts_str,
            "false_alarm_rate": headline_fa_rate,
            "false_alarm_rate_ci": list(headline_fa_ci),
            "false_alarm_counts": fa_counts_str,
            "unverified_rate": headline_unver_rate,
            "unverified_rate_ci": list(headline_unver_ci),
            "unverified_counts": unver_counts_str,
            "precision": headline_precision,
            "precision_ci": list(headline_prec_ci),
            "precision_counts": prec_counts_str,
        },
        "per_repeat": per_repeat_stats,
        "verdict_flips": {
            "multi_repeat_cases_count": len(multi_repeat_cases),
            "identical_verdicts_count": identical_count,
            "flipped_verdicts_count": flip_count,
            "flips": [
                {
                    "case_id": fc["base_id"],
                    "label": fc["label"],
                    "kind": fc["kind"],
                    "verdicts": fc["verdicts"],
                }
                for fc in flipped_cases
            ],
        },
        "confusion_matrix": conf,
        "recomputation_accuracy": recomputation_accuracy,
        "correct_cases_performance": correct_cases_performance,
        "failed_checks_distribution": failed_checks_counter,
        "breakdowns": {
            "by_magnitude_bucket": magnitude_buckets,
            "by_kind": by_kind_computed,
            "by_category": by_category_computed,
            "by_difficulty": by_difficulty_computed,
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

    md_lines: list[str] = []
    if is_partial:
        md_lines.append(f"PARTIAL RUN: {n_finished} of {expected_total} cases finished")
        md_lines.append("")

    md_lines.extend([
        "# Reviewer Agent Evaluation Summary",
        "",
        f"**Run Path**: `{run_path.resolve()}`  ",
        f"**Total Cases Finished**: {n_finished} (Unique: {n_unique_all}, Correct: {total_c}, Wrong: {total_w}, Errors: {errors_count})  ",
        f"**Resources**: {total_calls} LLM calls, {total_tokens:,} tokens, {total_time:.1f}s execution time  ",
        f"**Models Used**: {models_display}  ",
    ])
    if len(aggregated_by_model) > 1:
        md_lines.append("> Multiple models were used in this run  \n")

    md_lines.append("")
    md_lines.append("## Overall Metrics")
    md_lines.append("")
    if repeats > 1:
        md_lines.extend([
            "| Metric | Unique cases | Observations | Detected (mean per repeat) | Detection rate (mean over repeats) | 95% Wilson CI |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| **Detection Rate** (Sensitivity) | {n_unique_wrong} | {total_w} | {mean_det:.1f} | {headline_det_rate*100:.1f}% | [{headline_det_ci[0]*100:.1f}%, {headline_det_ci[1]*100:.1f}%] |",
            f"| **Detection for the right reason** | {n_unique_wrong} | {total_w} | {mean_right_reason:.1f} | {headline_right_reason_rate*100:.1f}% | [{headline_right_reason_ci[0]*100:.1f}%, {headline_right_reason_ci[1]*100:.1f}%] |",
            f"| **False Alarm Rate** (1 - Specificity) | {n_unique_correct} | {total_c} | {mean_fa:.1f} | {headline_fa_rate*100:.1f}% | [{headline_fa_ci[0]*100:.1f}%, {headline_fa_ci[1]*100:.1f}%] |",
            f"| **Unverified Rate** | {n_unique_all} | {n_finished} | {mean_unver:.1f} | {headline_unver_rate*100:.1f}% | [{headline_unver_ci[0]*100:.1f}%, {headline_unver_ci[1]*100:.1f}%] |",
            f"| **Precision** | {mean_fail_verdicts:.1f} | {total_flags} | {mean_det:.1f} | {headline_precision*100:.1f}% | [{headline_prec_ci[0]*100:.1f}%, {headline_prec_ci[1]*100:.1f}%] |",
        ])
    else:
        md_lines.extend([
            "| Metric | Rate | 95% Wilson CI | Counts |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Detection Rate** (Sensitivity) | {headline_det_rate*100:.1f}% | [{headline_det_ci[0]*100:.1f}%, {headline_det_ci[1]*100:.1f}%] | {det_counts_str} |",
            f"| **Detection for the right reason** | {headline_right_reason_rate*100:.1f}% | [{headline_right_reason_ci[0]*100:.1f}%, {headline_right_reason_ci[1]*100:.1f}%] | {right_reason_counts_str} |",
            f"| **False Alarm Rate** (1 - Specificity) | {headline_fa_rate*100:.1f}% | [{headline_fa_ci[0]*100:.1f}%, {headline_fa_ci[1]*100:.1f}%] | {fa_counts_str} |",
            f"| **Unverified Rate** | {headline_unver_rate*100:.1f}% | [{headline_unver_ci[0]*100:.1f}%, {headline_unver_ci[1]*100:.1f}%] | {unver_counts_str} |",
            f"| **Precision** | {headline_precision*100:.1f}% | [{headline_prec_ci[0]*100:.1f}%, {headline_prec_ci[1]*100:.1f}%] | {prec_counts_str} |",
        ])
    md_lines.append("")
    md_lines.append("Intervals treat the benchmark cases (not the repeats) as the sample; repeats only average out LLM randomness.  ")
    md_lines.append("'By luck' means the reviewer flagged an answer while its own recomputation was also wrong.")
    md_lines.append("")
    md_lines.extend([
        "## Per-Repeat Performance",
        "",
        "| Repeat | Finished Cases | Detection Rate (95% CI) | Detection Counts | False Alarm Rate (95% CI) | False Alarm Counts |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for s in per_repeat_stats:
        md_lines.append(
            f"| **{s['repeat']}** | {s['finished_cases']} | "
            f"{s['detection_rate']*100:.1f}% [{s['detection_rate_ci'][0]*100:.1f}%, {s['detection_rate_ci'][1]*100:.1f}%] | {s['detection_counts']} | "
            f"{s['false_alarm_rate']*100:.1f}% [{s['false_alarm_rate_ci'][0]*100:.1f}%, {s['false_alarm_rate_ci'][1]*100:.1f}%] | {s['false_alarm_counts']} |"
        )

    md_lines.extend([
        "",
        "## Correct Cases Performance",
        "",
        "| Cases | Passed | False Alarms | Unverified |",
        "| :--- | :--- | :--- | :--- |",
        f"| {total_c} | {correct_passed} | {total_fa} | {conf.get('correct_unverified', 0)} |",
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
        "## Verdict Flips Across Repeats",
        "",
        f"- **Cases evaluated in 2 or more repeats**: {len(multi_repeat_cases)}",
        f"- **Cases with identical verdicts across all repeats**: {identical_count}",
        f"- **Cases with differing verdicts (flips)**: {flip_count}",
        "",
    ])

    if flipped_cases:
        rep_headers = " | ".join(f"**{s['repeat']}**" for s in per_repeat_stats)
        rep_sep = " | ".join([":---:"] * len(per_repeat_stats))
        md_lines.extend([
            f"| Case ID | Label | Kind | {rep_headers} |",
            f"| :--- | :--- | :--- | {rep_sep} |",
        ])
        for fc in flipped_cases:
            v_cells = " | ".join(fc["verdicts"].get(s["repeat"], "—") for s in per_repeat_stats)
            md_lines.append(f"| `{fc['base_id']}` | {fc['label']} | `{fc['kind']}` | {v_cells} |")
        md_lines.append("")
    elif multi_repeat_cases:
        md_lines.append("*None. All cases evaluated in 2 or more repeats had identical verdicts across repeats.*")
        md_lines.append("")
    else:
        md_lines.append("*No cases evaluated across multiple repeats.*")
        md_lines.append("")

    # Breakdown by Magnitude Bucket
    md_lines.extend([
        "## Breakdown by Error Magnitude Bucket",
        "",
    ])
    if repeats > 1:
        md_lines.extend([
            "| Bucket | Unique cases | Observations | Detected (mean per repeat) | Detection rate (mean over repeats) | 95% Wilson CI |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for b_name in ["subtle", "moderate", "gross", "structural"]:
            if b_name in magnitude_buckets:
                b_stats = magnitude_buckets[b_name]
                md_lines.append(
                    f"| `{b_name}` | {b_stats['unique_cases']} | {b_stats['observations']} | "
                    f"{b_stats['detected_mean']:.1f} | {b_stats['detection_rate']*100:.1f}% | "
                    f"[{b_stats['detection_rate_ci'][0]*100:.1f}%, {b_stats['detection_rate_ci'][1]*100:.1f}%] |"
                )
    else:
        md_lines.extend([
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
    ])
    if repeats > 1:
        md_lines.extend([
            "| Error Kind | Unique cases | Observations | Detected (mean per repeat) | Detection rate (mean over repeats) | 95% Wilson CI |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for k_name, k_stats in by_kind_computed.items():
            md_lines.append(
                f"| `{k_name}` | {k_stats['unique_cases']} | {k_stats['observations']} | "
                f"{k_stats['detected_mean']:.1f} | {k_stats['detection_rate']*100:.1f}% | "
                f"[{k_stats['detection_rate_ci'][0]*100:.1f}%, {k_stats['detection_rate_ci'][1]*100:.1f}%] |"
            )
    else:
        md_lines.extend([
            "| Error Kind | Cases | Detected | Missed | Detection Rate |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for k_name, k_stats in by_kind_computed.items():
            if k_name != "correct":
                md_lines.append(
                    f"| `{k_name}` | {k_stats['count']} | {k_stats['detection']} | "
                    f"{k_stats['missed']} | {k_stats['detection_rate']*100:.1f}% |"
                )
    md_lines.append("")

    # Breakdown by Category
    md_lines.extend([
        "## Breakdown by Category",
        "",
    ])
    if repeats > 1:
        md_lines.extend([
            "| Category | Unique cases | Observations | Detected (mean per repeat) | Detection rate (mean over repeats) | 95% Wilson CI |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for cat, c_stats in sorted(by_category_computed.items()):
            md_lines.append(
                f"| `{cat}` | {c_stats['unique_cases']} | {c_stats['observations']} | "
                f"{c_stats['detected_mean']:.1f} | {c_stats['detection_rate']*100:.1f}% | "
                f"[{c_stats['detection_rate_ci'][0]*100:.1f}%, {c_stats['detection_rate_ci'][1]*100:.1f}%] |"
            )
    else:
        md_lines.extend([
            "| Category | Cases | Detected | Missed | Detection Rate |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for cat, c_stats in sorted(by_category_computed.items()):
            md_lines.append(
                f"| `{cat}` | {c_stats['count']} | {c_stats['detection']} | "
                f"{c_stats['missed']} | {c_stats['detection_rate']*100:.1f}% |"
            )
    md_lines.append("")

    # Breakdown by Difficulty
    md_lines.extend([
        "## Breakdown by Difficulty",
        "",
    ])
    if repeats > 1:
        md_lines.extend([
            "| Difficulty | Unique cases | Observations | Detected (mean per repeat) | Detection rate (mean over repeats) | 95% Wilson CI |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for diff, d_stats in sorted(by_difficulty_computed.items()):
            md_lines.append(
                f"| `{diff}` | {d_stats['unique_cases']} | {d_stats['observations']} | "
                f"{d_stats['detected_mean']:.1f} | {d_stats['detection_rate']*100:.1f}% | "
                f"[{d_stats['detection_rate_ci'][0]*100:.1f}%, {d_stats['detection_rate_ci'][1]*100:.1f}%] |"
            )
    else:
        md_lines.extend([
            "| Difficulty | Cases | Detected | Missed | Detection Rate |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for diff, d_stats in sorted(by_difficulty_computed.items()):
            md_lines.append(
                f"| `{diff}` | {d_stats['count']} | {d_stats['detection']} | "
                f"{d_stats['missed']} | {d_stats['detection_rate']*100:.1f}% |"
            )
    md_lines.append("")

    # Checks caught
    md_lines.extend([
        "## Detection Mechanism Distribution",
        "",
        "| Failed Check Type | Detected Cases Count |",
        "| :--- | :--- |",
    ])
    for check_type, c_cnt in sorted(failed_checks_counter.items()):
        md_lines.append(f"| `{check_type}` | {c_cnt} |")
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
    if is_partial:
        print(f"** PARTIAL RUN: {n_finished} of {expected_total} cases finished **")
    print("=" * 80)
    print(f"Total Cases:     {n_finished} (Unique: {n_unique_all}, Correct: {total_c}, Wrong: {total_w}, Errors: {errors_count})")
    print(f"Detection Rate:  {headline_det_rate*100:.1f}% ({det_counts_str}) [95% CI: {headline_det_ci[0]*100:.1f}% - {headline_det_ci[1]*100:.1f}%]")
    print(f"Right Reason:    {headline_right_reason_rate*100:.1f}% ({right_reason_counts_str}) [95% CI: {headline_right_reason_ci[0]*100:.1f}% - {headline_right_reason_ci[1]*100:.1f}%]")
    print(f"False Alarm Rate:{headline_fa_rate*100:.1f}% ({fa_counts_str}) [95% CI: {headline_fa_ci[0]*100:.1f}% - {headline_fa_ci[1]*100:.1f}%]")
    print(f"Unverified Rate: {headline_unver_rate*100:.1f}% [95% CI: {headline_unver_ci[0]*100:.1f}% - {headline_unver_ci[1]*100:.1f}%]")
    print(f"Precision:       {headline_precision*100:.1f}% [95% CI: {headline_prec_ci[0]*100:.1f}% - {headline_prec_ci[1]*100:.1f}%]")
    print(f"LLM Calls:       {total_calls} | Tokens: {total_tokens:,} | Seconds: {total_time:.1f}s")
    if aggregated_by_model:
        models_print = ", ".join(f"{m} ({d['calls']} calls, {d['tokens']:,} tokens)" for m, d in aggregated_by_model.items())
        print(f"Models Used:     {models_print}")
    if len(aggregated_by_model) > 1:
        print("Multiple models were used in this run")
    if multi_repeat_cases:
        print(f"Verdict Flips:   {flip_count} differing of {len(multi_repeat_cases)} cases with 2+ repeats ({identical_count} identical)")
    print(f"Summary saved:   {run_path / 'summary.md'}")
    print("=" * 80 + "\n")

    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Reviewer agent in isolation against benchmark.")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N cases (interleaved correct & wrong)")
    parser.add_argument("--repeats", type=int, default=None, help="Number of times to repeat each case (default 1)")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to store evaluation artifacts")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens allowed before stopping cleanly")
    parser.add_argument("--summarize-only", action="store_true", help="Generate summary from cases.jsonl without running cases")

    args = parser.parse_args()

    if args.summarize_only:
        if not args.run_dir:
            print("Error: --run-dir must be specified when using --summarize-only.", file=sys.stderr)
            sys.exit(1)
        summarize_run(
            run_dir=args.run_dir,
            repeats=args.repeats,
            limit=args.limit,
        )
        return

    run_evaluation(
        limit=args.limit,
        repeats=args.repeats or 1,
        run_dir=args.run_dir,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
