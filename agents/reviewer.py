"""
reviewer.py
Reviewer Agent: Independently verifies and validates analytical outputs through:
1. Independent recomputation via an alternative pandas implementation.
2. Rule-based invariant checks (non-negativity, nulls/infinities, aggregation sizes, takeaway alignment).
3. Advisory logic review for potential query/aggregation flaws.
Formulates structured review_feedback and audit reports for the feedback loop.
"""

from datetime import datetime, timezone
import json
import os
import re
from typing import Any, Callable
import numpy as np
import pandas as pd

from llm import call_llm
from state import AgentState
from tools.sandbox import run_code


# ============================================================================
# Helper: Results Matching with Tolerance
# ============================================================================

def results_match(a: Any, b: Any, tolerance: float = 0.01) -> bool:
    """
    Compares two analytical calculation results for equality within a given tolerance:
    - Scalars: compares numeric values within relative tolerance (default 1%),
      handling zero and NaN safely.
    - DataFrames: compares after sorting both by their first column and resetting
      index. Compares numeric columns within tolerance and text columns exactly.
      Row counts and column counts must match.
    """
    if a is None or b is None:
        return a is b

    # Convert Series to DataFrame via reset_index
    if isinstance(a, pd.Series):
        a = a.reset_index()
    if isinstance(b, pd.Series):
        b = b.reset_index()

    # Normalize 1x1 DataFrame to scalar if compared against a scalar
    is_a_scalar = isinstance(a, (int, float, str, bool, np.number, np.bool_))
    is_b_scalar = isinstance(b, (int, float, str, bool, np.number, np.bool_))

    if isinstance(a, pd.DataFrame) and a.shape == (1, 1) and is_b_scalar:
        a = a.iloc[0, 0]
        is_a_scalar = True
    elif isinstance(b, pd.DataFrame) and b.shape == (1, 1) and is_a_scalar:
        b = b.iloc[0, 0]
        is_b_scalar = True

    # 1. Scalar comparison
    if is_a_scalar and is_b_scalar:
        if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
            if pd.isna(a) and pd.isna(b):
                return True
            if pd.isna(a) or pd.isna(b):
                return False
            denom = max(abs(float(a)), abs(float(b)))
            if denom == 0:
                return True
            return (abs(float(a) - float(b)) / denom) <= tolerance
        return str(a).strip() == str(b).strip()

    # 2. DataFrame comparison
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        if len(a) == 0 and len(b) == 0:
            return True

        # Standard elementwise comparison if shapes match
        if len(a) == len(b) and a.shape[1] == b.shape[1]:
            # Align columns by name if set of columns is identical
            if set(a.columns) == set(b.columns):
                b_aligned = b[list(a.columns)]
            else:
                b_aligned = b

            # Sort both by first column and reset index
            try:
                a_sorted = a.sort_values(by=a.columns[0]).reset_index(drop=True)
                b_sorted = b_aligned.sort_values(by=b_aligned.columns[0]).reset_index(drop=True)
            except Exception:
                a_sorted = a.reset_index(drop=True)
                b_sorted = b_aligned.reset_index(drop=True)

            match_standard = True
            for i in range(a_sorted.shape[1]):
                col_a = a_sorted.iloc[:, i]
                col_b = b_sorted.iloc[:, i]

                is_a_num = pd.api.types.is_numeric_dtype(col_a)
                is_b_num = pd.api.types.is_numeric_dtype(col_b)

                if is_a_num and is_b_num:
                    diff = np.abs(col_a.astype(float) - col_b.astype(float))
                    denom = np.maximum(np.abs(col_a.astype(float)), np.abs(col_b.astype(float)))
                    denom = np.where(denom == 0, 1.0, denom)
                    rel_diff = diff / denom
                    if (rel_diff > tolerance).any():
                        match_standard = False
                        break
                else:
                    str_a = col_a.astype(str).str.strip()
                    str_b = col_b.astype(str).str.strip()
                    if (str_a != str_b).any():
                        match_standard = False
                        break

            if match_standard:
                return True

        # Layout-independent comparison fallback:
        # Collect all numeric values from both DataFrames, sort them, and compare
        nums_a: list[float] = []
        for col in a.select_dtypes(include=[np.number]).columns:
            nums_a.extend(a[col].dropna().astype(float).tolist())

        nums_b: list[float] = []
        for col in b.select_dtypes(include=[np.number]).columns:
            nums_b.extend(b[col].dropna().astype(float).tolist())

        if len(nums_a) > 0 and len(nums_a) == len(nums_b):
            nums_a.sort()
            nums_b.sort()
            diffs = [
                abs(va - vb) / max(abs(va), abs(vb), 1e-9)
                for va, vb in zip(nums_a, nums_b)
            ]
            if all(d <= tolerance for d in diffs):
                return True

        return False

    return False


# ============================================================================
# Helper: Value Shortener
# ============================================================================

def _shorten_val(val: Any) -> Any:
    """Provides a concise, JSON-friendly representation of analytical results."""
    if val is None:
        return "None"
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return round(float(val), 4)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, pd.DataFrame):
        return f"DataFrame(rows={len(val)}, cols={list(val.columns)})"
    s = str(val)
    return s[:120] + "..." if len(s) > 120 else s


# ============================================================================
# Check 2: Rule-Based Checks (No LLM)
# ============================================================================

def _extract_takeaway_numbers(takeaway: str, res: Any) -> list[str]:
    """
    Finds numbers cited in chart takeaway text and verifies their presence in the
    result data (within 1% tolerance, allowing for rounding and magnitudes like 1.2M).
    - Unit suffix bug: only treat a suffix as multiplier when directly attached (1.2M, 500k)
      or magnitude words ('thousand', 'million', 'billion'). Never treat ' m' with a space as million.
    - Ignore numbers that come from labels (column names, text cells, bracket labels like '0-10%').
    - Ignore numbers in ranges like '0-10%', '6.4-7.6'.
    - Leniently ignores years (1900-2099), small integers < 10, and percentages (%).
    """
    problems: list[str] = []

    # 1. Collect numbers from result numeric cells
    result_numbers: list[float] = []
    if isinstance(res, (int, float, np.number)):
        if not pd.isna(res) and np.isfinite(res):
            result_numbers.append(float(res))
    elif isinstance(res, pd.DataFrame):
        for col in res.select_dtypes(include=[np.number]).columns:
            for v in res[col].dropna():
                if np.isfinite(v):
                    result_numbers.append(float(v))

    # 2. Collect numbers from result labels/text columns and column names
    label_numbers: set[float] = set()
    if isinstance(res, pd.DataFrame):
        for col in res.columns:
            for num_str in re.findall(r"\d+(?:\.\d+)?", str(col)):
                try:
                    label_numbers.add(float(num_str))
                except ValueError:
                    pass
        for col in res.select_dtypes(include=["object", "category", "string"]).columns:
            for val in res[col].dropna().unique():
                for num_str in re.findall(r"\d+(?:\.\d+)?", str(val)):
                    try:
                        label_numbers.add(float(num_str))
                    except ValueError:
                        pass

    # 3. Collect numbers inside ranges in takeaway (e.g. 0-10%, 1.56–1.69, 6.4-7.6)
    range_numbers: set[float] = set()
    for rm in re.finditer(r"(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)", takeaway):
        try:
            range_numbers.add(float(rm.group(1)))
            range_numbers.add(float(rm.group(2)))
        except ValueError:
            pass

    # 4. Match numbers with precise suffix handling
    # Matches:
    # - $100, 15.71, 0.215
    # - 1.2M, 500k, 3B (suffix directly attached, no space, not followed by word chars)
    # - 1.2 million, 500 thousand, 3 billion (magnitude words)
    pattern = re.compile(
        r"(\$?\d+(?:,\d{3})*(?:\.\d+)?)(?:([kmbKMB])(?![a-zA-Z])|\s+(thousand|million|billion)\b)?(%?)",
        flags=re.IGNORECASE,
    )

    for m in pattern.finditer(takeaway):
        raw_num, direct_suffix, word_suffix, pct = m.groups()

        # Ignore percentage values
        if pct == "%":
            continue

        clean_num = raw_num.replace("$", "").replace(",", "")
        try:
            val = float(clean_num)
        except ValueError:
            continue

        # Ignore numbers in ranges (e.g. '0-10%')
        if val in range_numbers:
            continue

        # Ignore numbers that come from result labels or column names
        if val in label_numbers:
            continue

        # Ignore calendar years
        if not direct_suffix and not word_suffix and val.is_integer() and 1900 <= int(val) <= 2099:
            continue

        # Ignore small integers below 10
        if not direct_suffix and not word_suffix and val.is_integer() and 0 <= int(val) < 10:
            continue

        # Calculate multiplier
        multiplier = 1.0
        suffix_str = direct_suffix or word_suffix
        if suffix_str:
            s_low = suffix_str.lower()
            if s_low in ("k", "thousand"):
                multiplier = 1e3
            elif s_low in ("m", "million"):
                multiplier = 1e6
            elif s_low in ("b", "billion"):
                multiplier = 1e9

        scaled_val = val * multiplier
        raw_token = m.group(0).strip()

        # Check for presence in result_numbers
        matched = False
        for v in result_numbers:
            diff1 = abs(v - scaled_val) / max(abs(v), abs(scaled_val), 1e-9)
            if diff1 <= 0.01 or round(v) == round(scaled_val):
                matched = True
                break

            diff2 = abs(v - val) / max(abs(v), abs(val), 1e-9)
            if diff2 <= 0.01 or round(v) == round(val):
                matched = True
                break

        if not matched:
            problems.append(f"Takeaway mentions {raw_token} which is not in the result")

    return problems


def rule_checks(item: dict[str, Any], df: pd.DataFrame, chart: dict[str, Any] | None = None) -> list[str]:
    """
    Performs deterministic rule-based checks on an analytical result item:
    1. Checks if result is empty.
    2. Checks non-negativity on sum/count/sales/revenue columns.
    3. Checks for presence of NaN or infinite values.
    4. Checks for missing aggregation (> 50 rows).
    Note: Takeaway number checks are advisory warnings and handled separately in reviewer_node.
    """
    problems: list[str] = []
    res = item.get("result")

    # 1. Result is empty
    if res is None:
        problems.append("Result is None")
        return problems
    if isinstance(res, pd.DataFrame) and res.empty:
        problems.append("Result DataFrame is empty")
        return problems
    if isinstance(res, (list, dict)) and len(res) == 0:
        problems.append("Result is empty")
        return problems

    # 2. Non-negative column checks (count, total, quantity, sales, revenue)
    keywords = ("count", "total", "quantity", "sales", "revenue")
    if isinstance(res, pd.DataFrame):
        for col in res.select_dtypes(include=[np.number]).columns:
            col_lower = str(col).lower()
            if any(k in col_lower for k in keywords):
                if (res[col] < 0).any():
                    problems.append(f"Column '{col}' should be non-negative but contains negative values")
    elif isinstance(res, (int, float, np.number)):
        q_lower = str(item.get("question", "")).lower()
        if any(k in q_lower for k in keywords) and res < 0:
            problems.append(f"Result should be non-negative for this query but was {res}")

    # 3. NaN or Infinite values
    if isinstance(res, pd.DataFrame):
        if res.isna().any().any():
            problems.append("Result contains NaN values")
        num_cols = res.select_dtypes(include=[np.number])
        if np.isinf(num_cols).any().any():
            problems.append("Result contains infinite values")
    elif isinstance(res, float):
        if np.isnan(res) or np.isinf(res):
            problems.append("Result is NaN or infinite")

    # 4. Result size > 50 rows (likely missing aggregation)
    if isinstance(res, pd.DataFrame) and len(res) > 50:
        problems.append("Result has more than 50 rows (possible missing aggregation)")

    return problems


# ============================================================================
# Check 1: Independent Recomputation
# ============================================================================

def _clean_code_fence(text: str) -> str:
    """Strips markdown code fences from LLM responses."""
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:python)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _recompute_independently(
    question: str,
    original_code: str,
    original_res: Any,
    columns: list[str],
    df: pd.DataFrame,
    call_llm_fn: Callable[[str], str],
) -> tuple[str, Any, str, str]:
    """
    Check 1: Asks LLM to solve the question using an alternative pandas method,
    runs it via sandbox, and verifies agreement with original results.
    Returns (status: 'pass' | 'fail' | 'inconclusive', recomputed_res, note, recomp_code).
    """
    prompt = f"""Question: {question}

Original pandas code:
```python
{original_code}
```

Columns available:
{', '.join(columns)}

Solve the same question using a different pandas method than the original code
(for example, pivot_table instead of groupby, or sort_values instead of nlargest).
Store the answer in a variable called result. Use only pd and np, no imports.
Return ONLY the code."""

    reply = call_llm_fn(prompt)
    recomp_code = _clean_code_fence(reply)

    # Execute alternative code in sandbox
    res = run_code(recomp_code, df)

    # Retry once if execution encountered an error
    if res["status"] != "ok":
        retry_prompt = f"""Previous alternative code failed with: {res.get('error')}.
Fix it and return only the corrected alternative pandas code for: '{question}'.
Store the answer in a variable called result. Use only pd and np, no imports. Return ONLY the code."""
        retry_reply = call_llm_fn(retry_prompt)
        recomp_code = _clean_code_fence(retry_reply)
        res = run_code(recomp_code, df)

    if res["status"] != "ok":
        # Mark inconclusive (not a fail) per spec
        return "inconclusive", None, f"Recomputation code failed: {res.get('error')}", recomp_code

    recomputed_val = res.get("result")
    matches = results_match(original_res, recomputed_val)

    if matches:
        note = ""
        if isinstance(original_res, pd.DataFrame) and isinstance(recomputed_val, pd.DataFrame):
            if original_res.shape != recomputed_val.shape or list(original_res.columns) != list(recomputed_val.columns):
                note = "same values, different layout"
        return "pass", recomputed_val, note, recomp_code
    else:
        return "fail", recomputed_val, "Independent recomputation gave a different result", recomp_code


# ============================================================================
# Check 3: Logic Review (LLM Advisory Only)
# ============================================================================

def _logic_review(
    question: str,
    code: str,
    res: Any,
    call_llm_fn: Callable[[str], str],
) -> list[str]:
    """
    Check 3: Solicits an advisory logic review from LLM.
    Issues returned are non-blocking warnings only.
    """
    prompt = f"""Review this question, code, and result summary for potential logical issues
(wrong aggregation, missing filter, double counting, wrong column, or unsupported conclusions).

Question: {question}
Pandas Code:
```python
{code}
```
Result summary:
{_shorten_val(res)}

Return ONLY JSON: {{"issues": [list of short strings, empty if none]}}"""

    try:
        reply = call_llm_fn(prompt)
        cleaned = _clean_code_fence(reply)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "issues" in data and isinstance(data["issues"], list):
            return [str(s).strip() for s in data["issues"] if str(s).strip()]
    except Exception:
        pass
    return []


# ============================================================================
# LangGraph Node: reviewer_node
# ============================================================================

def reviewer_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph Reviewer Node:
    - Verifies analysis results with status 'ok'.
    - Runs Check 1 (independent recomputation), Check 2 (rule checks), Check 3 (logic warnings).
    - Fails an item if Check 1 mismatches or Check 2 finds violations.
    - Updates review_feedback (failures only), review_report (complete audit),
      caught_errors, and events.
    """
    events = list(state.get("events", []))
    caught_errors = list(state.get("caught_errors", []))
    retry_round = state.get("retry_count", 0)
    rerun_ids = state.get("rerun_ids", [])
    previous_report = state.get("review_report", [])
    prev_report_by_id = {
        r["id"]: r for r in previous_report if isinstance(r, dict) and "id" in r
    }

    analysis_results = state.get("analysis_results", [])
    charts = state.get("charts", [])
    charts_by_id = {c.get("id"): c for c in charts if isinstance(c, dict)}

    review_feedback: list[dict[str, Any]] = []
    review_report: list[dict[str, Any]] = []

    # 1. Load dataset
    csv_path = state.get("clean_df_path") or state.get("raw_df_path")
    if not csv_path or not os.path.exists(csv_path):
        events.append({
            "agent": "reviewer",
            "type": "error",
            "message": f"Reviewer could not find dataset at '{csv_path}'",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return {
            "review_feedback": [],
            "review_report": [],
            "caught_errors": caught_errors,
            "events": events,
        }

    df = pd.read_csv(csv_path)
    columns_list = list(df.columns)

    for item in analysis_results:
        # Skip items that already failed analysis
        if item.get("status") != "ok":
            continue

        item_id = item.get("id")

        # On re-run, reuse previous report entry for items not re-run by analyst
        if rerun_ids and item_id not in rerun_ids:
            if item_id in prev_report_by_id:
                reused = dict(prev_report_by_id[item_id])
                review_report.append(reused)
                if reused.get("verdict") == "fail":
                    review_feedback.append({
                        "id": item_id,
                        "verdict": "fail",
                        "reason": reused.get("reason", ""),
                        "suggested_fix": "",
                        "original_value": reused.get("original_value"),
                        "recomputed_value": reused.get("recomputed_value"),
                    })
                continue

        q = item.get("question", "")
        code = item.get("code", "")
        original_res = item.get("result")
        chart = charts_by_id.get(item_id)

        try:
            # Validate that original_res is a supported result type
            if not isinstance(
                original_res,
                (pd.DataFrame, pd.Series, int, float, str, bool, np.number, np.bool_),
            ):
                raise TypeError(f"Unsupported result object type: {type(original_res).__name__}")

            # ----------------------------------------------------------------
            # Check 1: Independent Recomputation
            # ----------------------------------------------------------------
            c1_status, recomp_res, c1_note, recomp_code = _recompute_independently(
                question=q,
                original_code=code,
                original_res=original_res,
                columns=columns_list,
                df=df,
                call_llm_fn=call_llm,
            )

            # ----------------------------------------------------------------
            # Check 2: Rule-Based Checks (Real result problems only)
            # ----------------------------------------------------------------
            c2_problems = rule_checks(item, df, chart=chart)

            # Takeaway number check (Warning only, does not fail item or trigger retry)
            takeaway_warning: str | None = None
            if chart and chart.get("takeaway"):
                takeaway_problems = _extract_takeaway_numbers(chart["takeaway"], original_res)
                if takeaway_problems:
                    takeaway_warning = "; ".join(takeaway_problems)

            # ----------------------------------------------------------------
            # Check 3: Logic Review (Advisory Only)
            # ----------------------------------------------------------------
            c3_warnings = _logic_review(
                question=q,
                code=code,
                res=original_res,
                call_llm_fn=call_llm,
            )

            # Consolidate warnings
            warnings: list[str] = list(c3_warnings)
            if c1_status == "inconclusive" and c1_note:
                warnings.append(c1_note)
            if c1_status == "pass" and c1_note:
                warnings.append(c1_note)
            if takeaway_warning:
                warnings.append(f"takeaway_mismatch: {takeaway_warning}")

            # Determine final verdict (only real result failures fail the item)
            is_fail = (c1_status == "fail") or (len(c2_problems) > 0)
            if is_fail:
                verdict = "fail"
            elif c1_status == "inconclusive":
                verdict = "unverified"
            else:
                verdict = "pass"

            # Formulate reason and suggested fix
            reasons: list[str] = []
            if c1_status == "fail":
                reasons.append(c1_note or "Independent recomputation gave a different result")
            if c2_problems:
                reasons.extend(c2_problems)
            if c1_status == "inconclusive" and not is_fail:
                reasons.append(c1_note or "Independent recomputation was inconclusive; could not verify")

            reason_str = "; ".join(reasons) if reasons else "All checks passed successfully."

            if c1_status == "fail":
                suggested_fix = (
                    "Re-check the aggregation and filters; an independent method gave a different value."
                )
            elif c2_problems:
                suggested_fix = f"Fix the following data issues: {'; '.join(c2_problems)}."
            else:
                suggested_fix = ""

            orig_short = _shorten_val(original_res)
            recomp_short = _shorten_val(recomp_res) if recomp_res is not None else None

            # Add to full audit report
            review_report.append({
                "id": item_id,
                "question": q,
                "verdict": verdict,
                "reason": reason_str,
                "warnings": warnings,
                "takeaway_warning": takeaway_warning,
                "original_value": orig_short,
                "recomputed_value": recomp_short,
            })

            # Record failing items in feedback and caught_errors
            if verdict == "fail":
                # Determine specific error type
                if c1_status == "fail":
                    error_type = "recomputation_mismatch"
                elif any("empty" in p.lower() for p in c2_problems):
                    error_type = "empty_result"
                else:
                    error_type = "rule_violation"

                orig_str = str(orig_short)[:300] if orig_short is not None else ""
                recomp_str = str(recomp_short)[:300] if recomp_short is not None else ""

                review_feedback.append({
                    "id": item_id,
                    "verdict": "fail",
                    "reason": reason_str,
                    "suggested_fix": suggested_fix,
                    "original_value": orig_str,
                    "recomputed_value": recomp_str,
                })
                caught_errors.append({
                    "id": item_id,
                    "question": q,
                    "reason": reason_str,
                    "type": error_type,
                    "retry_round": retry_round,
                    "original_value": orig_str,
                    "recomputed_value": recomp_str,
                    "original_code": code,
                    "recomputed_code": recomp_code,
                })

            events.append({
                "agent": "reviewer",
                "type": "info" if verdict == "pass" else "warning",
                "message": f"Review completed for item {item_id}: verdict '{verdict}'. {reason_str}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        except Exception as item_err:
            # Failure isolation: treat item as unverified with reason
            warn_msg = f"reviewer could not verify this item: {item_err}"
            review_report.append({
                "id": item_id,
                "question": q,
                "verdict": "unverified",
                "reason": warn_msg,
                "warnings": [warn_msg],
                "takeaway_warning": None,
                "original_value": _shorten_val(original_res),
                "recomputed_value": None,
            })
            events.append({
                "agent": "reviewer",
                "type": "warning",
                "message": f"Item {item_id} evaluation error: {item_err}. Marked as unverified.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    return {
        "review_feedback": review_feedback,
        "review_report": review_report,
        "caught_errors": caught_errors,
        "events": events,
    }
