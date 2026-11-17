"""
metrics.py
Evaluation Metrics: Statistical functions, Wilson score intervals, magnitude bucketing,
confusion matrix calculations, and subgroup breakdowns for agent evaluation.
"""

import math
from typing import Any
import numpy as np
import pandas as pd


def wilson_ci_from_rate(p: float, n: int | float, z: float = 1.96) -> tuple[float, float]:
    """
    Computes the Wilson score interval for a binomial rate p (between 0.0 and 1.0)
    given sample size n (integer or float).
    Returns (low, high) as fractions between 0.0 and 1.0.
    Applies no rounding to counts, safe for n <= 0 (returns (0.0, 0.0)).
    """
    if n <= 0:
        return 0.0, 0.0

    n_f = float(n)
    p_f = max(0.0, min(1.0, float(p)))
    z2 = z * z
    denom = 1.0 + z2 / n_f
    center = p_f + z2 / (2.0 * n_f)
    spread = z * math.sqrt(max(0.0, (p_f * (1.0 - p_f) / n_f) + (z2 / (4.0 * n_f * n_f))))

    low = max(0.0, (center - spread) / denom)
    high = min(1.0, (center + spread) / denom)
    return low, high


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """
    Computes the Wilson score interval for a binomial proportion.
    Returns (low, high) as fractions between 0.0 and 1.0.
    Safe for total <= 0 (returns (0.0, 0.0)).
    """
    if total <= 0:
        return 0.0, 0.0
    return wilson_ci_from_rate(float(successes) / float(total), total, z=z)


def _to_dataframe_if_possible(val: Any) -> pd.DataFrame | None:
    """Converts Series or dict with columns/rows to a DataFrame if possible."""
    if isinstance(val, pd.DataFrame):
        return val.copy(deep=True)
    if isinstance(val, pd.Series):
        return val.reset_index()
    if isinstance(val, dict) and "columns" in val and "rows" in val:
        return pd.DataFrame(val["rows"], columns=val["columns"])
    return None


def magnitude_bucket(reference_answer: Any, wrong_answer: Any) -> str:
    """
    Classifies the magnitude of an analytical error between reference and wrong answers:
    - 'subtle': relative numeric difference < 5% (< 0.05)
    - 'moderate': relative difference between 5% and 50% (0.05 <= diff <= 0.50)
    - 'gross': relative difference > 50% (> 0.50)
    - 'structural': tables with differing shapes (rows/cols) or mismatched text labels.
    """
    ref_df = _to_dataframe_if_possible(reference_answer)
    wrong_df = _to_dataframe_if_possible(wrong_answer)

    # If one is 1x1 DataFrame and other is scalar, unbox the 1x1 DataFrame
    if ref_df is not None and ref_df.shape == (1, 1) and wrong_df is None:
        reference_answer = ref_df.iloc[0, 0]
        ref_df = None
    if wrong_df is not None and wrong_df.shape == (1, 1) and ref_df is None:
        wrong_answer = wrong_df.iloc[0, 0]
        wrong_df = None

    # Table vs Table
    if ref_df is not None and wrong_df is not None:
        # Check shapes
        if ref_df.shape != wrong_df.shape:
            return "structural"
        if len(ref_df) == 0:
            return "subtle"

        # Separate text and numeric columns
        text_cols_ref = [c for c in ref_df.columns if not pd.api.types.is_numeric_dtype(ref_df[c])]
        text_cols_wrong = [c for c in wrong_df.columns if not pd.api.types.is_numeric_dtype(wrong_df[c])]

        if len(text_cols_ref) != len(text_cols_wrong):
            return "structural"

        # Compare text labels
        for c_r, c_w in zip(text_cols_ref, text_cols_wrong):
            s_r = sorted(ref_df[c_r].astype(str).str.strip().str.lower().tolist())
            s_w = sorted(wrong_df[c_w].astype(str).str.strip().str.lower().tolist())
            if s_r != s_w:
                return "structural"

        # Compare numeric values
        num_cols_ref = [c for c in ref_df.columns if pd.api.types.is_numeric_dtype(ref_df[c])]
        num_cols_wrong = [c for c in wrong_df.columns if pd.api.types.is_numeric_dtype(wrong_df[c])]

        if len(num_cols_ref) != len(num_cols_wrong):
            return "structural"

        all_ref_nums: list[float] = []
        for c in num_cols_ref:
            all_ref_nums.extend(ref_df[c].dropna().astype(float).tolist())
        all_wrong_nums: list[float] = []
        for c in num_cols_wrong:
            all_wrong_nums.extend(wrong_df[c].dropna().astype(float).tolist())

        if len(all_ref_nums) != len(all_wrong_nums) or len(all_ref_nums) == 0:
            return "structural"

        all_ref_nums.sort()
        all_wrong_nums.sort()

        max_rel_diff = 0.0
        for r, w in zip(all_ref_nums, all_wrong_nums):
            if r == 0.0:
                diff = abs(w)
            else:
                diff = abs(w - r) / abs(r)
            if diff > max_rel_diff:
                max_rel_diff = diff

        if max_rel_diff < 0.05:
            return "subtle"
        elif max_rel_diff <= 0.50:
            return "moderate"
        else:
            return "gross"

    # One is table and other is scalar
    if ref_df is not None or wrong_df is not None:
        return "structural"

    # Both are scalars
    ref = reference_answer
    wrong = wrong_answer

    if isinstance(ref, (int, float, np.number)) and isinstance(wrong, (int, float, np.number)):
        r = float(ref)
        w = float(wrong)
        if r == 0.0:
            diff = abs(w)
        else:
            diff = abs(w - r) / abs(r)

        if diff < 0.05:
            return "subtle"
        elif diff <= 0.50:
            return "moderate"
        else:
            return "gross"

    # Non-numeric scalar (string, bool)
    if str(ref).strip().lower() == str(wrong).strip().lower():
        return "subtle"
    return "structural"


def confusion(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Computes detection, false alarm, and unverified counts along with Wilson score interval rates.

    Args:
        cases: list of case dicts, each with 'is_wrong' (bool) and 'verdict' ('pass', 'fail', 'unverified', 'error')

    Returns:
        dict with:
        - detection: fail among wrong
        - missed: pass among wrong
        - wrong_unverified: unverified/error among wrong
        - total_wrong: total wrong cases
        - false_alarms: fail among correct
        - correct_passed: pass among correct
        - correct_unverified: unverified/error among correct
        - total_correct: total correct cases
        - total_cases: total cases
        - detection_rate, detection_rate_ci (over ALL wrong cases; unverified counts as not detected)
        - false_alarm_rate, false_alarm_rate_ci (over ALL correct cases)
        - unverified_rate, unverified_rate_ci (over ALL cases)
        - precision, precision_ci (detection / (detection + false_alarms), safe for 0)
    """
    detection = 0
    missed = 0
    wrong_unverified = 0

    false_alarms = 0
    correct_passed = 0
    correct_unverified = 0

    for c in cases:
        is_wrong = bool(c.get("is_wrong"))
        v = str(c.get("verdict", "")).strip().lower()

        if is_wrong:
            if v == "fail":
                detection += 1
            elif v == "pass":
                missed += 1
            else:
                wrong_unverified += 1
        else:
            if v == "fail":
                false_alarms += 1
            elif v == "pass":
                correct_passed += 1
            else:
                correct_unverified += 1

    total_wrong = detection + missed + wrong_unverified
    total_correct = false_alarms + correct_passed + correct_unverified
    total_cases = len(cases)
    total_unverified = wrong_unverified + correct_unverified

    # Rates with Wilson score confidence intervals
    detection_rate = (detection / total_wrong) if total_wrong > 0 else 0.0
    detection_rate_ci = wilson_ci(detection, total_wrong)

    false_alarm_rate = (false_alarms / total_correct) if total_correct > 0 else 0.0
    false_alarm_rate_ci = wilson_ci(false_alarms, total_correct)

    unverified_rate = (total_unverified / total_cases) if total_cases > 0 else 0.0
    unverified_rate_ci = wilson_ci(total_unverified, total_cases)

    prec_denom = detection + false_alarms
    precision = (detection / prec_denom) if prec_denom > 0 else 0.0
    precision_ci = wilson_ci(detection, prec_denom)

    return {
        "detection": detection,
        "missed": missed,
        "wrong_unverified": wrong_unverified,
        "total_wrong": total_wrong,
        "false_alarms": false_alarms,
        "correct_passed": correct_passed,
        "correct_unverified": correct_unverified,
        "total_correct": total_correct,
        "total_cases": total_cases,
        "detection_rate": detection_rate,
        "detection_rate_ci": detection_rate_ci,
        "false_alarm_rate": false_alarm_rate,
        "false_alarm_rate_ci": false_alarm_rate_ci,
        "unverified_rate": unverified_rate,
        "unverified_rate_ci": unverified_rate_ci,
        "precision": precision,
        "precision_ci": precision_ci,
    }


def breakdown(cases: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """
    Groups cases by a specified field (e.g. 'kind', 'category', 'difficulty',
    'magnitude_bucket') and computes confusion statistics for each group.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in cases:
        val = c.get(key)
        if val is None or val == "":
            g_key = "none"
        else:
            g_key = str(val)
        groups.setdefault(g_key, []).append(c)

    results: dict[str, dict[str, Any]] = {}
    for g_key in sorted(groups.keys()):
        g_cases = groups[g_key]
        stats = confusion(g_cases)
        stats["count"] = len(g_cases)
        results[g_key] = stats

    return results
