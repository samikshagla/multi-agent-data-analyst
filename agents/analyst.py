"""
analyst.py
Analyst Agent: Interprets cleaned dataset profile, formulates targeted business questions,
generates pandas analysis code via Gemini LLM, executes the code securely within the
isolated multiprocessing sandbox, and handles review critique feedback iterations.
"""

from datetime import datetime, timezone
import json
import os
import re
from typing import Any, Callable
import pandas as pd

from llm import call_llm, LLMUnavailableError, LLMConfigError
from state import AgentState
from tools.profiler import profile_dataframe, profile_to_text
from tools.sandbox import run_code


# ============================================================================
# Helpers: Prompt Cleaning & JSON Parsing
# ============================================================================

def _clean_code_fence(text: str) -> str:
    """Strips markdown code fences (```python ... ```) from LLM output."""
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:python)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_questions_json(text: str) -> list[str] | None:
    """Parses LLM response to extract a list of question strings."""
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Attempt direct JSON load
    try:
        data = json.loads(cleaned)
        if isinstance(data, list) and all(isinstance(q, str) for q in data) and len(data) > 0:
            return [q.strip() for q in data if q.strip()]
        if isinstance(data, dict) and "questions" in data and isinstance(data["questions"], list):
            return [str(q).strip() for q in data["questions"] if str(q).strip()]
    except Exception:
        pass

    # Regex search for JSON array
    match = re.search(r"\[\s*\"[^\"]+\".*?\]", cleaned, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list) and len(data) > 0:
                return [str(q).strip() for q in data if str(q).strip()]
        except Exception:
            pass

    return None


def _generate_default_questions(df: pd.DataFrame) -> list[str]:
    """Fallback generator creating 3 deterministic questions based on column dtypes."""
    questions: list[str] = []
    num_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    cat_cols = [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]

    # Question 1: Total records & general overview
    questions.append("What is the total number of records in the dataset?")

    # Question 2: Numeric aggregate
    if num_cols:
        questions.append(f"What are the mean, minimum, and maximum values of '{num_cols[0]}'?")
    else:
        questions.append("What are the summary statistics of the dataset columns?")

    # Question 3: Grouped breakdown
    if cat_cols and num_cols:
        questions.append(f"What is the average '{num_cols[0]}' broken down by '{cat_cols[0]}'?")
    elif cat_cols:
        questions.append(f"What are the top 5 most frequent categories in '{cat_cols[0]}'?")
    elif len(num_cols) > 1:
        questions.append(f"What is the total sum of '{num_cols[1]}'?")
    else:
        questions.append("What is the overall distribution of values across the dataset?")

    return questions[:3]


def _execute_with_retry(
    question: str,
    initial_code: str,
    df: pd.DataFrame,
    call_llm_fn: Callable[[str], str],
    events: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, int, str]:
    """
    Executes pandas code in sandbox. If not 'ok', sends error back to LLM to fix,
    up to 2 retries (maximum 3 total attempts).
    Returns: (final_sandbox_result, final_code, attempts_count, notes)
    """
    current_code = initial_code
    attempts = 1
    max_retries = 2  # up to 2 retries after initial failure

    sandbox_res = run_code(current_code, df)

    while sandbox_res["status"] != "ok" and attempts <= max_retries:
        err_msg = sandbox_res.get("error", "Unknown error")
        events.append({
            "agent": "analyst",
            "type": "warning",
            "message": f"Code execution attempt {attempts} failed: {err_msg}. Requesting LLM correction.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        fix_prompt = f"""Question: {question}

The following pandas code failed:
```python
{current_code}
```

This code failed with: {err_msg}.
Fix it and return only the corrected code using df, pd, and np.
Store the final answer in a variable called result (a DataFrame, Series, or single number).
Use only pd and np, no imports, no file access. Return ONLY the code, no explanation."""

        llm_reply = call_llm_fn(fix_prompt)
        current_code = _clean_code_fence(llm_reply)
        attempts += 1
        sandbox_res = run_code(current_code, df)

    notes = ""
    # Truncate large DataFrame results to 50 rows
    if sandbox_res["status"] == "ok" and isinstance(sandbox_res.get("result"), pd.DataFrame):
        df_res = sandbox_res["result"]
        if len(df_res) > 50:
            sandbox_res["result"] = df_res.head(50)
            notes = "Truncated to first 50 rows"

    return sandbox_res, current_code, attempts, notes


# ============================================================================
# LangGraph Node: analyst_node
# ============================================================================

def analyst_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph Analyst Node:
    - First run: Generates 3-5 business questions, writes pandas code, runs via sandbox
      with up to 2 retries, and records analysis_results.
    - Re-run after review: Redoes ONLY items flagged with verdict 'fail' in review_feedback,
      incorporating reviewer critiques and suggested fixes, and clears review_feedback.
    """
    events = list(state.get("events", []))
    existing_questions = list(state.get("questions", []))
    existing_results = list(state.get("analysis_results", []))
    review_feedback = list(state.get("review_feedback", []))

    try:
        # 1. Resolve CSV path
        csv_path = state.get("clean_df_path") or state.get("raw_df_path")
        if not csv_path or not os.path.exists(csv_path):
            raise FileNotFoundError(f"Cleaned dataset path not found at: '{csv_path}'")

        df = pd.read_csv(csv_path)

        # 2. Get profile text
        profile_data = state.get("profile")
        if not profile_data:
            profile_data = profile_dataframe(df)
        profile_text_block = profile_to_text(profile_data)

        # --------------------------------------------------------------------
        # Re-run Mode (review_feedback is present)
        # --------------------------------------------------------------------
        if review_feedback:
            failed_feedback_by_id = {
                fb["id"]: fb
                for fb in review_feedback
                if str(fb.get("verdict", "")).strip().lower() == "fail"
            }

            events.append({
                "agent": "analyst",
                "type": "info",
                "message": f"Processing review feedback: re-running {len(failed_feedback_by_id)} failed analysis item(s).",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            updated_results: list[dict[str, Any]] = []

            for item in existing_results:
                item_id = item.get("id")
                if item_id in failed_feedback_by_id:
                    fb = failed_feedback_by_id[item_id]
                    q = item.get("question", "")
                    old_code = item.get("code", "")
                    reason = fb.get("reason", "Needs correction")
                    suggested_fix = fb.get("suggested_fix", "")

                    events.append({
                        "agent": "analyst",
                        "type": "info",
                        "message": f"Revising question id {item_id}: '{q}' per reviewer feedback.",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                    redo_prompt = f"""Question: {q}

The previous pandas code was rejected by the reviewer.
Previous code:
```python
{old_code}
```

Reviewer critique: {reason}
Suggested fix: {suggested_fix}

Write corrected pandas code that answers the question using the dataframe called df.
Store the final answer in a variable called result (a DataFrame, Series, or single number).
Use only pd and np, no imports, no file access. Return ONLY the code, no explanation."""

                    llm_reply = call_llm(redo_prompt)
                    new_code = _clean_code_fence(llm_reply)

                    sandbox_res, final_code, attempts, notes = _execute_with_retry(
                        question=q,
                        initial_code=new_code,
                        df=df,
                        call_llm_fn=call_llm,
                        events=events,
                    )

                    is_ok = sandbox_res["status"] == "ok"
                    updated_item = {
                        "id": item_id,
                        "question": q,
                        "code": final_code,
                        "result": sandbox_res.get("result"),
                        "status": "ok" if is_ok else "failed",
                        "error": sandbox_res.get("error") if not is_ok else None,
                        "attempts": attempts,
                        "notes": notes,
                    }
                    updated_results.append(updated_item)
                else:
                    # Keep successful items completely untouched
                    updated_results.append(item)

            rerun_ids = list(failed_feedback_by_id.keys())
            return {
                "analysis_results": updated_results,
                "rerun_ids": rerun_ids,
                "review_feedback": [],  # Clear review feedback so reviewer starts fresh
                "events": events,
            }

        # --------------------------------------------------------------------
        # First Run Mode (review_feedback is empty)
        # --------------------------------------------------------------------
        questions_prompt = f"""You are a senior data analyst.
Analyze this dataset profile and propose 3-5 insightful business questions that can be answered from this data.

{profile_text_block}

Return ONLY a JSON list of questions (strings). Questions must be answerable using the columns given. Mix types: totals, comparisons across groups, trends over time (only if a date column exists), top/bottom N. Each question must have ONE single answer that fits in ONE table. Do not ask compound questions with 'and which...' or 'and how...'.
Do not return code or explanations."""

        # Generate questions with retry if JSON parsing fails
        questions_reply = call_llm(questions_prompt)
        questions = _parse_questions_json(questions_reply)

        if not questions:
            # Retry once
            events.append({
                "agent": "analyst",
                "type": "warning",
                "message": "First questions generation reply was not valid JSON. Retrying...",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            retry_reply = call_llm(questions_prompt + "\n\nCRITICAL: Return ONLY a valid JSON list of strings like [\"Question 1?\", \"Question 2?\", \"Question 3?\"].")
            questions = _parse_questions_json(retry_reply)

        if not questions:
            # Fall back to default questions derived from column dtypes
            fallback_msg = "Questions JSON parsing failed after retry. Using default questions."
            events.append({
                "agent": "analyst",
                "type": "warning",
                "message": fallback_msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            questions = _generate_default_questions(df)

        events.append({
            "agent": "analyst",
            "type": "info",
            "message": f"Generated {len(questions)} analytical business questions.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Answer each question via pandas code and execute in sandbox
        analysis_results: list[dict[str, Any]] = []

        for idx, q in enumerate(questions):
            code_prompt = f"""Write pandas code that answers the question using the dataframe called df.
Question: {q}

Columns available:
{', '.join(df.columns)}

Store the final answer in a variable called result (a DataFrame, Series, or single number).
Use only pd and np, no imports, no file access. Return ONLY the code, no explanation."""

            code_reply = call_llm(code_prompt)
            initial_code = _clean_code_fence(code_reply)

            sandbox_res, final_code, attempts, notes = _execute_with_retry(
                question=q,
                initial_code=initial_code,
                df=df,
                call_llm_fn=call_llm,
                events=events,
            )

            status = "ok" if sandbox_res["status"] == "ok" else "failed"
            events.append({
                "agent": "analyst",
                "type": "info" if status == "ok" else "error",
                "message": f"Question {idx + 1}/{len(questions)} ('{q}') completed with status '{status}' (attempts: {attempts}).",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            analysis_results.append({
                "id": idx,
                "question": q,
                "code": final_code,
                "result": sandbox_res.get("result"),
                "status": status,
                "error": sandbox_res.get("error") if status != "ok" else None,
                "attempts": attempts,
                "notes": notes,
            })

        return {
            "questions": questions,
            "analysis_results": analysis_results,
            "rerun_ids": [],
            "events": events,
        }

    except (LLMUnavailableError, LLMConfigError):
        raise
    except Exception as e:
        err_msg = f"Analyst agent encountered an unexpected error: {e}"
        events.append({
            "agent": "analyst",
            "type": "error",
            "message": err_msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # Graceful return with whatever results exist to prevent pipeline crash
        return {
            "questions": existing_questions,
            "analysis_results": existing_results,
            "rerun_ids": [],
            "events": events,
        }
