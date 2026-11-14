"""
sandbox.py
Code Execution Sandbox: Safely runs LLM-generated pandas code within an isolated
multiprocessing sandbox with AST static checks, restricted builtins, execution timeouts,
and protected original datasets.
"""

import ast
import multiprocessing
from typing import Any
import numpy as np
import pandas as pd

# Modules permitted in import statements
ALLOWED_MODULES = {"pandas", "numpy"}

# Function and builtin calls forbidden during AST static inspection
DISALLOWED_CALLS = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
}

# Restricted builtins exposed inside the sandbox exec environment
ALLOWED_BUILTINS = {
    "len": len,
    "range": range,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "isinstance": isinstance,
    "print": print,
}


def _validate_code_ast(code: str) -> tuple[bool, str | None, str]:
    """
    Safety Layer 1: Static AST validation.
    Inspects the Python syntax tree before execution to block unsafe imports,
    dangerous builtin calls, and double-underscore attribute inspection.

    Returns:
        tuple[bool, str | None, str]: (is_allowed, error_message, status)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})", "error"

    for node in ast.walk(tree):
        # Check standard imports (e.g., import os, import sys)
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    return (
                        False,
                        f"Importing module '{alias.name}' is blocked. Only pandas and numpy are allowed.",
                        "blocked",
                    )

        # Check from-imports (e.g., from os import path)
        elif isinstance(node, ast.ImportFrom):
            if not node.module:
                return False, "Relative imports are blocked.", "blocked"
            root = node.module.split(".")[0]
            if root not in ALLOWED_MODULES:
                return (
                    False,
                    f"Importing from '{node.module}' is blocked. Only pandas and numpy are allowed.",
                    "blocked",
                )

        # Check dangerous function calls (e.g., open(...), eval(...), getattr(...))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_CALLS:
                return False, f"Call to '{node.func.id}' is blocked for security.", "blocked"
            if isinstance(node.func, ast.Attribute) and node.func.attr in DISALLOWED_CALLS:
                return False, f"Call to '{node.func.attr}' is blocked for security.", "blocked"

        # Check double-underscore attribute access (e.g., obj.__class__, obj.__subclasses__)
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return (
                    False,
                    f"Attribute access starting with '__' ('{node.attr}') is blocked.",
                    "blocked",
                )

        # Check direct access to dunder names (e.g., __builtins__, __class__)
        elif isinstance(node, ast.Name):
            if node.id.startswith("__"):
                return False, f"Direct access to '{node.id}' is blocked.", "blocked"

    return True, None, "ok"


def _clean_series(s: pd.Series) -> pd.Series:
    """Converts Interval, Period, Timestamp, Categorical, and non-JSON-friendly values to strings."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s.astype(str)
    if (
        isinstance(s.dtype, pd.IntervalDtype)
        or isinstance(s.dtype, pd.PeriodDtype)
        or pd.api.types.is_datetime64_any_dtype(s.dtype)
    ):
        return s.astype(str)
    if pd.api.types.is_object_dtype(s.dtype):
        return s.map(
            lambda x: str(x)
            if isinstance(x, (pd.Interval, pd.Period, pd.Timestamp))
            else x
        )
    return s


def _clean_index(idx: pd.Index) -> pd.Index:
    """Converts non-JSON-friendly index values (Interval, Period, Timestamp, Categorical) to strings."""
    if isinstance(idx, pd.MultiIndex):
        new_levels = []
        for level in idx.levels:
            if (
                isinstance(level.dtype, pd.CategoricalDtype)
                or isinstance(level.dtype, pd.IntervalDtype)
                or isinstance(level.dtype, pd.PeriodDtype)
                or pd.api.types.is_datetime64_any_dtype(level.dtype)
            ):
                new_levels.append(level.astype(str))
            else:
                new_levels.append(
                    level.map(
                        lambda x: str(x)
                        if isinstance(x, (pd.Interval, pd.Period, pd.Timestamp))
                        else x
                    )
                )
        return idx.set_levels(new_levels)
    elif (
        isinstance(idx.dtype, pd.CategoricalDtype)
        or isinstance(idx.dtype, pd.IntervalDtype)
        or isinstance(idx.dtype, pd.PeriodDtype)
        or pd.api.types.is_datetime64_any_dtype(idx.dtype)
    ):
        return idx.astype(str)
    else:
        return idx.map(
            lambda x: str(x)
            if isinstance(x, (pd.Interval, pd.Period, pd.Timestamp))
            else x
        )


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Converts non-JSON-friendly values in DataFrame columns and index into strings or numbers."""
    cleaned = df.copy()
    cleaned.index = _clean_index(cleaned.index)
    if isinstance(cleaned.columns, pd.MultiIndex):
        cleaned.columns = pd.Index([
            "_".join(str(c) for c in col if str(c)).strip()
            for col in cleaned.columns
        ])
    else:
        cleaned.columns = _clean_index(cleaned.columns)

    for col in cleaned.columns:
        cleaned[col] = _clean_series(cleaned[col])
    return cleaned


def _convert_result(res: Any) -> Any:
    """
    Safety Layer 4: Result conversion.
    Converts Series to DataFrame, converts numpy numbers to Python numbers,
    and converts non-JSON values (Interval, Period, Timestamp, Categorical)
    in columns and index into plain strings or numbers.
    """
    # Convert pandas Series to DataFrame via reset_index
    if isinstance(res, pd.Series):
        res = res.reset_index()
        return _clean_dataframe(res)

    # DataFrames remain as DataFrames with sanitized index and columns
    if isinstance(res, pd.DataFrame):
        return _clean_dataframe(res)

    # Convert non-JSON scalar types
    if isinstance(res, (pd.Interval, pd.Period, pd.Timestamp)):
        return str(res)

    # Convert numpy numbers and booleans to plain Python types
    if isinstance(res, (np.integer, int)) and not isinstance(res, (bool, np.bool_)):
        return int(res)
    if isinstance(res, (np.floating, float)):
        return float(res)
    if isinstance(res, (np.bool_, bool)):
        return bool(res)

    return res


def _sandbox_worker(conn: Any, code: str, df: pd.DataFrame) -> None:
    """
    Safety Layer 3: Restricted in-process execution.
    Executes within an isolated child process with only permitted builtins
    and a deep copy of the DataFrame.
    """
    try:
        # Guarantee original DataFrame cannot be modified
        df_copy = df.copy(deep=True)

        # Provide restricted globals dictionary
        exec_scope = {
            "__builtins__": dict(ALLOWED_BUILTINS),
            "pd": pd,
            "pandas": pd,
            "np": np,
            "numpy": np,
            "df": df_copy,
        }

        # Run user/agent-generated code
        exec(code, exec_scope)

        # Verify 'result' variable was assigned
        if "result" not in exec_scope:
            conn.send({
                "status": "error",
                "result": None,
                "error": "code did not define a variable named result",
            })
            return

        raw_result = exec_scope["result"]

        # Check for disallowed aggregate structures (tuple, list, or dict containing DataFrames)
        is_tuple_or_list = isinstance(raw_result, (tuple, list))
        is_dict_of_dfs = isinstance(raw_result, dict) and any(
            isinstance(v, (pd.DataFrame, pd.Series)) for v in raw_result.values()
        )
        if is_tuple_or_list or is_dict_of_dfs:
            conn.send({
                "status": "error",
                "result": None,
                "error": "result must be ONE DataFrame, Series, or a single number, not a tuple/list. Combine the answer into a single table.",
            })
            return

        # Sanitize result to safe format
        safe_result = _convert_result(raw_result)
        conn.send({
            "status": "ok",
            "result": safe_result,
            "error": None,
        })

    except BaseException as e:
        # Return clean error message (exception type and message, no traceback)
        conn.send({
            "status": "error",
            "result": None,
            "error": f"{type(e).__name__}: {e}",
        })
    finally:
        conn.close()


def run_code(code: str, df: pd.DataFrame, timeout: int = 10) -> dict[str, Any]:
    """
    Safely executes LLM-generated pandas code inside an isolated subprocess.

    Safety measures applied:
    1. Static AST validation (blocks unsafe modules, calls, and dunder lookups).
    2. Subprocess isolation via multiprocessing (spawn-safe on Windows).
    3. Restricted builtins and deep copy of the DataFrame.
    4. Timeout enforcement (terminates runaway/infinite loops).
    5. Clean error handling (short exception messages, required 'result' variable).

    Args:
        code (str): Python pandas code string to execute.
        df (pd.DataFrame): Input dataframe available as 'df' inside code.
        timeout (int): Maximum execution time in seconds (default: 10).

    Returns:
        dict[str, Any]: Dictionary with status ('ok', 'error', 'blocked', 'timeout'),
                        result (computed value or None), and error (message or None).
    """
    # Step 1: Pre-execution static check using AST
    is_valid, error_msg, status = _validate_code_ast(code)
    if not is_valid:
        return {
            "status": status,
            "result": None,
            "error": error_msg,
        }

    # Step 2: Separate process execution (Windows spawn-safe)
    parent_conn, child_conn = multiprocessing.Pipe()
    process = multiprocessing.Process(
        target=_sandbox_worker,
        args=(child_conn, code, df),
    )
    process.start()

    # Step 3: Wait for process completion within timeout
    if parent_conn.poll(timeout):
        try:
            output = parent_conn.recv()
        except EOFError:
            output = {
                "status": "error",
                "result": None,
                "error": "Execution process ended unexpectedly without returning output.",
            }
        process.join()
        parent_conn.close()
        return output
    else:
        # Timeout reached: terminate process safely
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()

        parent_conn.close()
        return {
            "status": "timeout",
            "result": None,
            "error": f"Execution timed out after {timeout} seconds.",
        }
