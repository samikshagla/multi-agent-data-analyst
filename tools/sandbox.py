"""
sandbox.py
Code Execution Sandbox: Safely runs LLM-generated pandas code within an isolated
multiprocessing sandbox with AST static checks, restricted builtins, execution timeouts,
and protected original datasets.

KNOWN LIMITATIONS & SECURITY NOTE:
1. Best-Effort Defense-in-Depth:
   This sandbox layer is a best-effort defense-in-depth mechanism that reduces risk
   for local development, interactive demos, and evaluation. It is NOT an impenetrable
   operating-system security boundary. Running arbitrary untrusted code on a public
   or production server requires kernel-level isolation (e.g. gVisor, Firecracker microVM,
   or unprivileged OCI containers with read-only root filesystems and disabled network access).
2. Column Access via Subscripting:
   DataFrame columns whose names contain '__' or equal denylisted attribute names (e.g.
   'mro', 'code', 'func', 'values', 'pipe', 'style', 'plot') cannot be accessed via
   attribute notation (e.g. `df.mro`). They must be accessed using standard bracket
   subscript notation (e.g. `df['mro']` or `df['__id']`).
3. Platform-Specific Resource Limits:
   On Windows platforms, the POSIX `resource` module is unavailable, so kernel-level memory
   limits (RLIMIT_AS) cannot be enforced. Protection against excessive memory consumption
   on Windows relies on subprocess timeout enforcement (wall-clock termination) and post-execution
   result size / type inspection.
"""

import ast
import multiprocessing
import os
import pickle
import re
import shutil
import string
import sys
import tempfile
from typing import Any
import numpy as np
import pandas as pd

# Modules permitted in import statements
ALLOWED_MODULES = {"pandas", "numpy"}

# Plain names blocked anywhere in code (calls, references, aliases)
DENYLIST_NAMES = {
    "breakpoint",
    "help",
    "dir",
    "type",
    "object",
    "format_map",
    "vars",
    "locals",
    "globals",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "memoryview",
    "compile",
    "exec",
    "eval",
    "open",
    "input",
    "__import__",
}

# Attributes blocked on any object
DENYLIST_ATTRS = {
    # System / OS / Subprocess / Network
    "os",
    "sys",
    "io",
    "subprocess",
    "importlib",
    "builtins",
    "ctypes",
    "socket",
    "shutil",
    "pathlib",
    "pickle",
    "marshal",
    "compat",
    "util",
    "testing",
    "core",
    "errors",
    "_libs",
    "extensions",
    "style",
    "plot",
    "pipe",
    "eval_expr",
    # Writers (to_csv, to_pickle, to_sql, to_parquet, to_excel, to_json, etc.; tolist is allowed)
    "to_csv",
    "to_pickle",
    "to_sql",
    "to_parquet",
    "to_excel",
    "to_json",
    "to_html",
    "to_latex",
    "to_feather",
    "to_hdf",
    "to_clipboard",
    "to_stata",
    "to_xml",
    "to_orc",
    "to_markdown",
    "to_gbq",
    "tofile",
    # Numpy file IO and memory mapping
    "load",
    "save",
    "savez",
    "savetxt",
    "loadtxt",
    "genfromtxt",
    "fromfile",
    "memmap",
    "ctypeslib",
    "lib",
    # Python object reflection, frames, and bytecode inspection
    "mro",
    "subclasses",
    "bases",
    "func",
    "code",
    "gi_frame",
    "gi_code",
    "cr_frame",
    "cr_code",
    "tb_frame",
    "tb_next",
    "f_globals",
    "f_locals",
    "f_back",
    "f_code",
    "f_builtins",
    "co_code",
    "co_consts",
    "im_func",
    "im_self",
    "ag_frame",
}

# Top-level allowlist for pandas (pd.* and pandas.*)
PD_ALLOWLIST = {
    "DataFrame",
    "Series",
    "Index",
    "MultiIndex",
    "Categorical",
    "CategoricalDtype",
    "Timestamp",
    "Timedelta",
    "Period",
    "Interval",
    "IntervalIndex",
    "DatetimeIndex",
    "PeriodIndex",
    "TimedeltaIndex",
    "NaT",
    "NA",
    "concat",
    "merge",
    "merge_asof",
    "crosstab",
    "pivot",
    "pivot_table",
    "melt",
    "get_dummies",
    "cut",
    "qcut",
    "to_datetime",
    "to_numeric",
    "to_timedelta",
    "isna",
    "isnull",
    "notna",
    "notnull",
    "unique",
    "value_counts",
    "factorize",
    "date_range",
    "period_range",
    "timedelta_range",
    "interval_range",
    "Grouper",
    "NamedAgg",
    "array",
    "api",
    "IndexSlice",
}

# pd.api.types allowlist
PD_API_TYPES_ALLOWLIST = {
    "is_numeric_dtype",
    "is_string_dtype",
    "is_datetime64_any_dtype",
    "is_categorical_dtype",
    "is_bool_dtype",
    "is_integer_dtype",
    "is_float_dtype",
    "is_object_dtype",
    "is_period_dtype",
    "is_timedelta64_dtype",
}

# Top-level allowlist for numpy (np.* and numpy.*)
NP_ALLOWLIST = {
    "nan",
    "inf",
    "pi",
    "e",
    "newaxis",
    "where",
    "select",
    "mean",
    "median",
    "std",
    "var",
    "sum",
    "prod",
    "min",
    "max",
    "argmin",
    "argmax",
    "percentile",
    "quantile",
    "round",
    "around",
    "floor",
    "ceil",
    "abs",
    "absolute",
    "sqrt",
    "square",
    "exp",
    "log",
    "log10",
    "log2",
    "power",
    "clip",
    "cumsum",
    "cumprod",
    "diff",
    "unique",
    "sort",
    "argsort",
    "array",
    "asarray",
    "arange",
    "linspace",
    "zeros_like",
    "ones_like",
    "isnan",
    "isinf",
    "isfinite",
    "isin",
    "any",
    "all",
    "count_nonzero",
    "corrcoef",
    "cov",
    "average",
    "ptp",
    "nansum",
    "nanmean",
    "nanmedian",
    "nanstd",
    "nanvar",
    "nanmin",
    "nanmax",
    "nanpercentile",
    "nanquantile",
    "searchsorted",
    "digitize",
    "histogram",
    "bincount",
    "maximum",
    "minimum",
    "sign",
    "mod",
    "floor_divide",
    "true_divide",
    "divide",
    "multiply",
    "add",
    "subtract",
    "negative",
    "float64",
    "float32",
    "int64",
    "int32",
    "bool_",
    "datetime64",
    "timedelta64",
    "number",
    "integer",
    "floating",
    "ndarray",
}

# Aggregation names allowed in string dispatch (.agg, .apply, .transform, etc.)
AGG_ALLOWLIST = {
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "count",
    "std",
    "var",
    "size",
    "first",
    "last",
    "nunique",
    "prod",
    "sem",
    "skew",
    "kurt",
    "idxmin",
    "idxmax",
    "cumsum",
    "cumprod",
    "cummax",
    "cummin",
    "rank",
    "diff",
    "pct_change",
    "abs",
    "round",
    "quantile",
    "mode",
    "any",
    "all",
    "head",
    "tail",
}

# Callable names allowed as arguments in dispatch methods
CALLABLE_ALLOWLIST = {
    "len",
    "sum",
    "min",
    "max",
    "abs",
    "round",
    "str",
    "int",
    "float",
    "bool",
    "list",
    "set",
    "tuple",
    "sorted",
}

# Method names subjected to string/callable dispatch validation
DISPATCH_METHODS = {
    "agg",
    "aggregate",
    "apply",
    "transform",
    "map",
    "applymap",
    "pipe",
}

def _restricted_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    """
    Restricted import function exposed in child sandbox builtins.
    Returns the already-injected pandas or numpy module for top-level imports only
    (level == 0 and fromlist is empty or None). Never imports anything new and never
    calls the real __import__.
    """
    if level == 0 and not fromlist:
        if name == "pandas":
            return pd
        elif name == "numpy":
            return np
    raise ImportError("import not allowed")


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
    "__import__": _restricted_import,
}


def _check_dispatch_arg(target: ast.AST) -> tuple[bool, str | None]:
    """Validates an argument passed to a dispatch method (.agg, .apply, .transform, etc.)."""
    if isinstance(target, ast.Constant):
        if isinstance(target.value, str):
            if target.value not in AGG_ALLOWLIST:
                return False, f"function name '{target.value}' is not allowed"
        return True, None
    elif isinstance(target, (ast.List, ast.Tuple)):
        for elt in target.elts:
            ok, msg = _check_dispatch_arg(elt)
            if not ok:
                return False, msg
        return True, None
    elif isinstance(target, ast.Dict):
        for val in target.values:
            ok, msg = _check_dispatch_arg(val)
            if not ok:
                return False, msg
        return True, None
    elif isinstance(target, ast.Name):
        if target.id not in CALLABLE_ALLOWLIST:
            return False, f"function name '{target.id}' is not allowed"
        return True, None
    elif isinstance(target, ast.Lambda):
        return True, None
    elif isinstance(target, ast.Attribute):
        # Attribute callable: must be an allowed numpy function (e.g. np.mean)
        if isinstance(target.value, ast.Name) and target.value.id in ("np", "numpy") and target.attr in NP_ALLOWLIST:
            return True, None
        return False, f"function name '{target.attr}' is not allowed"
    else:
        return False, "Disallowed argument in dispatch call."


def _get_max_depth(tree: ast.AST) -> int:
    """Computes the maximum AST nesting depth using an iterative stack with early exit."""
    stack = [(tree, 1)]
    max_d = 1
    while stack:
        node, d = stack.pop()
        if d > max_d:
            max_d = d
        if max_d > 40:
            return max_d  # Early exit as soon as limit is exceeded
        for child in ast.iter_child_nodes(node):
            stack.append((child, d + 1))
    return max_d


def _validate_code_ast(code: str) -> tuple[bool, str | None, str]:
    """
    Safety Layer 1: Static AST validation.
    Inspects the Python syntax tree before execution to block unsafe imports,
    dangerous builtin calls, attribute traversal, query/eval injection,
    and excessive complexity.

    Returns:
        tuple[bool, str | None, str]: (is_allowed, error_message, status)
    """
    # Check 1: Code character length (maximum 20,000 characters)
    if len(code) > 20000:
        return False, "Code exceeds maximum length (20,000 characters).", "blocked"

    # Check 2: Parse AST
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})", "error"

    # Check 3: Complexity & nesting depth limits
    node_count = 0
    for _ in ast.walk(tree):
        node_count += 1
        if node_count > 2000:
            return False, "Code exceeds maximum complexity (more than 2,000 AST nodes).", "blocked"

    if _get_max_depth(tree) > 40:
        return False, "Code exceeds maximum nesting depth (more than 40 levels).", "blocked"

    # Annotate parent references for contextual checks
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore

    # Check 4: Walk all syntax nodes
    for node in ast.walk(tree):
        # 4a. Check standard imports: ONLY "import pandas", "import pandas as <name>",
        # "import numpy", "import numpy as <name>" (exact top-level names, no dotted names).
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ("pandas", "numpy"):
                    return (
                        False,
                        f"Importing module '{alias.name}' is blocked. Only 'import pandas' and 'import numpy' are allowed.",
                        "blocked",
                    )
                if alias.asname and (alias.asname in DENYLIST_NAMES or alias.asname.startswith("__")):
                    return (
                        False,
                        f"Import alias '{alias.asname}' is blocked.",
                        "blocked",
                    )

        # 4b. Check from-imports: ImportFrom is not allowed at all (no star, no relative imports).
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module if node.module else "."
            return (
                False,
                f"Importing from '{mod_name}' is blocked. 'from ... import ...' statements are not allowed.",
                "blocked",
            )

        # 4c. String literals containing '__'
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "__" in node.value:
                return False, "String literal containing '__' is blocked.", "blocked"

        # 4d. Plain names (denylist + dunder names, blocking aliasing like f = getattr)
        elif isinstance(node, ast.Name):
            if node.id.startswith("__"):
                return False, f"Direct access to '{node.id}' is blocked.", "blocked"
            if node.id in DENYLIST_NAMES:
                return False, f"Access to '{node.id}' is blocked.", "blocked"

        # 4e. Attribute access checks
        elif isinstance(node, ast.Attribute):
            # Block any attribute starting with an underscore (including private/dunder)
            if node.attr.startswith("__"):
                return (
                    False,
                    f"Attribute access starting with '__' ('{node.attr}') is blocked.",
                    "blocked",
                )
            if node.attr.startswith("_"):
                return (
                    False,
                    f"Attribute access starting with '_' ('{node.attr}') is blocked.",
                    "blocked",
                )

            # Block any reader attribute (read_csv, read_pickle, read_json, etc.)
            if node.attr.startswith("read_"):
                return False, f"Attribute '{node.attr}' is not allowed.", "blocked"

            # Block denylisted attributes (writers, system modules, internals, frame tricks)
            if node.attr in DENYLIST_ATTRS:
                return False, f"Attribute '{node.attr}' is not allowed.", "blocked"

            # Top-level namespace allowlist for pd / pandas
            if isinstance(node.value, ast.Name) and node.value.id in ("pd", "pandas"):
                if node.attr not in PD_ALLOWLIST:
                    return False, f"Attribute 'pd.{node.attr}' is not allowed.", "blocked"

            # Top-level namespace allowlist for np / numpy
            if isinstance(node.value, ast.Name) and node.value.id in ("np", "numpy"):
                if node.attr not in NP_ALLOWLIST:
                    return False, f"Attribute 'np.{node.attr}' is not allowed.", "blocked"

            # Allow pd.api.types only (block pd.api.extensions and any other access under pd.api)
            if node.attr == "api":
                parent = getattr(node, "parent", None)
                if not (isinstance(parent, ast.Attribute) and parent.attr == "types"):
                    return False, "Access to 'api' is only allowed as pd.api.types.", "blocked"

            if isinstance(node.value, ast.Attribute) and getattr(node.value, "attr", None) == "api":
                if node.attr != "types":
                    return (
                        False,
                        f"Access to 'api.{node.attr}' is blocked. Only pd.api.types is allowed.",
                        "blocked",
                    )

            # pd.api.types.* allowlist
            if (
                isinstance(node.value, ast.Attribute)
                and node.value.attr == "types"
                and isinstance(getattr(node.value, "value", None), ast.Attribute)
                and node.value.value.attr == "api"
            ):
                if node.attr not in PD_API_TYPES_ALLOWLIST:
                    return (
                        False,
                        f"Attribute 'pd.api.types.{node.attr}' is not allowed.",
                        "blocked",
                    )

        # 4f. Call checks (including .query(), .eval(), .format(), and string dispatch)
        elif isinstance(node, ast.Call):
            # Format string checks: .format() and .format_map()
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("format", "format_map"):
                # Must be a plain string literal receiver (blocks dynamic strings, variables, joins, concats)
                if not (isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str)):
                    return False, f"Calls to .{node.func.attr}() require a plain string literal receiver.", "blocked"

                fmt_str = node.func.value.value
                try:
                    formatter = string.Formatter()
                    for literal_text, field_name, format_spec, conversion in formatter.parse(fmt_str):
                        if field_name is not None:
                            if "." in field_name or "[" in field_name:
                                return (
                                    False,
                                    f"Replacement field '{field_name}' in format string is not allowed.",
                                    "blocked",
                                )
                except ValueError as e:
                    return False, f"Invalid format string: {e}", "blocked"

            # Function name is a plain Name
            if isinstance(node.func, ast.Name):
                if node.func.id in DENYLIST_NAMES or node.func.id.startswith("__"):
                    return False, f"Call to '{node.func.id}' is blocked.", "blocked"

            # Function name is an Attribute
            elif isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                if attr_name in DENYLIST_NAMES or attr_name in DENYLIST_ATTRS or attr_name.startswith("_"):
                    return False, f"Call to '{attr_name}' is blocked.", "blocked"

                # Check string / callable dispatch methods (.agg, .apply, .transform, .map, .applymap, .pipe)
                if attr_name in DISPATCH_METHODS:
                    if node.args:
                        ok, msg = _check_dispatch_arg(node.args[0])
                        if not ok:
                            return False, msg, "blocked"
                        for arg in node.args[1:]:
                            if isinstance(arg, (ast.Constant, ast.Name, ast.List, ast.Tuple, ast.Dict)):
                                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                    if arg.value not in AGG_ALLOWLIST:
                                        return False, f"function name '{arg.value}' is not allowed", "blocked"

                    for kw in node.keywords:
                        if kw.arg in ("func", "f", "mapper", "arg"):
                            ok, msg = _check_dispatch_arg(kw.value)
                            if not ok:
                                return False, msg, "blocked"
                        elif isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            if kw.arg not in ("axis", "na_action", "raw", "result_type", "engine", "engine_kwargs"):
                                if kw.value.value not in AGG_ALLOWLIST:
                                    return False, f"function name '{kw.value.value}' is not allowed", "blocked"

                # Guard for .query(...) and .eval(...)
                if attr_name in ("query", "eval"):
                    if not node.args:
                        return False, f"Call to .{attr_name}() requires at least one argument.", "blocked"

                    for arg in node.args:
                        # Must be a plain string literal (no variables, no f-strings, no concatenation)
                        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                            return False, f"Arguments to .{attr_name}() must be plain string literals.", "blocked"

                        expr_str = arg.value
                        if "@" in expr_str:
                            return False, f"Expression in .{attr_name}() contains '@', which is blocked.", "blocked"
                        if "__" in expr_str:
                            return False, f"Expression in .{attr_name}() contains '__', which is blocked.", "blocked"

                        # Block assignment in eval (e.g. a = 1)
                        if attr_name == "eval" and re.search(r"(?<![=<>!~])=(?!=)", expr_str):
                            return False, f"Assignment in .{attr_name}() is blocked.", "blocked"

                        # Extract identifier tokens and ensure none are in the denylists
                        tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr_str))
                        forbidden = (
                            DENYLIST_NAMES
                            | DENYLIST_ATTRS
                            | {"pd", "pandas", "np", "numpy", "os", "sys", "subprocess", "io"}
                        )
                        found = tokens & forbidden
                        if found:
                            bad_tok = sorted(found)[0]
                            return False, f"Expression in .{attr_name}() contains disallowed identifier '{bad_tok}'.", "blocked"

                    # Check keyword arguments (block inplace=True)
                    for kw in node.keywords:
                        if kw.arg == "inplace":
                            return False, f"Keyword 'inplace' in .{attr_name}() is blocked.", "blocked"

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


def _sandbox_worker(conn: Any, code: str, df: pd.DataFrame, timeout: int = 10) -> None:
    """
    Safety Layer 3: Restricted in-process execution.
    Executes within an isolated child process with:
    - Completely cleared os.environ
    - Fresh temporary directory as cwd
    - POSIX resource limits (address space, CPU time, open files, file creation)
    - Whitelisted builtins and cloned DataFrame
    - Result size and serialization caps (200,000 cells / 10 MB)
    """
    temp_dir = None
    try:
        # Step A: Wipe environment variables in child process
        os.environ.clear()

        # Step B: Switch to fresh isolated temporary directory
        temp_dir = tempfile.mkdtemp(prefix="sandbox_")
        os.chdir(temp_dir)

        # Step C: On POSIX systems, apply strict resource constraints (after pandas/numpy import)
        if sys.platform != "win32":
            try:
                import resource
                # Memory limit: ~1.5 GB
                as_limit = int(1.5 * 1024 * 1024 * 1024)
                resource.setrlimit(resource.RLIMIT_AS, (as_limit, as_limit))
                # CPU limit: timeout + 2 seconds
                cpu_limit = max(1, int(timeout) + 2)
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
                # File descriptor limit: 32
                resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
                # File size limit: 0 (prevents writing files to disk)
                resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
            except (ImportError, AttributeError, ValueError, OSError):
                pass

        # Step D: Deep copy DataFrame
        df_copy = df.copy(deep=True)

        # Step E: Restricted globals dictionary
        exec_scope = {
            "__builtins__": dict(ALLOWED_BUILTINS),
            "pd": pd,
            "pandas": pd,
            "np": np,
            "numpy": np,
            "df": df_copy,
        }

        # Step F: Run code
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

        # Validate result type: must be DataFrame, Series, scalar, or numpy scalar
        allowed_scalar_types = (
            int,
            float,
            bool,
            str,
            np.number,
            np.bool_,
            pd.Timestamp,
            pd.Timedelta,
            pd.Interval,
            pd.Period,
        )
        if not isinstance(raw_result, (pd.DataFrame, pd.Series, allowed_scalar_types)) and raw_result is not None:
            conn.send({
                "status": "error",
                "result": None,
                "error": f"Result must be a DataFrame, Series, or scalar, not {type(raw_result).__name__}.",
            })
            return

        # Scalar string length limit (10,000 characters)
        if isinstance(raw_result, str) and len(raw_result) > 10_000:
            conn.send({
                "status": "error",
                "result": None,
                "error": f"Result string exceeds maximum allowed length ({len(raw_result):,} characters > 10,000 character limit).",
            })
            return

        # Scalar integer bit length limit (10,000 bits) to prevent DoS from huge powers (e.g. 2**10**7)
        if isinstance(raw_result, int) and not isinstance(raw_result, bool) and raw_result.bit_length() > 10_000:
            conn.send({
                "status": "error",
                "result": None,
                "error": f"Result integer exceeds maximum allowed size ({raw_result.bit_length():,} bits > 10,000 bit limit).",
            })
            return

        # Check table cell count limit (200,000 cells)
        if isinstance(raw_result, (pd.DataFrame, pd.Series)):
            cell_count = raw_result.size
            if cell_count > 200_000:
                conn.send({
                    "status": "error",
                    "result": None,
                    "error": f"Result table exceeds maximum allowed size ({cell_count:,} cells > 200,000 cell limit).",
                })
                return

        # Sanitize result to safe format
        safe_result = _convert_result(raw_result)

        # Check serialized size limit (10 MB)
        try:
            serialized = pickle.dumps(safe_result)
            if len(serialized) > 10 * 1024 * 1024:
                conn.send({
                    "status": "error",
                    "result": None,
                    "error": f"Result exceeds maximum serialized size ({len(serialized):,} bytes > 10 MB limit).",
                })
                return
        except Exception:
            pass

        conn.send({
            "status": "ok",
            "result": safe_result,
            "error": None,
        })

    except BaseException as e:
        conn.send({
            "status": "error",
            "result": None,
            "error": f"{type(e).__name__}: {e}",
        })
    finally:
        try:
            conn.close()
        except Exception:
            pass
        if temp_dir:
            try:
                os.chdir(tempfile.gettempdir())
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


def run_code(code: str, df: pd.DataFrame, timeout: int = 10) -> dict[str, Any]:
    """
    Safely executes LLM-generated pandas code inside an isolated subprocess.

    Safety measures applied:
    1. Static AST validation (blocks unsafe modules, dangerous attributes, dunder traversal,
       aliasing, and query/eval injection).
    2. Subprocess isolation via explicit 'spawn' multiprocessing context.
    3. Child environment wiped (os.environ cleared, temporary directory cwd, POSIX limits).
    4. Restricted builtins and deep copy of the DataFrame.
    5. Result table cell count and serialized size caps (200,000 cells / 10 MB).
    6. Timeout enforcement (terminates runaway/infinite loops).
    7. Clean error handling (short exception messages, required 'result' variable).

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

    # Step 2: Separate process execution using explicit 'spawn' start method
    # Set single-threaded execution for BLAS/OpenMP libraries to prevent thread exhaustion under POSIX limits
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    process = ctx.Process(
        target=_sandbox_worker,
        args=(child_conn, code, df, timeout),
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
        process.join(timeout=2)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
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
