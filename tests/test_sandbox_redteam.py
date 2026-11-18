"""
tests/test_sandbox_redteam.py
Red-team attack suite against tools/sandbox.py.
Exercises known sandbox bypasses, file system leaks, attribute traversal,
query/eval injection, code aliasing, and denial-of-service payloads.
"""

import os
import numpy as np
import pandas as pd
import pytest
from tools.sandbox import run_code


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})


# 1. Filesystem & os leaks via pd.io.common.os
def test_attack_pd_io_os_listdir(df):
    code = "result = pd.io.common.os.listdir('.')"
    res = run_code(code, df)
    # Target: must be blocked. If status == 'ok', attack succeeds (vulnerability).
    assert res["status"] == "blocked", f"Attack succeeded with result: {res['result']}"


def test_attack_pd_io_os_getcwd(df):
    code = "result = pd.io.common.os.getcwd()"
    res = run_code(code, df)
    assert res["status"] == "blocked", f"Attack succeeded with result: {res['result']}"


# 2. File read attacks via pandas readers
@pytest.mark.parametrize("reader_call", [
    "pd.read_csv('data/sample.csv')",
    "pd.read_pickle('x.pkl')",
    "pd.read_json('x.json')",
    "pd.read_excel('x.xlsx')",
    "pd.read_parquet('x.parquet')",
    "pd.read_sql('SELECT 1', 'sqlite:///')",
])
def test_attack_pd_readers(df, reader_call):
    code = f"result = {reader_call}"
    res = run_code(code, df)
    assert res["status"] == "blocked", f"Attack '{reader_call}' succeeded: status={res['status']}, err={res['error']}"


# 3. File write attacks via pandas writers
@pytest.mark.parametrize("writer_call", [
    "df.to_csv('x.csv')",
    "df.to_pickle('x.pkl')",
    "df.to_json('x.json')",
    "df.to_sql('table', 'sqlite:///')",
    "df.to_excel('x.xlsx')",
    "df.to_parquet('x.parquet')",
    "df.to_html('x.html')",
    "df.to_feather('x.feather')",
    "df.to_hdf('x.h5', 'key')",
    "df.to_clipboard()",
])
def test_attack_pd_writers(df, writer_call):
    code = f"{writer_call}\nresult = 1"
    res = run_code(code, df)
    assert res["status"] == "blocked", f"Attack '{writer_call}' succeeded: status={res['status']}"


# 4. Numpy file and memory mapping attacks
@pytest.mark.parametrize("np_call", [
    "np.load('x.npy')",
    "np.save('x.npy', df['a'])",
    "np.loadtxt('x.txt')",
    "np.fromfile('x.bin')",
    "np.memmap('x.mmap')",
])
def test_attack_numpy_file_io(df, np_call):
    code = f"result = {np_call}"
    res = run_code(code, df)
    assert res["status"] == "blocked", f"Numpy attack '{np_call}' succeeded: status={res['status']}"


# 5. Internal module traversal
@pytest.mark.parametrize("attr_access", [
    "np.ctypeslib",
    "np.lib.npyio",
    "np.testing",
    "pd.testing",
    "pd.compat",
    "pd.util",
    "pd.core",
    "pd._libs",
    "pd.api.extensions",
])
def test_attack_module_traversal(df, attr_access):
    code = f"result = str({attr_access})"
    res = run_code(code, df)
    assert res["status"] == "blocked", f"Traversal '{attr_access}' succeeded: status={res['status']}, res={res['result']}"


# 6. Format string traversal into dunder attributes
def test_attack_format_string_class(df):
    code = "result = '{0.__class__}'.format(df)"
    res = run_code(code, df)
    assert res["status"] == "blocked", f"Format class attack succeeded: {res['result']}"


def test_attack_format_string_globals(df):
    code = "result = '{0.__init__.__globals__}'.format(df)"
    res = run_code(code, df)
    assert res["status"] == "blocked", f"Format globals attack succeeded: {res['result']}"


# 7. Query and Eval injection
def test_attack_query_at_escape(df):
    code = "result = df.query('@pd.io.common.os.getcwd() == 1')"
    res = run_code(code, df)
    assert res["status"] == "blocked", f"df.query attack succeeded: status={res['status']}, res={res['result']}"


def test_attack_query_dynamic_string(df):
    code = "col = 'a'\nresult = df.query(f'{col} > 1')"
    res = run_code(code, df)
    assert res["status"] == "blocked", f"Dynamic query string succeeded: status={res['status']}"


def test_attack_df_eval_write(df):
    code = "df.eval('a = 999', inplace=True)\nresult = df['a'].sum()"
    res = run_code(code, df)
    assert res["status"] == "blocked", f"df.eval attack succeeded: status={res['status']}"


# 8. Unsafe pandas attributes: style, plot, pipe
@pytest.mark.parametrize("unsafe_method", [
    "df.style",
    "df.plot",
    "df.pipe(print)",
])
def test_attack_unsafe_pd_methods(df, unsafe_method):
    code = f"result = str({unsafe_method})"
    res = run_code(code, df)
    assert res["status"] == "blocked", f"Method '{unsafe_method}' succeeded: status={res['status']}"


# 9. Builtins aliasing and plain name access
@pytest.mark.parametrize("builtin_name", [
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "vars",
    "dir",
    "type",
    "object",
    "__import__",
    "compile",
    "exec",
    "eval",
    "open",
    "input",
    "breakpoint",
    "help",
    "globals",
    "locals",
    "format_map",
    "memoryview",
])
def test_attack_builtin_name_alias(df, builtin_name):
    code = f"f = {builtin_name}\nresult = 1"
    res = run_code(code, df)
    assert res["status"] == "blocked", f"Builtin alias '{builtin_name}' succeeded: status={res['status']}"


# 10. Lambda or list comprehension hiding dangerous calls
def test_attack_lambda_eval(df):
    code = "f = lambda x: eval(x)\nresult = f('1+1')"
    res = run_code(code, df)
    assert res["status"] == "blocked", f"Lambda eval succeeded: status={res['status']}"


def test_attack_comprehension_dunder(df):
    code = "result = [x for x in [df] if '{0.__class__}'.format(x)][0]"
    res = run_code(code, df)
    assert res["status"] == "blocked", f"Comprehension attack succeeded: status={res['status']}"


# 11. Huge output denial of service
def test_attack_huge_output(df):
    code = "result = pd.DataFrame({'a': range(10**7)})"
    res = run_code(code, df)
    # Must be rejected / blocked, not status 'ok' returning a 10M row DataFrame
    assert res["status"] != "ok", f"Huge output succeeded with status 'ok'"


# 12. Reading os.environ
def test_attack_read_environ_via_pd(df):
    code = "result = str(pd.io.common.os.environ.get('PATH'))"
    res = run_code(code, df)
    assert res["status"] == "blocked", f"Reading os.environ succeeded: status={res['status']}, res={res['result']}"


# 13. Resource & Complexity limits
def test_attack_code_too_long(df):
    code = "# padding comment\n" * 1500 + "result = 1"
    assert len(code) > 20000
    res = run_code(code, df)
    assert res["status"] == "blocked"
    assert "20,000 characters" in res["error"]


def test_attack_nesting_too_deep(df):
    expr = "1"
    for _ in range(45):
        expr = f"({expr} + 1)"
    code = f"result = {expr}"
    res = run_code(code, df)
    assert res["status"] == "blocked"
    assert "nesting depth" in res["error"]


def test_attack_ast_node_count_too_large(df):
    # Construct an AST with > 2,000 nodes without exceeding depth
    code = "result = [" + ", ".join(["1"] * 2100) + "]"
    res = run_code(code, df)
    assert res["status"] == "blocked"
    assert "2,000 AST nodes" in res["error"]


def test_attack_memory_bomb(df):
    code = "result = np.ones(10**10)"
    res = run_code(code, df)
    assert res["status"] == "error"
    assert "Memory" in res["error"] or "unable to allocate" in res["error"].lower()


# 14. Verified Legal Patterns (Must Pass with status 'ok')
def test_legal_groupby_agg(df):
    code = "result = df.groupby('a')['b'].agg(['sum', 'mean']).reset_index()"
    res = run_code(code, df)
    assert res["status"] == "ok"
    assert isinstance(res["result"], pd.DataFrame)


def test_legal_pivot_table_merge_cut_qcut():
    df_legal = pd.DataFrame({"cat": ["A", "B", "A", "B"], "val": [10.0, 20.0, 30.0, 40.0]})
    code = """
p = df.pivot_table(index='cat', values='val', aggfunc='mean').reset_index()
b = pd.DataFrame({'cat': ['A', 'B'], 'factor': [1, 2]})
m = pd.merge(p, b, on='cat')
m['bin'] = pd.cut(m['val'], bins=2)
m['qbin'] = pd.qcut(m['val'], q=2)
result = m
"""
    res = run_code(code, df_legal)
    assert res["status"] == "ok"


def test_legal_accessors_and_apply():
    df_legal = pd.DataFrame({
        "txt": [" hello ", "WORLD"],
        "date_str": ["2026-01-01", "2026-02-01"],
        "num": [10, 20],
    })
    code = """
df['txt_clean'] = df['txt'].str.strip().str.lower()
df['dt'] = pd.to_datetime(df['date_str'])
df['month'] = df['dt'].dt.month
df['doubled'] = df['num'].apply(lambda x: x * 2)
result = df
"""
    res = run_code(code, df_legal)
    assert res["status"] == "ok"


def test_legal_query_where_nan_types():
    df_legal = pd.DataFrame({"a": [1, 6, 10], "b": ["x", "x", "y"], "c": ["1.5", "2.5", "NaN"]})
    code = """
filtered = df.query("a > 5 and b == 'x'")
val = np.where(filtered['a'] > 5, filtered['a'], np.nan)
is_num = pd.api.types.is_numeric_dtype(df['a'])
num_c = pd.to_numeric(df['c'], errors='coerce')
result = float(num_c.mean())
"""
    res = run_code(code, df_legal)
    assert res["status"] == "ok"


def test_legal_values_sorting_cleaning():
    df_legal = pd.DataFrame({"cat": ["A", "B", "A"], "val": [1.234, 5.678, np.nan]})
    code = """
counts = df['cat'].value_counts()
top = df.nlargest(2, 'val')
sorted_df = df.sort_values(by='val', ascending=False)
rounded = df['val'].round(2)
filled = df['val'].fillna(0.0).astype(float)
result = float(filled.sum())
"""
    res = run_code(code, df_legal)
    assert res["status"] == "ok"
    assert res["result"] == 6.912


# 15. Real Reviewer Snippets from Evaluation Logs (Must Pass with status 'ok')
def test_reviewer_snippet_pivot_table_loc_bool():
    df_sample = pd.DataFrame({
        "is_peak_hour": [True, False, True, True],
        "discount_applied_usd": [2.5, 1.0, 3.0, 4.5],
    })
    code = "result = df.pivot_table(index='is_peak_hour', values='discount_applied_usd', aggfunc='sum').loc[True]"
    res = run_code(code, df_sample)
    assert res["status"] == "ok"


def test_reviewer_snippet_pivot_table_sort_head_reset_index():
    df_sample = pd.DataFrame({
        "cuisine_category": ["Italian", "Mexican", "Indian", "Chinese", "Thai", "American"],
        "customer_rating": [4.2, 4.5, 4.8, 3.9, 4.6, 4.1],
    })
    code = "result = df.pivot_table(index='cuisine_category', values='customer_rating', aggfunc='mean').sort_values(by='customer_rating', ascending=False).head(5).reset_index()"
    res = run_code(code, df_sample)
    assert res["status"] == "ok"
    assert len(res["result"]) == 5


def test_reviewer_snippet_groupby_mean_loc_mean():
    df_sample = pd.DataFrame({
        "weather_condition": ["Rainy", "Heavy Rain / Storm", "Clear", "Rainy"],
        "total_fulfillment_time_min": [30.0, 34.0, 25.0, 32.0],
    })
    code = "result = df.groupby('weather_condition')['total_fulfillment_time_min'].mean().loc[['Rainy', 'Heavy Rain / Storm']].mean()"
    res = run_code(code, df_sample)
    assert res["status"] == "ok"
    assert abs(res["result"] - 32.5) < 1e-5


def test_reviewer_snippet_query_weather_condition_mean():
    df_sample = pd.DataFrame({
        "weather_condition": ["Rainy", "Heavy Rain / Storm", "Clear", "Rainy"],
        "total_fulfillment_time_min": [30.0, 34.0, 25.0, 32.0],
    })
    code = "result = df.query(\"weather_condition in ['Rainy', 'Heavy Rain / Storm']\")['total_fulfillment_time_min'].mean()"
    res = run_code(code, df_sample)
    assert res["status"] == "ok"
    assert abs(res["result"] - 32.0) < 1e-5

