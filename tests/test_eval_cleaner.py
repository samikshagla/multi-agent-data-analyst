"""
test_eval_cleaner.py
Unit tests for the Data Cleaning evaluation pipeline:
- baseline_clean deterministic transformations
- cleaner_metrics accuracy, damage detection, and case tolerance
- TokenBudget limit enforcement and BudgetExceeded exception
- run_cleaner_eval execution, monkeypatching, and per-seed resumability
- --max-tokens CLI option on run_reviewer_eval
"""

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pytest

from eval.baseline_cleaning import baseline_clean
from eval.budget import BudgetExceeded, TokenBudget
from eval.cleaner_metrics import cleaner_metrics
from eval.inject_data_errors import make_dirty_dataset
from eval.run_cleaner_eval import run_cleaner_evaluation
from eval.run_reviewer_eval import run_evaluation as run_reviewer_evaluation
import llm


# ============================================================================
# 1. baseline_clean Tests
# ============================================================================

def test_baseline_clean_transformations():
    """
    Tests baseline_clean on a synthetic dataframe:
    - Drops exact duplicate rows
    - Trims whitespace from string columns
    - Converts '$12.50' string currency columns to numeric float
    - Title-cases low cardinality text columns (< 5% unique values)
    - Leaves ID-like columns untouched
    """
    # 30 base rows + 1 exact duplicate = 31 rows
    # 1 unique city across 30 rows = 1/30 = 0.033 < 0.05 (triggers title-casing)
    n = 30
    df = pd.DataFrame({
        "order_id": [f"ord_{i}" for i in range(n)],
        "customer_id": [f"cust_{i}" for i in range(n)],  # ID-like column
        "city": ["  new york  "] * n,                     # text column with whitespace and lowercase
        "price": ["$12.50", " $15.00 ", "N/A"] * 10,     # currency strings
    })
    # Add an exact duplicate of row 0
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert len(df) == 31

    cleaned_df, logs = baseline_clean(df)

    # 1. Duplicate row dropped (31 -> 30)
    assert len(cleaned_df) == 30
    assert any("duplicate" in log.lower() for log in logs)

    # 2. Whitespace trimmed & title cased for low-cardinality city
    assert cleaned_df["city"].unique().tolist() == ["New York"]

    # 3. ID columns preserved without title casing
    assert cleaned_df["order_id"].iloc[0] == "ord_0"
    assert cleaned_df["customer_id"].iloc[0] == "cust_0"

    # 4. Price converted to numeric float
    assert pd.api.types.is_numeric_dtype(cleaned_df["price"])
    assert cleaned_df["price"].iloc[0] == pytest.approx(12.50)
    assert cleaned_df["price"].iloc[1] == pytest.approx(15.00)
    assert pd.isna(cleaned_df["price"].iloc[2])


# ============================================================================
# 2. cleaner_metrics Tests
# ============================================================================

@pytest.fixture
def clean_eval_data():
    """Provides a small, controlled ground-truth dataset and manifest."""
    clean_df = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "category": ["Electronics", "Clothing", "Food", "Electronics", "Clothing"],
        "amount": [100.0, 50.0, 20.0, 80.0, 60.0],
    })

    dirty_df = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5, 5],  # Injected duplicate of row index 4 (order_id 5)
        "category": ["Electronics", "CLOTHING", "Food", "Electronics", "Clothing", "Clothing"],
        "amount": ["$100.00", "50.0", "N/A", "-80.0", "60.0", "60.0"],
    })

    manifest = {
        "key_column": "order_id",
        "duplicate_rows": {"count": 1, "source_indices": [4], "injected_indices": [5]},
        "missing_values": {
            "amount": {"count": 1, "indices": [2]}
        },
        "inconsistent_text": {
            "column": "category",
            "indices": [1],
        },
        "impossible_values": {
            "column": "amount",
            "indices": [3],
        },
        "numbers_stored_as_text": {
            "column": "amount",
            "indices": [2, 0],  # pos 0 -> mod 0 (N/A), pos 1 -> mod 1 ($100.00)
        },
        "true_means": {
            "amount": 62.0,
        },
    }
    return clean_df, dirty_df, manifest


def test_cleaner_metrics_perfect_scores(clean_eval_data):
    """Verifies that a perfectly restored DataFrame receives perfect scores and zero damage."""
    clean_df, dirty_df, manifest = clean_eval_data

    cleaned_df = clean_df.copy(deep=True)
    metrics = cleaner_metrics(clean_df, dirty_df, cleaned_df, manifest)

    # Rows metric
    assert metrics["rows"]["expected_rows"] == 5
    assert metrics["rows"]["cleaned_rows"] == 5
    assert metrics["rows"]["valid_rows_lost"] == 0
    assert metrics["rows"]["duplicate_rows_remaining"] == 0
    assert metrics["rows"]["duplicate_removal_rate"] == 1.0

    # Missing values: index 2 has clean 20.0
    assert metrics["missing_values"]["amount"]["remaining_missing"] == 0
    assert metrics["missing_values"]["amount"]["rate_handled"] == 1.0
    assert metrics["missing_values"]["amount"]["valid_rows_removed"] == 0

    # Inconsistent text: category is normalized
    assert metrics["inconsistent_text"]["share_normalized"] == 1.0

    # Negative values: no negatives in clean_df
    assert metrics["negative_values"]["remaining_negatives"] == 0

    # Numbers stored as text: amount is numeric float and recovered
    assert metrics["numbers_stored_as_text"]["is_numeric"] is True
    assert metrics["numbers_stored_as_text"]["recovery_share"] == 1.0

    # Damage: zero unintended changes
    assert metrics["damage"]["unintended_change_rate"] == 0.0
    assert metrics["damage"]["unintentionally_changed_cells"] == 0

    # Statistical drift
    assert metrics["statistical_drift"]["amount"]["relative_drift"] == pytest.approx(0.0)


def test_cleaner_metrics_detects_lost_valid_rows(clean_eval_data):
    """Verifies valid_rows_lost is detected when valid rows are erroneously dropped."""
    clean_df, dirty_df, manifest = clean_eval_data

    # Cleaned dataframe dropped order_id 1 and 2
    cleaned_df = clean_df.iloc[2:].copy(deep=True)
    metrics = cleaner_metrics(clean_df, dirty_df, cleaned_df, manifest)

    assert metrics["rows"]["valid_rows_lost"] == 2


def test_cleaner_metrics_detects_remaining_duplicates(clean_eval_data):
    """Verifies remaining duplicate rows are properly detected."""
    clean_df, dirty_df, manifest = clean_eval_data

    # Duplicate row was kept
    cleaned_df = pd.concat([clean_df, clean_df.iloc[[4]]], ignore_index=True)
    metrics = cleaner_metrics(clean_df, dirty_df, cleaned_df, manifest)

    assert metrics["rows"]["duplicate_rows_remaining"] == 1
    assert metrics["rows"]["duplicate_removal_rate"] == 0.0


def test_cleaner_metrics_damage_detection_and_case_tolerance(clean_eval_data):
    """
    Verifies that:
    - Case-only changes to uncorrupted text cells are NOT counted as damage.
    - True changes to uncorrupted cells ARE counted as damage and reported in top columns.
    """
    clean_df, dirty_df, manifest = clean_eval_data

    # 1. Case-only modification on uncorrupted row 0 category: 'Electronics' -> 'ELECTRONICS'
    cleaned_df = clean_df.copy(deep=True)
    cleaned_df.loc[0, "category"] = "ELECTRONICS"
    metrics = cleaner_metrics(clean_df, dirty_df, cleaned_df, manifest)
    assert metrics["damage"]["unintended_change_rate"] == 0.0
    assert metrics["damage"]["unintentionally_changed_cells"] == 0

    # 2. Genuine modification on uncorrupted cell: row 0 category: 'Electronics' -> 'Groceries'
    cleaned_df.loc[0, "category"] = "Groceries"
    metrics = cleaner_metrics(clean_df, dirty_df, cleaned_df, manifest)
    assert metrics["damage"]["unintentionally_changed_cells"] == 1
    assert metrics["damage"]["unintended_change_rate"] > 0.0
    assert metrics["damage"]["top_changed_columns"][0]["column"] == "category"


def test_cleaner_metrics_currency_recovery_and_statistical_drift(clean_eval_data):
    """Verifies currency recovery calculation and statistical drift."""
    clean_df, dirty_df, manifest = clean_eval_data

    cleaned_df = clean_df.copy(deep=True)
    # Modify amount values slightly
    cleaned_df["amount"] = [100.0, 50.0, 20.0, 80.0, 70.0]  # mean is 64.0 vs true 62.0
    metrics = cleaner_metrics(clean_df, dirty_df, cleaned_df, manifest)

    # Relative drift: |64.0 - 62.0| / 62.0 = 2.0 / 62.0 ≈ 0.032258
    assert metrics["statistical_drift"]["amount"]["relative_drift"] == pytest.approx(2.0 / 62.0, rel=1e-3)
    # Row 0 ($100.00 currency) recovered with 100.0
    assert metrics["numbers_stored_as_text"]["recovery_share"] == 1.0


# ============================================================================
# 3. TokenBudget & BudgetExceeded Tests
# ============================================================================

def test_token_budget_raises_budget_exceeded(monkeypatch: pytest.MonkeyPatch):
    """Verifies TokenBudget raises BudgetExceeded when tokens consumed exceed max_tokens."""
    fake_stats = {"total_tokens": 1000}
    monkeypatch.setattr(llm, "get_usage_stats", lambda: fake_stats)

    budget = TokenBudget(max_tokens=500)
    # No usage yet
    budget.check()
    assert budget.tokens_used == 0

    # Consume 400 tokens -> below limit
    fake_stats["total_tokens"] = 1400
    budget.check()
    assert budget.tokens_used == 400

    # Consume 150 more tokens -> 550 >= 500 limit
    fake_stats["total_tokens"] = 1550
    with pytest.raises(BudgetExceeded) as exc_info:
        budget.check()

    assert exc_info.value.used_tokens == 550
    assert exc_info.value.max_tokens == 500


# ============================================================================
# 4. run_cleaner_eval Tests (Resumability & Monkeypatching)
# ============================================================================

def test_run_cleaner_eval_with_monkeypatch_and_resumability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Verifies run_cleaner_eval:
    - Runs cleanly with monkeypatched cleaner_node without real LLM calls
    - Saves progress to seeds.jsonl, summary.json, and summary.md
    - Resumes gracefully and skips already evaluated seeds
    """
    # Create a small sample CSV
    csv_file = tmp_path / "tiny_sample.csv"
    sample_df = pd.DataFrame({
        "order_id": list(range(1, 21)),
        "metro_city": ["Mumbai", "Delhi", "Bangalore", "Pune"] * 5,
        "cuisine_category": ["Burgers & Fast Food", "Healthy & Bowls"] * 10,
        "order_value_usd": [15.0 + i for i in range(20)],
    })
    sample_df.to_csv(csv_file, index=False)

    run_dir = tmp_path / "cleaner_eval_run"

    cleaner_call_count = {"count": 0}

    def mock_cleaner_node(state: dict[str, Any]) -> dict[str, Any]:
        cleaner_call_count["count"] += 1
        raw_path = state.get("raw_df_path", "")
        df_raw = pd.read_csv(raw_path)
        # Mock cleaning: baseline clean
        clean_df, _ = baseline_clean(df_raw)
        out_path = tmp_path / f"mock_clean_{cleaner_call_count['count']}.csv"
        clean_df.to_csv(out_path, index=False)
        return {
            "clean_df_path": str(out_path),
            "profile": {},
            "cleaning_log": ["Mock dropped duplicate rows"],
            "events": [{"agent": "cleaner", "type": "info", "message": "Mock action"}],
        }

    monkeypatch.setattr("eval.run_cleaner_eval.cleaner_node", mock_cleaner_node)

    # 1. Run with seed 42 only
    run_cleaner_evaluation(
        seeds=[42],
        run_dir=str(run_dir),
        csv_path=str(csv_file),
    )

    assert cleaner_call_count["count"] == 1
    seeds_file = run_dir / "seeds.jsonl"
    summary_file = run_dir / "summary.json"
    summary_md = run_dir / "summary.md"

    assert seeds_file.exists()
    assert summary_file.exists()
    assert summary_md.exists()

    with open(summary_file, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    assert summary_data["n_seeds"] == 1
    assert summary_data["seeds"] == [42]

    # 2. Re-run with seeds [42, 43]: should resume and only call mock_cleaner_node for seed 43
    run_cleaner_evaluation(
        seeds=[42, 43],
        run_dir=str(run_dir),
        csv_path=str(csv_file),
    )

    assert cleaner_call_count["count"] == 2  # Called only once for seed 43!

    with open(summary_file, "r", encoding="utf-8") as f:
        summary_data_resumed = json.load(f)
    assert summary_data_resumed["n_seeds"] == 2
    assert summary_data_resumed["seeds"] == [42, 43]


def test_run_cleaner_eval_budget_exceeded_stops_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verifies run_cleaner_eval stops cleanly when budget is exceeded before seed 2."""
    csv_file = tmp_path / "tiny_sample.csv"
    sample_df = pd.DataFrame({
        "order_id": list(range(1, 10)),
        "metro_city": ["Mumbai"] * 9,
        "cuisine_category": ["Healthy"] * 9,
        "order_value_usd": [20.0] * 9,
    })
    sample_df.to_csv(csv_file, index=False)

    run_dir = tmp_path / "budget_cleaner_run"

    fake_stats = {"total_tokens": 0, "calls": 0, "by_model": {}}
    monkeypatch.setattr(llm, "get_usage_stats", lambda: fake_stats)

    def mock_cleaner_node(state: dict[str, Any]) -> dict[str, Any]:
        # Consume 600 tokens
        fake_stats["total_tokens"] += 600
        out_path = tmp_path / "mock.csv"
        sample_df.to_csv(out_path, index=False)
        return {
            "clean_df_path": str(out_path),
            "profile": {},
            "cleaning_log": [],
            "events": [],
        }

    monkeypatch.setattr("eval.run_cleaner_eval.cleaner_node", mock_cleaner_node)

    # Budget is 500. Seed 1 consumes 600. Before seed 2, budget.check() raises BudgetExceeded and exits cleanly.
    with pytest.raises(SystemExit) as exc_info:
        run_cleaner_evaluation(
            seeds=[42, 43],
            max_tokens=500,
            run_dir=str(run_dir),
            csv_path=str(csv_file),
        )

    assert exc_info.value.code == 1
    # Seed 42 was written to seeds.jsonl before pause
    seeds_file = run_dir / "seeds.jsonl"
    assert seeds_file.exists()
    lines = seeds_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["seed"] == 42


# ============================================================================
# 5. run_reviewer_eval --max-tokens Option Tests
# ============================================================================

def test_reviewer_eval_max_tokens_stops_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verifies that --max-tokens option on reviewer eval stops cleanly before exceeding budget."""
    run_dir = tmp_path / "reviewer_budget_run"

    fake_stats = {"total_tokens": 0, "calls": 0, "by_model": {}}
    monkeypatch.setattr(llm, "get_usage_stats", lambda: fake_stats)

    def mock_reviewer_node(state: dict[str, Any]) -> dict[str, Any]:
        fake_stats["total_tokens"] += 600
        return {
            "review_report": [
                {
                    "id": 0,
                    "question": "q",
                    "verdict": "pass",
                    "reason": "ok",
                    "recomputed_value": "1",
                    "recomputed_result": 1,
                }
            ],
            "review_feedback": [],
            "caught_errors": [],
            "events": [],
        }

    monkeypatch.setattr("eval.run_reviewer_eval.reviewer_node", mock_reviewer_node)

    # Limit = 3 cases, max_tokens = 500
    # Case 1 consumes 600 tokens. Before case 2, budget.check() raises BudgetExceeded.
    with pytest.raises(SystemExit) as exc_info:
        run_reviewer_evaluation(
            limit=3,
            repeats=1,
            run_dir=str(run_dir),
            csv_path="data/sample.csv",
            max_tokens=500,
        )

    assert exc_info.value.code == 1

    # Case 1 progress was saved
    cases_file = run_dir / "cases.jsonl"
    assert cases_file.exists()
    lines = cases_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


# ============================================================================
# 5. Diagnostic and Harness Regression Tests
# ============================================================================

def test_make_dirty_dataset_preserves_id_and_key_columns():
    """
    Verifies that make_dirty_dataset never changes the key or ID-like columns (no nulls,
    all original values preserved), for seeds 42, 43, 44 on data/sample.csv.
    """
    csv_path = Path("data/sample.csv")
    if not csv_path.exists():
        pytest.skip("data/sample.csv not found")

    clean_df = pd.read_csv(csv_path)
    clean_keys = set(clean_df["order_id"])

    for seed in [42, 43, 44]:
        dirty_df, manifest = make_dirty_dataset(clean_df, seed=seed)
        key_col = manifest["key_column"]
        assert key_col == "order_id"
        assert "order_id" in manifest["excluded_columns"]

        # No nulls in key column
        assert dirty_df[key_col].notna().all()

        # All original clean keys are present in dirty_df
        dirty_keys = set(dirty_df[key_col])
        assert clean_keys.issubset(dirty_keys)

        # ID-like columns (order_id, customer_id, cot_market_dynamics_rationale)
        for col in manifest["excluded_columns"]:
            assert col not in manifest["missing_values"]
            if manifest["inconsistent_text"]:
                assert col != manifest["inconsistent_text"]["column"]
            if manifest["impossible_values"]:
                assert col != manifest["impossible_values"]["column"]
            if manifest["numbers_stored_as_text"]:
                assert col != manifest["numbers_stored_as_text"]["column"]


def test_known_answer_baseline_metrics_sample_csv():
    """
    KNOWN-ANSWER test: for seeds 42, 43, 44, baseline_clean gives:
    - duplicate_removal_rate == 1.0
    - valid_rows_lost == 0
    - no rows removed for missing values (row_removed_count == 0, valid_rows_removed == 0)
    - 0 missing values handled by row removal (row_removed_rate == 0.0)
    - text normalisation == 1.0 (100%)
    - key nulls == 0
    - unintended change rate <= 0.001
    """
    csv_path = Path("data/sample.csv")
    if not csv_path.exists():
        pytest.skip("data/sample.csv not found")

    clean_df = pd.read_csv(csv_path)

    for seed in [42, 43, 44]:
        dirty_df, manifest = make_dirty_dataset(clean_df, seed=seed)
        cleaned_df, _ = baseline_clean(dirty_df)
        metrics = cleaner_metrics(clean_df, dirty_df, cleaned_df, manifest)

        # 1. Duplicates
        assert metrics["rows"]["duplicate_removal_rate"] == 1.0
        assert metrics["rows"]["duplicate_rows_remaining"] == 0

        # 2. Row preservation & identifier integrity
        assert metrics["rows"]["valid_rows_lost"] == 0
        assert metrics["rows"]["key_nulls_in_cleaned"] == 0
        assert metrics["identifier_integrity"]["key_nulls_in_cleaned"] == 0
        assert metrics["rows"]["key_duplicates_in_cleaned"] == 0

        # 3. Missing values
        for col, mv_data in metrics["missing_values"].items():
            assert mv_data["valid_rows_removed"] == 0
            assert mv_data["row_removed_count"] == 0
            assert mv_data["row_removed_rate"] == 0.0

        # 4. Text normalisation
        assert metrics["inconsistent_text"]["share_normalized"] == 1.0

        # 5. Damage / Unintended change rate
        assert metrics["damage"]["unintended_change_rate"] <= 0.001


def test_cleaned_df_with_nan_keys_does_not_create_phantom_duplicates():
    """
    Verifies that a cleaned_df with NaN keys does NOT create phantom duplicates.
    Series.duplicated() in pandas treats NaNs as duplicates, which previously collapsed
    duplicate removal rate to 0.4%.
    """
    clean_df = pd.DataFrame({
        "order_id": ["o1", "o2", "o3", "o4", "o5"],
        "val": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    # Dirty df has 1 exact duplicate of row index 4 (o5)
    dirty_df = pd.concat([clean_df, clean_df.iloc[[4]]], ignore_index=True)
    manifest = {
        "key_column": "order_id",
        "duplicate_rows": {"count": 1, "source_indices": [4], "injected_indices": [5]},
        "missing_values": {},
        "true_means": {"val": 30.0},
    }

    # Cleaned df drops the duplicate, but has 2 NaN keys in different rows
    cleaned_df = clean_df.copy()
    cleaned_df.loc[0, "order_id"] = np.nan
    cleaned_df.loc[1, "order_id"] = np.nan

    metrics = cleaner_metrics(clean_df, dirty_df, cleaned_df, manifest)

    # Exact duplicate rows in cleaned_df must be 0 (no phantom duplicates)
    assert metrics["rows"]["duplicate_rows_remaining"] == 0
    assert metrics["rows"]["duplicate_removal_rate"] == 1.0

    # Key nulls must be detected
    assert metrics["rows"]["key_nulls_in_cleaned"] == 2
    assert metrics["identifier_integrity"]["key_nulls_in_cleaned"] == 2

    # Rows with NaN keys are counted as lost
    assert metrics["rows"]["valid_rows_lost"] == 2


def test_row_removal_credited_only_when_key_really_absent():
    """
    Verifies that missing value row removal is credited ONLY when the row key is really absent
    from cleaned_df, not merely because a cell is missing.
    """
    clean_df = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "amount": [100.0, 200.0, 300.0],
    })
    dirty_df = clean_df.copy()
    dirty_df.loc[1, "amount"] = np.nan  # injected missing at row 1 (o2)
    manifest = {
        "key_column": "order_id",
        "duplicate_rows": {"count": 0},
        "missing_values": {"amount": {"count": 1, "indices": [1]}},
        "true_means": {"amount": 200.0},
    }

    # Scenario A: Row o2 is STILL PRESENT in cleaned_df, but amount is still missing
    cleaned_a = dirty_df.copy()
    metrics_a = cleaner_metrics(clean_df, dirty_df, cleaned_a, manifest)
    mv_a = metrics_a["missing_values"]["amount"]
    assert mv_a["row_removed_count"] == 0
    assert mv_a["valid_rows_removed"] == 0
    assert mv_a["still_missing_count"] == 1
    assert mv_a["filled_count"] == 0

    # Scenario B: Row o2 was ACTUALLY REMOVED from cleaned_df
    cleaned_b = clean_df[clean_df["order_id"] != "o2"].copy()
    metrics_b = cleaner_metrics(clean_df, dirty_df, cleaned_b, manifest)
    mv_b = metrics_b["missing_values"]["amount"]
    assert mv_b["row_removed_count"] == 1
    assert mv_b["valid_rows_removed"] == 1
    assert mv_b["still_missing_count"] == 0
    assert mv_b["filled_count"] == 0


def test_fill_missing_mode_on_key_column_increases_key_duplicates():
    """
    Verifies that if a cleaner imputes nulls in the key column using mode,
    key_duplicates_in_cleaned reflects the resulting duplicate non-null keys.
    """
    clean_df = pd.DataFrame({
        "order_id": [f"ord_{i}" for i in range(20)],
        "val": [float(i) for i in range(20)],
    })
    dirty_df = clean_df.copy()
    manifest = {
        "key_column": "order_id",
        "duplicate_rows": {"count": 0},
        "missing_values": {},
        "true_means": {},
    }

    # Cleaner imputed 5 keys with 'ord_0'
    cleaned_df = clean_df.copy()
    for i in range(1, 6):
        cleaned_df.loc[i, "order_id"] = "ord_0"

    metrics = cleaner_metrics(clean_df, dirty_df, cleaned_df, manifest)
    # 5 duplicates of 'ord_0'
    assert metrics["rows"]["key_duplicates_in_cleaned"] == 5
    assert metrics["identifier_integrity"]["key_duplicates_in_cleaned"] == 5
    # Since 5 unique keys were replaced, 5 valid keys were lost
    assert metrics["rows"]["valid_rows_lost"] == 5


def test_baseline_only_and_recompute_metrics_only_workflow(tmp_path: Path):
    """
    Verifies that:
    1. --baseline-only runs without LLM, creates seeds.jsonl, summary.md, summary.json,
       and saves artifacts under <run-dir>/artifacts/.
    2. Adding a fake LLM output and running --recompute-metrics-only reloads artifacts,
       recomputes metrics, and updates the summaries with zero LLM calls.
    """
    run_dir = tmp_path / "baseline_and_recompute_run"

    clean_csv = tmp_path / "test_sample.csv"
    sample_df = pd.DataFrame({
        "order_id": [f"id_{i}" for i in range(50)],
        "cat": ["A" if i % 2 == 0 else "B" for i in range(50)],
        "num": [float(i) for i in range(50)],
    })
    sample_df.to_csv(clean_csv, index=False)

    # 1. Run baseline-only
    run_path = run_cleaner_evaluation(
        seeds=[42],
        run_dir=str(run_dir),
        csv_path=str(clean_csv),
        baseline_only=True,
    )
    assert Path(run_path).exists()

    # Verify artifacts exist
    artifacts_dir = run_dir / "artifacts"
    dirty_file = artifacts_dir / "seed_42_dirty.csv"
    manifest_file = artifacts_dir / "seed_42_manifest.json"
    base_file = artifacts_dir / "seed_42_baseline.csv"
    summary_md = run_dir / "summary.md"
    summary_json = run_dir / "summary.json"

    assert dirty_file.exists()
    assert manifest_file.exists()
    assert base_file.exists()
    assert summary_md.exists()
    assert summary_json.exists()

    with open(summary_json, "r", encoding="utf-8") as f:
        s_data = json.load(f)
    assert s_data["total_llm_calls"] == 0
    assert s_data["baseline_means"]["duplicate_removal_rate"] == 1.0

    # 2. Simulate LLM output saved to artifacts
    llm_file = artifacts_dir / "seed_42_llm.csv"
    # LLM cleaned df: copy baseline
    base_df = pd.read_csv(base_file)
    base_df.to_csv(llm_file, index=False)

    # 3. Run recompute-metrics-only
    run_cleaner_evaluation(
        run_dir=str(run_dir),
        csv_path=str(clean_csv),
        recompute_metrics_only=True,
    )

    with open(summary_json, "r", encoding="utf-8") as f:
        recomputed_data = json.load(f)

    # Both baseline and cleaner metrics now present
    assert recomputed_data["baseline_means"]["duplicate_removal_rate"] == 1.0
    assert recomputed_data["cleaner_means"]["duplicate_removal_rate"] == 1.0
    assert recomputed_data["total_llm_calls"] == 0
