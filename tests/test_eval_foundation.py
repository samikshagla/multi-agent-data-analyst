"""
test_eval_foundation.py
Comprehensive test suite for the eval foundation package:
- answer_correct comparator (scalars, tolerances, tables, row reordering, column names,
  extra columns, ignorable index, one-cell tables, text normalisation)
- make_dirty_dataset (determinism, untouched input, manifest counts, type pollution, duplicates)
- benchmark integrity on data/sample.csv (20 items, unique IDs, status ok, no NaN, no empty)
- compute_ground_truth and save/load ground truth round-trip serialization
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from eval.benchmark import BENCHMARK
from eval.ground_truth import compute_ground_truth, save_ground_truth, load_ground_truth
from eval.inject_data_errors import make_dirty_dataset
from eval.scoring import answer_correct


# ============================================================================
# 1. Tests for answer_correct comparator
# ============================================================================

def test_answer_correct_equal_scalars():
    ok, _ = answer_correct(42.0, 42.0)
    assert ok is True

    ok, _ = answer_correct(100, 100)
    assert ok is True

    ok, _ = answer_correct("Berlin", "berlin  ")
    assert ok is True

    ok, _ = answer_correct(True, True)
    assert ok is True

    ok, _ = answer_correct(True, False)
    assert ok is False


def test_answer_correct_scalars_within_and_outside_tolerance():
    # 0.8% difference -> within 1% tol
    ok, _ = answer_correct(100.0, 100.8, tol=0.01)
    assert ok is True

    # 1.5% difference -> outside 1% tol
    ok, reason = answer_correct(100.0, 101.5, tol=0.01)
    assert ok is False
    assert "101.5" in reason and "100.0" in reason

    # Zero handling
    ok, _ = answer_correct(0.0, 0.0000001, tol=0.01)
    assert ok is True

    ok, reason = answer_correct(0.0, 0.05, tol=0.01)
    assert ok is False
    assert "0.05" in reason and "0.0" in reason


def test_answer_correct_scalar_vs_one_cell_table():
    # Scalar vs 1x1 DataFrame
    df_cell = pd.DataFrame({"metric": [99.5]})
    ok, _ = answer_correct(99.5, df_cell)
    assert ok is True

    ok, _ = answer_correct(df_cell, 99.5)
    assert ok is True

    # One cell table with an index column
    df_cell_indexed = pd.DataFrame({"index": [0], "val": [99.5]})
    ok, _ = answer_correct(99.5, df_cell_indexed)
    assert ok is True

    # Scalar vs 2-row DataFrame must fail
    df_two_rows = pd.DataFrame({"metric": [99.5, 10.0]})
    ok, reason = answer_correct(99.5, df_two_rows)
    assert ok is False
    assert "2 rows" in reason


def test_answer_correct_tables_reordered_rows():
    ref = pd.DataFrame({
        "city": ["Berlin", "London", "Paris"],
        "revenue": [100.0, 200.0, 300.0],
    })
    cand = pd.DataFrame({
        "city": ["Paris", "Berlin", "London"],
        "revenue": [300.0, 100.0, 200.0],
    })
    ok, _ = answer_correct(ref, cand)
    assert ok is True


def test_answer_correct_tables_different_column_names():
    ref = pd.DataFrame({
        "channel": ["Online", "Dine-In"],
        "orders": [500, 300],
    })
    cand = pd.DataFrame({
        "order_channel_renamed": ["Online", "Dine-In"],
        "count_val": [500, 300],
    })
    ok, _ = answer_correct(ref, cand)
    assert ok is True


def test_answer_correct_tables_wrong_value():
    ref = pd.DataFrame({
        "category": ["Food", "Drink"],
        "amount": [50.0, 20.0],
    })
    cand = pd.DataFrame({
        "category": ["Food", "Drink"],
        "amount": [50.0, 28.0],  # 28.0 vs 20.0
    })
    ok, reason = answer_correct(ref, cand)
    assert ok is False
    assert "28" in reason and "20" in reason


def test_answer_correct_tables_different_row_count():
    ref = pd.DataFrame({
        "cuisine": ["Italian", "Mexican", "Asian"],
        "score": [4.5, 4.2, 4.8],
    })
    cand = pd.DataFrame({
        "cuisine": ["Italian", "Mexican"],
        "score": [4.5, 4.2],
    })
    ok, reason = answer_correct(ref, cand)
    assert ok is False
    assert "row count 2 vs 3" in reason


def test_answer_correct_tables_extra_column():
    ref = pd.DataFrame({
        "cuisine": ["Italian", "Mexican"],
        "score": [4.5, 4.2],
    })
    # Extra data column: should fail
    cand_extra = pd.DataFrame({
        "cuisine": ["Italian", "Mexican"],
        "score": [4.5, 4.2],
        "bonus_col": [10, 20],
    })
    ok, reason = answer_correct(ref, cand_extra)
    assert ok is False
    assert "3 vs 2" in reason

    # Extra ignorable index column: should pass
    cand_with_index = pd.DataFrame({
        "level_0": [0, 1],
        "cuisine": ["Italian", "Mexican"],
        "score": [4.5, 4.2],
    })
    ok, _ = answer_correct(ref, cand_with_index)
    assert ok is True


def test_answer_correct_text_case_differences():
    ref = pd.DataFrame({
        "city": ["Berlin", "New York"],
        "avg_price": [15.2, 30.5],
    })
    cand = pd.DataFrame({
        "city": ["  BERLIN  ", "new york "],
        "avg_price": [15.2, 30.5],
    })
    ok, _ = answer_correct(ref, cand)
    assert ok is True


# ============================================================================
# 2. Tests for make_dirty_dataset
# ============================================================================

def test_make_dirty_dataset_deterministic_and_preserves_input():
    df = pd.DataFrame({
        "order_id": [f"ord_{i}" for i in range(100)],
        "city": ["Berlin" if i % 2 == 0 else "London" for i in range(100)],
        "channel": ["Online" if i % 3 == 0 else "Takeaway" for i in range(100)],
        "tier": ["$ Budget" if i % 4 == 0 else "$$ Moderate" for i in range(100)],
        "amount": [float(i * 10) for i in range(100)],
        "distance": [float(i * 0.5) for i in range(100)],
        "fee": [float(i * 0.2) for i in range(100)],
        "time": [int(10 + i) for i in range(100)],
    })
    df_copy_before = df.copy(deep=True)

    dirty1, manifest1 = make_dirty_dataset(df, seed=42)
    dirty2, manifest2 = make_dirty_dataset(df, seed=42)

    # 1. Deterministic with the same seed
    pd.testing.assert_frame_equal(dirty1, dirty2)
    assert manifest1 == manifest2

    # 2. Input df is completely untouched
    pd.testing.assert_frame_equal(df, df_copy_before)

    # 3. Manifest contains clean row count, columns, and true means before corruption
    assert manifest1["clean_row_count"] == 100
    assert manifest1["clean_columns"] == list(df.columns)
    assert len(manifest1["true_means"]) > 0
    for col, mean_val in manifest1["true_means"].items():
        assert abs(mean_val - df[col].mean()) < 1e-6


def test_make_dirty_dataset_injected_corruptions():
    df = pd.DataFrame({
        "id": [f"id_{i}" for i in range(100)],
        "city": ["Berlin" if i % 2 == 0 else "Paris" for i in range(100)],
        "notes": ["ok" for _ in range(100)],
        "status": ["active" for _ in range(100)],
        "val1": [float(i + 1) for i in range(100)],
        "val2": [float((i + 1) * 2) for i in range(100)],
        "val3": [float((i + 1) * 3) for i in range(100)],
        "val4": [float(5.0) for _ in range(100)],
    })

    dirty, manifest = make_dirty_dataset(df, seed=99)

    # Duplicates present: 3% of 100 = 3 rows
    dup_manifest = manifest["duplicate_rows"]
    assert dup_manifest is not None
    assert dup_manifest["count"] == 3
    assert len(dirty) == 100 + 3

    # Numbers stored as text: column dtype becomes object with string entries
    num_text_manifest = manifest["numbers_stored_as_text"]
    assert num_text_manifest is not None
    text_col = num_text_manifest["column"]
    assert dirty[text_col].dtype == object
    assert any(isinstance(val, str) for val in dirty[text_col])

    # Impossible values: negative values injected in a previously non-negative column
    imp_manifest = manifest["impossible_values"]
    assert imp_manifest is not None
    imp_col = imp_manifest["column"]
    assert (dirty[imp_col] < 0).any()

    # Missing values: NaN injected in numeric and None in text
    missing_manifest = manifest["missing_values"]
    assert len(missing_manifest) > 0


# ============================================================================
# 3. Tests for benchmark integrity on data/sample.csv
# ============================================================================

def test_benchmark_integrity():
    csv_path = "data/sample.csv"
    if not os.path.exists(csv_path):
        pytest.skip("data/sample.csv not found")

    assert len(BENCHMARK) == 20

    # IDs are unique integers from 0 to 19
    ids = [item["id"] for item in BENCHMARK]
    assert len(set(ids)) == 20
    assert min(ids) == 0 and max(ids) == 19

    # Categories breakdown
    categories = [item["category"] for item in BENCHMARK]
    assert categories.count("simple aggregation") == 5
    assert categories.count("group-by aggregation") == 5
    assert categories.count("filter then aggregate") == 4
    assert categories.count("percentages or shares") == 4
    assert categories.count("top-N or highest/lowest by group") == 2

    # Permitted wrong variant kinds
    allowed_kinds = {
        "missing_filter",
        "wrong_aggregation",
        "wrong_column",
        "wrong_denominator",
        "wrong_groupby_key",
        "off_by_one_threshold",
        "wrong_top_n_direction",
        "double_counting",
    }

    # Every reference code and wrong variant runs with status 'ok'
    # and every variant is judged INCORRECT by answer_correct (discrimination check)
    ground_truth = compute_ground_truth(BENCHMARK, csv_path)
    assert len(ground_truth) == 20

    for item in BENCHMARK:
        item_id = item["id"]
        ref_data = ground_truth[item_id]
        assert ref_data["status"] == "ok"
        ans = ref_data["answer"]
        assert ans is not None

        if isinstance(ans, pd.DataFrame):
            assert len(ans) > 0, f"Benchmark item {item_id} returned empty DataFrame"
            assert not ans.isna().any().any(), f"Benchmark item {item_id} DataFrame contains NaN values"
        else:
            assert not pd.isna(ans), f"Benchmark item {item_id} returned NaN scalar"

        variants = ref_data.get("wrong_variants", [])
        assert len(variants) >= 1, f"Benchmark item {item_id} must have at least 1 wrong variant"

        for v in variants:
            assert v["kind"] in allowed_kinds, f"Item {item_id} variant '{v['label']}' has invalid kind '{v['kind']}'"
            assert v["status"] == "ok", f"Item {item_id} variant '{v['label']}' did not run with status ok"
            v_ans = v["answer"]
            assert v_ans is not None

            # Discrimination check: must be judged INCORRECT against reference
            is_correct, reason = answer_correct(ans, v_ans)
            assert not is_correct, (
                f"Benchmark item {item_id} variant '{v['label']}' ({v['kind']}) is NON-DISCRIMINATING! "
                f"Candidate answer matches reference answer."
            )


def test_ground_truth_json_contains_variants():
    """Verify that eval/ground_truth.json contains certified answers and wrong variants."""
    gt_path = "eval/ground_truth.json"
    if not os.path.exists(gt_path):
        pytest.skip("eval/ground_truth.json not found")

    loaded_gt = load_ground_truth(gt_path)
    assert len(loaded_gt) == 20

    for item_id, item_data in loaded_gt.items():
        assert item_data["status"] == "ok"
        assert item_data["answer"] is not None
        assert "wrong_variants" in item_data
        assert len(item_data["wrong_variants"]) >= 1
        for v in item_data["wrong_variants"]:
            assert v["status"] == "ok"
            assert "label" in v and v["label"]
            assert "kind" in v and v["kind"]
            assert "answer" in v and v["answer"] is not None


# ============================================================================
# 4. Tests for compute_ground_truth and save/load round trip
# ============================================================================

def test_compute_ground_truth_and_save_load_round_trip(tmp_path: Path):
    fake_csv = tmp_path / "tiny_sample.csv"
    df = pd.DataFrame({
        "city": ["Berlin", "London", "Tokyo"],
        "sales": [100.0, 250.0, 400.0],
    })
    df.to_csv(fake_csv, index=False)

    fake_benchmark = [
        {
            "id": 0,
            "category": "simple aggregation",
            "difficulty": "easy",
            "question": "Total sales?",
            "reference_code": "result = df['sales'].sum()",
            "wrong_variants": [
                {
                    "label": "Average sales instead of sum",
                    "code": "result = df['sales'].mean()",
                    "kind": "wrong_aggregation",
                }
            ],
        },
        {
            "id": 1,
            "category": "group-by aggregation",
            "difficulty": "easy",
            "question": "Sales by city?",
            "reference_code": "result = df.groupby('city')['sales'].sum().reset_index()",
            "wrong_variants": [
                {
                    "label": "Average sales by city",
                    "code": "result = df.groupby('city')['sales'].mean().reset_index()",
                    "kind": "wrong_aggregation",
                }
            ],
        },
    ]

    # 1. Compute ground truth
    gt = compute_ground_truth(fake_benchmark, str(fake_csv))
    assert 0 in gt and 1 in gt
    assert gt[0]["status"] == "ok"
    assert gt[0]["answer"] == 750.0
    assert len(gt[0]["wrong_variants"]) == 1
    assert gt[0]["wrong_variants"][0]["answer"] == 250.0

    assert isinstance(gt[1]["answer"], pd.DataFrame)
    assert len(gt[1]["answer"]) == 3
    assert len(gt[1]["wrong_variants"]) == 1
    assert isinstance(gt[1]["wrong_variants"][0]["answer"], pd.DataFrame)

    # 2. Save ground truth
    gt_file = tmp_path / "ground_truth.json"
    save_ground_truth(gt, str(gt_file))
    assert gt_file.exists()

    # 3. Load ground truth back
    loaded_gt = load_ground_truth(str(gt_file))
    assert 0 in loaded_gt and 1 in loaded_gt
    assert loaded_gt[0]["answer"] == 750.0
    assert loaded_gt[0]["wrong_variants"][0]["answer"] == 250.0
    assert isinstance(loaded_gt[1]["answer"], pd.DataFrame)
    assert len(loaded_gt[1]["answer"]) == 3
    assert list(loaded_gt[1]["answer"].columns) == ["city", "sales"]
    assert isinstance(loaded_gt[1]["wrong_variants"][0]["answer"], pd.DataFrame)

    # 4. Score loaded vs original
    ok0, _ = answer_correct(gt[0]["answer"], loaded_gt[0]["answer"])
    assert ok0 is True
    ok1, _ = answer_correct(gt[1]["answer"], loaded_gt[1]["answer"])
    assert ok1 is True
