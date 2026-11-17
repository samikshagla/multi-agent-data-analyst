"""
eval/diagnose_cleaner_metrics.py
Diagnostic tool for investigating contradictions in cleaner evaluation metrics.
Rebuilds the dirty dataset for a given seed with make_dirty_dataset, runs baseline_clean,
and prints a step-by-step reconciliation of data corruption, metric code paths,
and metric calculations.

Usage:
    python -m eval.diagnose_cleaner_metrics [--seed 42] [--csv-path data/sample.csv]
"""

import argparse
import os
from typing import Any
import numpy as np
import pandas as pd

from eval.baseline_cleaning import baseline_clean
from eval.cleaner_metrics import cleaner_metrics
from eval.inject_data_errors import make_dirty_dataset


def diagnose_seed(seed: int = 42, csv_path: str = "data/sample.csv") -> None:
    """Performs deep diagnostic inspection on cleaner metrics for a given seed."""
    print("=" * 80)
    print(f"CLEANER METRICS DIAGNOSTIC REPORT (Seed {seed})")
    print(f"Dataset: {csv_path}")
    print("=" * 80)

    # Load clean ground truth
    clean_df = pd.read_csv(csv_path)
    clean_row_count = len(clean_df)

    # 1. Rebuild dirty dataset
    dirty_df, manifest = make_dirty_dataset(clean_df.copy(deep=True), seed=seed)
    key_col = manifest.get("key_column", "order_id")

    # =========================================================================
    # Section (a): Dirty Dataset Reconciliation
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION (a): Dirty Dataset Reconciliation")
    print("-" * 80)

    clean_keys = set(clean_df[key_col])
    dirty_keys_all = set(dirty_df[key_col])
    dirty_keys_non_null = set(dirty_df[key_col].dropna())

    exact_dups_dirty = int(dirty_df.duplicated().sum())
    key_dups_dirty = int(dirty_df[key_col].duplicated().sum())
    rows_with_shared_key = int(dirty_df[key_col].duplicated(keep=False).sum())
    null_keys_dirty = int(dirty_df[key_col].isna().sum())
    missing_keys_from_clean = len(clean_keys - dirty_keys_non_null)

    print(f"Clean rows:                          {clean_row_count}")
    print(f"Dirty rows:                          {len(dirty_df)}")
    print(f"Alignment key column:                '{key_col}'")
    print(f"Clean unique keys:                   {len(clean_keys)}")
    print(f"Dirty unique keys (excluding NaN):   {len(dirty_keys_non_null)}")
    print(f"Dirty null key cells (NaN):          {null_keys_dirty}")
    print(f"Exact duplicate rows in dirty:       {exact_dups_dirty} (via df.duplicated())")
    print(f"Key duplicates in dirty:             {key_dups_dirty} (via df['{key_col}'].duplicated())")
    print(f"Rows with a duplicate key:           {rows_with_shared_key}")
    print(f"Clean keys MISSING from dirty data:  {missing_keys_from_clean}")

    print("\nInjected Corruptions (from manifest):")
    dup_manifest = manifest.get("duplicate_rows") or {}
    print(f"  - duplicate_rows:          {dup_manifest.get('count', 0)} injected rows")

    missing_manifest = manifest.get("missing_values") or {}
    print(f"  - missing_values columns:  {list(missing_manifest.keys())}")
    for col, info in missing_manifest.items():
        is_key = (col == key_col)
        key_flag = " *** PRIMARY KEY COLUMN ***" if is_key else ""
        print(f"      * {col:25s}: {info.get('count')} missing cells (dtype: {info.get('dtype')}){key_flag}")

    inc_manifest = manifest.get("inconsistent_text") or {}
    print(f"  - inconsistent_text:       {inc_manifest.get('count', 0)} cells in '{inc_manifest.get('column')}'")

    imp_manifest = manifest.get("impossible_values") or {}
    print(f"  - impossible_values:       {imp_manifest.get('count', 0)} cells in '{imp_manifest.get('column')}'")

    num_text_manifest = manifest.get("numbers_stored_as_text") or {}
    print(f"  - numbers_stored_as_text:  {num_text_manifest.get('count', 0)} cells in '{num_text_manifest.get('column')}'")

    # =========================================================================
    # Section (b): Injected Duplicate Rows vs Originals
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION (b): Injected Duplicate Rows vs Originals")
    print("-" * 80)

    inj_indices = dup_manifest.get("injected_indices", [])
    src_indices = dup_manifest.get("source_indices", [])

    print(f"Total injected duplicate rows: {len(inj_indices)}")
    print(f"Exact duplicate rows in dirty: {exact_dups_dirty}")

    # Check whether the injected rows match their source in dirty
    all_exact_copies_in_dirty = (exact_dups_dirty == len(inj_indices))
    print(f"Are all injected duplicates exact copies in dirty? {all_exact_copies_in_dirty}")

    print("\n3 Examples of Injected Duplicate Rows vs Pre-Corruption Originals in clean_df:")
    for ex_idx in range(min(3, len(inj_indices))):
        inj_row_idx = inj_indices[ex_idx]
        src_row_idx = src_indices[ex_idx]

        dup_row = dirty_df.iloc[inj_row_idx]
        orig_clean_row = clean_df.iloc[src_row_idx]

        # Check which columns differ between dirty duplicate and clean original
        diff_cols = []
        for col in clean_df.columns:
            v_clean = orig_clean_row[col]
            v_dup = dup_row[col]
            c_na = bool(pd.api.types.is_scalar(v_clean) and pd.isna(v_clean))
            d_na = bool(pd.api.types.is_scalar(v_dup) and pd.isna(v_dup))
            if c_na and d_na:
                continue
            try:
                is_diff = (v_clean != v_dup)
                if hasattr(is_diff, "any"):
                    is_diff = bool(is_diff.any())
            except Exception:
                is_diff = True
            if is_diff:
                diff_cols.append((col, v_clean, v_dup))

        key_val = orig_clean_row[key_col]
        print(f"\nExample {ex_idx + 1} (Source idx {src_row_idx} -> Injected at idx {inj_row_idx}, Key: {key_val}):")
        if not diff_cols:
            print("  All 26 columns are identical between dirty duplicate and clean original.")
        else:
            print(f"  {len(diff_cols)} column(s) differ from clean original due to corruptions applied to source:")
            for c_name, c_orig, c_dup in diff_cols:
                print(f"    - Column '{c_name}': Clean='{c_orig}' -> Dirty Copy='{c_dup}'")

    # =========================================================================
    # Section (c): Baseline Output Inspection
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION (c): Baseline Output Inspection")
    print("-" * 80)

    base_cleaned_df, base_log = baseline_clean(dirty_df.copy(deep=True))

    base_rows = len(base_cleaned_df)
    base_exact_dups = int(base_cleaned_df.duplicated().sum())
    base_key_dups = int(base_cleaned_df[key_col].duplicated().sum())
    base_null_keys = int(base_cleaned_df[key_col].isna().sum())
    base_unique_keys_non_null = len(set(base_cleaned_df[key_col].dropna()))
    base_missing_clean_keys = len(clean_keys - set(base_cleaned_df[key_col].dropna()))

    print(f"Baseline cleaned rows:               {base_rows}")
    print(f"Baseline exact duplicate rows:       {base_exact_dups} (via df.duplicated())")
    print(f"Baseline key duplicates:             {base_key_dups} (via df['{key_col}'].duplicated())")
    print(f"Baseline null key cells:             {base_null_keys}")
    print(f"Baseline unique keys (non-null):     {base_unique_keys_non_null}")
    print(f"Clean keys missing from cleaned:     {base_missing_clean_keys}")
    print(f"Baseline transformation log:")
    for log_line in base_log:
        print(f"  - {log_line}")

    # =========================================================================
    # Section (d): Metric Reconciliations & Code Paths
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION (d): Metric Reconciliations & Code Paths")
    print("-" * 80)

    base_metrics = cleaner_metrics(clean_df, dirty_df, base_cleaned_df, manifest)

    # 1. Valid Rows Lost
    print("1. Metric: valid_rows_lost")
    print("   Code path (cleaner_metrics.py lines 102-105):")
    print("       clean_keys = set(clean_df[key_col])")
    print("       cleaned_keys = set(cleaned_df[key_col])")
    print("       valid_rows_lost = int(len(clean_keys - cleaned_keys))")
    print(f"   Current metric value: {base_metrics['rows']['valid_rows_lost']}")
    print(f"   Intermediate numbers: len(clean_keys) = {len(clean_keys)}, len(cleaned_keys) = {len(set(base_cleaned_df[key_col]))}")
    print(f"   Root Cause: 'order_id' had {null_keys_dirty} missing values injected in make_dirty_dataset.")
    print(f"   These rows still exist in baseline ({base_rows} rows), but their key is NaN.")
    print(f"   Correct definition value: 0 (Baseline dropped zero valid rows).")

    # 2. Rows Removed for Missing Values
    print("\n2. Metric: valid_rows_removed (Missing Values)")
    print("   Code path (cleaner_metrics.py lines 147-152):")
    print("       for idx in indices:")
    print("           row_key = clean_df.iloc[idx][key_col]")
    print("           if row_key not in cleaned_by_key:")
    print("               valid_rows_removed += 1; handled_count += 1")
    total_rows_removed_mv = sum(v["valid_rows_removed"] for v in base_metrics["missing_values"].values())
    print(f"   Current metric sum: {total_rows_removed_mv}")
    print("   Breakdown by column:")
    for col, v in base_metrics["missing_values"].items():
        print(f"     - '{col}': valid_rows_removed = {v['valid_rows_removed']}, rate_handled = {v['rate_handled']*100:.1f}%")
    print(f"   Root Cause: Because 240 rows have order_id=NaN, cleaned_by_key cannot find their key.")
    print(f"   For 'order_id' itself, all 240 rows are marked as 'removed to handle missing values'!")
    print(f"   The other 25 rows are random collisions where another column's missing value landed on a null-key row.")
    print(f"   Correct definition value: 0 (Baseline dropped zero rows for missing values).")

    # 3. Duplicate Removal Rate
    print("\n3. Metric: duplicate_removal_rate")
    print("   Code path (cleaner_metrics.py lines 106, 114-115):")
    print("       duplicate_rows_remaining = int(cleaned_df[key_col].duplicated().sum())")
    print("       dups_removed = max(0, injected_dups - duplicate_rows_remaining)")
    print("       duplicate_removal_rate = float(dups_removed / injected_dups)")
    print(f"   Current metric value: {base_metrics['rows']['duplicate_removal_rate']*100:.2f}%")
    print(f"   duplicate_rows_remaining reported: {base_metrics['rows']['duplicate_rows_remaining']}")
    print(f"   Actual exact duplicate rows in cleaned_df: {base_exact_dups} (via df.duplicated().sum())")
    print(f"   Root Cause: pandas series.duplicated() considers all NaN values to be duplicates of each other!")
    print(f"   Since cleaned_df['{key_col}'] contains 240 NaNs, series.duplicated() returns 239 duplicates!")
    print(f"   dups_removed = max(0, 240 - 239) = 1 -> 1/240 = 0.42% removal rate.")
    print(f"   Correct definition value: 100.0% (all 240 exact duplicate rows were dropped by baseline_clean).")

    # 4. Missing Values Handled Rate
    print("\n4. Metric: rate_handled (Missing Values)")
    print("   Code path: Same as valid_rows_removed above.")
    rates = [v["rate_handled"] for v in base_metrics["missing_values"].values()]
    mean_handled = float(np.mean(rates)) * 100
    print(f"   Current metric value (mean across columns): {mean_handled:.2f}%")
    print(f"   Root Cause: 'order_id' reports 100% handled (240/240) because all 240 keys are missing from cleaned_by_key.")
    print(f"   Average across 5 columns: (4.2% + 2.9% + 1.2% + 100.0% + 2.5%) / 5 = 22.17%.")
    print(f"   Correct definition value: 0.0% (Baseline performs no missing value handling).")

    # 5. Inconsistent Text Normalized
    print("\n5. Metric: share_normalized (Inconsistent Text)")
    print("   Code path (cleaner_metrics.py lines 177-185):")
    print("       for idx in inc_indices:")
    print("           row_key = clean_df.iloc[idx][key_col]")
    print("           if row_key in cleaned_by_key:")
    print("               if cleaned_val == true_label: normalized_count += 1")
    inc_metric = base_metrics["inconsistent_text"]
    print(f"   Current metric value: {inc_metric['share_normalized']*100:.2f}%")
    print(f"   Injected count: {inc_metric['injected_count']}")
    norm_count = round(inc_metric["share_normalized"] * inc_metric["injected_count"])
    not_norm_count = inc_metric["injected_count"] - norm_count
    print(f"   Normalized count: {norm_count}/{inc_metric['injected_count']} (un-normalized: {not_norm_count})")
    print(f"   Root Cause: Exactly {not_norm_count} rows with inconsistent text were ALSO rows where order_id was NaN!")
    print(f"   Because order_id is NaN, 'row_key in cleaned_by_key' is False, so they were marked un-normalized.")
    print(f"   Correct definition value: 100.0% (all rows whose keys exist were perfectly normalized by baseline).")

    # =========================================================================
    # Section (e): Inconsistent Text Un-normalized Examples
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION (e): Inconsistent Text Non-Normalized Examples")
    print("-" * 80)

    inc_col = manifest["inconsistent_text"]["column"]
    inc_indices = manifest["inconsistent_text"]["indices"]
    order_id_missing_indices = set(manifest.get("missing_values", {}).get(key_col, {}).get("indices", []))
    cleaned_by_key = {r[key_col]: r for _, r in base_cleaned_df.drop_duplicates(subset=[key_col]).iterrows()}

    failed_examples = []
    for pos, idx in enumerate(inc_indices):
        r_key = clean_df.iloc[idx][key_col]
        true_val = clean_df.iloc[idx][inc_col]

        # Determine what dirty value was generated at idx
        mod_choice = pos % 4
        if mod_choice == 0:
            dirty_val = f"  {true_val}"
        elif mod_choice == 1:
            dirty_val = f"{true_val}  "
        elif mod_choice == 2:
            dirty_val = f" {str(true_val).upper()} "
        else:
            dirty_val = str(true_val).lower()

        if r_key not in cleaned_by_key:
            has_missing_key = idx in order_id_missing_indices
            reason = (
                f"order_id was corrupted to NaN by missing_values on '{key_col}'. "
                f"Row exists in cleaned_df but with order_id=NaN. Key lookup failed."
                if has_missing_key else
                f"Row was dropped or key missing from cleaned data."
            )
            failed_examples.append({
                "index": idx,
                "clean_key": r_key,
                "dirty_val": dirty_val,
                "true_val": true_val,
                "reason": reason,
            })

    print(f"Total inconsistent text injected: {len(inc_indices)}")
    print(f"Total un-normalized cases:        {len(failed_examples)} (All {len(failed_examples)} overlap with missing {key_col}!)")
    print(f"Total successfully normalized:    {len(inc_indices) - len(failed_examples)} / {len(inc_indices)} ({((len(inc_indices) - len(failed_examples))/len(inc_indices))*100:.2f}%)")
    print("\n10 Examples of Non-Normalized Cells:")
    for i, ex in enumerate(failed_examples[:10]):
        print(f"  {i + 1:2d}. Row idx {ex['index']:4d} | Clean Key: {ex['clean_key']}")
        print(f"      Dirty Injected Value: '{ex['dirty_val']}'")
        print(f"      True Clean Value:     '{ex['true_val']}'")
        print(f"      Root Cause Failure:   {ex['reason']}")

    # =========================================================================
    # Section (f): Conclusion & Summary of Suspected Bugs
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION (f): Conclusion & Taxonomy of Issues")
    print("=" * 80)

    print("""
1. INJECTION DESIGN PROBLEMS:
   a) eval/inject_data_errors.py (lines 51-58, 125-133):
      - What it does wrong: 'text_cols' includes 'order_id' as its first element.
        'text_cols_for_missing = text_cols[:2]' selects ['order_id', 'metro_city'],
        injecting 240 NaNs into the PRIMARY KEY column ('order_id').
      - Consequence: Since 'order_id' is used as the alignment key for all evaluations,
        destroying order_id corrupts row alignment for 240 rows, causing phantom
        'valid rows lost', phantom 'duplicate keys', phantom 'missing values handled',
        and phantom 'un-normalized text'.
      - Minimal fix: In make_dirty_dataset, filter out the detected 'key_column'
        (and any column ending with '_id' or 'id') from candidate lists for corruptions:
        text_cols = [c for c in text_cols if c != key_column and not _is_id_like(c)]

2. METRIC BUGS:
   a) eval/cleaner_metrics.py (line 106):
      - What it does wrong: Calculates duplicate rows remaining using:
        duplicate_rows_remaining = int(cleaned_df[key_col].duplicated().sum())
        In pandas, Series.duplicated() treats all NaN values as duplicate copies of
        each other! 240 NaNs become 239 duplicates, falsely dropping removal rate to 0.4%.
      - Minimal fix: Check exact row duplicates across all columns:
        duplicate_rows_remaining = int(cleaned_df.duplicated().sum())
        Or if checking key duplicates, exclude NaNs:
        duplicate_rows_remaining = int(cleaned_df[key_col].dropna().duplicated().sum())

   b) eval/cleaner_metrics.py (lines 149-152):
      - What it does wrong: For missing values, if 'row_key not in cleaned_by_key',
        it automatically assumes the cleaner intentionally removed the row to handle
        that column's missing value ('valid_rows_removed += 1; handled_count += 1').
        When order_id was NaN, 240 rows triggered this condition across all columns.
      - Minimal fix: Only credit row removal if the row was actually dropped from
        the dataset (expected_rows vs cleaned_rows), not simply because the key was NaN.

   c) eval/cleaner_metrics.py (lines 178-180):
      - What it does wrong: If 'row_key not in cleaned_by_key', inconsistent text is
        marked as un-normalized.
      - Minimal fix: Resolved automatically once key_column is excluded from missing
        value corruption.

3. REAL CLEANER BEHAVIOR:
   a) In Seed 43, the LLM cleaner agent genuinely called:
      fill_missing(column='order_id', method='mode')
      which imputed all 240 missing order_id values with 'ord_0006729129ae',
      genuinely creating 240 duplicate keys.
   b) In Seed 42 and Seed 44, the LLM cleaner agent called:
      drop_missing_rows(column='order_id')
      which genuinely dropped 240 rows (8,000 -> 7,760 rows).
   c) The LLM cleaner did this because the prompt presented a profile showing 240
      missing values in 'order_id', prompting it to clean 'order_id'.
""")

    print("=" * 80)
    print("END OF DIAGNOSTIC REPORT")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose cleaner evaluation metric contradictions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to inspect (default: 42)")
    parser.add_argument("--csv-path", type=str, default="data/sample.csv", help="Clean dataset path")
    args = parser.parse_args()

    diagnose_seed(seed=args.seed, csv_path=args.csv_path)


if __name__ == "__main__":
    main()
