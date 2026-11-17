INVALID: this run had a harness bug (the primary key column was corrupted during error injection). Kept as a record. See cleaner_run2_summary.md for corrected numbers.

# Cleaner Agent Evaluation Report (n = 3 seeds)

- **Seeds**: `42, 43, 44`
- **Dataset**: `data/sample.csv`
- **Generated**: `2026-11-16 03:53:23 UTC`
- **Total LLM Calls**: `3`
- **Total Tokens Consumed**: `19,881`
- **Runtime**: `105.1s`
- **Models Used**: `openai/gpt-oss-120b (3 calls, 19,881 tokens)`

## 1. Metrics Comparison (n = 3 seeds)

| Metric | Baseline Mean | LLM Cleaner Mean | Baseline (Per Seed) | LLM Cleaner (Per Seed) |
|:---|:---:|:---:|:---:|:---:|
| Duplicate Removal Rate | 0.4% | 66.7% | 0.4%, 0.4%, 0.4% | 100.0%, 0.0%, 100.0% |
| Valid Rows Lost | 240.0 | 240.0 | 240.0, 240.0, 240.0 | 240.0, 240.0, 240.0 |
| Duplicate Rows Remaining | 239.0 | 80.0 | 239.0, 239.0, 239.0 | 0.0, 240.0, 0.0 |
| Missing Values Handled Rate | 22.5% | 93.6% | 22.2%, 22.6%, 22.7% | 100.0%, 100.0%, 80.8% |
| Valid Rows Removed (Missing) | 269.3 | 269.3 | 265.0, 270.0, 273.0 | 265.0, 270.0, 273.0 |
| Inconsistent Text Normalized | 96.3% | 96.3% | 96.2%, 96.5%, 96.2% | 96.2%, 96.5%, 96.2% |
| Negative Values Remaining | 40.0 | 39.3 | 40.0, 40.0, 40.0 | 39.0, 40.0, 39.0 |
| Numbers as Text: Numeric Dtype | 100.0% | 100.0% | 100.0%, 100.0%, 100.0% | 100.0%, 100.0%, 100.0% |
| Numbers as Text: True Value Recovery | 95.4% | 63.8% | 93.8%, 98.8%, 93.8% | 62.5%, 66.2%, 62.5% |
| Damage (Unintended Change Rate) | 0.0% | 0.0% | 0.0%, 0.0%, 0.0% | 0.0%, 0.0%, 0.0% |
| Mean Statistical Drift | 0.3% | 1.4% | 0.3%, 0.3%, 0.4% | 1.4%, 1.2%, 1.7% |

## 2. Damage Analysis (Unintended Changes to Uncorrupted Cells)

- **Baseline Unintended Change Rate**: `0.0%`
  - Top changed columns: None (0 unintended changes)
- **LLM Cleaner Unintended Change Rate**: `0.0%`
  - Top changed columns: `metro_city` (1 cells), `cuisine_category` (1 cells), `order_channel` (1 cells), `order_value_usd` (1 cells), `delivery_distance_km` (1 cells)

## 3. Error Types Not Handled

### Baseline (Deterministic Script)
- Missing values: Baseline performs no imputation or null row removal.
- Negative values: Baseline performs no range filtering or clipping.

### LLM Cleaner Agent
- Negative values: Cleaner has no action for domain range constraints or absolute value conversion.
- Numbers stored as text: Cleaner's convert_dtype coerces formatted strings like '$12.50' to NaN rather than extracting numeric figures.

## 4. Execution & Plan Adherence

- **Seed 42**: LLM Plan | Steps applied: 25, skipped: 0 | Tokens: 6,666 | Time: 9.8s
- **Seed 43**: LLM Plan | Steps applied: 24, skipped: 0 | Tokens: 6,466 | Time: 15.1s
- **Seed 44**: LLM Plan | Steps applied: 23, skipped: 0 | Tokens: 6,749 | Time: 43.0s

## 5. Limitations & Caveats

1. **Single Dataset**: Evaluation conducted solely on `data/sample.csv` (delivery orders).
2. **Synthetic Perturbation**: Errors injected by programmatic rules; distributions may differ from organic messy data.
3. **Sample Size**: Evaluated across n=3 seeds (42, 43, 44); variance exists across different random seeds.
4. **Probabilistic Reasoning**: LLM plans can vary between identical prompts and runs.
5. **Alignment Assumption**: Accuracy and damage metrics require a unique key column (`order_id`) to track row identities.
