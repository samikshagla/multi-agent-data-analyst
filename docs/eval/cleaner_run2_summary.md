# Cleaner Agent Evaluation Report (n = 3 seeds)

- **Seeds**: `42, 43, 44`
- **Dataset**: `data/sample.csv`
- **Generated**: `2026-11-17 03:51:32 UTC`
- **Total LLM Calls**: `3`
- **Total Tokens Consumed**: `21,784`
- **Runtime**: `115.6s`
- **Models Used**: `openai/gpt-oss-120b (3 calls, 21,784 tokens)`

## 1. Metrics Comparison (n = 3 seeds)

| Metric | Baseline Mean | LLM Cleaner Mean | Baseline (Per Seed) | LLM Cleaner (Per Seed) |
|:---|:---:|:---:|:---:|:---:|
| Duplicate Removal Rate | 100.0% | 100.0% | 100.0%, 100.0%, 100.0% | 100.0%, 100.0%, 100.0% |
| Valid Rows Lost | 0.0 | 0.0 | 0.0, 0.0, 0.0 | 0.0, 0.0, 0.0 |
| Duplicate Rows Remaining | 0.0 | 0.0 | 0.0, 0.0, 0.0 | 0.0, 0.0, 0.0 |
| Key Nulls in Cleaned | 0.0 | 0.0 | 0.0, 0.0, 0.0 | 0.0, 0.0, 0.0 |
| Key Duplicates in Cleaned | 0.0 | 0.0 | 0.0, 0.0, 0.0 | 0.0, 0.0, 0.0 |
| Missing Values Handled Rate | 0.1% | 60.0% | 0.1%, 0.1%, 0.0% | 60.0%, 60.0%, 60.0% |
| Valid Rows Removed (Missing) | 0.0 | 0.0 | 0.0, 0.0, 0.0 | 0.0, 0.0, 0.0 |
| Inconsistent Text Normalized | 100.0% | 100.0% | 100.0%, 100.0%, 100.0% | 100.0%, 100.0%, 100.0% |
| Negative Values Remaining | 40.0 | 40.0 | 40.0, 40.0, 40.0 | 40.0, 40.0, 40.0 |
| Numbers as Text: Numeric Dtype | 100.0% | 100.0% | 100.0%, 100.0%, 100.0% | 100.0%, 100.0%, 100.0% |
| Numbers as Text: True Value Recovery | 99.2% | 66.2% | 98.8%, 98.8%, 100.0% | 66.2%, 66.2%, 66.2% |
| Damage (Unintended Change Rate) | 0.0% | 0.0% | 0.0%, 0.0%, 0.0% | 0.0%, 0.0%, 0.0% |
| Mean Statistical Drift | 0.3% | 1.5% | 0.3%, 0.3%, 0.4% | 1.3%, 1.6%, 1.6% |

## 2. Missing Value Handling Breakdown

| Column | Injected | Baseline: Filled | Baseline: Removed | Baseline: Still Missing | Cleaner: Filled | Cleaner: Removed | Cleaner: Still Missing |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `cuisine_category` | 720 | 0 (0.0%) | 0 (0.0%) | 720 (100.0%) | 0 (0.0%) | 0 (0.0%) | 720 (100.0%) |
| `delivery_distance_km` | 720 | 0 (0.0%) | 0 (0.0%) | 720 (100.0%) | 720 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| `discount_applied_usd` | 720 | 0 (0.0%) | 0 (0.0%) | 720 (100.0%) | 720 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| `metro_city` | 720 | 0 (0.0%) | 0 (0.0%) | 720 (100.0%) | 0 (0.0%) | 0 (0.0%) | 720 (100.0%) |
| `order_value_usd` | 720 | 2 (0.3%) | 0 (0.0%) | 718 (99.7%) | 720 (100.0%) | 0 (0.0%) | 0 (0.0%) |

## 3. Damage Analysis (Unintended Changes to Uncorrupted Cells)

- **Baseline Unintended Change Rate**: `0.0%`
  - Top changed columns: None (0 unintended changes)
- **LLM Cleaner Unintended Change Rate**: `0.0%`
  - Top changed columns: None (0 unintended changes)

## 4. Error Types Not Handled

### Baseline (Deterministic Script)
- Missing values: Baseline performs no imputation or null row removal.
- Negative values: Baseline performs no range filtering or clipping.

### LLM Cleaner Agent
- Negative values: Cleaner has no action for domain range constraints or absolute value conversion.
- Numbers stored as text: Cleaner's convert_dtype coerces formatted strings like '$12.50' to NaN rather than extracting numeric figures.

## 5. Execution & Plan Adherence

- **Seed 42**: LLM Plan | Steps applied: 22, skipped: 1 | Tokens: 7,636 | Time: 9.5s
- **Seed 43**: LLM Plan | Steps applied: 22, skipped: 0 | Tokens: 7,414 | Time: 30.2s
- **Seed 44**: LLM Plan | Steps applied: 14, skipped: 0 | Tokens: 6,734 | Time: 56.9s

## 6. Limitations & Caveats

1. **Single Dataset**: Evaluation conducted solely on `data/sample.csv` (delivery orders).
2. **Synthetic Perturbation**: Errors injected by programmatic rules; distributions may differ from organic messy data.
3. **Sample Size**: Evaluated across n=3 seeds (42, 43, 44); variance exists across different random seeds.
4. **Probabilistic Reasoning**: LLM plans can vary between identical prompts and runs.
5. **Alignment Assumption**: Accuracy and damage metrics require a unique key column (`order_id`) to track row identities.
