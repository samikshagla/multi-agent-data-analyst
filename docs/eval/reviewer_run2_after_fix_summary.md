# Reviewer Agent Evaluation Summary

**Run Path**: `E:\multi-data-analyst-agent\eval\runs\20261116_092817_reviewer`  
**Total Cases Finished**: 120 (Unique: 60, Correct: 40, Wrong: 80, Errors: 0)  
**Resources**: 249 LLM calls, 138,099 tokens, 712.0s execution time  
**Models Used**: `openai/gpt-oss-120b` (249 calls, 138,099 tokens)  

## Overall Metrics

| Metric | Unique cases | Observations | Detected (mean per repeat) | Detection rate (mean over repeats) | 95% Wilson CI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Detection Rate** (Sensitivity) | 40 | 80 | 39.5 | 98.8% | [89.1%, 99.9%] |
| **Detection for the right reason** | 40 | 80 | 37.0 | 92.5% | [80.1%, 97.4%] |
| **False Alarm Rate** (1 - Specificity) | 20 | 40 | 0.5 | 2.5% | [0.3%, 20.0%] |
| **Unverified Rate** | 60 | 120 | 0.0 | 0.0% | [0.0%, 6.0%] |
| **Precision** | 40.0 | 80 | 39.5 | 98.8% | [89.1%, 99.9%] |

Intervals treat the benchmark cases (not the repeats) as the sample; repeats only average out LLM randomness.  
'By luck' means the reviewer flagged an answer while its own recomputation was also wrong.

## Per-Repeat Performance

| Repeat | Finished Cases | Detection Rate (95% CI) | Detection Counts | False Alarm Rate (95% CI) | False Alarm Counts |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **r1** | 60 | 100.0% [91.2%, 100.0%] | 40/40 | 0.0% [0.0%, 16.1%] | 0/20 |
| **r2** | 60 | 97.5% [87.1%, 99.6%] | 39/40 | 5.0% [0.9%, 23.6%] | 1/20 |

## Correct Cases Performance

| Cases | Passed | False Alarms | Unverified |
| :--- | :--- | :--- | :--- |
| 40 | 39 | 1 | 0 |

## Was the Reviewer's Own Recomputation Correct?

| Cohort | Total Cases | Recompute Matched Reference | Rate |
| :--- | :--- | :--- | :--- |
| **Correct Cases** | 40 | 39 | 97.5% |
| **Wrong Cases** | 80 | 74 | 92.5% |

### Detected Wrong Cases Breakdown
Among the **79** detected wrong cases:
- **Detected for the right reason** (recompute matched reference): **74** (93.7%)
- **Detected by luck** (recompute was also wrong): **5** (6.3%)

## Verdict Flips Across Repeats

- **Cases evaluated in 2 or more repeats**: 60
- **Cases with identical verdicts across all repeats**: 58
- **Cases with differing verdicts (flips)**: 2

| Case ID | Label | Kind | **r1** | **r2** |
| :--- | :--- | :--- | :---: | :---: |
| `q04_v2` | Used net_effective_order_price as denominator instead of order_value_usd | `wrong_denominator` | fail | pass |
| `q11_correct` | Correct reference code | `correct` | pass | fail |

## Breakdown by Error Magnitude Bucket

| Bucket | Unique cases | Observations | Detected (mean per repeat) | Detection rate (mean over repeats) | 95% Wilson CI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `subtle` | 6 | 12 | 5.5 | 91.7% | [51.7%, 99.1%] |
| `moderate` | 6 | 12 | 6.0 | 100.0% | [61.0%, 100.0%] |
| `gross` | 22 | 44 | 22.0 | 100.0% | [85.1%, 100.0%] |
| `structural` | 6 | 12 | 6.0 | 100.0% | [61.0%, 100.0%] |

- `subtle`: small sample (n=6), interpret with care
- `moderate`: small sample (n=6), interpret with care
- `structural`: small sample (n=6), interpret with care

## Breakdown by Error Kind

| Error Kind | Unique cases | Observations | Detected (mean per repeat) | Detection rate (mean over repeats) | 95% Wilson CI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `correct` | 20 | 40 | 0.5 | 2.5% | [0.3%, 20.0%] |
| `missing_filter` | 5 | 10 | 5.0 | 100.0% | [56.6%, 100.0%] |
| `off_by_one_threshold` | 1 | 2 | 1.0 | 100.0% | [20.7%, 100.0%] |
| `wrong_aggregation` | 15 | 30 | 15.0 | 100.0% | [79.6%, 100.0%] |
| `wrong_column` | 9 | 18 | 9.0 | 100.0% | [70.1%, 100.0%] |
| `wrong_denominator` | 6 | 12 | 5.5 | 91.7% | [51.7%, 99.1%] |
| `wrong_groupby_key` | 2 | 4 | 2.0 | 100.0% | [34.2%, 100.0%] |
| `wrong_top_n_direction` | 2 | 4 | 2.0 | 100.0% | [34.2%, 100.0%] |

## Breakdown by Category

| Category | Unique cases | Observations | Detected (mean per repeat) | Detection rate (mean over repeats) | 95% Wilson CI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `filter then aggregate` | 8 | 16 | 8.0 | 100.0% | [67.6%, 100.0%] |
| `group-by aggregation` | 10 | 20 | 10.0 | 100.0% | [72.2%, 100.0%] |
| `percentages or shares` | 8 | 16 | 8.0 | 100.0% | [67.6%, 100.0%] |
| `simple aggregation` | 10 | 20 | 9.5 | 95.0% | [65.5%, 99.5%] |
| `top-N or highest/lowest by group` | 4 | 8 | 4.0 | 100.0% | [51.0%, 100.0%] |

## Breakdown by Difficulty

| Difficulty | Unique cases | Observations | Detected (mean per repeat) | Detection rate (mean over repeats) | 95% Wilson CI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `easy` | 12 | 24 | 12.0 | 100.0% | [75.7%, 100.0%] |
| `hard` | 8 | 16 | 8.0 | 100.0% | [67.6%, 100.0%] |
| `medium` | 20 | 40 | 19.5 | 97.5% | [80.0%, 99.7%] |

## Detection Mechanism Distribution

| Failed Check Type | Detected Cases Count |
| :--- | :--- |
| `recomputation_mismatch` | 79 |

## Missed Wrong Cases (Verdicts = Pass)

- **Case**: `q04_v2_r2` [wrong_denominator] (subtle)
  - **Label**: Used net_effective_order_price as denominator instead of order_value_usd
  - **Question**: *What is the overall ratio of total delivery_fee_usd to total order_value_usd across the entire dataset?*
  - **Analyst Result**: `0.0992`
  - **Reviewer Recomputed**: `0.0992`
  - **Reference Answer**: `0.0980`
  - **Reason**: All checks passed successfully.

## False Alarms (Verdicts = Fail on Correct Cases)

- **Case**: `q11_correct_r2`
  - **Question**: *What is the average total_fulfillment_time_min for orders delivered in 'Rainy' or 'Heavy Rain / Storm' weather_condition?*
  - **Reason**: Independent recomputation gave a different result
  - **Failed Check**: `recomputation_mismatch`
  - **Analyst Result**: `31.2609`
  - **Reviewer Recomputed**: 
| index | 0 |
| --- | --- |
| total_fulfillment_time_min | 32.03262667338511 |
  - **Reference Answer**: `31.2609`

---
> [!NOTE] Evaluation Limitations
> Evaluated on 20 standardized questions on `data/sample.csv`. The chart takeaway verification check is not exercised in this isolated reviewer evaluation setup. Results may vary slightly between LLM invocations.