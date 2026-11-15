# Reviewer Agent Evaluation Summary

**Run Path**: `E:\multi-data-analyst-agent\eval\runs\20261115_133310_reviewer`  
**Total Cases**: 60 (Correct: 20, Wrong: 40, Errors: 0)  
**Resources**: 125 LLM calls, 69,434 tokens, 407.9s execution time  

## Overall Metrics

| Metric | Rate | 95% Wilson CI | Counts |
| :--- | :--- | :--- | :--- |
| **Detection Rate** (Sensitivity) | 97.5% | [87.1%, 99.6%] | 39/40 |
| **False Alarm Rate** (1 - Specificity) | 10.0% | [2.8%, 30.1%] | 2/20 |
| **Unverified Rate** | 0.0% | [0.0%, 6.0%] | 0/60 |
| **Precision** | 95.1% | [83.9%, 98.7%] | 39/41 |

## Breakdown by Error Magnitude Bucket

| Bucket | Cases | Detected | Missed | Unverified | Detection Rate (95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `subtle` | 6 | 6 | 0 | 0 | 100.0% [61.0%, 100.0%] |
| `moderate` | 6 | 6 | 0 | 0 | 100.0% [61.0%, 100.0%] |
| `gross` | 22 | 22 | 0 | 0 | 100.0% [85.1%, 100.0%] |
| `structural` | 6 | 5 | 1 | 0 | 83.3% [43.6%, 97.0%] |
| `none` | 20 | 0 | 0 | 0 | 0.0% [0.0%, 0.0%] |

## Breakdown by Error Kind

| Error Kind | Cases | Detected | Missed | Detection Rate |
| :--- | :--- | :--- | :--- | :--- |
| `missing_filter` | 5 | 5 | 0 | 100.0% |
| `off_by_one_threshold` | 1 | 1 | 0 | 100.0% |
| `wrong_aggregation` | 15 | 15 | 0 | 100.0% |
| `wrong_column` | 9 | 9 | 0 | 100.0% |
| `wrong_denominator` | 6 | 6 | 0 | 100.0% |
| `wrong_groupby_key` | 2 | 2 | 0 | 100.0% |
| `wrong_top_n_direction` | 2 | 1 | 1 | 50.0% |

## Detection Mechanism Distribution

| Failed Check Type | Detected Cases Count |
| :--- | :--- |
| `recomputation_mismatch` | 39 |

## Missed Wrong Cases (Verdicts = Pass)

- **Case**: `q18_v1` [wrong_top_n_direction] (structural)
  - **Label**: Selected top 5 highest ratings instead of lowest 5
  - **Question**: *What are the 5 cuisine_category options with the lowest average customer_rating, sorted from lowest to highest rating?*
  - **Recomputed value**: `DataFrame(rows=5, cols=['cuisine_category', 'customer_rating'])`

## False Alarms (Verdicts = Fail on Correct Cases)

- **Case**: `q10_correct`
  - **Question**: *What is the total discount_applied_usd for all orders where is_peak_hour is True?*
  - **Reason**: Independent recomputation gave a different result
  - **Recomputed value**: `DataFrame(rows=1, cols=['index', True])`
- **Case**: `q11_correct`
  - **Question**: *What is the average total_fulfillment_time_min for orders delivered in 'Rainy' or 'Heavy Rain / Storm' weather_condition?*
  - **Reason**: Independent recomputation gave a different result
  - **Recomputed value**: `DataFrame(rows=1, cols=['index', 0])`

---
> [!NOTE] Evaluation Limitations
> Evaluated on 20 standardized questions on `data/sample.csv`. The chart takeaway verification check is not exercised in this isolated reviewer evaluation setup. Results may vary slightly between LLM invocations.