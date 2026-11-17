# End-to-End System Comparison: A vs B vs C

**Run Path**: `E:\multi-data-analyst-agent\eval\runs\e2e_full`  
**Cases Finished**: 24 of 24 expected (Unique Items: 12, Repeats: 2)  
**Resources**: 115 LLM calls, 67,365 tokens  
**Models Used**: `openai/gpt-oss-120b` (115 calls, 67,365 tokens)  
System C includes System B's first attempt, so B is not counted twice.  

## 1. System Accuracy Comparison

| System | Description | Unique items | Observations | Correct (mean per repeat) | Accuracy rate (mean over repeats) | 95% Wilson CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **System A** | Single prompt (no retry, no review) | 12 | 24 | 12.0 | 100.0% | [75.7%, 100.0%] |
| **System B** | Analyst agent (self-repair on errors) | 12 | 24 | 12.0 | 100.0% | [75.7%, 100.0%] |
| **System C** | Analyst + Reviewer feedback loop | 12 | 24 | 12.0 | 100.0% | [75.7%, 100.0%] |

Intervals treat the unique benchmark questions (not the repeats) as the sample; repeats average out LLM variability.

## 2. Paired Analysis: System B vs System C

System C starts directly from System B's first-pass code and result, allowing exact paired diagnosis of reviewer impact:

| Paired Outcome | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **Both correct** | 24 | 100.0% | B was correct, reviewer did not harm |
| **FIXED by reviewer** | **0** | **0.0%** | **B was wrong -> reviewer feedback led to correct answer** |
| **HARMED by reviewer** | **0** | **0.0%** | **B was correct -> reviewer falsely failed and analyst corrupted answer** |
| **Both wrong** | 0 | 0.0% | B was wrong and reviewer could not fix it |

*No items were fixed by the reviewer loop in this run.*  

*No items were harmed by the reviewer loop in this run.*  

### Paired Analysis: System A vs System B (Impact of Self-Repair)

| Paired Outcome | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **Both correct** | 24 | 100.0% | Both single-prompt and analyst agent correct |
| **B improved** | 0 | 0.0% | A failed syntax/execution, B self-repaired |
| **B worsened** | 0 | 0.0% | A succeeded, B failed |
| **Both wrong** | 0 | 0.0% | Both systems gave wrong answers |

## 3. Reviewer Behavior on Benchmark Subset

- **First-pass answers flagged**: 2 of 24 (8.3%)
- **True flags** (flagged an actually wrong answer): **0**
- **False alarms** (flagged an actually correct answer): **2**
- **Correct but unresolved items** (final answer right, reviewer verdict 'fail'): **0**
  Such items would be shown as "Unresolved" in the app even though the answer is right.
- **Total reviewer retries used**: 5
- **Average reviewer rounds per item**: 1.21

### False Alarm Diagnoses:
- **Item 11** (r1): Flagged because: `Independent recomputation gave a different result` -> Outcome: **analyst kept correct answer**
- **Item 11** (r2): Flagged because: `Independent recomputation gave a different result` -> Outcome: **analyst kept correct answer**

## 4. Cost and Resource Overhead

| System | LLM Calls | Total Tokens | Execution Time | Overhead vs System B |
| :--- | :---: | :---: | :---: | :---: |
| **System A** (Single prompt) | 24 | 10,506 | 103.9s | 0.95x tokens |
| **System B** (Analyst agent) | 24 | 11,039 | 99.0s | 1.00x (baseline) |
| **System C** (Analyst + Reviewer) | 91 | 56,859 | 338.4s | **5.15x tokens** (3.79x calls, 3.42x time) |

*The reviewer loop added 67 LLM calls and 45,820 tokens beyond the initial analyst generation.*

## 5. Per-Item Results Table

| Item ID | Category | Difficulty | System A | System B | System C | Final reviewer verdict | Correct but unresolved | Reviewer Rounds | Paired (B->C) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `1` (r1) | simple aggregation | easy | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `4` (r1) | simple aggregation | medium | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `9` (r1) | group-by aggregation | hard | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `10` (r1) | filter then aggregate | easy | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `11` (r1) | filter then aggregate | medium | PASS | PASS | PASS | pass | - | 3 | both_correct |
| `12` (r1) | filter then aggregate | medium | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `13` (r1) | filter then aggregate | hard | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `15` (r1) | percentages or shares | medium | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `16` (r1) | percentages or shares | hard | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `17` (r1) | top-N or highest/lowest by group | medium | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `18` (r1) | top-N or highest/lowest by group | medium | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `19` (r1) | percentages or shares | hard | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `1` (r2) | simple aggregation | easy | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `4` (r2) | simple aggregation | medium | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `9` (r2) | group-by aggregation | hard | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `10` (r2) | filter then aggregate | easy | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `11` (r2) | filter then aggregate | medium | PASS | PASS | PASS | pass | - | 4 | both_correct |
| `12` (r2) | filter then aggregate | medium | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `13` (r2) | filter then aggregate | hard | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `15` (r2) | percentages or shares | medium | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `16` (r2) | percentages or shares | hard | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `17` (r2) | top-N or highest/lowest by group | medium | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `18` (r2) | top-N or highest/lowest by group | medium | PASS | PASS | PASS | pass | - | 1 | both_correct |
| `19` (r2) | percentages or shares | hard | PASS | PASS | PASS | unverified | - | 1 | both_correct |

## 6. Correctness Flips Across Repeats

- **Items evaluated in 2 or more repeats**: 12
- **Items with differing correctness across repeats**: 0

*None. All items had identical correctness verdicts across repeats.*  

---
> [!NOTE] Evaluation Limitations
> - **Small Subset**: Evaluated on 12 questions on a single dataset (`data/sample.csv`).
> - **Model Homogeneity**: The same LLM plays both the analyst and reviewer roles; common failure modes and reasoning biases can be correlated.
> - **Component Isolation**: The data cleaner and visualizer are not included in this evaluation.
> - **Stochasticity**: LLM outputs vary between runs; small sample size (n = 12) leads to wide confidence intervals.
> - **Tuning Overlap**: The benchmark questions were previously inspected during development to find and fix two reviewer comparison bugs.
> 
> With n = 12 the paired counts are the main result; differences of one or two questions are not statistically meaningful.