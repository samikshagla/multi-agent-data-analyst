"""
verify_benchmark.py
Benchmark Verification Script: Validates and executes all 20 benchmark questions on
data/sample.csv along with all wrong variants. Evaluates discrimination using
eval/scoring.py answer_correct, prints detailed findings, and outputs a complete summary.

Usage:
    python -m eval.verify_benchmark
"""

import sys
from collections import Counter
import pandas as pd

from eval.benchmark import BENCHMARK
from eval.ground_truth import compute_ground_truth
from eval.scoring import answer_correct


def _format_answer_preview(ans: any, max_rows: int = 5, indent: str = "  ") -> str:
    """Formats preview of DataFrame or scalar answer."""
    if isinstance(ans, pd.DataFrame):
        n_rows = len(ans)
        header = f"DataFrame (rows={n_rows}, cols={ans.shape[1]})"
        if n_rows > max_rows:
            header += f" [showing first {max_rows} rows]:"
        else:
            header += ":"
        preview = ans.head(max_rows).to_string(index=False)
        lines = [header] + [f"{indent}{line}" for line in preview.splitlines()]
        return "\n".join(lines)
    else:
        return str(ans)


def main(csv_path: str = "data/sample.csv") -> None:
    print("=" * 80)
    print("MULTI-AGENT DATA ANALYST BENCHMARK & DISCRIMINATION VERIFICATION")
    print(f"Dataset: {csv_path}")
    print(f"Total benchmark questions: {len(BENCHMARK)}")
    print("=" * 80)

    try:
        ground_truth = compute_ground_truth(BENCHMARK, csv_path)
    except Exception as e:
        print(f"\n[FATAL ERROR] Failed to compute ground truth: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nExecuting benchmark items & wrong variants:\n")

    total_variants = 0
    non_discriminating_variants: list[tuple[int, str, str]] = []
    items_without_variants: list[int] = []

    for item in BENCHMARK:
        item_id = item["id"]
        cat = item["category"]
        diff = item["difficulty"]
        q = item["question"]
        code = item["reference_code"]
        ref_ans = ground_truth[item_id]["answer"]
        variants = ground_truth[item_id].get("wrong_variants", [])

        print(f"--- [Item {item_id:02d}] ({diff.upper()}) [{cat}] ---")
        print(f"Question:       {q}")
        print(f"Reference Code: {code}")
        print(f"Reference Answer:")
        for line in _format_answer_preview(ref_ans, max_rows=5, indent="  ").splitlines():
            print(f"  {line}")

        if not variants:
            items_without_variants.append(item_id)
            print("  (No wrong variants defined)")
        else:
            print(f"  Wrong Variants ({len(variants)}):")
            for v_idx, v in enumerate(variants, start=1):
                total_variants += 1
                v_label = v["label"]
                v_code = v["code"]
                v_kind = v["kind"]
                v_ans = v["answer"]

                is_correct, reason = answer_correct(ref_ans, v_ans)
                if is_correct:
                    verdict = "[*** NON-DISCRIMINATING ***]"
                    non_discriminating_variants.append((item_id, v_label, v_kind))
                else:
                    verdict = "[DISCRIMINATING]"

                print(f"    ({v_idx}) {verdict} Kind: {v_kind}")
                print(f"        Label:  {v_label}")
                print(f"        Code:   {v_code}")
                if is_correct:
                    print(f"        Result: Matches reference (mistake did not change answer!)")
                else:
                    print(f"        Reason: {reason}")
        print()

    # Category summary
    categories = [item["category"] for item in BENCHMARK]
    counts = Counter(categories)

    print("=" * 80)
    print("BENCHMARK SUMMARY & DISCRIMINATION REPORT:")
    print(f"  - Total Benchmark Items:             {len(BENCHMARK)}")
    print(f"  - Total Wrong Variants Tested:       {total_variants}")
    print(f"  - Non-Discriminating Variants:       {len(non_discriminating_variants)}")
    if non_discriminating_variants:
        print("    WARNING: Non-discriminating variants found:")
        for nd_id, nd_label, nd_kind in non_discriminating_variants:
            print(f"      * Item {nd_id} [{nd_kind}]: {nd_label}")
    else:
        print("    All wrong variants are properly DISCRIMINATING.")

    if items_without_variants:
        print(f"  - Items without variants (IDs):      {items_without_variants}")
    else:
        print("  - Items without variants:            None (all items have variants)")

    print("\nCATEGORY DISTRIBUTION:")
    for cat, count in sorted(counts.items(), key=lambda x: x[0]):
        print(f"  - {cat:35s}: {count} questions")
    print(f"Total: {len(BENCHMARK)} questions across {len(counts)} categories")
    print("=" * 80)

    if non_discriminating_variants:
        sys.exit(1)


if __name__ == "__main__":
    main()
