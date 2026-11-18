"""
check_sandbox_corpus.py
Regression Corpus Static AST Verification:
Extracts every piece of pandas code produced by the analyst, the reviewer, the
benchmark, and the wrong variants from across the codebase, then runs static AST
checks to ensure no legitimate analyst or benchmark code is blocked.
"""

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.benchmark import BENCHMARK
from tools.sandbox import _validate_code_ast


def extract_corpus_snippets() -> list[dict[str, Any]]:
    """Extracts all pandas code snippets from the benchmark, evaluation runs, and ground truth."""
    snippets: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(source: str, item_id: str, code: str, kind: str = "code") -> None:
        if not code or not isinstance(code, str):
            return
        code_clean = code.strip()
        key = (source, str(item_id), code_clean)
        if key in seen:
            return
        seen.add(key)
        snippets.append({
            "source": source,
            "item_id": str(item_id),
            "code": code_clean,
            "kind": kind,
        })

    # 1. Benchmark reference codes and wrong variants
    for item in BENCHMARK:
        item_id = item.get("id", "unknown")
        if "reference_code" in item:
            _add("eval/benchmark.py", f"item_{item_id}_ref", item["reference_code"], kind="benchmark_ref")
        for v_idx, v in enumerate(item.get("wrong_variants", [])):
            if "code" in v:
                _add("eval/benchmark.py", f"item_{item_id}_wrong_{v_idx}", v["code"], kind="benchmark_wrong")

    # 2. Ground truth json
    gt_path = Path("eval/ground_truth.json")
    if gt_path.exists():
        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                for item_id, item_val in gt_data.items():
                    if isinstance(item_val, dict):
                        if "reference_code" in item_val:
                            _add("eval/ground_truth.json", f"gt_{item_id}_ref", item_val["reference_code"], kind="ground_truth_ref")
                        if "code" in item_val:
                            _add("eval/ground_truth.json", f"gt_{item_id}_code", item_val["code"], kind="ground_truth_code")
                        for v_idx, v in enumerate(item_val.get("wrong_variants", [])):
                            if isinstance(v, dict) and "code" in v:
                                _add("eval/ground_truth.json", f"gt_{item_id}_wrong_{v_idx}", v["code"], kind="ground_truth_wrong")
        except Exception:
            pass

    # 3. eval/runs/*/cases.jsonl
    for fpath in sorted(glob.glob("eval/runs/**/cases.jsonl", recursive=True)):
        rel_path = fpath.replace("\\", "/")
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    d = json.loads(line_str)
                    cid = d.get("case_id", f"line_{idx}")

                    # Reviewer run analyst code under test
                    if "code" in d and isinstance(d["code"], str):
                        _add(rel_path, f"{cid}_code", d["code"], kind="reviewer_analyst_code")

                    # E2E runs
                    if "system_a" in d and isinstance(d["system_a"], dict):
                        c = d["system_a"].get("code")
                        if c:
                            _add(rel_path, f"{cid}_system_a", c, kind="e2e_system_a")

                    if "system_b" in d and isinstance(d["system_b"], dict):
                        c = d["system_b"].get("code")
                        if c:
                            _add(rel_path, f"{cid}_system_b", c, kind="e2e_system_b")

                    if "system_c" in d and isinstance(d["system_c"], dict):
                        c = d["system_c"].get("final_code")
                        if c:
                            _add(rel_path, f"{cid}_system_c_final", c, kind="e2e_system_c_final")
                        for r_idx, r in enumerate(d["system_c"].get("rounds", [])):
                            if isinstance(r, dict):
                                rc = r.get("analyst_code")
                                if rc:
                                    _add(rel_path, f"{cid}_system_c_r{r_idx}", rc, kind="e2e_system_c_round")
        except Exception:
            pass

    # 4. eval/runs/*/seeds.jsonl
    for fpath in sorted(glob.glob("eval/runs/**/seeds.jsonl", recursive=True)):
        rel_path = fpath.replace("\\", "/")
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    d = json.loads(line_str)
                    for k, v in d.items():
                        if isinstance(v, str) and ("result = " in v or "df." in v):
                            _add(rel_path, f"seed_{idx}_{k}", v, kind="seed_code")
        except Exception:
            pass

    return snippets


def check_corpus() -> tuple[int, int, list[dict[str, Any]]]:
    """Runs static AST check on every snippet in the corpus."""
    snippets = extract_corpus_snippets()
    blocked: list[dict[str, Any]] = []

    for s in snippets:
        is_valid, err_msg, status = _validate_code_ast(s["code"])
        if not is_valid:
            blocked.append({
                "source": s["source"],
                "item_id": s["item_id"],
                "kind": s["kind"],
                "error": err_msg,
                "status": status,
                "code": s["code"],
            })

    return len(snippets), len(blocked), blocked


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify sandbox static AST checks against all real analyst and benchmark corpus code."
    )
    parser.parse_args()

    total, num_blocked, blocked = check_corpus()

    print("=" * 70)
    print("SANDBOX REAL-OUTPUT REGRESSION CORPUS CHECK")
    print("=" * 70)
    print(f"Total snippets checked: {total}")
    print(f"Total snippets blocked: {num_blocked}")

    if num_blocked > 0:
        print("\nBLOCKED SNIPPETS DETAILS:")
        for idx, b in enumerate(blocked, start=1):
            print(f"\n[{idx}] {b['source']} ({b['item_id']}, {b['kind']})")
            print(f"    Status: {b['status']}")
            print(f"    Reason: {b['error']}")
            print("    Code snippet:")
            for line in b["code"].splitlines():
                print(f"      {line}")
        print("\nFAIL: Legitimate corpus code was blocked by sandbox rules.")
        return 1
    else:
        print("\nSUCCESS: All real-world and benchmark pandas code passed static AST checks.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
