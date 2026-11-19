"""
test_sandbox_corpus.py
Tests that sandbox static AST checks permit all legitimate benchmark code,
wrong variants, and real LLM-produced code without false positives.
"""

import pytest
from eval.benchmark import BENCHMARK
from eval.check_sandbox_corpus import check_corpus, extract_corpus_snippets
from tools.sandbox import _validate_code_ast


def test_benchmark_reference_codes_all_pass_ast():
    """All 20 benchmark reference codes must pass AST validation with status 'ok'."""
    assert len(BENCHMARK) == 20
    for item in BENCHMARK:
        item_id = item["id"]
        code = item["reference_code"]
        is_valid, err, status = _validate_code_ast(code)
        assert is_valid is True, f"Benchmark item {item_id} reference code was blocked: {err}\nCode: {code}"
        assert status == "ok"
        assert err is None


def test_benchmark_wrong_variants_all_pass_ast():
    """
    All plausible wrong analyst variants in benchmark must pass AST validation.
    They represent logical errors (e.g. wrong aggregation, wrong denominator),
    not sandbox violations, and must be executed in eval to test the reviewer.
    """
    total_variants = 0
    for item in BENCHMARK:
        item_id = item["id"]
        for v_idx, v in enumerate(item.get("wrong_variants", [])):
            total_variants += 1
            code = v["code"]
            is_valid, err, status = _validate_code_ast(code)
            assert is_valid is True, f"Benchmark item {item_id} variant {v_idx} ({v.get('kind')}) was blocked: {err}\nCode: {code}"
            assert status == "ok"
            assert err is None
    assert total_variants > 0


def test_real_llm_corpus_zero_blocked():
    """
    Runs static AST check on the full extracted corpus of real LLM outputs
    and benchmark cases, asserting zero false-positive blocks.
    """
    from pathlib import Path
    total, num_blocked, blocked = check_corpus()
    has_runs = Path("eval/runs").is_dir() and any(Path("eval/runs").glob("**/*.jsonl"))
    min_expected = 400 if has_runs else 100
    assert total >= min_expected, f"Expected at least {min_expected} snippets in corpus, found {total}"
    assert num_blocked == 0, f"Corpus check blocked {num_blocked} snippets:\n" + "\n".join(
        f"  - [{b['source']} {b['item_id']}] {b['error']}" for b in blocked
    )
