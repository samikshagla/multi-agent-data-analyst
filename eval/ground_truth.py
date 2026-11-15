"""
ground_truth.py
Ground Truth Computation and Persistence: Runs reference benchmark codes and wrong
analyst variants inside the sandbox to compute certified outputs, and saves/loads them to/from JSON.
"""

import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from tools.sandbox import run_code


def _to_native_scalar(val: Any) -> Any:
    """Converts numpy numbers and pandas scalars to native Python types."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return float(val)
    return str(val)


def _serialize_answer(val: Any) -> Any:
    """Serializes DataFrame or scalar into JSON-serializable structure."""
    if isinstance(val, pd.DataFrame):
        columns = [str(c) for c in val.columns]
        rows: list[list[Any]] = []
        for row in val.itertuples(index=False):
            rows.append([_to_native_scalar(cell) for cell in row])
        return {"columns": columns, "rows": rows}
    elif isinstance(val, pd.Series):
        df_series = val.reset_index()
        return _serialize_answer(df_series)
    else:
        return _to_native_scalar(val)


def _deserialize_answer(ans: Any) -> Any:
    """Deserializes JSON answer into DataFrame or native scalar."""
    if isinstance(ans, dict) and "columns" in ans and "rows" in ans:
        return pd.DataFrame(ans["rows"], columns=ans["columns"])
    return ans


def compute_ground_truth(
    benchmark: list[dict[str, Any]],
    csv_path: str,
) -> dict[int, dict[str, Any]]:
    """
    Runs each reference_code and its wrong_variants with tools/sandbox.py run_code
    on a copy of the dataframe and returns:
    id -> {
        "status": "ok",
        "answer": <reference_result>,
        "wrong_variants": [
            {"label": ..., "code": ..., "kind": ..., "status": "ok", "answer": <variant_result>}
        ]
    }
    Fails loudly if any reference code or wrong variant returns a status other than 'ok'.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    ground_truth: dict[int, dict[str, Any]] = {}

    for item in benchmark:
        item_id = int(item["id"])
        code = item.get("reference_code", "")

        # 1. Run reference code
        res = run_code(code, df.copy(deep=True))
        if res.get("status") != "ok":
            err = res.get("error", "Unknown error")
            raise RuntimeError(
                f"Reference code for benchmark item id {item_id} failed with status "
                f"'{res.get('status')}': {err}\nCode:\n{code}"
            )

        # 2. Run wrong variants if defined
        computed_variants: list[dict[str, Any]] = []
        for v in item.get("wrong_variants", []):
            v_code = v.get("code", "")
            v_label = v.get("label", "")
            v_kind = v.get("kind", "")

            v_res = run_code(v_code, df.copy(deep=True))
            if v_res.get("status") != "ok":
                v_err = v_res.get("error", "Unknown error")
                raise RuntimeError(
                    f"Wrong variant '{v_label}' for benchmark item id {item_id} failed with status "
                    f"'{v_res.get('status')}': {v_err}\nCode:\n{v_code}"
                )

            computed_variants.append({
                "label": v_label,
                "code": v_code,
                "kind": v_kind,
                "status": "ok",
                "answer": v_res.get("result"),
            })

        ground_truth[item_id] = {
            "status": "ok",
            "answer": res.get("result"),
            "wrong_variants": computed_variants,
        }

    return ground_truth


def save_ground_truth(
    ground_truth: dict[int, dict[str, Any]],
    path: str = "eval/ground_truth.json",
) -> None:
    """
    Saves computed ground truth dictionary to a readable JSON file.
    DataFrames are stored as {"columns": [...], "rows": [[...]]} and scalars as numbers.
    Wrong variants are stored under each item id next to the correct answer.
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    serialized: dict[str, Any] = {}
    for item_id, item_data in ground_truth.items():
        variants_serialized = []
        for v in item_data.get("wrong_variants", []):
            variants_serialized.append({
                "label": v.get("label", ""),
                "code": v.get("code", ""),
                "kind": v.get("kind", ""),
                "status": v.get("status", "ok"),
                "answer": _serialize_answer(v.get("answer")),
            })

        serialized[str(item_id)] = {
            "status": item_data.get("status", "ok"),
            "answer": _serialize_answer(item_data.get("answer")),
            "wrong_variants": variants_serialized,
        }

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)


def load_ground_truth(
    path: str = "eval/ground_truth.json",
) -> dict[int, dict[str, Any]]:
    """
    Loads ground truth from JSON. DataFrames stored as {"columns": [...], "rows": [...]}
    are reconstructed as pandas DataFrames, and scalar values as native numbers.
    Keys are returned as integer IDs.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground truth file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    result: dict[int, dict[str, Any]] = {}
    for key_str, item in raw_data.items():
        item_id = int(key_str)
        ans_val = _deserialize_answer(item.get("answer"))

        variants_deserialized = []
        for v in item.get("wrong_variants", []):
            variants_deserialized.append({
                "label": v.get("label", ""),
                "code": v.get("code", ""),
                "kind": v.get("kind", ""),
                "status": v.get("status", "ok"),
                "answer": _deserialize_answer(v.get("answer")),
            })

        result[item_id] = {
            "status": item.get("status", "ok"),
            "answer": ans_val,
            "wrong_variants": variants_deserialized,
        }

    return result
