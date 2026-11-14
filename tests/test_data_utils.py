"""
test_data_utils.py
Unit tests for components/data_utils.py using pytest.
Validates load_csv_safe (valid, empty, header-only, single-row, latin-1, invalid/binary, limits),
dataset_snapshot (known missing, duplicates, column type breakdown),
and save_upload (path traversal sanitization, collision avoidance / no overwrite).
"""

import io
from pathlib import Path
import pandas as pd
import pytest

from components.data_utils import (
    load_csv_safe,
    dataset_snapshot,
    save_upload,
    MAX_ROWS,
    MAX_COLS,
)


class MockUploadedFile:
    """Simulates Streamlit's UploadedFile object with .name, .read(), and .seek()."""
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._buffer = io.BytesIO(content)

    def read(self, *args):
        return self._buffer.read(*args)

    def seek(self, pos, whence=0):
        return self._buffer.seek(pos, whence)

    def getbuffer(self):
        return self._buffer.getbuffer()


def test_load_csv_safe_valid_file(tmp_path: Path):
    """Valid CSV file loads cleanly into a DataFrame with no error message."""
    csv_file = tmp_path / "valid.csv"
    csv_file.write_text("id,name,val\n1,alpha,10.5\n2,beta,20.0\n3,gamma,30.2\n", encoding="utf-8")

    df, err = load_csv_safe(csv_file)
    assert err is None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["id", "name", "val"]


def test_load_csv_safe_empty_file(tmp_path: Path):
    """Empty 0-byte file returns an appropriate error message."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_bytes(b"")

    df, err = load_csv_safe(empty_file)
    assert df is None
    assert err is not None
    assert "empty" in err.lower()


def test_load_csv_safe_header_only_file(tmp_path: Path):
    """File with only column headers and no rows returns an informative error."""
    header_file = tmp_path / "header_only.csv"
    header_file.write_text("col_a,col_b,col_c\n", encoding="utf-8")

    df, err = load_csv_safe(header_file)
    assert df is None
    assert err is not None
    assert "headers" in err.lower() or "no data" in err.lower()


def test_load_csv_safe_single_row_file(tmp_path: Path):
    """File with only one row of data returns an error requiring at least 2 rows."""
    single_row_file = tmp_path / "single_row.csv"
    single_row_file.write_text("col_a,col_b\n1,2\n", encoding="utf-8")

    df, err = load_csv_safe(single_row_file)
    assert df is None
    assert err is not None
    assert "only one row" in err.lower() or "at least 2" in err.lower()


def test_load_csv_safe_latin1_encoded(tmp_path: Path):
    """Latin-1 encoded file is successfully decoded and loaded."""
    latin1_file = tmp_path / "latin1.csv"
    # Content with Latin-1 special accented characters
    raw_content = "ville,café,score\nMontréal,Espresso,95\nGenève,Latte,88\n"
    latin1_file.write_bytes(raw_content.encode("latin-1"))

    df, err = load_csv_safe(latin1_file)
    assert err is None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "café" in df.columns or "caf" in df.columns[1]
    assert df.iloc[0]["ville"] == "Montréal"


def test_load_csv_safe_non_csv_text(tmp_path: Path):
    """Corrupt or non-CSV binary data returns a clean error."""
    bin_file = tmp_path / "corrupt.csv"
    # Unparseable binary data with null bytes and invalid tokens
    bin_file.write_bytes(b"\x00\xff\xfe\x00\x12\x34\x56\x78\x90\xaa\xbb\xcc\xdd\xee\xff")

    df, err = load_csv_safe(bin_file)
    assert df is None
    assert err is not None
    assert "could not parse" in err.lower() or "could not decode" in err.lower() or "empty" in err.lower()


def test_load_csv_safe_exceeding_column_limit(tmp_path: Path):
    """File exceeding the 100 columns demo limit is rejected with a clear message."""
    col_names = [f"col_{i}" for i in range(105)]
    line1 = ",".join(col_names) + "\n"
    line2 = ",".join(["1"] * 105) + "\n"
    line3 = ",".join(["2"] * 105) + "\n"

    huge_cols_file = tmp_path / "wide.csv"
    huge_cols_file.write_text(line1 + line2 + line3, encoding="utf-8")

    df, err = load_csv_safe(huge_cols_file)
    assert df is None
    assert err is not None
    assert "exceeds demo limits" in err
    assert "105 columns" in err


def test_load_csv_safe_from_bytesio():
    """File-like BytesIO streams are handled seamlessly."""
    stream = io.BytesIO(b"x,y\n10,20\n30,40\n")
    df, err = load_csv_safe(stream)
    assert err is None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_dataset_snapshot_known_counts():
    """dataset_snapshot accurately tallies missing %, duplicates, and column types."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Alice", None],
        "age": [25, 30, 25, 40],
        "signup_date": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-04"],
        "score": [100.0, None, 100.0, 95.0],
    })
    # Row 0 and Row 2 are exact duplicate rows: ("Alice", 25, "2023-01-01", 100.0)
    # Total cells: 4 * 4 = 16
    # Missing cells: 2 (name[3], score[1]) -> missing_pct = (2/16)*100 = 12.5%
    # Duplicate rows: 1

    snap = dataset_snapshot(df)

    assert snap["rows"] == 4
    assert snap["columns"] == 4
    assert snap["duplicate_rows"] == 1
    assert snap["missing_pct"] == 12.5
    assert snap["numeric_cols"] == 2      # age, score
    assert snap["date_like_cols"] == 1    # signup_date
    assert snap["text_cols"] == 1         # name


def test_save_upload_sanitizes_dangerous_path(tmp_path: Path):
    """save_upload strips path traversal characters and saves safely into target folder."""
    mock_file = MockUploadedFile(
        name="../../x y.csv",
        content=b"city,temp\nSeattle,65\nAustin,85\n",
    )

    saved_path = save_upload(mock_file, upload_dir=tmp_path)

    # 1. File must reside inside tmp_path
    assert saved_path.parent == tmp_path
    # 2. Filename must not contain '..' or spaces
    assert ".." not in saved_path.name
    assert " " not in saved_path.name
    assert "x_y.csv" in saved_path.name
    # 3. File exists and contents match
    assert saved_path.exists()
    assert saved_path.read_bytes() == b"city,temp\nSeattle,65\nAustin,85\n"


def test_save_upload_never_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """save_upload avoids collision and never overwrites an existing file."""
    mock_file1 = MockUploadedFile("dataset.csv", b"a,b\n1,2\n3,4\n")
    mock_file2 = MockUploadedFile("dataset.csv", b"a,b\n5,6\n7,8\n")

    # Freeze timestamp so both calls would generate the exact same candidate path
    class FixedNow:
        @staticmethod
        def strftime(fmt: str) -> str:
            return "20261114_120000"

    class FixedDateTime:
        @staticmethod
        def now():
            return FixedNow()

    monkeypatch.setattr("components.data_utils.datetime", FixedDateTime)

    path1 = save_upload(mock_file1, upload_dir=tmp_path)
    path2 = save_upload(mock_file2, upload_dir=tmp_path)

    assert path1 != path2
    assert path1.exists()
    assert path2.exists()
    assert path1.read_bytes() == b"a,b\n1,2\n3,4\n"
    assert path2.read_bytes() == b"a,b\n5,6\n7,8\n"
