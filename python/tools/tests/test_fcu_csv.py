"""Tests for the shared CSV I/O helpers (fcu_csv.py)."""

from __future__ import annotations

import pytest

from fcu_csv import (
    append_measurement_csv,
    read_measurements_csv,
    write_measurements_csv,
)


def test_append_measurement_csv_writes_header_once(tmp_path) -> None:
    path = tmp_path / "measurements.csv"
    append_measurement_csv(str(path), 700.0, 100)
    append_measurement_csv(str(path), 750.0, 200)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "wavelengthNm;positionSteps"
    assert lines.count("wavelengthNm;positionSteps") == 1
    assert len(lines) == 3

    wavelengths_nm, positions_steps = read_measurements_csv(str(path))
    assert wavelengths_nm == [700.0, 750.0]
    assert positions_steps == [100.0, 200.0]


def test_write_measurements_csv_overwrites(tmp_path) -> None:
    path = tmp_path / "spline.csv"
    write_measurements_csv(str(path), [(700.0, 1.0), (750.0, 2.0)])

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "wavelengthNm;positionSteps"
    assert lines[1] == "700.000000;1"
    assert lines[2] == "750.000000;2"

    # A second write overwrites rather than appending.
    write_measurements_csv(str(path), [(800.0, 3.0)])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_read_measurements_csv_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        read_measurements_csv("/nonexistent/path/measurements.csv")


def test_read_measurements_csv_rejects_empty_file(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        read_measurements_csv(str(path))
