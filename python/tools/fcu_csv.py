"""Semicolon-delimited CSV readers/writers for the Sirah FCU calibration
tools' cross-component contract.

Three CSV shapes are in play, all semicolon-delimited with a header
row (see ``dev-docs/sirah-fcu-calibration.md``):

- **Measurements** — ``wavelengthNm;positionSteps``. Produced by
  ``fcu_measure.py``, consumed by ``fcu_fit.py``.
- **Spline import** — the same ``wavelengthNm;positionSteps`` shape,
  exported by ``fcu_fit.py`` for Blackchirp's Spline scheme.
- **Polynomial import** — ``order;forward;inverse``, one row per
  polynomial order ``0..n`` ascending, exported by ``fcu_fit.py`` for
  Blackchirp's Polynomial scheme.
"""

from __future__ import annotations

import csv
import os
from typing import Iterable, List, Sequence, Tuple

MEASUREMENT_HEADER = ("wavelengthNm", "positionSteps")
POLYNOMIAL_HEADER = ("order", "forward", "inverse")


def _format_wavelength(value: float) -> str:
    """Format a wavelength in nm with fixed (never scientific) decimal notation."""
    return f"{value:.6f}"


def _format_position(value: float) -> str:
    """Format a motor position in steps as a plain integer (never scientific notation).

    Motor positions are always integer-valued step counts; formatting
    with ``%g``-style notation risks emitting e.g. ``1e+06``, which a
    strict integer parser (as Blackchirp's CSV import may use for an
    integer-typed sub-key) would reject.
    """
    return str(int(round(value)))


def read_measurements_csv(path: str) -> Tuple[List[float], List[float]]:
    """Read a ``wavelengthNm;positionSteps`` measurements CSV.

    Args:
        path: Path to the CSV file, semicolon-delimited with a header
            row matching :data:`MEASUREMENT_HEADER`.

    Returns:
        A ``(wavelengths_nm, positions_steps)`` pair of equal-length
        lists, in file order.

    Raises:
        ValueError: If the file has no header/data rows, or a row
            cannot be parsed as two numbers.
    """
    wavelengths: List[float] = []
    positions: List[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError(f"{path}: empty measurements file")

    # Tolerate a missing header (treat the first row as data if it
    # parses as two numbers); otherwise require it to match the
    # documented column names and skip it.
    start = 0
    try:
        float(rows[0][0])
        float(rows[0][1])
    except (ValueError, IndexError):
        start = 1

    for row in rows[start:]:
        if len(row) < 2:
            raise ValueError(f"{path}: expected 2 columns, got {row!r}")
        wavelengths.append(float(row[0]))
        positions.append(float(row[1]))

    if not wavelengths:
        raise ValueError(f"{path}: no data rows")
    return wavelengths, positions


def append_measurement_csv(
    path: str, wavelength_nm: float, position_steps: float
) -> None:
    """Append one measurement row, writing the header only if the file is new.

    Args:
        path: Output CSV path. Created (with header) if it does not
            already exist or is empty; otherwise appended to.
        wavelength_nm: Fundamental wavelength in nanometers.
        position_steps: Raw motor position in steps.
    """
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        if write_header:
            writer.writerow(MEASUREMENT_HEADER)
        writer.writerow(
            [_format_wavelength(wavelength_nm), _format_position(position_steps)]
        )


def write_measurements_csv(path: str, points: Iterable[Tuple[float, float]]) -> None:
    """Write a complete ``wavelengthNm;positionSteps`` CSV (the Spline export).

    Args:
        path: Output CSV path (overwritten).
        points: ``(wavelength_nm, position_steps)`` pairs, written in
            the given order.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(MEASUREMENT_HEADER)
        for wl, pos in points:
            writer.writerow([_format_wavelength(wl), _format_position(pos)])


def write_polynomial_csv(
    path: str, forward_coeffs_asc: Sequence[float], inverse_coeffs_asc: Sequence[float]
) -> None:
    """Write an ``order;forward;inverse`` polynomial-coefficient CSV.

    Args:
        path: Output CSV path (overwritten).
        forward_coeffs_asc: Wavelength (nm) -> position coefficients,
            ascending order (``c0 + c1*x + c2*x^2 + ...``).
        inverse_coeffs_asc: Position -> wavelength (nm) coefficients,
            ascending order.

    Raises:
        ValueError: If the two coefficient lists differ in length.
    """
    if len(forward_coeffs_asc) != len(inverse_coeffs_asc):
        raise ValueError(
            "forward/inverse coefficient lists must have the same length "
            f"({len(forward_coeffs_asc)} != {len(inverse_coeffs_asc)})"
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(POLYNOMIAL_HEADER)
        for order, (fwd, inv) in enumerate(zip(forward_coeffs_asc, inverse_coeffs_asc)):
            writer.writerow([order, f"{fwd:.10g}", f"{inv:.10g}"])
