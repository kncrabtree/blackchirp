"""Tests for the offline reference fitter (fcu_fit.py)."""

from __future__ import annotations

import numpy as np
import pytest

from fcu_calibration_math import PhysicalParams, physical_forward
from fcu_csv import read_measurements_csv, write_polynomial_csv
from fcu_fit import (
    ascending,
    build_spline_points,
    fit_physical,
    fit_polynomial,
    normalization,
)


def test_fit_physical_recovers_synthetic_parameters() -> None:
    """A noiseless synthetic table below-tolerance-recovers the true parameters.

    Mirrors the prototype validation described in
    dev-docs/sirah-fcu-calibration.md: the 5-parameter (here: default
    3-parameter) fit recovers a synthetic table.
    """
    truth = PhysicalParams(
        crystal="bbo",
        cut_angle_deg=32.3,
        temperature=290.0,
        linear_offset_mm=-76.0,
        angle_offset_deg=31.5,
        screw_pitch_mm=-0.25,
        lever_length_mm=134.599318,
        motor_resolution=4800.0,
        invert=False,
    )
    wavelengths_nm = np.linspace(650.0, 900.0, 12)
    positions_steps = [physical_forward(truth, lam) for lam in wavelengths_nm]

    initial = PhysicalParams(
        crystal="bbo",
        cut_angle_deg=32.3,
        temperature=293.0,
        linear_offset_mm=-76.543335,
        angle_offset_deg=31.329809,
        screw_pitch_mm=-0.25,
        lever_length_mm=134.599318,
        motor_resolution=4800.0,
        invert=False,
    )
    fitted, rms = fit_physical(wavelengths_nm, positions_steps, initial)

    assert rms == pytest.approx(0.0, abs=1e-4)
    assert fitted.temperature == pytest.approx(truth.temperature, abs=1e-4)
    assert fitted.linear_offset_mm == pytest.approx(truth.linear_offset_mm, abs=1e-4)
    assert fitted.angle_offset_deg == pytest.approx(truth.angle_offset_deg, abs=1e-4)
    # cut angle and screw pitch were held fixed, not fit
    assert fitted.cut_angle_deg == initial.cut_angle_deg
    assert fitted.screw_pitch_mm == initial.screw_pitch_mm


def test_fit_polynomial_reproduces_synthetic_cubic() -> None:
    """A degree-3 fit of exact cubic data reproduces it in both directions."""
    true_desc = [
        2.5,
        -100.0,
        40000.0,
        500.0,
    ]  # descending, numpy.polyfit's native order
    wavelengths_nm = np.linspace(600.0, 900.0, 20)
    positions_steps = np.polyval(true_desc, wavelengths_nm)

    fit = fit_polynomial(wavelengths_nm, positions_steps, degree=3)

    assert fit.forward_rms == pytest.approx(0.0, abs=1e-4)
    np.testing.assert_allclose(fit.forward(wavelengths_nm), positions_steps, atol=1e-4)
    assert fit.wavelength_center == pytest.approx(750.0)
    assert fit.wavelength_scale == pytest.approx(150.0)


def test_normalization_rounds_center_and_half_span() -> None:
    assert normalization([560.0, 566.1234, 563.0], 3) == (563.062, 3.062)
    assert normalization([16397800.0, 16426900.0], 0) == (16412350.0, 14550.0)
    assert normalization([5.0, 5.0], 3) == (5.0, 1.0)


def test_normalized_high_degree_fit_survives_csv_round_trip(tmp_path) -> None:
    """A quintic over a narrow band near 1.64e7 steps evaluates accurately
    from its exported coefficients, and forward and inverse agree.

    In raw units this fit is badly conditioned and its two directions
    disagree by hundreds of steps; normalized, both stay within a step.
    """
    wavelengths_nm = np.linspace(560.0, 566.1, 25)
    positions_steps = (
        16397800.0
        + 4770.0 * (wavelengths_nm - 560.0)
        + 15.0 * (wavelengths_nm - 560.0) ** 2
    )
    fit = fit_polynomial(wavelengths_nm, positions_steps, degree=5)

    path = tmp_path / "poly.csv"
    write_polynomial_csv(
        str(path), ascending(fit.forward_desc), ascending(fit.inverse_desc)
    )
    rows = [
        line.split(";") for line in path.read_text(encoding="utf-8").splitlines()[1:]
    ]
    fwd = [float(r[1]) for r in rows]
    inv = [float(r[2]) for r in rows]

    def horner(coeffs, x):
        result = 0.0
        for c in reversed(coeffs):
            result = result * x + c
        return result

    for lam, pos in zip(wavelengths_nm, positions_steps):
        u = (lam - fit.wavelength_center) / fit.wavelength_scale
        p = horner(fwd, u)
        assert p == pytest.approx(pos, abs=1.0)
        v = (p - fit.position_center) / fit.position_scale
        # 2e-4 nm is about one step on this curve.
        assert horner(inv, v) == pytest.approx(lam, abs=2e-4)


def test_ascending_reverses_polyfit_order() -> None:
    assert ascending(np.array([3.0, 2.0, 1.0])) == [1.0, 2.0, 3.0]


def test_build_spline_points_sorts_and_dedupes_wavelength() -> None:
    wavelengths_nm = [800.0, 700.0, 700.0, 750.0]
    positions_steps = [3.0, 1.0, 1.5, 2.0]
    points = build_spline_points(wavelengths_nm, positions_steps)
    assert points == [(700.0, 1.0), (750.0, 2.0), (800.0, 3.0)]


def test_write_polynomial_csv_header_and_order_column(tmp_path) -> None:
    path = tmp_path / "poly.csv"
    write_polynomial_csv(
        str(path),
        forward_coeffs_asc=[1.0, 2.0, 3.0],
        inverse_coeffs_asc=[4.0, 5.0, 6.0],
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "order;forward;inverse"
    assert lines[1] == "0;1;4"
    assert lines[2] == "1;2;5"
    assert lines[3] == "2;3;6"


def test_read_measurements_csv_round_trips(tmp_path) -> None:
    path = tmp_path / "measurements.csv"
    path.write_text(
        "wavelengthNm;positionSteps\n700.0;123456\n750.5;234567\n", encoding="utf-8"
    )

    wavelengths_nm, positions_steps = read_measurements_csv(str(path))
    assert wavelengths_nm == [700.0, 750.5]
    assert positions_steps == [123456.0, 234567.0]
