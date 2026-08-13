"""Tests for the Physical-scheme math port (fcu_calibration_math.py).

These pin the port against the same behavior as
``src/data/lif/fcucalibration.cpp``: a forward/inverse round trip
across each supported crystal's tuning band.
"""

from __future__ import annotations

import math

import pytest

from fcu_calibration_math import PhysicalParams, physical_forward, physical_inverse

_CUT_ANGLE_DEG = {"bbo": 32.3, "kdp": 41.0}


def _params(crystal: str) -> PhysicalParams:
    return PhysicalParams(
        crystal=crystal,
        cut_angle_deg=_CUT_ANGLE_DEG[crystal],
        temperature=293.0,
        linear_offset_mm=-76.543335,
        angle_offset_deg=31.329809,
        screw_pitch_mm=-0.25,
        lever_length_mm=134.599318,
        motor_resolution=4800.0,
        invert=False,
    )


@pytest.mark.parametrize("crystal", ["bbo", "kdp"])
@pytest.mark.parametrize("lam_nm", [600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0])
def test_physical_forward_inverse_round_trip(crystal: str, lam_nm: float) -> None:
    params = _params(crystal)
    pos = physical_forward(params, lam_nm)
    assert math.isfinite(pos)

    recovered_nm = physical_inverse(params, pos)
    assert math.isfinite(recovered_nm)
    assert recovered_nm == pytest.approx(lam_nm, abs=0.05)


def test_physical_forward_invert_flips_sign_of_alpha_int() -> None:
    """--invert selects the other phase-match branch, so positions differ."""
    params = _params("bbo")
    inverted = PhysicalParams(**{**params.__dict__, "invert": True})
    assert physical_forward(params, 750.0) != physical_forward(inverted, 750.0)


def test_crystal_name_is_case_insensitive_and_normalized() -> None:
    params = PhysicalParams(
        crystal="bbo",
        cut_angle_deg=32.3,
        temperature=293.0,
        linear_offset_mm=-76.543335,
        angle_offset_deg=31.329809,
        screw_pitch_mm=-0.25,
        lever_length_mm=134.599318,
        motor_resolution=4800.0,
    )
    assert params.crystal == "BBO"


def test_unknown_crystal_raises() -> None:
    with pytest.raises(ValueError):
        PhysicalParams(
            crystal="linbo3",
            cut_angle_deg=0.0,
            temperature=293.0,
            linear_offset_mm=0.0,
            angle_offset_deg=0.0,
            screw_pitch_mm=-0.25,
            lever_length_mm=134.6,
            motor_resolution=4800.0,
        )
