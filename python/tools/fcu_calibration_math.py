"""Physical (Type-I SHG) tuning-curve math for a Sirah FCU doubling crystal.

This module is a line-for-line port of the ``Physical`` scheme in
Blackchirp's C++ evaluator, ``src/data/lif/fcucalibration.cpp``
(``FcuCalibration::physical()`` / ``physicalForward()`` /
``physicalInverse()`` / ``phaseMatchAngleDeg()``). It exists so the
offline fitter (``fcu_fit.py``) fits against *exactly* the same
Sellmeier equations, dispersion constants, and forward/inverse maps
that Blackchirp evaluates, so fitted parameters reproduce the recorded
calibration when typed into Blackchirp's Physical scheme fields.

Keep this module's formulas byte-for-byte consistent with the C++
source above. If the C++ evaluator changes, mirror the change here.

Sellmeier references:
    BBO: Eimerl, "Optical, mechanical, and thermal properties of
    barium borate," IEEE J. Quantum Electron. 23, 575 (1987).

    KDP: Zernike, "Refractive indices of ammonium dihydrogen phosphate
    and potassium dihydrogen phosphate between 2000 A and 1.5 mu,"
    J. Opt. Soc. Am. 54, 1215 (1964).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi

#: Sellmeier dn/dT zero point (arbitrary "temperature" units), matching
#: ``kReferenceTemperature`` in fcucalibration.cpp.
REFERENCE_TEMPERATURE = 293.0

BBO_DNO_DT = -16.6e-6
BBO_DNE_DT = -9.3e-6
KDP_DNO_DT = -3.4e-5
KDP_DNE_DT = -2.4e-5

CRYSTALS = ("BBO", "KDP")

#: Physical inverse root-find grid, matching physicalInverse()'s
#: lamMin/lamMax/steps in fcucalibration.cpp.
_INVERSE_LAM_MIN_NM = 380.0
_INVERSE_LAM_MAX_NM = 1000.0
_INVERSE_STEPS = 4000
_INVERSE_BISECTION_ITERS = 100


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` to ``[lo, hi]``."""
    return max(lo, min(hi, value))


def _normalize_crystal(crystal: str) -> str:
    """Validate and upper-case a crystal-type string.

    Args:
        crystal: ``"bbo"`` or ``"kdp"``, any case.

    Returns:
        The normalized (upper-case) crystal name.

    Raises:
        ValueError: If ``crystal`` is not one of the supported types.
    """
    c = crystal.upper()
    if c not in CRYSTALS:
        raise ValueError(
            f"Unknown crystal type {crystal!r}; expected one of {CRYSTALS}"
        )
    return c


def bbo_no2(lam_um: float) -> float:
    """BBO ordinary-index Sellmeier, n_o^2(lam_um) (Eimerl 1987)."""
    lam2 = lam_um * lam_um
    return 2.7405 + 0.0184 / (lam2 - 0.0179) - 0.0155 * lam2


def bbo_ne2(lam_um: float) -> float:
    """BBO extraordinary-index Sellmeier, n_e^2(lam_um) (Eimerl 1987)."""
    lam2 = lam_um * lam_um
    return 2.3730 + 0.0128 / (lam2 - 0.0156) - 0.0044 * lam2


def kdp_no2(lam_um: float) -> float:
    """KDP ordinary-index Sellmeier, n_o^2(lam_um) (Zernike 1964)."""
    lam2 = lam_um * lam_um
    return (
        2.259276 + 0.01008956 / (lam2 - 0.012942625) + 13.00522 * lam2 / (lam2 - 400.0)
    )


def kdp_ne2(lam_um: float) -> float:
    """KDP extraordinary-index Sellmeier, n_e^2(lam_um) (Zernike 1964)."""
    lam2 = lam_um * lam_um
    return (
        2.132668
        + 0.008637494 / (lam2 - 0.012281043)
        + 3.2279924 * lam2 / (lam2 - 400.0)
    )


def sellmeier_no(crystal: str, lam_um: float, temperature: float) -> float:
    """Ordinary index n_o(lam_um) at a dispersion-fit ``temperature``.

    Args:
        crystal: ``"bbo"`` or ``"kdp"``.
        lam_um: Wavelength in micrometers.
        temperature: Arbitrary-units dispersion-fit knob; a linear
            dn/dT is applied to ``n`` itself (not ``n^2``) about the
            293-unit reference point.

    Returns:
        The ordinary refractive index.
    """
    crystal = _normalize_crystal(crystal)
    if crystal == "BBO":
        n2, dndt = bbo_no2(lam_um), BBO_DNO_DT
    else:
        n2, dndt = kdp_no2(lam_um), KDP_DNO_DT
    return math.sqrt(n2) + dndt * (temperature - REFERENCE_TEMPERATURE)


def sellmeier_ne(crystal: str, lam_um: float, temperature: float) -> float:
    """Extraordinary index n_e(lam_um); see :func:`sellmeier_no`."""
    crystal = _normalize_crystal(crystal)
    if crystal == "BBO":
        n2, dndt = bbo_ne2(lam_um), BBO_DNE_DT
    else:
        n2, dndt = kdp_ne2(lam_um), KDP_DNE_DT
    return math.sqrt(n2) + dndt * (temperature - REFERENCE_TEMPERATURE)


def raw_phase_match_s2(crystal: str, lam_fund_nm: float, temperature: float) -> float:
    """Unclamped Type-I SHG phase-match sin^2(theta_pm).

    Outside ``[0, 1]`` the crystal cannot phase-match this fundamental
    at all; :func:`phase_match_angle_deg` clamps this to report a
    boundary angle, but :func:`physical_inverse`'s root find needs the
    unclamped value to restrict its search to the invertible band.

    Args:
        crystal: ``"bbo"`` or ``"kdp"``.
        lam_fund_nm: Fundamental wavelength in nanometers.
        temperature: Dispersion-fit temperature (arbitrary units).

    Returns:
        sin^2(theta_pm), not clamped to ``[0, 1]``.
    """
    lam_um = lam_fund_nm / 1000.0
    lam2_um = lam_um / 2.0
    no_fund = sellmeier_no(crystal, lam_um, temperature)
    no_sh = sellmeier_no(crystal, lam2_um, temperature)
    ne_sh = sellmeier_ne(crystal, lam2_um, temperature)
    return (1.0 / (no_fund * no_fund) - 1.0 / (no_sh * no_sh)) / (
        1.0 / (ne_sh * ne_sh) - 1.0 / (no_sh * no_sh)
    )


def phase_match_angle_deg(
    crystal: str, lam_fund_nm: float, temperature: float = 293.0
) -> float:
    """Type-I second-harmonic phase-match angle (degrees).

    Args:
        crystal: ``"bbo"`` or ``"kdp"``.
        lam_fund_nm: Fundamental wavelength in nanometers.
        temperature: Dispersion-fit temperature (arbitrary units).
            Defaults to the 293-unit reference point.

    Returns:
        The phase-match angle in degrees, clamped to ``[0, 90]``.
    """
    s2 = _clamp(raw_phase_match_s2(crystal, lam_fund_nm, temperature), 0.0, 1.0)
    return math.asin(math.sqrt(s2)) * RAD_TO_DEG


@dataclass
class PhysicalParams:
    """The five Physical-scheme calibration parameters plus fixed mechanics.

    Field names and grouping mirror ``FcuCalibration::PhysicalParams``
    in fcucalibration.h. ``lever_length_mm`` and ``motor_resolution``
    are fixed sine-bar mechanics, not fit parameters, but travel with
    the rest so :func:`physical_forward` takes a single argument.

    Attributes:
        crystal: ``"BBO"`` or ``"KDP"``.
        cut_angle_deg: Crystal optic-axis-to-face cut angle (degrees).
        temperature: Dispersion-fit temperature (arbitrary units).
        linear_offset_mm: Sine-bar linear offset (mm).
        angle_offset_deg: Sine-bar angle offset (degrees).
        screw_pitch_mm: Lead-screw pitch (mm/rev).
        lever_length_mm: Sine-bar lever length (mm); fixed mechanics.
        motor_resolution: Motor steps per revolution; fixed mechanics.
        invert: Selects the phase-match relation's +/- branch.
    """

    crystal: str
    cut_angle_deg: float
    temperature: float
    linear_offset_mm: float
    angle_offset_deg: float
    screw_pitch_mm: float
    lever_length_mm: float
    motor_resolution: float
    invert: bool = False

    def __post_init__(self) -> None:
        self.crystal = _normalize_crystal(self.crystal)


def physical_forward(params: PhysicalParams, lam_nm: float) -> float:
    """Fundamental wavelength (nm) -> motor position (steps).

    Mirrors ``FcuCalibration::physicalForward()``: phase-match angle,
    crystal-face Snell refraction using the fundamental's ordinary
    index, then the sine-bar geometry.

    Args:
        params: The Physical scheme parameters.
        lam_nm: Fundamental wavelength in nanometers.

    Returns:
        The motor position in steps.
    """
    theta_pm_rad = (
        phase_match_angle_deg(params.crystal, lam_nm, params.temperature) * DEG_TO_RAD
    )
    sign = -1.0 if params.invert else 1.0
    alpha_int = sign * (theta_pm_rad - params.cut_angle_deg * DEG_TO_RAD)

    # Snell refraction at the crystal face, using the ordinary index at
    # the fundamental (not the second harmonic) as the refraction index.
    n = sellmeier_no(params.crystal, lam_nm / 1000.0, params.temperature)
    alpha_ext = math.asin(_clamp(n * math.sin(alpha_int), -1.0, 1.0))

    x = params.linear_offset_mm - params.lever_length_mm * math.sin(
        params.angle_offset_deg * DEG_TO_RAD - alpha_ext
    )
    return (params.motor_resolution / params.screw_pitch_mm) * x


def physical_inverse(params: PhysicalParams, pos: float) -> float:
    """Motor position (steps) -> fundamental wavelength (nm).

    Mirrors ``FcuCalibration::physicalInverse()``: a coarse grid search
    over the plausible fundamental range restricted to where the
    crystal can phase-match (``raw_phase_match_s2`` in ``[0, 1]``),
    followed by bisection refinement.

    Args:
        params: The Physical scheme parameters.
        pos: Motor position in steps.

    Returns:
        The fundamental wavelength in nanometers, or ``nan`` if no
        bracket was found.
    """
    have_prev = False
    prev_lam = prev_f = 0.0
    bracket_lo = bracket_hi = None

    for i in range(_INVERSE_STEPS + 1):
        lam = (
            _INVERSE_LAM_MIN_NM
            + (_INVERSE_LAM_MAX_NM - _INVERSE_LAM_MIN_NM) * i / _INVERSE_STEPS
        )
        s2 = raw_phase_match_s2(params.crystal, lam, params.temperature)
        if s2 < 0.0 or s2 > 1.0:
            have_prev = False
            continue

        f = physical_forward(params, lam) - pos
        if have_prev and ((prev_f <= 0.0 <= f) or (prev_f >= 0.0 >= f)):
            bracket_lo, bracket_hi = prev_lam, lam
            break
        prev_lam, prev_f, have_prev = lam, f, True

    if bracket_lo is None:
        return math.nan

    lo, hi = bracket_lo, bracket_hi
    flo = physical_forward(params, lo) - pos
    for _ in range(_INVERSE_BISECTION_ITERS):
        mid = 0.5 * (lo + hi)
        fmid = physical_forward(params, mid) - pos
        if (flo <= 0.0 <= fmid) or (flo >= 0.0 >= fmid):
            hi = mid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def with_free_params(params: PhysicalParams, **updates: float) -> PhysicalParams:
    """Return a copy of ``params`` with the given fields replaced.

    Thin wrapper around :func:`dataclasses.replace` used by the
    least-squares residual function to materialize a trial parameter
    set from the optimizer's free-parameter vector.
    """
    return replace(params, **updates)
