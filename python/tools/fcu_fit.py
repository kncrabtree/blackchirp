#!/usr/bin/env python
"""Offline reference fitter for a Sirah FCU doubling-crystal calibration.

Reads a measurements CSV (``wavelengthNm;positionSteps``, produced by
``fcu_measure.py`` or hand-recorded per the Sirah Autotracker service
manual's calibration procedure, section 6) and fits/visualizes the
tuning curve under Blackchirp's three ``FcuCalibration`` schemes:

- **Physical** — a nonlinear least-squares fit of the Type-I SHG
  forward map in :mod:`fcu_calibration_math`, which mirrors
  ``src/data/lif/fcucalibration.cpp`` exactly. Prints the five fitted
  parameters (crystal cut angle, temperature, linear offset, angle
  offset, screw pitch) for direct entry into Blackchirp's Physical
  scheme fields.
- **Polynomial** — ``numpy.polyfit`` forward (wavelength -> position)
  and inverse (position -> wavelength), each in a normalized input
  variable (center and half-span of the measured range); exports an
  ``order;forward;inverse`` CSV for Blackchirp's Polynomial scheme and
  prints the four normalization values for entry alongside it.
- **Spline** — passes the measurement points through as the spline
  table (a ``scipy.interpolate.PchipInterpolator`` gives a monotone
  preview curve only); exports a ``wavelengthNm;positionSteps`` CSV
  for Blackchirp's Spline scheme.

All three fits are visualized together (tuning curve + residuals) and
can be saved headlessly with ``--save``.

Example:
    python fcu_fit.py measurements.csv --crystal bbo --cut-angle 32.3 \\
        --save calibration.png
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from fcu_calibration_math import (
    PhysicalParams,
    physical_forward,
    with_free_params,
)
from fcu_csv import (
    read_measurements_csv,
    write_measurements_csv,
    write_polynomial_csv,
)

#: Free-parameter fields fit by default (manual's default: temperature +
#: linear offset + angle offset, holding cut angle and screw pitch fixed).
_DEFAULT_FREE_FIELDS = ("temperature", "linear_offset_mm", "angle_offset_deg")


def fit_physical(
    wavelengths_nm: Sequence[float],
    positions_steps: Sequence[float],
    initial: PhysicalParams,
    fit_cut_angle: bool = False,
    fit_screw_pitch: bool = False,
) -> Tuple[PhysicalParams, float]:
    """Nonlinear least-squares fit of the Physical forward map.

    Args:
        wavelengths_nm: Measured fundamental wavelengths (nm).
        positions_steps: Measured motor positions (steps), same order.
        initial: Initial guess / fixed values for the five parameters
            plus the fixed mechanics (lever length, motor resolution).
        fit_cut_angle: Also fit ``cut_angle_deg`` (default: fixed).
        fit_screw_pitch: Also fit ``screw_pitch_mm`` (default: fixed).

    Returns:
        A ``(fitted_params, rms_residual_steps)`` pair.
    """
    free_fields = list(_DEFAULT_FREE_FIELDS)
    if fit_cut_angle:
        free_fields.append("cut_angle_deg")
    if fit_screw_pitch:
        free_fields.append("screw_pitch_mm")

    wl = np.asarray(wavelengths_nm, dtype=float)
    pos = np.asarray(positions_steps, dtype=float)
    x0 = np.array([getattr(initial, f) for f in free_fields], dtype=float)

    def residuals(x: np.ndarray) -> np.ndarray:
        trial = with_free_params(initial, **dict(zip(free_fields, x)))
        predicted = np.array([physical_forward(trial, lam) for lam in wl])
        return predicted - pos

    result = least_squares(residuals, x0)
    fitted = with_free_params(initial, **dict(zip(free_fields, result.x)))
    rms = float(np.sqrt(np.mean(result.fun**2)))
    return fitted, rms


@dataclass
class PolynomialFit:
    """Forward and inverse polynomial fits in normalized input variables.

    Blackchirp evaluates the forward list at
    ``(wavelength - wavelength_center) / wavelength_scale`` and the
    inverse list at ``(position - position_center) / position_scale``;
    outputs are raw steps and nm. Coefficients are in ``numpy.polyfit``'s
    native descending-power order; see :func:`ascending` to convert for
    the CSV export / Horner-evaluation contract.

    Attributes:
        forward_desc: Wavelength -> position coefficients (descending).
        inverse_desc: Position -> wavelength coefficients (descending).
        forward_rms: Forward RMS residual (steps).
        inverse_rms: Inverse RMS residual (nm).
        wavelength_center: Forward input center (nm).
        wavelength_scale: Forward input scale (nm).
        position_center: Inverse input center (steps).
        position_scale: Inverse input scale (steps).
    """

    forward_desc: np.ndarray
    inverse_desc: np.ndarray
    forward_rms: float
    inverse_rms: float
    wavelength_center: float
    wavelength_scale: float
    position_center: float
    position_scale: float

    def forward(self, wavelengths_nm) -> np.ndarray:
        """Evaluate wavelength (nm) -> position (steps)."""
        u = (np.asarray(wavelengths_nm, dtype=float) - self.wavelength_center) / (
            self.wavelength_scale
        )
        return np.polyval(self.forward_desc, u)

    def inverse(self, positions_steps) -> np.ndarray:
        """Evaluate position (steps) -> wavelength (nm)."""
        u = (np.asarray(positions_steps, dtype=float) - self.position_center) / (
            self.position_scale
        )
        return np.polyval(self.inverse_desc, u)


def normalization(values: Sequence[float], decimals: int) -> Tuple[float, float]:
    """Center and half-span of ``values``, rounded to ``decimals`` places.

    Rounding keeps the printed values exact when typed into Blackchirp's
    settings; the fit uses the rounded values, so nothing is lost. A
    zero span (a single distinct value) yields a scale of 1.

    Args:
        values: Sample values (wavelengths in nm or positions in steps).
        decimals: Decimal places to round the center and scale to.

    Returns:
        ``(center, scale)``.
    """
    v = np.asarray(values, dtype=float)
    center = round(float((v.max() + v.min()) / 2.0), decimals)
    scale = round(float((v.max() - v.min()) / 2.0), decimals)
    if scale == 0.0:
        scale = 1.0
    return center, scale


def fit_polynomial(
    wavelengths_nm: Sequence[float], positions_steps: Sequence[float], degree: int
) -> PolynomialFit:
    """Forward and inverse polynomial fits via ``numpy.polyfit``.

    Each direction is fit in a normalized input variable spanning roughly
    [-1, 1]. Fitting in raw units (wavelengths near 500 nm, positions
    near 10^7 steps) is badly conditioned beyond low order: the
    coefficients become huge and cancel heavily, so even a
    full-precision export evaluates with errors of many steps.

    Args:
        wavelengths_nm: Measured fundamental wavelengths (nm).
        positions_steps: Measured motor positions (steps), same order.
        degree: Polynomial degree.

    Returns:
        The fit, with its normalization.
    """
    wl = np.asarray(wavelengths_nm, dtype=float)
    pos = np.asarray(positions_steps, dtype=float)
    wl_center, wl_scale = normalization(wl, 3)
    pos_center, pos_scale = normalization(pos, 0)
    forward_desc = np.polyfit((wl - wl_center) / wl_scale, pos, degree)
    inverse_desc = np.polyfit((pos - pos_center) / pos_scale, wl, degree)
    fit = PolynomialFit(
        forward_desc=forward_desc,
        inverse_desc=inverse_desc,
        forward_rms=0.0,
        inverse_rms=0.0,
        wavelength_center=wl_center,
        wavelength_scale=wl_scale,
        position_center=pos_center,
        position_scale=pos_scale,
    )
    fit.forward_rms = float(np.sqrt(np.mean((fit.forward(wl) - pos) ** 2)))
    fit.inverse_rms = float(np.sqrt(np.mean((fit.inverse(pos) - wl) ** 2)))
    return fit


def ascending(coeffs_desc: np.ndarray) -> List[float]:
    """Reverse ``numpy.polyfit``'s descending-power coefficients to ascending.

    Blackchirp's Horner evaluation (``FcuCalibration``'s ``horner()``)
    expects ``c0 + c1*x + c2*x^2 + ...``; ``numpy.polyfit`` returns the
    highest power first.
    """
    return [float(c) for c in reversed(list(coeffs_desc))]


#: Minimum distinct-wavelength point count for Blackchirp's Spline scheme.
#: Blackchirp fits the imported points with GSL's ``gsl_interp_steffen``
#: (see ``makeSteffenSpline`` in ``fcucalibration.cpp``), whose documented
#: ``gsl_interp_type_min_size`` is 3; fewer points fail at import time.
_MIN_SPLINE_POINTS = 3


def build_spline_points(
    wavelengths_nm: Sequence[float], positions_steps: Sequence[float]
) -> List[Tuple[float, float]]:
    """Sort measurement points by wavelength, deduplicating repeats.

    Args:
        wavelengths_nm: Measured fundamental wavelengths (nm).
        positions_steps: Measured motor positions (steps), same order.

    Returns:
        ``(wavelength_nm, position_steps)`` pairs sorted ascending by
        wavelength, with later duplicate-wavelength rows dropped (the
        first occurrence wins) so the spline scheme's imported point
        table has distinct wavelengths.

    Raises:
        ValueError: If fewer than :data:`_MIN_SPLINE_POINTS`
            distinct-wavelength points remain, or the positions are not
            strictly monotonic (all increasing or all decreasing) in
            wavelength order. Both conditions mirror the validation
            ``FcuCalibration::spline()`` performs on import
            (``fcucalibration.cpp``); failing them here surfaces the
            error at fit time instead of inside Blackchirp.
    """
    pts = sorted(zip(wavelengths_nm, positions_steps), key=lambda p: p[0])
    deduped: List[Tuple[float, float]] = []
    seen = set()
    for wl, pos in pts:
        if wl in seen:
            continue
        seen.add(wl)
        deduped.append((float(wl), float(pos)))

    if len(deduped) < _MIN_SPLINE_POINTS:
        raise ValueError(
            f"Spline scheme requires at least {_MIN_SPLINE_POINTS} "
            f"distinct-wavelength points, got {len(deduped)}"
        )

    increasing = all(deduped[i][1] > deduped[i - 1][1] for i in range(1, len(deduped)))
    decreasing = all(deduped[i][1] < deduped[i - 1][1] for i in range(1, len(deduped)))
    if not increasing and not decreasing:
        raise ValueError(
            "Spline scheme requires positions strictly monotonic in "
            "wavelength order (all increasing or all decreasing); the "
            "measurements do not satisfy this"
        )

    return deduped


def print_physical_result(
    fitted: PhysicalParams, rms: float, fit_cut_angle: bool, fit_screw_pitch: bool
) -> None:
    """Print the fitted Physical parameters for Blackchirp entry."""

    def tag(fitted_flag: bool) -> str:
        return "" if fitted_flag else "  (held fixed)"

    print("Physical scheme parameters (Blackchirp Physical scheme fields):")
    print(f"  Crystal type      = {fitted.crystal}")
    print(f"  Cut angle (deg)   = {fitted.cut_angle_deg:.6f}{tag(fit_cut_angle)}")
    print(f"  Temperature       = {fitted.temperature:.6f}")
    print(f"  Linear offset(mm) = {fitted.linear_offset_mm:.6f}")
    print(f"  Angle offset(deg) = {fitted.angle_offset_deg:.6f}")
    print(f"  Screw pitch (mm)  = {fitted.screw_pitch_mm:.6f}{tag(fit_screw_pitch)}")
    print(f"  Invert            = {fitted.invert}")
    print(
        f"  (fixed mechanics) lever length = {fitted.lever_length_mm:.6f} mm, "
        f"motor resolution = {fitted.motor_resolution:.6f} steps/rev"
    )
    print(f"  RMS residual = {rms:.4f} steps")


def plot_calibration(
    wavelengths_nm: Sequence[float],
    positions_steps: Sequence[float],
    physical_fitted: PhysicalParams,
    poly_fit: PolynomialFit,
    spline_points: Sequence[Tuple[float, float]],
    save_path: str | None,
) -> None:
    """Plot data points, each fitted model's tuning curve, and residuals.

    Mirrors the Sirah Autotracker service manual's Table dialog: motor
    position vs wavelength on top, a residual trace underneath.

    Args:
        wavelengths_nm: Measured fundamental wavelengths (nm).
        positions_steps: Measured motor positions (steps), same order.
        physical_fitted: The fitted Physical parameters.
        poly_fit: The fitted Polynomial scheme.
        spline_points: Sorted, deduplicated ``(wavelength, position)``
            points for the Spline scheme.
        save_path: If given, save the figure here instead of showing
            it interactively (for headless use).
    """
    import matplotlib

    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import PchipInterpolator

    wl = np.asarray(wavelengths_nm, dtype=float)
    pos = np.asarray(positions_steps, dtype=float)

    lam_grid = np.linspace(wl.min(), wl.max(), 400)
    physical_grid = np.array(
        [physical_forward(physical_fitted, lam) for lam in lam_grid]
    )
    poly_grid = poly_fit.forward(lam_grid)

    spline_wl = np.array([p[0] for p in spline_points])
    spline_pos = np.array([p[1] for p in spline_points])
    spline_curve = PchipInterpolator(spline_wl, spline_pos)(lam_grid)

    physical_resid = (
        np.array([physical_forward(physical_fitted, lam) for lam in wl]) - pos
    )
    poly_resid = poly_fit.forward(wl) - pos

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 7), gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_top.plot(wl, pos, "ko", label="Measured", zorder=5)
    ax_top.plot(lam_grid, physical_grid, "-", label="Physical fit")
    ax_top.plot(lam_grid, poly_grid, "--", label="Polynomial fit")
    ax_top.plot(lam_grid, spline_curve, ":", label="Spline (monotone preview)")
    ax_top.set_ylabel("Motor position (steps)")
    ax_top.set_title("Sirah FCU calibration")
    ax_top.legend()

    ax_bottom.axhline(0.0, color="k", linewidth=0.8)
    ax_bottom.plot(wl, physical_resid, "o-", label="Physical")
    ax_bottom.plot(wl, poly_resid, "s--", label="Polynomial")
    ax_bottom.set_xlabel("Fundamental wavelength (nm)")
    ax_bottom.set_ylabel("Residual (steps)")
    ax_bottom.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def _default_sibling_path(measurements_path: str, suffix: str) -> str:
    """Derive a default export path alongside ``measurements_path``."""
    stem, _ = os.path.splitext(measurements_path)
    return f"{stem}_{suffix}.csv"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit and visualize a Sirah FCU doubling-crystal calibration from a "
            "wavelengthNm;positionSteps measurements CSV, under all three of "
            "Blackchirp's FcuCalibration schemes (Physical / Polynomial / Spline)."
        )
    )
    parser.add_argument(
        "measurements", help="Path to the wavelengthNm;positionSteps measurements CSV."
    )

    physical = parser.add_argument_group("Physical scheme")
    physical.add_argument(
        "--crystal",
        choices=("bbo", "kdp"),
        default="bbo",
        help="Doubling crystal (default: bbo).",
    )
    physical.add_argument(
        "--cut-angle",
        type=float,
        required=True,
        help="Crystal cut angle (deg), initial guess/fixed value. Table 6-1 of the "
        "Sirah Autotracker service manual, per crystal/band.",
    )
    physical.add_argument(
        "--temperature",
        type=float,
        default=293.0,
        help="Dispersion-fit temperature initial guess, arbitrary units (default: 293, the "
        "Sellmeier dn/dT reference point).",
    )
    physical.add_argument(
        "--linear-offset",
        type=float,
        default=-76.543335,
        help="Sine-bar linear offset (mm) initial guess (default: the registered SirahFcu "
        "stage default).",
    )
    physical.add_argument(
        "--angle-offset",
        type=float,
        default=31.329809,
        help="Sine-bar angle offset (deg) initial guess (default: the registered SirahFcu "
        "stage default).",
    )
    physical.add_argument(
        "--screw-pitch",
        type=float,
        default=-0.25,
        help="Lead-screw pitch (mm/rev), initial guess/fixed value (default: the registered "
        "SirahFcu stage default).",
    )
    physical.add_argument(
        "--lever-length",
        type=float,
        default=134.599318,
        help="Sine-bar lever length (mm); fixed mechanics, not fit (default: the registered "
        "SirahFcu stage default).",
    )
    physical.add_argument(
        "--motor-resolution",
        type=float,
        default=4800.0,
        help="Motor steps per revolution; fixed mechanics, not fit (default: the registered "
        "SirahFcu stage default).",
    )
    physical.add_argument(
        "--invert",
        action="store_true",
        help="Select the phase-match relation's - branch.",
    )
    physical.add_argument(
        "--fit-cut-angle",
        action="store_true",
        help="Also fit the cut angle (default: held fixed at --cut-angle, per the manual's "
        "default calibration procedure).",
    )
    physical.add_argument(
        "--fit-screw-pitch",
        action="store_true",
        help="Also fit the screw pitch (default: held fixed at --screw-pitch, per the "
        "manual's default calibration procedure).",
    )

    poly = parser.add_argument_group("Polynomial scheme")
    poly.add_argument(
        "--poly-degree",
        type=int,
        default=2,
        help="Polynomial degree (default: 2). Over a narrow tuning band the "
        "scatter between measurements usually limits the residual, so a "
        "higher degree fits that scatter rather than the tuning curve.",
    )
    poly.add_argument(
        "--poly-output",
        default=None,
        help="Output path for the order;forward;inverse CSV (default: "
        "<measurements>_poly.csv).",
    )

    spline = parser.add_argument_group("Spline scheme")
    spline.add_argument(
        "--spline-output",
        default=None,
        help="Output path for the wavelengthNm;positionSteps spline CSV (default: "
        "<measurements>_spline.csv).",
    )

    viz = parser.add_argument_group("Visualization")
    viz.add_argument(
        "--save",
        default=None,
        metavar="PNG",
        help="Save the plot here instead of showing it (headless use).",
    )
    viz.add_argument("--no-plot", action="store_true", help="Skip plotting entirely.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: fit all three schemes, print results, export CSVs, plot."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    wavelengths_nm, positions_steps = read_measurements_csv(args.measurements)
    print(f"Read {len(wavelengths_nm)} measurement points from {args.measurements}")

    initial = PhysicalParams(
        crystal=args.crystal,
        cut_angle_deg=args.cut_angle,
        temperature=args.temperature,
        linear_offset_mm=args.linear_offset,
        angle_offset_deg=args.angle_offset,
        screw_pitch_mm=args.screw_pitch,
        lever_length_mm=args.lever_length,
        motor_resolution=args.motor_resolution,
        invert=args.invert,
    )
    physical_fitted, physical_rms = fit_physical(
        wavelengths_nm,
        positions_steps,
        initial,
        args.fit_cut_angle,
        args.fit_screw_pitch,
    )
    print()
    print_physical_result(
        physical_fitted, physical_rms, args.fit_cut_angle, args.fit_screw_pitch
    )

    poly_fit = fit_polynomial(wavelengths_nm, positions_steps, args.poly_degree)
    poly_output = args.poly_output or _default_sibling_path(args.measurements, "poly")
    write_polynomial_csv(
        poly_output, ascending(poly_fit.forward_desc), ascending(poly_fit.inverse_desc)
    )
    print()
    print(f"Polynomial scheme (degree={args.poly_degree}):")
    print(
        "  Forward (wavelength[nm] -> position[steps]) RMS residual = "
        f"{poly_fit.forward_rms:.4f} steps"
    )
    print(
        "  Inverse (position[steps] -> wavelength[nm]) RMS residual = "
        f"{poly_fit.inverse_rms:.6f} nm"
    )
    print(f"  Exported to {poly_output}")
    print("  Normalization (enter in the Polynomial scheme fields):")
    print(f"    Polynomial Wavelength Center (nm)  = {poly_fit.wavelength_center:.3f}")
    print(f"    Polynomial Wavelength Scale (nm)   = {poly_fit.wavelength_scale:.3f}")
    print(f"    Polynomial Position Center (steps) = {poly_fit.position_center:.0f}")
    print(f"    Polynomial Position Scale (steps)  = {poly_fit.position_scale:.0f}")

    spline_points = build_spline_points(wavelengths_nm, positions_steps)
    spline_output = args.spline_output or _default_sibling_path(
        args.measurements, "spline"
    )
    write_measurements_csv(spline_output, spline_points)
    print()
    print("Spline scheme:")
    print(
        f"  {len(spline_points)} distinct-wavelength points pass through exactly (RMS residual = 0)."
    )
    print(f"  Exported to {spline_output}")

    if not args.no_plot:
        plot_calibration(
            wavelengths_nm,
            positions_steps,
            physical_fitted,
            poly_fit,
            spline_points,
            args.save,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
