"""Tests for :class:`blackchirp.BCLifTrace`.

The fixtures used here are sparse 6×6 LIF scans bundled under
``python/example-data/`` — ``v2-lif-ref/`` (with reference channel)
and ``v2-lif-noref/`` (single channel). Numerical accuracy of the
underlying smoothing and integration kernels is taken on faith; these
tests exercise that every kwarg surface returns finite values of the
expected shape and dtype.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_trace_shapes_with_ref(v2_lif_ref_exp):
    trace = v2_lif_ref_exp.lif.get_trace(0, 0)
    assert trace.has_ref()
    x = trace.x()
    y = trace.lif()
    r = trace.ref()
    assert x.shape == (trace.lifsize,)
    assert y.shape == (trace.lifsize,)
    assert r.shape == (trace.refsize,)
    assert np.issubdtype(x.dtype, np.floating)
    assert np.issubdtype(y.dtype, np.floating)
    assert np.issubdtype(r.dtype, np.floating)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(r))


def test_trace_x_spacing_matches_params(v2_lif_ref_exp):
    trace = v2_lif_ref_exp.lif.get_trace(0, 0)
    x = trace.x()
    spacing = float(trace.params["spacing"])
    assert x[0] == 0.0
    assert pytest.approx(x[1] - x[0]) == spacing
    assert pytest.approx(x[-1]) == (trace.lifsize - 1) * spacing


@pytest.mark.parametrize(
    "units,scale",
    [("s", 1.0), ("ms", 1e3), ("us", 1e6), ("μs", 1e6), ("ns", 1e9)],
)
def test_trace_x_units(v2_lif_ref_exp, units, scale):
    trace = v2_lif_ref_exp.lif.get_trace(0, 0)
    spacing = float(trace.params["spacing"])
    x = trace.x(units=units)
    assert x.shape == (trace.lifsize,)
    assert pytest.approx(x[1] - x[0]) == spacing * scale


def test_trace_x_units_invalid(v2_lif_ref_exp):
    trace = v2_lif_ref_exp.lif.get_trace(0, 0)
    with pytest.raises(ValueError):
        trace.x(units="hours")


def test_trace_xy_propagates_units(v2_lif_ref_exp):
    trace = v2_lif_ref_exp.lif.get_trace(0, 0)
    spacing = float(trace.params["spacing"])
    x_ns, _, _ = trace.xy(units="ns")
    assert pytest.approx(x_ns[1] - x_ns[0]) == spacing * 1e9


def test_trace_xy_with_ref_returns_three(v2_lif_ref_exp):
    out = v2_lif_ref_exp.lif.get_trace(0, 0).xy()
    assert len(out) == 3
    x, y, r = out
    assert x.shape == y.shape == r.shape


def test_trace_xy_no_ref_returns_two(v2_lif_noref_exp):
    out = v2_lif_noref_exp.lif.get_trace(0, 0).xy()
    assert len(out) == 2
    x, y = out
    assert x.shape == y.shape


def test_trace_ref_none_when_single_channel(v2_lif_noref_exp):
    trace = v2_lif_noref_exp.lif.get_trace(0, 0)
    assert not trace.has_ref()
    assert trace.ref() is None


@pytest.mark.parametrize(
    "low_pass,savgol",
    [
        (True, False),
        (False, True),
        (True, True),
        (False, False),
    ],
)
def test_smooth_combinations(v2_lif_ref_exp, low_pass, savgol):
    trace = v2_lif_ref_exp.lif.get_trace(0, 0)
    out = trace.smooth(
        low_pass=low_pass,
        savgol=savgol,
        low_pass_alpha=0.4,
        savgol_window=11,
        savgol_poly=3,
    )
    assert out.shape == (trace.lifsize,)
    assert np.all(np.isfinite(out))


def test_smooth_default_passes_through(v2_lif_ref_exp):
    trace = v2_lif_ref_exp.lif.get_trace(0, 0)
    out = trace.smooth()
    assert np.all(np.isfinite(out))
    assert out.shape == (trace.lifsize,)


def test_integrate_default_gates(v2_lif_ref_exp):
    trace = v2_lif_ref_exp.lif.get_trace(0, 0)
    val = trace.integrate()
    assert isinstance(val, float)
    assert np.isfinite(val)


def test_integrate_override_gates(v2_lif_ref_exp):
    trace = v2_lif_ref_exp.lif.get_trace(0, 0)
    val = trace.integrate(lif_start=400, lif_end=2000)
    assert isinstance(val, float)
    assert np.isfinite(val)


def test_integrate_ratio_path_with_ref(v2_lif_ref_exp):
    trace = v2_lif_ref_exp.lif.get_trace(0, 0)
    no_ref_call = trace.integrate(ref_start=0, ref_end=0)
    ratioed = trace.integrate(ref_start=100, ref_end=300)
    assert np.isfinite(ratioed)
    assert ratioed != no_ref_call


def test_integrate_no_ref_branch(v2_lif_noref_exp):
    trace = v2_lif_noref_exp.lif.get_trace(0, 0)
    val = trace.integrate(ref_start=100, ref_end=300)
    assert np.isfinite(val)


def test_integrate_with_smoothing_overrides(v2_lif_ref_exp):
    trace = v2_lif_ref_exp.lif.get_trace(0, 0)
    val = trace.integrate(
        low_pass_alpha=0.5,
        savgol_enabled=True,
        savgol_window=15,
        savgol_poly=3,
    )
    assert np.isfinite(val)
