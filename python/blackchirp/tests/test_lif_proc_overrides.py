"""Coverage for the per-call processing override surface.

Every kwarg accepted by ``BCLifTrace.integrate`` is also accepted by
the aggregating helpers ``BCLIF.image``, ``BCLIF.delay_slice`` and
``BCLIF.laser_slice``. These tests exercise each kwarg one at a time
to confirm no exception is raised and that the populated points
remain finite.
"""

from __future__ import annotations

import numpy as np
import pytest

_KWARG_CASES = [
    {"lif_start": 400},
    {"lif_end": 1500},
    {"ref_start": 80},
    {"ref_end": 320},
    {"low_pass_alpha": 0.3},
    {"savgol_window": 9},
    {"savgol_poly": 2},
    {"savgol_enabled": True},
]


@pytest.mark.parametrize("kwargs", _KWARG_CASES, ids=lambda k: ",".join(k))
def test_image_accepts_each_kwarg(v2_lif_ref_exp, kwargs):
    _, _, img = v2_lif_ref_exp.lif.image(**kwargs)
    finite = np.isfinite(img)
    assert finite.any()


@pytest.mark.parametrize("kwargs", _KWARG_CASES, ids=lambda k: ",".join(k))
def test_delay_slice_accepts_each_kwarg(v2_lif_ref_exp, kwargs):
    _, vals = v2_lif_ref_exp.lif.delay_slice(0, **kwargs)
    finite = np.isfinite(vals)
    assert finite.any()


@pytest.mark.parametrize("kwargs", _KWARG_CASES, ids=lambda k: ",".join(k))
def test_laser_slice_accepts_each_kwarg(v2_lif_ref_exp, kwargs):
    _, vals = v2_lif_ref_exp.lif.laser_slice(0, **kwargs)
    finite = np.isfinite(vals)
    assert finite.any()


def test_image_with_all_kwargs_combined(v2_lif_ref_exp):
    _, _, img = v2_lif_ref_exp.lif.image(
        lif_start=300,
        lif_end=1800,
        ref_start=50,
        ref_end=350,
        low_pass_alpha=0.2,
        savgol_window=11,
        savgol_poly=3,
        savgol_enabled=True,
    )
    assert np.isfinite(img).any()
