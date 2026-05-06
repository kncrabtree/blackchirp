"""Sweep every kwarg of ``BCFid.ft`` one at a time."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_us": 0.0},
        {"start_us": 5.0},
        {"end_us": 10.0},
        {"winf": "Hamming"},
        {"zpf": 1},
        {"zpf": 2},
        {"rdc": True},
        {"rdc": False},
        {"expf_us": 5.0},
        {"autoscale_MHz": 50.0},
        {"units_power": 3},
        {"units_power": 6},
        {"frame": 0},
    ],
)
def test_ft_kwarg(v2_ftmw_exp, kwargs):
    fid = v2_ftmw_exp.ftmw.get_fid()
    x, y = fid.ft(**kwargs)
    assert x.ndim == 1
    assert np.all(np.isfinite(y))
    assert y.shape[0] == x.shape[0]


def test_ft_combined_kwargs(v2_ftmw_exp):
    fid = v2_ftmw_exp.ftmw.get_fid()
    x, y = fid.ft(start_us=3.0, end_us=12.0, winf="Hanning", zpf=1, rdc=True)
    assert np.all(np.isfinite(y))


def test_ft_does_not_mutate_data(v2_ftmw_exp):
    """``ft`` must leave ``self.data`` untouched so repeated calls
    produce identical results — the windowed slice is a view of
    ``self.data`` and was historically mutated by the FidRemoveDC
    subtraction and the exp-decay multiplication."""
    fid = v2_ftmw_exp.ftmw.get_fid()
    data_before = fid.data.copy()
    x1, y1 = fid.ft(rdc=True, expf_us=5.0)
    np.testing.assert_array_equal(fid.data, data_before)
    x2, y2 = fid.ft(rdc=True, expf_us=5.0)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
