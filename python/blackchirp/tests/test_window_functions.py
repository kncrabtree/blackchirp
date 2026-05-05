"""Exercise every window-function form through ``BCFid.ft``."""

from __future__ import annotations

import numpy as np
import pytest

from blackchirp.bcfid import _WINDOW_INT_MAP, _WINDOW_MAP


@pytest.mark.parametrize("name", list(_WINDOW_MAP.keys()))
def test_window_by_name(any_exp, name):
    fid = any_exp.ftmw.get_fid()
    x, y = fid.ft(winf=name)
    assert np.all(np.isfinite(y))
    assert y.shape[0] == x.shape[0]


@pytest.mark.parametrize("intval", list(_WINDOW_INT_MAP.keys()))
def test_window_by_int_in_proc(v2_ftmw_exp, intval):
    """Integer-form window value in ``proc`` resolves correctly."""
    fid = v2_ftmw_exp.ftmw.get_fid()
    fid.proc["FidWindowFunction"] = intval
    x, y = fid.ft()
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize("intstr", [str(i) for i in _WINDOW_INT_MAP.keys()])
def test_window_by_int_string_in_proc(v2_ftmw_exp, intstr):
    """Integer-shaped string (e.g. ``"3"``) in ``proc`` resolves correctly."""
    fid = v2_ftmw_exp.ftmw.get_fid()
    fid.proc["FidWindowFunction"] = intstr
    x, y = fid.ft()
    assert np.all(np.isfinite(y))


def test_window_unknown_raises(v2_ftmw_exp):
    fid = v2_ftmw_exp.ftmw.get_fid()
    with pytest.raises(ValueError):
        fid.ft(winf="NoSuchWindow")


def test_window_int_out_of_range_raises(v2_ftmw_exp):
    fid = v2_ftmw_exp.ftmw.get_fid()
    fid.proc["FidWindowFunction"] = 99
    with pytest.raises(ValueError):
        fid.ft()
