"""Exercise every sideband form through ``apply_lo`` / ``process_sideband``."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("value", [0, 1, "UpperSideband", "LowerSideband"])
def test_apply_lo(v2_ftmw_exp, value):
    fid = v2_ftmw_exp.ftmw.get_fid()
    fid.fidparams.loc["sideband"] = value
    f = np.array([100.0, 200.0])
    out = fid.apply_lo(f)
    assert out.shape == f.shape
    if value in (1, "LowerSideband"):
        assert fid.is_lower_sideband() is True
        np.testing.assert_allclose(out, fid.fidparams.probefreq - f)
    else:
        assert fid.is_lower_sideband() is False
        np.testing.assert_allclose(out, fid.fidparams.probefreq + f)


def test_apply_lo_unknown_sideband_raises(v2_ftmw_exp):
    fid = v2_ftmw_exp.ftmw.get_fid()
    fid.fidparams.loc["sideband"] = "NotASideband"
    with pytest.raises(ValueError):
        fid.apply_lo(np.array([100.0]))


@pytest.mark.parametrize("which", ["upper", "lower", "both"])
@pytest.mark.parametrize("avg", ["harmonic", "geometric"])
def test_process_sideband_branches(v2_ftmw_exp, which, avg):
    x, y = v2_ftmw_exp.ftmw.process_sideband(which=which, avg=avg)
    assert x.ndim == 1
    assert np.all(np.isfinite(y))


def test_process_sideband_invalid_which(v2_ftmw_exp):
    with pytest.raises(ValueError):
        v2_ftmw_exp.ftmw.process_sideband(which="bogus")


def test_process_sideband_invalid_avg(v2_ftmw_exp):
    with pytest.raises(ValueError):
        v2_ftmw_exp.ftmw.process_sideband(avg="bogus")
