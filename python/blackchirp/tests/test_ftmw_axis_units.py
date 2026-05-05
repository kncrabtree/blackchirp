"""Coverage for the time- and frequency-axis ``units`` kwargs on
``BCFid.x``, ``BCFid.xy``, ``BCFid.ft``, and
``BCFTMW.process_sideband``. The defaults preserve the historical
return units (seconds for time, MHz for frequency); other choices
rescale the axis array linearly without affecting intensities.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize(
    "units,scale",
    [("s", 1.0), ("ms", 1e3), ("us", 1e6), ("μs", 1e6), ("ns", 1e9)],
)
def test_fid_x_units(mtbe_exp, units, scale):
    fid = mtbe_exp.ftmw.get_fid(0)
    spacing = float(fid.fidparams.spacing)
    x = fid.x(units=units)
    assert x.shape == (int(fid.fidparams["size"]),)
    assert pytest.approx(x[1] - x[0]) == spacing * scale


def test_fid_x_units_invalid(mtbe_exp):
    fid = mtbe_exp.ftmw.get_fid(0)
    with pytest.raises(ValueError):
        fid.x(units="years")


def test_fid_xy_propagates_units(mtbe_exp):
    fid = mtbe_exp.ftmw.get_fid(0)
    spacing = float(fid.fidparams.spacing)
    x_us, _ = fid.xy(units="us")
    assert pytest.approx(x_us[1] - x_us[0]) == spacing * 1e6


@pytest.mark.parametrize(
    "freq_units,scale",
    [("Hz", 1e6), ("kHz", 1e3), ("MHz", 1.0), ("GHz", 1e-3), ("THz", 1e-6)],
)
def test_ft_freq_units(mtbe_exp, freq_units, scale):
    fid = mtbe_exp.ftmw.get_fid(0)
    x_mhz, y_mhz = fid.ft()
    x_other, y_other = fid.ft(freq_units=freq_units)
    np.testing.assert_allclose(x_other, x_mhz * scale, rtol=1e-12)
    np.testing.assert_allclose(y_other, y_mhz)


def test_ft_freq_units_invalid(mtbe_exp):
    fid = mtbe_exp.ftmw.get_fid(0)
    with pytest.raises(ValueError):
        fid.ft(freq_units="parsecs")


def test_process_sideband_freq_units(v2_ftmw_exp):
    ftmw = v2_ftmw_exp.ftmw
    if ftmw.numfids < 2 or not ftmw.is_multi_segment():
        pytest.skip("Sideband fixture requires multi-segment LO_Scan")
    x_mhz, y_mhz = ftmw.process_sideband(which="upper")
    x_ghz, y_ghz = ftmw.process_sideband(which="upper", freq_units="GHz")
    np.testing.assert_allclose(x_ghz, x_mhz * 1e-3, rtol=1e-12)
    np.testing.assert_allclose(y_ghz, y_mhz)


def test_process_sideband_freq_units_invalid(v2_ftmw_exp):
    ftmw = v2_ftmw_exp.ftmw
    if not ftmw.is_multi_segment():
        pytest.skip("Sideband fixture requires multi-segment LO_Scan")
    with pytest.raises(ValueError):
        ftmw.process_sideband(which="upper", freq_units="furlongs")
