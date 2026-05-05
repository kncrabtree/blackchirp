"""Exercise every ``FtUnits`` form through ``BCFid.ft``."""

from __future__ import annotations

import numpy as np
import pytest

from blackchirp.bcfid import _FT_UNITS_INT_MAP, _FT_UNITS_MAP


@pytest.mark.parametrize("name,exponent", list(_FT_UNITS_MAP.items()))
def test_ftunits_string_in_proc(v2_ftmw_exp, name, exponent):
    fid = v2_ftmw_exp.ftmw.get_fid()
    fid.proc["FtUnits"] = name
    _, y_named = fid.ft()
    fid2 = v2_ftmw_exp.ftmw.get_fid()
    fid2.proc["FtUnits"] = "FtV"
    _, y_base = fid2.ft()
    np.testing.assert_allclose(y_named, y_base * (10**exponent))


@pytest.mark.parametrize("intval,name", list(_FT_UNITS_INT_MAP.items()))
def test_ftunits_int_in_proc(v2_ftmw_exp, intval, name):
    fid = v2_ftmw_exp.ftmw.get_fid()
    fid.proc["FtUnits"] = intval
    _, y_int = fid.ft()
    fid2 = v2_ftmw_exp.ftmw.get_fid()
    fid2.proc["FtUnits"] = name
    _, y_name = fid2.ft()
    np.testing.assert_allclose(y_int, y_name)


def test_ftunits_int_string_in_proc(v2_ftmw_exp):
    fid = v2_ftmw_exp.ftmw.get_fid()
    fid.proc["FtUnits"] = "6"
    _, y_str = fid.ft()
    fid2 = v2_ftmw_exp.ftmw.get_fid()
    fid2.proc["FtUnits"] = "FtuV"
    _, y_name = fid2.ft()
    np.testing.assert_allclose(y_str, y_name)


def test_ftunits_unknown_raises(v2_ftmw_exp):
    fid = v2_ftmw_exp.ftmw.get_fid()
    fid.proc["FtUnits"] = "NotARealUnit"
    with pytest.raises(ValueError):
        fid.ft()


def test_ftunits_kwarg_overrides_proc(v2_ftmw_exp):
    fid = v2_ftmw_exp.ftmw.get_fid()
    fid.proc["FtUnits"] = "FtuV"
    _, y_kwarg = fid.ft(units_power=0)
    fid2 = v2_ftmw_exp.ftmw.get_fid()
    fid2.proc["FtUnits"] = "FtV"
    _, y_base = fid2.ft()
    np.testing.assert_allclose(y_kwarg, y_base)
