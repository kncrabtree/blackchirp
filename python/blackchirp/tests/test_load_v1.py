"""Smoke-test the v1 ``mtbe`` fixture through every public method."""

from __future__ import annotations

import numpy as np
import pandas as pd

from blackchirp import BCExperiment, BCFTMW, BCFid


def test_loads(mtbe_exp):
    assert isinstance(mtbe_exp, BCExperiment)
    assert mtbe_exp.num >= 0
    assert isinstance(mtbe_exp.header, pd.DataFrame)
    assert isinstance(mtbe_exp.objectives, pd.DataFrame)
    assert isinstance(mtbe_exp.hardware, pd.DataFrame)
    assert isinstance(mtbe_exp.clocks, pd.DataFrame)
    assert isinstance(mtbe_exp.log, pd.DataFrame)
    assert isinstance(mtbe_exp.version, pd.DataFrame)


def test_hardware_normalised_to_driver(mtbe_exp):
    assert "driver" in mtbe_exp.hardware.columns
    assert "subKey" not in mtbe_exp.hardware.columns


def test_no_markers_attribute(mtbe_exp):
    # The mtbe fixture has no markers.csv; the attribute should be absent
    # rather than set to an empty DataFrame.
    assert not hasattr(mtbe_exp, "markers")


def test_header_helpers(mtbe_exp):
    keys = mtbe_exp.header_unique_keys()
    assert isinstance(keys, set)
    assert len(keys) > 0

    rows = mtbe_exp.header_rows("Experiment", "Number")
    assert isinstance(rows, pd.DataFrame)
    assert not rows.empty

    val = mtbe_exp.header_value("Experiment", "Number")
    assert val != ""

    # header_unit returns either a string (possibly empty) — exercise the path
    unit = mtbe_exp.header_unit("Experiment", "Number")
    assert isinstance(unit, str)


def test_ftmw_present(mtbe_exp):
    assert isinstance(mtbe_exp.ftmw, BCFTMW)
    assert mtbe_exp.ftmw.numfids >= 1
    assert mtbe_exp.ftmw.is_multi_segment() is False
    assert mtbe_exp.ftmw.num_backups() == mtbe_exp.ftmw.numfids - 1


def test_get_fid_default(mtbe_exp):
    fid = mtbe_exp.ftmw.get_fid()
    assert isinstance(fid, BCFid)
    t = fid.x()
    assert t.ndim == 1
    assert len(t) == int(fid.fidparams["size"])
    tt, dd = fid.xy()
    assert tt.shape == (len(t),)
    assert dd.shape[0] == len(t)


def test_apply_lo_and_sideband(mtbe_exp):
    fid = mtbe_exp.ftmw.get_fid()
    f = np.array([10.0, 20.0, 30.0])
    out = fid.apply_lo(f)
    assert out.shape == f.shape
    assert isinstance(fid.is_lower_sideband(), bool)


def test_default_ft(mtbe_exp):
    fid = mtbe_exp.ftmw.get_fid()
    x, y = fid.ft()
    assert x.ndim == 1
    assert np.all(np.isfinite(y))
    assert y.shape[0] == x.shape[0]
