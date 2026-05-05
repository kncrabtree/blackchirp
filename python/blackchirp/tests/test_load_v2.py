"""Smoke-test the v2 FTMW fixture, plus v2-specific behaviour."""

from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd
import pytest

from blackchirp import BCExperiment, BCFTMW, BCFid


def test_loads(v2_ftmw_exp):
    assert isinstance(v2_ftmw_exp, BCExperiment)
    assert isinstance(v2_ftmw_exp.ftmw, BCFTMW)


def test_markers_present(v2_ftmw_exp):
    assert hasattr(v2_ftmw_exp, "markers")
    df = v2_ftmw_exp.markers
    assert isinstance(df, pd.DataFrame)
    assert {
        "Channel",
        "Name",
        "Role",
        "TimingMode",
        "StartUs",
        "EndUs",
        "Enabled",
    } <= set(df.columns)


def test_hardware_driver_column(v2_ftmw_exp):
    assert "driver" in v2_ftmw_exp.hardware.columns
    assert "subKey" not in v2_ftmw_exp.hardware.columns


def test_blackmanharris_window_string(v2_ftmw_exp):
    # v2-ftmw fixture has FidWindowFunction=BlackmanHarris in processing.csv;
    # FT must resolve it without falling through to boxcar.
    fid = v2_ftmw_exp.ftmw.get_fid()
    x_bh, y_bh = fid.ft()
    x_box, y_box = fid.ft(winf="None")
    # The two FTs must differ — if BlackmanHarris had silently fallen
    # through to boxcar these would have been identical.
    assert not np.allclose(y_bh, y_box)


def test_subkey_legacy_header_loads(tmp_path, v2_ftmw_path):
    """Hardware reader accepts the legacy ``subKey`` column header."""
    dst = tmp_path / "v2-ftmw-legacy"
    shutil.copytree(v2_ftmw_path, dst)
    hw_path = dst / "hardware.csv"
    text = hw_path.read_text()
    text = text.replace("key;driver", "key;subKey", 1)
    hw_path.write_text(text)

    exp = BCExperiment(str(dst))
    assert "driver" in exp.hardware.columns
    assert "subKey" not in exp.hardware.columns


def test_process_sideband_lower(v2_ftmw_exp):
    # v2-ftmw fidparams report sideband=LowerSideband; process_sideband
    # must dispatch correctly via is_lower_sideband().
    x, y = v2_ftmw_exp.ftmw.process_sideband(which="lower")
    assert x.ndim == 1
    assert np.all(np.isfinite(y))
