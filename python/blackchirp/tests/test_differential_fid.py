"""Tests for ``BCFTMW.get_differential_fid`` and related helpers."""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from blackchirp import BCExperiment, BCFid


def _shots(fp, idx):
    return int(fp.iloc[idx].shots)


def test_defaults_match_cumulative(any_exp):
    diff = any_exp.ftmw.get_differential_fid()
    assert isinstance(diff, BCFid)
    expected_shots = _shots(any_exp.ftmw.fidparams, 0)
    assert int(diff.fidparams.shots) == expected_shots
    assert diff.shots == expected_shots


def test_explicit_full_range_matches_cumulative(any_exp):
    diff = any_exp.ftmw.get_differential_fid(start=0, end=-1)
    assert int(diff.fidparams.shots) == _shots(any_exp.ftmw.fidparams, 0)


def test_start_one_end_default(any_exp):
    fp = any_exp.ftmw.fidparams
    nb = any_exp.ftmw.num_backups()
    if nb < 1:
        pytest.skip("fixture has no backups")
    diff = any_exp.ftmw.get_differential_fid(start=1)
    expected = _shots(fp, 0) - _shots(fp, 1)
    assert int(diff.fidparams.shots) == expected


def test_start_zero_end_one(any_exp):
    nb = any_exp.ftmw.num_backups()
    if nb < 1:
        pytest.skip("fixture has no backups")
    diff = any_exp.ftmw.get_differential_fid(start=0, end=1)
    expected = _shots(any_exp.ftmw.fidparams, 1)
    assert int(diff.fidparams.shots) == expected


def test_two_bound_internal(any_exp):
    nb = any_exp.ftmw.num_backups()
    if nb < 2:
        pytest.skip("fixture has fewer than two backups")
    fp = any_exp.ftmw.fidparams
    diff = any_exp.ftmw.get_differential_fid(start=1, end=2)
    expected = _shots(fp, 2) - _shots(fp, 1)
    assert int(diff.fidparams.shots) == expected


def test_all_backup_pairs_arithmetic(any_exp):
    """Across every backup, shot counts subtract exactly."""
    nb = any_exp.ftmw.num_backups()
    fp = any_exp.ftmw.fidparams
    for k in range(1, nb + 1):
        diff = any_exp.ftmw.get_differential_fid(start=k, end=-1)
        assert int(diff.fidparams.shots) == _shots(fp, 0) - _shots(fp, k)


def test_ft_runs_on_differential(any_exp):
    nb = any_exp.ftmw.num_backups()
    if nb < 1:
        pytest.skip("fixture has no backups")
    diff = any_exp.ftmw.get_differential_fid(start=1)
    x, y = diff.ft()
    assert x.ndim == 1
    assert np.all(np.isfinite(y))


def test_start_out_of_range(any_exp):
    nb = any_exp.ftmw.num_backups()
    with pytest.raises(ValueError):
        any_exp.ftmw.get_differential_fid(start=nb + 1)


def test_negative_start(any_exp):
    with pytest.raises(ValueError):
        any_exp.ftmw.get_differential_fid(start=-1)


def test_end_out_of_range(any_exp):
    nb = any_exp.ftmw.num_backups()
    with pytest.raises(ValueError):
        any_exp.ftmw.get_differential_fid(start=0, end=nb + 5)


def test_end_zero_invalid(any_exp):
    """end=0 is not a valid upper bound; only -1 (final) or [1, nb] is."""
    with pytest.raises(ValueError):
        any_exp.ftmw.get_differential_fid(start=0, end=0)


def test_start_equals_end(any_exp):
    nb = any_exp.ftmw.num_backups()
    if nb < 1:
        pytest.skip("fixture has no backups")
    with pytest.raises(ValueError):
        any_exp.ftmw.get_differential_fid(start=1, end=1)


def test_start_greater_than_end(any_exp):
    nb = any_exp.ftmw.num_backups()
    if nb < 2:
        pytest.skip("fixture has fewer than two backups")
    with pytest.raises(ValueError):
        any_exp.ftmw.get_differential_fid(start=2, end=1)


def test_multi_segment_refused(tmp_path, v2_ftmw_path):
    """Stub a fixture as ``LO_Scan`` and assert the API refuses."""
    dst = tmp_path / "v2-ftmw-loscan"
    shutil.copytree(v2_ftmw_path, dst)
    obj_path = dst / "objectives.csv"
    text = obj_path.read_text().replace("FtmwType;Forever", "FtmwType;LO_Scan")
    obj_path.write_text(text)

    exp = BCExperiment(str(dst))
    assert exp.ftmw.is_multi_segment() is True
    assert exp.ftmw.num_backups() == 0
    with pytest.raises(ValueError):
        exp.ftmw.get_differential_fid()


def test_differential_rawdata_subtracts(any_exp):
    """Sanity check: the differential raw data really is the subtraction."""
    nb = any_exp.ftmw.num_backups()
    if nb < 1:
        pytest.skip("fixture has no backups")
    end_fid = any_exp.ftmw.get_fid(0)
    start_fid = any_exp.ftmw.get_fid(1)
    diff = any_exp.ftmw.get_differential_fid(start=1, end=-1)
    expected_raw = end_fid._rawdata - start_fid._rawdata
    np.testing.assert_array_equal(diff._rawdata, expected_raw)
