"""Verify the error-handling sweep: missing-key paths raise rather than return ''."""

from __future__ import annotations

import os
import shutil

import pytest

from blackchirp import BCExperiment


def test_header_value_missing_raises(v2_ftmw_exp):
    with pytest.raises(KeyError):
        v2_ftmw_exp.header_value("NoSuchObject", "NoSuchKey")


def test_header_unit_missing_row_raises(v2_ftmw_exp):
    with pytest.raises(KeyError):
        v2_ftmw_exp.header_unit("NoSuchObject", "NoSuchKey")


def test_header_value_idx_past_end_raises(v2_ftmw_exp):
    with pytest.raises(KeyError):
        v2_ftmw_exp.header_value("Experiment", "Number", idx=99)


def test_header_unit_present_row_no_unit_returns_empty(v2_ftmw_exp):
    """Row exists but has no unit cell — sentinel return is intentional."""
    rows = v2_ftmw_exp.header_rows()
    no_unit = rows[rows.Units == ""]
    if no_unit.empty:
        pytest.skip("fixture has no rows with empty Units")
    r = no_unit.iloc[0]
    out = v2_ftmw_exp.header_unit(r.ObjKey, r.ValueKey)
    assert out == ""


def test_header_rows_no_match_returns_empty_df(v2_ftmw_exp):
    """Empty DataFrame on no-match is intentional, not an error."""
    df = v2_ftmw_exp.header_rows("NoSuchObject")
    assert df.empty


def test_get_fid_out_of_range_raises(v2_ftmw_exp):
    with pytest.raises(ValueError):
        v2_ftmw_exp.ftmw.get_fid(num=999)


def test_get_fid_negative_raises(v2_ftmw_exp):
    with pytest.raises(ValueError):
        v2_ftmw_exp.ftmw.get_fid(num=-1)


def test_missing_experiment_number_raises(tmp_path, v2_ftmw_path):
    """Constructor surfaces a clear KeyError when header.csv is incomplete."""
    dst = tmp_path / "v2-ftmw-no-num"
    shutil.copytree(v2_ftmw_path, dst)
    hp = dst / "header.csv"
    lines = hp.read_text().splitlines()
    filtered = [ln for ln in lines if not ln.startswith("Experiment;;;Number;")]
    hp.write_text("\n".join(filtered) + "\n")
    with pytest.raises(KeyError):
        BCExperiment(str(dst))


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        BCExperiment(str(tmp_path / "does-not-exist"))
