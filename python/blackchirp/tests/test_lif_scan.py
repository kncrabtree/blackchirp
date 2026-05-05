"""Tests for :class:`blackchirp.BCLIF` scan-level helpers.

The bundled fixtures are sparse 6×6 scans, so the aggregating helpers
are guaranteed to encounter missing scan points at every present
laser index.
"""

from __future__ import annotations

import numpy as np


def test_numtraces_matches_lifparams(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    assert lif.numtraces == len(lif.lifparams)


def test_has_ref_flag(v2_lif_ref_exp, v2_lif_noref_exp):
    assert v2_lif_ref_exp.lif.has_ref is True
    assert v2_lif_noref_exp.lif.has_ref is False


def test_delay_axis_shape_and_units(v2_lif_ref_exp):
    arr, units = v2_lif_ref_exp.lif.delay_axis()
    assert arr.shape == (v2_lif_ref_exp.lif.delay_points,)
    assert np.issubdtype(arr.dtype, np.floating)
    assert isinstance(units, str)
    assert units == "μs"


def test_laser_axis_shape_and_units(v2_lif_ref_exp):
    arr, units = v2_lif_ref_exp.lif.laser_axis()
    assert arr.shape == (v2_lif_ref_exp.lif.laser_points,)
    assert np.issubdtype(arr.dtype, np.floating)
    assert units == "nm"


def test_image_shape(v2_lif_ref_exp):
    delays, lasers, img = v2_lif_ref_exp.lif.image()
    lif = v2_lif_ref_exp.lif
    assert delays.shape == (lif.delay_points,)
    assert lasers.shape == (lif.laser_points,)
    assert img.shape == (lif.delay_points, lif.laser_points)


def test_image_nan_at_missing_points(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    _, _, img = lif.image()
    present = lif._index  # pylint: disable=protected-access
    for d in range(lif.delay_points):
        for l in range(lif.laser_points):
            if (l, d) in present:
                assert np.isfinite(img[d, l])
            else:
                assert np.isnan(img[d, l])


def test_image_fill_zero_substitutes_zero(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    _, _, img_nan = lif.image()
    _, _, img_zero = lif.image(fill=0.0)
    nan_mask = np.isnan(img_nan)
    assert nan_mask.any()
    assert np.all(img_zero[nan_mask] == 0.0)
    finite_mask = ~nan_mask
    np.testing.assert_allclose(img_nan[finite_mask], img_zero[finite_mask])


def test_delay_slice_present_and_missing(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    delays, vals = lif.delay_slice(0)
    assert delays.shape == (lif.delay_points,)
    assert vals.shape == (lif.delay_points,)
    for d in range(lif.delay_points):
        if (0, d) in lif._index:  # pylint: disable=protected-access
            assert np.isfinite(vals[d])
        else:
            assert np.isnan(vals[d])


def test_delay_slice_fill_zero(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    _, nan_vals = lif.delay_slice(2)
    _, zero_vals = lif.delay_slice(2, fill=0.0)
    nan_mask = np.isnan(nan_vals)
    assert nan_mask.any()
    assert np.all(zero_vals[nan_mask] == 0.0)


def test_laser_slice_present_and_missing(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    lasers, vals = lif.laser_slice(0)
    assert lasers.shape == (lif.laser_points,)
    assert vals.shape == (lif.laser_points,)
    for l in range(lif.laser_points):
        if (l, 0) in lif._index:  # pylint: disable=protected-access
            assert np.isfinite(vals[l])
        else:
            assert np.isnan(vals[l])


def test_laser_slice_fill_zero(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    _, nan_vals = lif.laser_slice(0)
    _, zero_vals = lif.laser_slice(0, fill=0.0)
    nan_mask = np.isnan(nan_vals)
    assert nan_mask.any()
    assert np.all(zero_vals[nan_mask] == 0.0)


def test_noref_image_shape(v2_lif_noref_exp):
    lif = v2_lif_noref_exp.lif
    delays, lasers, img = lif.image()
    assert delays.shape == (lif.delay_points,)
    assert lasers.shape == (lif.laser_points,)
    assert img.shape == (lif.delay_points, lif.laser_points)
