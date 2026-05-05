"""End-to-end exercise of the no-reference LIF code path."""

from __future__ import annotations

import numpy as np


def test_noref_trace_skips_ref_branch(v2_lif_noref_exp):
    trace = v2_lif_noref_exp.lif.get_trace(0, 0)
    assert trace.ref() is None
    assert not trace.has_ref()
    direct = trace.integrate()
    with_ref_kwargs = trace.integrate(ref_start=100, ref_end=300)
    np.testing.assert_allclose(direct, with_ref_kwargs)


def test_noref_image_all_finite_at_present_points(v2_lif_noref_exp):
    lif = v2_lif_noref_exp.lif
    _, _, img = lif.image()
    for l, d in lif._index:  # pylint: disable=protected-access
        assert np.isfinite(img[d, l])


def test_noref_smooth_chain(v2_lif_noref_exp):
    trace = v2_lif_noref_exp.lif.get_trace(0, 0)
    out = trace.smooth(low_pass=True, savgol=True, low_pass_alpha=0.4)
    assert out.shape == (trace.lifsize,)
    assert np.all(np.isfinite(out))
