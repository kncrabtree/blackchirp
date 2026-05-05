"""``BCLIF.get_trace`` raises ``KeyError`` for sparse-scan holes."""

from __future__ import annotations

import pytest


def test_get_trace_missing_raises(v2_lif_ref_exp):
    lif = v2_lif_ref_exp.lif
    missing = None
    for d in range(lif.delay_points):
        for l in range(lif.laser_points):
            if (l, d) not in lif._index:  # pylint: disable=protected-access
                missing = (l, d)
                break
        if missing is not None:
            break
    assert missing is not None, "Sparse fixture should have at least one hole"
    with pytest.raises(KeyError):
        lif.get_trace(*missing)


def test_get_trace_present_succeeds(v2_lif_ref_exp):
    trace = v2_lif_ref_exp.lif.get_trace(0, 0)
    assert trace.lifsize > 0
