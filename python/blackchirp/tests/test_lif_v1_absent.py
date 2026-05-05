"""v1 fixture (mtbe) has no LIF directory; ``BCExperiment.lif`` is absent."""

from __future__ import annotations


def test_mtbe_has_no_lif_attribute(mtbe_exp):
    assert not hasattr(mtbe_exp, "lif")
