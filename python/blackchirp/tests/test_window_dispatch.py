"""Pure-unit tests for the ``_WINDOW_MAP`` / ``_WINDOW_INT_MAP`` lookups."""

from __future__ import annotations

import pytest

from blackchirp.bcfid import (
    _FT_UNITS_INT_MAP,
    _FT_UNITS_MAP,
    _WINDOW_INT_MAP,
    _WINDOW_MAP,
)


def test_window_map_keys_match():
    """Every integer in _WINDOW_INT_MAP maps to a key in _WINDOW_MAP."""
    for name in _WINDOW_INT_MAP.values():
        assert name in _WINDOW_MAP


def test_window_int_map_dense():
    """Integer keys cover the contiguous range observed in C++."""
    keys = sorted(_WINDOW_INT_MAP.keys())
    assert keys == list(range(min(keys), max(keys) + 1))


def test_ft_units_round_trip():
    for name, exp in _FT_UNITS_MAP.items():
        assert _FT_UNITS_INT_MAP[exp] == name


def test_window_payload_types():
    """Each window payload is a string or a (str, ...) tuple — what scipy accepts."""
    for spec in _WINDOW_MAP.values():
        assert isinstance(spec, str) or (
            isinstance(spec, tuple) and isinstance(spec[0], str)
        )
