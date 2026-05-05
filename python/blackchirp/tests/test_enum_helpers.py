"""Pure-unit tests for ``_resolve_enum``."""

from __future__ import annotations

import pytest

from blackchirp._enum_helpers import _resolve_enum

_NAMES = {"Foo": 1, "Bar": 2, "Baz": 3}
_INTS = {0: "Foo", 1: "Bar", 2: "Baz"}


def test_name_passthrough():
    assert _resolve_enum("Foo", _NAMES) == "Foo"


def test_int_string_with_int_map():
    assert _resolve_enum("1", _NAMES, int_map=_INTS) == "Bar"


def test_int_with_int_map():
    assert _resolve_enum(2, _NAMES, int_map=_INTS) == "Baz"


def test_default_used_for_empty_string():
    assert _resolve_enum("", _NAMES, default="Foo") == "Foo"


def test_default_used_for_none():
    assert _resolve_enum(None, _NAMES, default="Bar") == "Bar"


def test_empty_no_default_raises():
    with pytest.raises(ValueError):
        _resolve_enum("", _NAMES)


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        _resolve_enum("NotARealKey", _NAMES)


def test_int_without_int_map_raises():
    with pytest.raises(ValueError):
        _resolve_enum(1, _NAMES)


def test_int_string_without_int_map_raises():
    with pytest.raises(ValueError):
        _resolve_enum("1", _NAMES)


def test_int_out_of_range_raises():
    with pytest.raises(ValueError):
        _resolve_enum(99, _NAMES, int_map=_INTS)


def test_unsupported_type_raises():
    with pytest.raises(ValueError):
        _resolve_enum(1.5, _NAMES, int_map=_INTS)
