"""Internal helpers for enum-cell parsing.

Blackchirp CSV files may store an enum value either as a Q_ENUM name
(``"BlackmanHarris"``) or as the underlying integer (``"3"``). The
on-disk representation has changed across versions, so the Python
module accepts both forms. ``_resolve_enum`` centralises the lookup
so that every call site uses the same dual-form parse.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np


def _resolve_enum(
    value,
    name_map: Mapping[str, object],
    *,
    int_map: Optional[Mapping[int, str]] = None,
    default: Optional[str] = None,
) -> Optional[str]:
    """Map a CSV cell value to a canonical enum name.

    The cell may already be the canonical name (``"BlackmanHarris"``),
    a string that parses as an integer (``"3"``), or an actual integer.
    Integer codes delivered as numpy scalars (as happens when pandas
    upcasts a mixed-dtype row to ``float64``) are also accepted.
    ``name_map`` is the canonical name → payload mapping the caller
    cares about; ``_resolve_enum`` only inspects its keys.

    Args:
        value: Raw CSV cell value (``str``, ``int``, a numpy integer or
            integral-valued float scalar, or already an enum name).
        name_map: Mapping whose keys are the canonical enum names.
            The values are unused by this helper but typically carry
            payload (e.g. a scipy window spec) the caller will look
            up after resolution.
        int_map: Optional integer → name mapping used when ``value`` is
            (or parses as) an integer. Required when integer-form
            inputs must be accepted.
        default: Returned when ``value`` is ``None`` or the empty
            string. ``None`` (the default) means "no fallback — raise".

    Returns:
        The canonical name (a key of ``name_map``).

    Raises:
        ValueError: If ``value`` cannot be resolved to a key in
            ``name_map``.
    """

    if value is None or value == "":
        if default is not None:
            return default
        raise ValueError("Empty enum value with no default supplied")

    # A pandas row spanning mixed-dtype columns is upcast to a single
    # float64 Series, so an integer-coded enum cell arrives here as a
    # numpy scalar (e.g. numpy.float64(1.0)) rather than a Python int.
    # Coerce to the underlying Python scalar, then fold an integral
    # float back to int so it flows through the integer-code path.
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        value = int(value)

    if isinstance(value, str):
        s = value.strip()
        if s in name_map:
            return s
        try:
            i = int(s)
        except ValueError:
            raise ValueError(
                f"Unknown enum value {value!r} (expected one of {list(name_map.keys())})"
            ) from None
        if int_map is None:
            raise ValueError(
                f"Integer-form enum value {value!r} given but no integer map provided"
            )
        if i not in int_map:
            raise ValueError(
                f"Unknown enum integer {i} (expected one of {list(int_map.keys())})"
            )
        return int_map[i]

    if isinstance(value, (int,)) and not isinstance(value, bool):
        if int_map is None:
            raise ValueError(
                f"Integer-form enum value {value!r} given but no integer map provided"
            )
        if value not in int_map:
            raise ValueError(
                f"Unknown enum integer {value} (expected one of {list(int_map.keys())})"
            )
        return int_map[value]

    raise ValueError(f"Unsupported enum value type: {type(value).__name__}")
