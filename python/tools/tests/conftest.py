"""Test configuration for the Sirah FCU calibration tools' pytest suite.

Adds ``python/tools`` (the tools' own location, sibling of this
``tests`` directory) to ``sys.path`` so tests can ``import fcu_fit``
etc. directly — these are standalone operator scripts, not part of the
installed ``blackchirp`` package.
"""

from __future__ import annotations

import os
import sys

_TOOLS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
