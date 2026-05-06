.. index::
   single: BCExperiment
   single: Python module; BCExperiment
   single: experiment; loading from disk

BCExperiment
============

``BCExperiment`` is the entry point for loading a Blackchirp experiment
from disk into Python. Pass it the path to a single experiment folder
(the directory containing ``version.csv``) or a path to the data-storage
root together with a numeric experiment ``num``; it resolves the on-disk
location, reads every per-experiment CSV present, and exposes each as a
pandas ``DataFrame`` attribute named after the file.

The constructor always reads ``version.csv``, ``header.csv``,
``objectives.csv``, ``log.csv``, and ``hardware.csv``. Optional files
(``clocks.csv``, ``auxdata.csv``, ``chirps.csv``, ``markers.csv``) are
loaded when present; the corresponding attribute is omitted (or set to
``None`` for ``clocks``) when the file is absent. A ``clocks.csv`` is
required for any experiment that contains an ``fid/`` subdirectory and
its absence raises ``FileNotFoundError`` at construction time.

If the experiment folder contains an ``fid/`` subdirectory, a
:class:`~blackchirp.BCFTMW` is constructed and exposed as ``ftmw``. If it
contains a ``lif/`` subdirectory, a :class:`~blackchirp.BCLIF` is
constructed and exposed as ``lif``. Both attributes are present on a
combined CP-FTMW + LIF experiment; either may be present alone.

The header table — by far the largest and most useful of the loaded
DataFrames — is queried through the ``header_unique_keys``,
``header_rows``, ``header_value``, and ``header_unit`` helpers. The
on-disk schema is described in detail on the
:doc:`Data Storage </user_guide/data_storage>` user-guide page; the
helpers there are direct counterparts to the column meanings documented
on that page.

API Reference
-------------

.. autoclass:: blackchirp.BCExperiment
   :members:
