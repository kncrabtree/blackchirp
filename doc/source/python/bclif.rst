.. index::
   single: BCLIF
   single: Python module; BCLIF
   single: LIF; Python container
   single: laser-induced fluorescence; Python module

BCLIF
=====

``BCLIF`` is the LIF-scan container exposed by
:class:`~blackchirp.BCExperiment` as ``exp.lif`` whenever the experiment
folder contains a ``lif/`` subdirectory. On construction it reads
``lif/lifparams.csv`` and ``lif/processing.csv`` and pulls the
``DelayPoints / DelayStart / DelayStep`` and
``LaserPoints / LaserStart / LaserStep`` rows out of the experiment
header so the full delay × laser scan grid is known up-front. Per-point
trace files are read lazily through ``get_trace``; opening an LIF
experiment is therefore cheap regardless of grid size.

The class provides three aggregating helpers — ``delay_slice``,
``laser_slice``, and ``image`` — that integrate every present scan point
with one processing-override surface, returning numpy arrays sized
against the full scan axes. Scan grids are routinely incomplete:
acquisitions can stop partway through, leaving some ``(lIndex, dIndex)``
positions with no trace file. Those positions are filled with ``np.nan``
by default; pass ``fill=0.0`` (or any other numeric value) when a
zero-baseline image is preferred. The ``has_ref`` attribute reports
whether *any* point in the scan recorded a reference channel, derived
from the ``refsize`` column of ``lifparams.csv``.

The on-disk format that ``BCLIF`` reads — the ``lif/`` subdirectory
layout, the meaning of each ``lifparams.csv`` column, the
``processing.csv`` integration-gate settings — is documented in detail
on the :doc:`LIF Data Storage </user_guide/lif/data_storage>`
user-guide page.

API Reference
-------------

.. autoclass:: blackchirp.BCLIF
   :members:
