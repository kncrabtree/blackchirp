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
on the :doc:`LIF Data Storage </user_guide/data_storage/lif>`
user-guide page.

When the experiment's LIF setup runs the tunable laser through one or
more optical conversion stages — a doubling crystal, a mixing crystal —
before the beam reaches the sample, ``BCLIF`` also exposes the recorded
frequency-conversion topology, read from ``liftopology.csv`` via
``BCExperiment.liftopology`` (``None`` for a bare laser with no
conversion stages); see the file's column reference on the
:doc:`LIF Data Storage </user_guide/data_storage/lif>` page. The
boolean ``has_topology`` reports whether a topology was recorded;
``stages``, ``final_stage``, and ``laser_key`` name every stage in it,
the stage whose output is the excitation beam reaching the sample, and
the laser the topology was built around.

Three accessors translate between the laser's tuning value and any
beam in the topology, in any of ``cm-1`` / ``nm`` / ``GHz`` / ``eV``:
``fundamental`` answers "what laser setting produced this beam value?";
``at_stage`` is the inverse, answering "what beam value does this laser
setting produce?"; and ``stage_frequencies`` returns a table of every
beam in the topology for a single laser setting or excitation-beam
value. All three fall back to the identity relationship (the laser
setting *is* the excitation-beam value) when ``has_topology`` is
``False``.

API Reference
-------------

.. autoclass:: blackchirp.BCLIF
   :members:
