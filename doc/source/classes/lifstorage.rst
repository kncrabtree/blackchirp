.. index::
   single: LifStorage
   single: LIF; storage
   single: LIF; delay/laser grid
   single: LifTrace; storage
   single: BC::Key::LifStorage

LifStorage
==========

``LifStorage`` extends :doc:`datastoragebase` to manage
:cpp:class:`LifTrace` waveforms for Laser-Induced Fluorescence (LIF)
acquisitions.  The scan grid is fixed at construction: ``d_delayPoints``
delay steps and ``d_laserPoints`` laser positions.  Each cell
``(di, li)`` maps to the flat index ``di * d_laserPoints + li``, which
is used as the stem for the per-cell CSV files written under the
``lif/`` subdirectory of the experiment directory.

``LifStorage`` is owned by :cpp:class:`LifConfig` alongside the
digitizer configuration.  The LIF experiment setup and data-viewing
workflow are described in :doc:`/user_guide/lif`.

Acquisition lifecycle
---------------------

``start()`` arms the storage; ``finish()`` disarms it.  As the
acquisition sweeps the grid, the calling code passes each incoming
waveform to ``addTrace()``, which accumulates it into ``d_currentTrace``
(seeding from the completed-cell map if the cell has been partially
accumulated in a previous sweep).  At each grid-step boundary,
``advance()`` sets the ``d_nextNew`` flag and flushes the current cell
to disk via ``save()``.  ``save()`` also rewrites the ``lif/lifparams.csv``
index file, which records the grid coordinates, shot count, and
calibration scalars for every completed cell.

Trace access
------------

``getLifTrace()`` returns the trace for cell ``(di, li)``.  When the
cell matches the one being accumulated it returns the in-memory
``d_currentTrace`` directly.  Otherwise it searches the completed-cell
map ``d_data``; if the experiment is no longer acquiring and the cell
is absent from memory, it falls back to ``loadLifTrace()`` for a disk
read.  ``currentLifTrace()`` returns ``d_currentTrace`` without
acquiring the mutex and is intended for use on the acquisition thread
only.  ``currentTraceShots()`` returns the shot count of the trace
being accumulated; ``completedShots()`` sums shot counts across all
completed cells (plus the current cell if it is in progress).

Processing settings
-------------------

``writeProcessingSettings()`` serializes a
``LifTrace::LifProcSettings`` struct to ``lif/processing.csv`` using
the keys in ``BC::Key::LifStorage``:

+------------------+------------------------------------------------------+
| Key              | Meaning                                              |
+==================+======================================================+
| ``lifGateStart`` | Start sample index of the LIF integration gate       |
+------------------+------------------------------------------------------+
| ``lifGateEnd``   | End sample index of the LIF integration gate         |
+------------------+------------------------------------------------------+
| ``refGateStart`` | Start sample index of the reference integration gate |
+------------------+------------------------------------------------------+
| ``refGateEnd``   | End sample index of the reference integration gate   |
+------------------+------------------------------------------------------+
| ``lowPassAlpha`` | Exponential low-pass smoothing factor (0 = disabled) |
+------------------+------------------------------------------------------+
| ``savGol``       | Savitzky-Golay filter enabled flag                   |
+------------------+------------------------------------------------------+
| ``sgWin``        | Savitzky-Golay window length (must be odd)           |
+------------------+------------------------------------------------------+
| ``sgPoly``       | Savitzky-Golay polynomial order                      |
+------------------+------------------------------------------------------+

``readProcessingSettings()`` reads the same file back into a
``LifProcSettings`` struct.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: LifStorage
   :members:
   :protected-members:
   :undoc-members:
