.. index::
   single: FtWorker
   single: FFT; FtWorker
   single: FidProcessingSettings
   single: FtWindowFunction
   single: FtUnits
   single: DeconvolutionMethod
   single: SidebandProcessingData
   single: FilterResult
   single: LO scan; sideband co-averaging

FtWorker
========

``FtWorker`` is the analysis-pipeline class that converts :cpp:class:`Fid`
objects into :cpp:class:`Ft` magnitude spectra. Three slot methods cover
the three usage patterns: ``doFT()`` for a single FT, ``doFtDiff()`` for
the difference between two spectra, and ``processSideband()`` for the
incremental LO-scan stitching that builds a broadband spectrum from a
series of per-LO-step FTs. Each slot is invoked through
``QtConcurrent::run`` rather than via a dedicated ``QThread``; results
are returned both by signal (``ftDone``, ``ftDiffDone``,
``sidebandDone``) and as the slot's return value, so the same instance
serves both the asynchronous UI path (``FtmwViewWidget``) and the
synchronous batch path (``BatchManager``).

The transform itself uses the GSL mixed-radix real FFT, which is most
efficient when the FID length factors into powers of 2, 3, and 5. The
frequency-domain plot, peak finder, and overlay system in both
blackchirp and the blackchirp-viewer consume the resulting
:cpp:class:`Ft` via ``ExperimentViewWidget``; the user-facing controls
that drive ``FidProcessingSettings`` are described in
:doc:`/user_guide/data_storage` and the FT-domain UI documentation.

.. highlight:: cpp

Preprocessing pipeline
----------------------

Before the FFT, ``FtWorker`` applies an optional pipeline of
time-domain steps driven by ``FidProcessingSettings``:

- **Truncation** to a sub-window via ``startUs`` / ``endUs``.
- **DC removal** (subtract the FID mean).
- **Exponential apodization** (``expFilter``).
- **Zero-padding** (``zeroPadFactor``).
- **Window-function apodization** — one of ``FtWindowFunction``:
  Bartlett, Blackman, Blackman-Harris, Hamming, Hanning,
  Kaiser-Bessel.

The magnitude scale of the output :cpp:class:`Ft` is controlled by
``FidProcessingSettings::units``. The enumeration values map directly
to powers of ten relative to the digitized voltage: e.g. ``FtmV = 3``
multiplies by ``10³ / rawSize``, so the spectrum reads in mV per FFT
bin. The window-function cache is keyed on ``(window, length)`` and
guarded by ``pu_winfLock`` so repeated FFTs at the same size pay the
window cost only once.

Threading
---------

``FtWorker`` is a ``QObject`` but is *not* moved to a dedicated
``QThread``. Callers invoke ``doFT()``, ``doFtDiff()``, and
``processSideband()`` through ``QtConcurrent::run``, which schedules
each call on the global thread pool. The signals ``ftDone``,
``fidDone``, ``ftDiffDone``, and ``sidebandDone`` are emitted from the
thread-pool thread, so connections to UI slots require queued delivery
(the default when the receiver lives on a different thread). When
called from a non-UI thread (e.g. a :cpp:class:`BatchManager`), the
return values of ``doFT()`` and ``filterFid()`` can be used directly
without waiting for the signals. The ``id`` parameter of ``doFT()``
selects the path: pass ``id >= 0`` to get signal delivery
(asynchronous) or ``id = -1`` to suppress signals (synchronous).

GSL workspace allocation is guarded by ``pu_fftLock`` and spline
allocation by ``pu_splineLock``, so the same ``FtWorker`` instance can
be called concurrently from multiple thread-pool threads provided each
concurrent call uses a different code path.

Resource management
-------------------

GSL wavetables, workspaces, and spline objects are allocated lazily on
first use and freed by the destructor. When idle-cleanup is enabled
via ``setIdleCleanupEnabled()``, resources are also freed automatically
after a 5-minute inactivity timeout (``cleanupResources()``). Call
``resetIdleTimer()`` after each processing operation to restart the
countdown.

API Reference
-------------

.. doxygenclass:: FtWorker
   :members:
   :protected-members:
   :undoc-members:
