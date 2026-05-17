.. index::
   single: FTMW pipeline
   single: WaveformBuffer; producer/consumer
   single: FtmwDigitizer; emitShot
   single: pre-accumulation
   single: flush marker; segment boundary
   single: drain timer
   single: AcquisitionManager; drainFtmwBuffer
   single: QFutureWatcher; FtmwProcessingResult
   single: FtmwConfig; addBatchFids
   single: parseBatchParallel
   single: FidStorageBase; cache
   single: FidStorageBase; processing settings
   single: FtmwViewWidget; live update timer
   single: FtWorker; QtConcurrent dispatch
   single: PeakFinder

FTMW Acquisition and Visualization
==================================

The FTMW pipeline carries waveform data from the digitizer hardware
thread to the on-screen FT spectrum without ever pinning either of
those threads to the other's pace. The producer side is
:cpp:class:`FtmwDigitizer` running on its own ``"<hwKey>Thread"``; the
consumer side is :cpp:class:`AcquisitionManager` running on
``AcquisitionManagerThread``; the visualization side is
``FtmwViewWidget`` and ``FtWorker`` straddling the GUI thread and the
Qt thread pool. A bounded :cpp:class:`WaveformBuffer` decouples the
producer from the consumer; a per-segment ``FidStorageBase``
accumulator decouples the consumer from the renderer.

This page traces that pipeline end to end. The
:cpp:class:`WaveformBuffer` API, the
:ref:`AcquisitionManager state machine
<acquisitionmanager-state-machine>`, and the per-class
:cpp:class:`FtmwConfig`, :cpp:class:`FidStorageBase`, and
:cpp:class:`FtWorker` contracts each have their own pages — this one
covers the *flow* across them.

The end-to-end picture
----------------------

.. mermaid::

   flowchart LR
       A["FtmwDigitizer<br/>(digitizer thread)<br/>emitShot()"]
       B["WaveformBuffer<br/>(SPSC ring, drop-newest)"]
       C["AcquisitionManager<br/>(AM thread)<br/>drainFtmwBuffer()"]
       D["QtConcurrent worker<br/>(thread pool)<br/>addBatchFids()"]
       E["FidStorageBase<br/>(in-progress FidList<br/>+ segment cache)"]
       F["FtmwViewWidget<br/>(GUI thread)<br/>live-update timer"]
       G["FtWorker<br/>(thread pool)<br/>doFT()"]
       H["MainFtPlot, FidPlot<br/>(GUI thread)"]
       A --> B --> C --> D --> E
       E --> F --> G --> H

Why a ring buffer
-----------------

The pipeline uses a bounded SPSC ring buffer instead of per-shot Qt
signal emission for three reasons. At the throughput target
(~20 kFID/s with firmware block averaging) per-shot ``QMetaCallEvent``
allocation and event-loop dispatch — once from the digitizer to
:cpp:class:`HardwareManager` and once from there to
:cpp:class:`AcquisitionManager` — becomes the bottleneck before the
hardware does. The Qt event queue has no backpressure, so a slow
consumer causes unbounded memory growth. And the
:cpp:class:`HardwareManager` hop adds nothing to the data path; it
exists only because cross-thread access from the AM to the digitizer
object is otherwise awkward.

A bounded ring buffer with drop-newest overflow gives bounded memory,
race-free producer writes, and a natural fallback (pre-accumulation,
described below) when the consumer falls behind. See
:doc:`/classes/waveformbuffer` for the SPSC discipline, the overflow
counter, and the ``WaveformEntry`` struct layout.

Producer: FtmwDigitizer::emitShot
---------------------------------

Each :cpp:class:`FtmwDigitizer` subclass produces raw waveform bytes on
its hardware thread and calls the base-class
``FtmwDigitizer::emitShot(data)``. The base class handles four cases in
order:

#. **Acquisition is gated.** ``setAcquisitionGated(true)`` short-circuits
   ``emitShot``; the bytes are dropped silently. Gating engages at
   segment transitions (LO scan, DR scan) and disengages when the
   next segment's clocks are confirmed by the hardware.
#. **Discard countdown is active.** A small leading-shot discard
   counter swallows the first shot after gating releases, so the
   producer never publishes a half-segment-aligned waveform.
#. **The buffer has space.** Write the entry as
   ``preAccumulated = false`` with ``shotCount`` equal to the
   digitizer's ``shotIncrement`` (1 for single-shot, ``d_numAverages``
   for firmware block averaging).
#. **The buffer is full.** Switch into pre-accumulation mode: parse
   the raw bytes into a ``QVector<qint64>`` accumulator, sum
   subsequent shots into the same accumulator, and flush as soon as a
   slot becomes free.

The pre-accumulation accumulator uses ``qint64`` per sample so that
summed values do not overflow even for 1-byte digitizer modes. The
parsing path used for accumulation is the same byte-unpacking
function the consumer would otherwise call — see
:cpp:func:`BC::Analysis::parseWaveform` with
``ParseMode::Accumulate``. When a slot opens, the accumulator is
serialized into a ``QByteArray`` of ``recordLength × numRecords ×
sizeof(qint64)`` bytes and written with ``preAccumulated = true``
and ``shotCount`` equal to the sum of all accumulated shot
increments.

Pre-accumulation only engages on backpressure. The default
steady-state path is plain raw writes; the accumulator state resets
after each flush.

Buffer ownership and the ``FtmwConfig`` pointer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :cpp:class:`WaveformBuffer` is created in
:cpp:func:`FtmwDigitizer::hwPrepareForExperiment` (sized at 10 slots, with
each slot pre-reserved to ``recordLength × bytesPerPoint × numRecords``
bytes to avoid per-shot heap allocation) and destroyed when the
``FtmwDigitizer`` is torn down. A non-owning pointer is stashed on
:cpp:class:`FtmwConfig` via ``setWaveformBuffer``; the AM retrieves it
via ``exp->ftmwConfig()->waveformBuffer()`` rather than reaching across
to the hardware object directly.

Segment boundaries
------------------

Multi-segment acquisitions (LO scan, DR scan) need clean boundaries:
no shot from segment *N* may leak into segment *N + 1*'s accumulator.
Two mechanisms enforce that.

On the producer side, ``setAcquisitionGated(true)`` blocks
``emitShot`` from publishing more entries and resets any partial
pre-accumulation state. The producer also exposes
``writeFlushMarker``, which writes a sentinel ``WaveformEntry`` with
``flushMarker = true`` and empty data into the ring. Sentinels obey
the same drop-newest overflow rule, so a flush marker may be lost
when the buffer is full — the gating flag is the load-bearing
guarantee that no segment-*N* data is published after the boundary,
not the marker itself.

On the consumer side, ``drainFtmwBuffer`` breaks out of its read
loop the moment it encounters a flush marker. With no entries to
process, the AM still calls ``advance()`` (which returns ``true`` if
a segment boundary was crossed), re-emits ``newClockSettings`` for
the next segment, and runs the completion check. Acquisition
resumes draining once the next segment's clocks settle and gating
releases.

Consumer: AcquisitionManager drain loop
---------------------------------------

The drain loop is built from three pieces: a 20 ms ``QTimer``, a
``std::atomic<bool>`` abort flag, and a
``QFutureWatcher<FtmwProcessingResult>`` that hops worker results back
onto the AM thread.

``beginExperiment`` constructs all three when the experiment has FTMW
enabled and a buffer pointer is available. The timer
(``p_drainTimer``) fires every 20 ms. The watcher wraps a
``QFuture<FtmwProcessingResult>`` returned by ``QtConcurrent::run``
and routes its ``finished`` signal to ``onProcessingComplete`` on the
AM thread.

Each tick, ``drainFtmwBuffer`` runs four steps:

#. Refuse to start if not in the ``Acquiring`` state, if the buffer
   is empty, or if the previous worker has not yet finished.
#. Read entries one at a time with ``WaveformBuffer::read``. The read
   moves the slot's ``QByteArray`` out of the ring, so each call is
   constant-time and never copies the payload. Entries are pushed
   into a local ``std::vector<WaveformEntry>``. Flush markers break
   the read loop; entries that arrive while the experiment is
   complete or while ``d_processingPaused`` is set are skipped.
#. If the batch is empty (only sentinels were drained or every entry
   was skipped), still call ``ftmw->advance()`` so segment-boundary
   and 60-second autosave logic runs, then return.
#. Otherwise, stop the drain timer and dispatch the batch via
   ``QtConcurrent::run`` to a thread-pool lambda that consumes the
   moved vector.

The lambda checks ``d_abortProcessing`` before every entry it
processes and short-circuits if the flag is set. Worst-case latency
between an abort request and the worker stopping is one ``addFids``
call, which the sub-bundle scope notes can run several hundred
milliseconds for very large waveforms.

When the worker finishes, ``onProcessingComplete`` runs on the AM
thread:

#. Read ``pu_processingWatcher->result()`` — the ``FtmwProcessingResult``
   struct carries the entry count, a success flag, and optional error
   and warning strings.
#. If the result reports failure, log it and abort the experiment.
   Warnings are logged but not fatal.
#. Call ``ftmw->advance()``; if a segment boundary was crossed, emit
   ``newClockSettings`` to retune the next segment.
#. Emit ``ftmwUpdateProgress(perMilComplete)`` (drives the main
   window's progress bar) and run ``checkComplete`` (which kicks off
   the next backup snapshot if one is due, and finishes the
   acquisition if the experiment objective has been met).
#. Restart the drain timer if the AM is still acquiring.

``finishAcquisition`` is the matching teardown. It stops and deletes
the drain timer, sets the abort flag, waits for the in-flight worker
(if any) via ``QFutureWatcher::waitForFinished``, then resets the
watcher and emits ``endAcquisition``.

Parse and accumulate
--------------------

The dispatched worker performs the expensive byte-unpacking and FID
accumulation off the AM thread. Two paths exist depending on whether
the build defines ``BC_CUDA``.

**Non-CUDA path (default).** The worker calls
``FtmwConfig::addBatchFids(entries)``. ``addBatchFids`` is a single
parallel pass over the flat sample index space ``[0, L)`` where
``L = recordLength × numRecords``:

.. code-block:: cpp

   constexpr int kMinChunkSamples = 8192;
   int P = qBound(1,
                  static_cast<int>(L / kMinChunkSamples),
                  QThread::idealThreadCount());

Each of the ``P`` chunk threads handles all ``N`` entries for its
slice of the index space. Entry 0 writes the slice with
``ParseMode::Write``; entries 1..N-1 add into the same slice with
``ParseMode::Accumulate``. There is no inter-chunk synchronization —
each chunk owns a disjoint range of ``dst[]``. Pre-accumulated
entries skip the byte-unpacking work and are reinterpreted directly
from their ``qint64`` payload. ``parseBatchParallel`` falls back to
serial execution when ``P == 1``, which spares the
``QtConcurrent::blockingMap`` overhead for small records.

The combined ``QVector<qint64>`` becomes a ``FidList`` whose
``totalShots`` is the sum of every entry's ``shotCount``. Chirp
scoring (``d_chirpScoringEnabled``) and phase correction
(``d_phaseCorrectionEnabled``) operate on the combined result, not
per-entry; this is correct because both decisions are statistical
ones against the running average held by ``FidStorageBase``. The
worker then calls ``p_fidStorage->addFids`` once per drain cycle —
one mutex acquisition for the entire batch.

**CUDA path** (``#ifdef BC_CUDA``). The batch optimization above is
not available, so the worker iterates entries serially:
``addPreAccumulatedFids`` for entries with ``preAccumulated = true``
(which bypass the GPU averager because their bytes are already a
``qint64`` accumulator) and ``addFids`` for raw entries (which can
route through ``GpuAverager::parseAndAdd`` or ``parseAndRollAvg``
depending on the acquisition mode). The bypass is appropriate
because pre-accumulation only engages when the consumer is already
behind; sending those entries back through the GPU averager would
sum them twice.

The shared parsing function lives in
``data/analysis/waveformparser.{cpp,h}``. Both
``FtmwConfig::parseWaveform`` (single-entry, non-CUDA) and
``parseBatchParallel`` use it.

FidStorage: accumulator and cache
---------------------------------

:cpp:class:`FidStorageBase` is the storage half of the FTMW pipeline.
A single instance per experiment serves three purposes: it owns the
in-progress co-averaged ``FidList`` for the current segment, it
publishes thread-safe snapshots of that list for the GUI, and it
maintains a segment-indexed cache of historical ``FidList`` data so
the viewer can browse earlier segments without re-reading from disk
each time.

Two mutexes guard the storage. The accumulator (``d_currentFidList``)
sits behind ``pu_mutex`` (declared on
:cpp:class:`DataStorageBase`); ``addFids``, ``setFidsData``,
``getCurrentFidList``, and ``currentSegmentShots`` all take this
lock. The cache (``d_cache``, ``d_cacheKeys``) sits behind
``pu_baseMutex`` (declared on :cpp:class:`FidStorageBase` itself);
``loadFidList``, ``saveFidList``, and ``updateCache`` use it.

Three concrete subclasses cover the standard acquisition modes:

- ``FidSingleStorage`` — single-segment acquisition. Implements
  ``backup()`` to snapshot the in-progress FID under an incrementing
  index (used for target-shots, target-duration, and forever modes).
- ``FidMultiStorage`` — multi-segment storage indexed by segment
  number; used for LO scan and DR scan acquisitions.
- ``FidPeakUpStorage`` — rolling-average peak-up mode. Constructed
  with experiment number ``-1``, so ``save()`` is a silent no-op:
  peak-up performs no disk I/O. ``addFids`` rolls the new shots into
  a fixed-target rolling average rather than co-averaging.

The cache services live ``loadFidList(i)`` calls from the viewer.
When a request arrives for a segment not in the cache, the file is
read from ``fid/<i>.csv``, parsed back into a ``FidList`` using the
stored template, inserted into the cache, and (when full) evicts the
oldest entry per ``d_cacheKeys`` insertion order. The default cache
budget is ``d_maxCacheSize`` — roughly 256 MB of FID data; eviction
is triggered when adding the next entry would push the total beyond
the budget.

Visualization: FtmwViewWidget and FtWorker
------------------------------------------

The FTMW tab is implemented by ``FtmwViewWidget``
(``gui/widget/ftmwviewwidget.{cpp,h}``). It hosts the live FID and FT
plots, two configurable side-by-side plot pairs, the processing and
plot toolbars, the peak-find widget, and the overlay controls. The
widget hierarchy is straightforward; the developer-relevant
complexity is the data flow that keeps the spectrum on screen
without blocking either the AM or the GUI thread.

Threading model
~~~~~~~~~~~~~~~

``FtmwViewWidget`` constructs an :cpp:class:`FtWorker` instance with
the widget itself as parent, so the worker object lives on the GUI
thread. Worker methods are not invoked directly, however —
``FtmwViewWidget`` dispatches each ``doFT``, ``doFtDiff``, and
``processSideband`` call through ``QtConcurrent::run``, which schedules
the actual work on the global thread pool. A
``QFutureWatcher<void>`` per logical "plot slot" (live, main,
plot1, plot2) tracks completion. The worker emits its result signals
(``ftDone``, ``fidDone``, ``ftDiffDone``) back to the widget via
``Qt::QueuedConnection`` — even though sender and receiver are
nominally on the same thread, the worker code is *executing* on the
pool thread, so the queued connection hops the result onto the GUI
thread for the plot update.

The :cpp:class:`FtWorker` API page covers GSL workspace allocation,
the read/write locks that guard reentrancy, and the idle-cleanup
timer that frees workspaces after five minutes of inactivity.

Mutex coordination with the AM writer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AM worker mutates ``FidStorageBase`` from the thread-pool worker
thread (via ``addBatchFids`` → ``addFids``); ``FtmwViewWidget`` reads
it via ``getCurrentFidList()``, which copies under the
``pu_mutex``. The two never see partially-written state. The cost is
one ``FidList`` copy per refresh, which is bounded because the
viewer's refresh cadence is independent of the producer.

Refresh trigger
~~~~~~~~~~~~~~~

Live-plot refresh is driven by a periodic ``QObject::startTimer``
(``d_liveTimerId``) on the widget itself. The interval is set by the
**Refresh** spinbox on the FTMW toolbar (default 500 ms; persisted
under ``BC::Key::FtmwView::refresh``); the spinbox is enabled when an
FTMW experiment starts and disabled when it ends. Each tick,
``updateLiveFidList()`` reads ``ps_fidStorage->getCurrentFidList()``,
walks the active plot slots, and dispatches a fresh ``doFT`` per
slot. The progress bar in the main window tracks
``AcquisitionManager::ftmwUpdateProgress`` separately.

The view widget also listens to
:cpp:func:`AcquisitionManager::backupComplete` so that the backup
list refreshes whenever a concurrent backup snapshot finishes
writing to disk.

Plot classes
~~~~~~~~~~~~

The frequency-domain plot is :cpp:class:`MainFtPlot` (a
``FtPlot`` subclass); the time-domain plot is :cpp:class:`FidPlot`.
Both inherit from :cpp:class:`ZoomPanPlot` and use
:cpp:class:`BlackchirpPlotCurve` for their data series; the shared
zoom/pan/curve-customization machinery is documented on those API
pages. The user-facing controls for the toolbars are described in
:doc:`/user_guide/plot_controls`.

Processing settings persistence
-------------------------------

``FtWorker::FidProcessingSettings`` carries the eight knobs that
control time-domain preprocessing and FFT output: ``startUs``,
``endUs``, ``expFilter``, ``zeroPadFactor``, ``removeDC``,
``units``, ``autoScaleIgnoreMHz``, and ``windowFunction``. Every FT
in the application uses one of these structs, so the same struct
that drives the live plot also drives the viewer.

``FidStorageBase::writeProcessingSettings`` serializes the struct
into ``fid/processing.csv`` using the keys declared in
``BC::Key::FidStorage`` (``fidStart``, ``fidEnd``, ``fidExp``,
``zpf``, ``rdc``, ``units``, ``autoscaleIgnore``, ``winf``) via the
generic ``DataStorageBase::writeMetadata`` helper.
``readProcessingSettings`` reverses the operation. Storing the
processing settings alongside the FID data on disk lets the
``blackchirp-viewer`` reproduce the same plot view from a loaded
experiment without re-asking the user. The user-facing meaning of
each knob is described in :doc:`/user_guide/data_storage`.

The peak-find parameters travel the same way through a sibling pair,
:cpp:func:`FidStorageBase::writePeakFindSettings` /
``readPeakFindSettings``, which serialize a ``PeakFindSettings``
aggregate (min/max frequency, SNR, neighborhood half-width, window
size, polynomial order — keys in ``BC::Key::PeakStorage``) to
``fid/peakfind.csv`` via the same ``writeMetadata`` helper. The
viewer reloads it so a re-opened experiment restores the peak-search
configuration as well as the FT view.

Pointers
--------

**Peak finder.** :cpp:class:`PeakFinder`
(``data/analysis/peakfinder.{cpp,h}``) consumes an
:cpp:class:`Ft` and produces a peak list. The peak-find widget
hosted on ``FtmwViewWidget`` exposes the controls and the result
table.

**Overlays.** The overlay system (``OverlayBase``, ``OverlayStorage``,
the overlay parsers under ``data/processing/parsers/``) is covered
in :doc:`/developer_guide/persistence`.
