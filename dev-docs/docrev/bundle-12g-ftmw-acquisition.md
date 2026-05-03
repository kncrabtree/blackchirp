# Bundle 12g — Developer Guide: FTMW Acquisition and Visualization

**Status:** complete

<!--
Status log:
- 2026-05-03: not started → complete. Page landed at
  doc/source/developer_guide/ftmw_acquisition.rst with end-to-end
  Mermaid diagram and prose covering the producer/consumer ring,
  drain loop, parse-and-accumulate (CUDA and non-CUDA), FidStorage
  accumulator + cache, and the FtmwViewWidget/FtWorker layer.
  Authorized source-tree change: bumped
  FidStorageBase::d_maxCacheSize from 1 << 25 to 1 << 28 (~256 MB),
  matching the doxygen intent. Sub-bundle scope had two stale
  claims about visualization that the page corrected against the
  current code: (1) FtWorker is parented to FtmwViewWidget on the
  GUI thread and dispatched via QtConcurrent::run, not moved to a
  helper thread; (2) live-plot refresh is driven by the widget's
  own QObject::startTimer (default 500 ms via
  BC::Key::FtmwView::refresh), not by AcquisitionManager::ftmwUpdateProgress
  (which drives the main-window progress bar). Build clean except
  for the expected forward-link to /developer_guide/persistence
  (12i, not yet landed). Content commit 43517240.
-->

Sub-page of the Developer Guide chapter. Documents the FTMW data
pipeline end-to-end: from `FtmwScope::emitShot` on the digitizer
thread, through the `WaveformBuffer` ring buffer, into the
`AcquisitionManager` drain loop and worker, into `FidStorageBase`
accumulation, and out into the visualization layer
(`FtmwViewWidget`, `MainFtPlot`, `FidPlot`, `FtWorker`).

Two complementary halves: **acquisition** (producer/consumer ring
buffer, threading rules, parse-and-accumulate) and **visualization**
(the `FidStorage` cache, `FtWorker` threading, queued plot updates,
mutex coordination with the acquisition writer). Both halves are
worth a dedicated page because the API ref documents the pieces but
not the full pipeline.

## Scope

Single Sphinx file:
`doc/source/developer_guide/ftmw_acquisition.rst`.

The page should answer the following for a contributor:

1. **Why a ring buffer.** Brief rationale, drawn from the design
   document research:

   - Per-shot Qt signal emission across the digitizer thread →
     HardwareManager → AcquisitionManager creates two
     `QMetaCallEvent` allocations and event-loop dispatches per
     shot. With ~20k FIDs/sec target throughput this becomes the
     bottleneck before hardware is.
   - The Qt signal queue has no backpressure: a slow consumer
     causes unbounded queue growth.
   - Replacing the signal hops with a bounded SPSC ring buffer
     gives bounded memory, drop-newest backpressure, low-latency
     producer→consumer notification via `QSemaphore`, and a
     natural pre-accumulation fallback when the consumer falls
     behind.

   This is rationale, not API documentation. Keep it tight (≤ 1
   paragraph) and forward-link to
   `:doc:`/classes/waveformbuffer`` for the SPSC discipline,
   overflow policy, and `WaveformEntry` struct.

2. **Producer side: `FtmwScope::emitShot`.**

   - Each `FtmwScope` subclass produces raw waveform bytes and
     calls the base-class `emitShot(data)` from its own thread.
     The base class (in `FtmwScope::emitShot`) handles three
     cases:
     1. Acquisition is gated (segment boundary in progress) —
        the shot is dropped silently.
     2. The buffer has space — write the entry as-is with
        `preAccumulated = false`.
     3. The buffer is full and pre-accumulation is engaged —
        parse the raw bytes into a `QVector<qint64>` accumulator,
        wait for a slot, then flush as a single
        `preAccumulated = true` entry.
   - Pre-accumulation buffers raw bytes interpreted as
     1/2/4-byte integers per `d_bytesPerPoint`; the accumulator
     uses `qint64` per sample so summed values fit. Pre-
     accumulation only engages on backpressure (drop-newest is
     the default policy).
   - The buffer is created in `hwPrepareForExperiment` and
     destroyed in `endAcquisition`; a non-owning pointer is
     stored on `FtmwConfig` via `setWaveformBuffer` so the AM
     can find it through `exp->ftmwConfig()->waveformBuffer()`.

3. **Segment boundaries: gating and flush markers.**

   - Multi-segment acquisitions (LO scan, DR scan) need clean
     boundaries — no shots from segment N must leak into segment
     N+1's accumulator. `setAcquisitionGated(true)` causes
     `emitShot` to short-circuit; on the consumer side, the
     producer writes a flush-marker entry
     (`writeFlushMarker`) to signal the boundary.
   - The AM drain loop, on encountering a flush marker, breaks
     out of its read loop, calls `advance()` (which may emit
     `newClockSettings` for the next segment), and returns. The
     AM resumes draining once the next segment starts (gating
     released).

4. **Consumer side: the AM drain loop.**

   - `beginExperiment` creates a `QFutureWatcher<FtmwProcessingResult>`
     and a 20 ms `QTimer` (`p_drainTimer`). Both live on the AM
     thread.
   - `drainFtmwBuffer` (timer slot) reads available entries via
     fast `WaveformBuffer::read` calls (which move
     `QByteArray`s, not deep-copy), pushes them into a
     `std::vector<WaveformEntry>`, and dispatches the batch to
     `QtConcurrent::run`. While the worker runs, the drain
     timer is paused.
   - `onProcessingComplete` (the future watcher's `finished`
     slot) consumes the result on the AM thread, calls
     `advance()`, emits `ftmwUpdateProgress`, calls
     `checkComplete`, and restarts the drain timer.
   - The atomic `d_abortProcessing` flag lets `finishAcquisition`
     signal the worker to exit early; worst-case latency is one
     `addBatchFids` call.

5. **Parse-and-accumulate inside the worker.**

   - Non-CUDA path: `FtmwConfig::addBatchFids` calls
     `BC::Analysis::parseBatchParallel`, which processes every
     entry in one chunked parallel pass over the flat sample
     index space `[0, recordLength × numRecords)`. Chunk count
     `P = qBound(1, L/8192, idealThreadCount())`; serial
     fallback when `P == 1`. Each chunk thread handles all N
     entries for its index slice, producing a single combined
     `qint64` buffer with no inter-chunk synchronization.
   - The combined buffer becomes a `FidList` with
     `totalShots = sum of entry shotCounts`. Chirp scoring and
     phase correction (when enabled) operate on the combined
     result. `FidStorageBase::addFids` is then called once per
     drain cycle (one mutex acquisition).
   - CUDA path: when `BC_CUDA` is defined, the worker iterates
     entries serially: `addPreAccumulatedFids` for entries with
     `preAccumulated = true` (which bypass the GPU averager
     since accumulation has already happened), `addFids` for
     raw entries (which can route through `GpuAverager`).
     Pre-accumulation only occurs under backpressure, so the
     bypass is appropriate.
   - The shared parsing function lives in
     `data/analysis/waveformparser.{cpp,h}`. Both
     `FtmwConfig::parseWaveform` (single-entry,
     non-CUDA) and `parseBatchParallel` use it.

6. **`FidStorageBase` and the FidStorage cache.**

   - `FidStorageBase` is the storage half of the FTMW pipeline.
     It accumulates the in-progress `FidList` for the current
     segment behind a `pu_mutex` (in the base class) and
     publishes it via `getCurrentFidList()` for the GUI. Cache
     handling (segment-indexed `FidList` cache up to ~200 MB by
     default) is guarded by a separate `pu_baseMutex` in
     `FidStorageBase`.
   - Concrete subclasses:
     - `FidSingleStorage` — single segment with optional backup
       snapshots (target-shots, target-duration, forever).
     - `FidMultiStorage` — multi-segment for LO scan and DR
       scan.
     - `FidPeakUpStorage` — rolling-average peak-up
       (`d_number == -1`, no disk I/O, dummy experiment).
   - The cache services `FtmwViewWidget` requests for
     historical segments (e.g., during overlay browsing). When
     a `loadFidList(i)` call arrives for a segment not in the
     cache, the file is read from `fid/<i>.csv`, inserted into
     the cache, and (when full) evicts the oldest entry.

7. **Visualization: `FtmwViewWidget`, `MainFtPlot`, `FidPlot`,
   `FtWorker`.**

   - `FtmwViewWidget` (`gui/widget/ftmwviewwidget.{cpp,h}`) is
     the FTMW tab's container. It hosts a `MainFtPlot` (the
     primary FT spectrum), a `FidPlot` (the time-domain FID),
     processing- and plot-toolbars, the peak-find widget, and
     the overlay controls. Widget hierarchy is obvious from
     using the program; the developer-relevant complexity is
     the data flow.
   - `FtWorker` (`data/analysis/ftworker.{cpp,h}`) does the FT
     itself. It owns no GUI state; it accepts a `Fid`/`FidList`
     plus a `FidProcessingSettings` struct (window, ZPF,
     apodization, units, autoscale-ignore, DC removal) and
     returns an `Ft`. `FtWorker` runs on its own helper thread
     (created by `FtmwViewWidget`), and the view widget hops
     work onto it via queued connections, queued back via
     `Ft` payload signals.
   - Coordination with the acquisition writer: the AM mutates
     `FidStorageBase` from the AM thread (via the worker's
     `addBatchFids` call); `FtmwViewWidget` reads via
     `getCurrentFidList()` (which copies under the storage
     mutex) and dispatches the FT job to `FtWorker`. The two
     never see partially-written state; the cost is the copy on
     each refresh, which is bounded because `getCurrentFidList`
     is called at most once per progress signal.
   - Refresh trigger: `AcquisitionManager::ftmwUpdateProgress`
     is emitted on every drain cycle's `onProcessingComplete`.
     `FtmwViewWidget` listens, throttles (rendering every shot
     is wasteful), and dispatches a fresh FT job to
     `FtWorker`.
   - Backup-list refresh: the view widget listens to
     `AcquisitionManager::backupComplete` to refresh its list of
     restorable backups.

8. **Processing settings persistence.**

   - `FidStorageBase::writeProcessingSettings` /
     `readProcessingSettings` serialize the
     `FidProcessingSettings` struct to `fid/processing.csv`
     using the keys in `BC::Key::FidStorage` (`fidStart`,
     `fidEnd`, `fidExp`, `zpf`, `rdc`, `units`,
     `autoscaleIgnore`, `winf`). Storing processing settings
     alongside the FID data lets the viewer reproduce the same
     plot view from disk-loaded experiments without re-asking
     the user.

9. **Peak finding and overlays — pointer only.**

   - `PeakFinder` (`data/analysis/peakfinder.{cpp,h}`) consumes
     an `Ft` and produces a `PeakList`. The peak-find widget
     hosts the controls and the result table.
   - The overlay system (`OverlayBase`, `OverlayStorage`, the
     parsers under `data/processing/parsers/`) is covered in
     `:doc:`/developer_guide/persistence`` (12i); this page
     mentions it as a one-line pointer.

## Out of scope

- The `WaveformBuffer` class API — already on
  `:doc:`/classes/waveformbuffer``.
- The AM state machine — already on
  `:doc:`/classes/acquisitionmanager`` with the
  `acquisitionmanager-state-machine` anchor.
- LIF acquisition — bundle 12h.
- The persistence model and the file-parser ecosystem — bundle
  12i.
- The user-facing FTMW-tab walkthrough (which knob does what) —
  user guide, `:doc:`/user_guide/data_storage`` and the FTMW
  configuration / data viewing pages.
- CUDA build internals beyond the bypass note — out of scope.

## Sources

### Related source files

- `src/hardware/core/ftmwdigitizer/ftmwscope.{cpp,h}` — the
  base-class `emitShot`, pre-accumulation logic,
  `setAcquisitionGated`, `flushPreAccumulated`,
  `parseAndAccumulate`.
- `src/data/storage/waveformbuffer.{cpp,h}` — the buffer.
- `src/data/experiment/ftmwconfig.{cpp,h}` —
  `addBatchFids`, `addFids`, `addPreAccumulatedFids`, the
  WaveformBuffer pointer, chirp scoring / phase correction
  hooks.
- `src/data/experiment/ftmwconfigtypes.{cpp,h}` —
  the six concrete subclasses and their `_init` /
  `createStorage` overrides.
- `src/data/analysis/waveformparser.{cpp,h}` — the shared
  parse function and `parseBatchParallel`.
- `src/acquisition/acquisitionmanager.{cpp,h}` — the drain loop,
  `QFutureWatcher<FtmwProcessingResult>`, `onProcessingComplete`,
  `clockSettingsComplete`, the abort flag.
- `src/data/storage/fidstoragebase.{cpp,h}` — base accumulator,
  cache, processing settings I/O.
- `src/data/storage/fidsinglestorage.{cpp,h}`,
  `fidmultistorage.{cpp,h}`,
  `fidpeakupstorage.{cpp,h}` — concretes.
- `src/data/analysis/ftworker.{cpp,h}` — FT worker.
- `src/gui/widget/ftmwviewwidget.{cpp,h}` — view widget.
- `src/gui/plot/mainftplot.{cpp,h}`, `fidplot.{cpp,h}` — plots.
- `src/gui/widget/peakfindwidget.{cpp,h}` — peak finder.

### Related dev-docs

- `dev-docs/digitizer-data-flow.md` — research material for the
  ring-buffer rationale, the producer/consumer split, the
  flush-marker protocol, the parallel batch parse. Do not link.

### Related user-guide pages

Forward-link, do not duplicate:

- `doc/source/user_guide/ftmw_configuration.rst`
- `doc/source/user_guide/data_storage.rst`
- `doc/source/user_guide/experiment/`-prefixed pages, as they
  exist after user-guide bundles have landed.
- `doc/source/user_guide/plot_controls.rst` — for ZoomPanPlot
  shared controls.

### Related API reference pages

- `doc/source/classes/waveformbuffer.rst`
- `doc/source/classes/ftmwconfig.rst`
- `doc/source/classes/fid.rst`
- `doc/source/classes/fidstoragebase.rst`
- `doc/source/classes/ftworker.rst`
- `doc/source/classes/ft.rst`
- `doc/source/classes/blackchirpcsv.rst`
- `doc/source/classes/acquisitionmanager.rst`
  (use `:ref:`acquisitionmanager-state-machine`` for the drain
  loop reference)
- `doc/source/classes/zoompanplot.rst`,
  `doc/source/classes/blackchirpplotcurve.rst` — for the plot
  base classes; one-line forward-links are sufficient.

## Sphinx file deltas

**Created:**

- `doc/source/developer_guide/ftmw_acquisition.rst`.

## Page structure

H1 intro: 1–2 paragraphs framing the FTMW pipeline as
producer → ring buffer → consumer worker → storage → plots.

H2 sections (`-` underlines):

- *Why a ring buffer* — design rationale.
- *Producer: FtmwScope::emitShot* — drop-newest, pre-
  accumulation.
- *Segment boundaries* — gating + flush markers.
- *Consumer: AM drain loop* — drain timer, worker dispatch,
  `QFutureWatcher` hop-back, abort atomic.
- *Parse and accumulate* — non-CUDA parallel batch parse vs.
  CUDA serial path.
- *FidStorage: accumulator and cache*
- *Visualization: FtmwViewWidget and FtWorker* — the FT
  threading model, refresh throttling, mutex coordination.
- *Processing settings persistence* — `processing.csv`.
- *Pointers* — peak finder and overlays as one-paragraph
  forwards.

A simple ASCII or Mermaid diagram showing
`FtmwScope (digi thread) → WaveformBuffer → AM (drain timer) →
worker (QtConcurrent) → FidStorage → FtmwViewWidget → FtWorker
(helper thread)` is the most useful single visual.

## Acceptance criteria

- The producer side documents drop-newest, pre-accumulation
  trigger, and the gated/flush-marker protocol.
- The consumer side documents the drain timer, worker dispatch
  via `QtConcurrent`, and the future-watcher result hop-back.
- The non-CUDA parallel batch parse is described in enough
  detail that a contributor can locate
  `BC::Analysis::parseBatchParallel` and reason about its
  thread-count choice.
- The CUDA-path serial loop and the
  `addPreAccumulatedFids` bypass are documented.
- `FidStorageBase` is described as base accumulator + cache,
  with the three concrete subclasses named.
- The cache size limit (~200 MB default) and the LRU eviction
  policy are documented.
- `FtmwViewWidget` ↔ `FtWorker` interaction is documented as a
  helper-thread + queued-connection pattern; the mutex
  coordination with the AM writer is explicit.
- Refresh trigger is described as
  `ftmwUpdateProgress` with throttling.
- Processing settings persistence to `fid/processing.csv` is
  described with the eight-key list.
- A diagram (Mermaid or ASCII) shows the end-to-end pipeline.
- No duplication of per-class API content.
- No rendered link points into `dev-docs/`.
