# Bundle 12c — Developer Guide: Architecture

**Status:** complete

<!--
Status log:
- 2026-05-03: not started → complete. doc/source/developer_guide/architecture.rst
  landed with intro, source tree, orchestration singletons,
  MainWindow-as-wiring-hub, threading model, cross-thread call
  patterns, and a Mermaid ownership/signal-flow diagram. Toolchain
  picked up sphinxcontrib-mermaid (added to extensions) plus a
  _static/mermaid_force_light.js shim that adds the 'light' class to
  <html> so the package's auto-theme heuristic short-circuits before
  matchMedia falls through to the user's OS dark-mode preference.
  Authorized source-tree change: removed the qmake-era
  src/modules/lif/hardware/lifhw_h.h aggregator (untracked,
  referenced nowhere) and its now-empty parent dirs; src/config/ is
  .gitignored on a clean clone, so it dropped from the source-tree
  walk too. Content commit a9e333cf.
- (entries appended in reverse chronological order; most recent first)
-->

Sub-page of the Developer Guide chapter. The chapter's load-bearing
overview: the layout of `src/`, the orchestration singletons that
coordinate the program, and the threading model that connects them.

This page is the spine of the developer guide. Every other sub-page
expects the reader to have absorbed:

- which directory under `src/` owns which responsibility,
- the names of the orchestration singletons and how they relate, and
- the threading model and the rules for crossing thread boundaries.

Subsequent sub-pages (12d–12n) reference these elements by name and
do not re-introduce them.

## Scope

Single Sphinx file: `doc/source/developer_guide/architecture.rst`.

The page should answer the following for a contributor:

1. **What is in each top-level directory under `src/`?** Walk the
   tree at one level of depth:

   - `src/acquisition/` — the experiment-execution layer.
     `AcquisitionManager` drives the acquisition loop;
     `acquisition/batch/` holds `BatchManager` and its concrete
     subclasses (`BatchSingle`, `BatchSequence`).
   - `src/data/` — data model, persistence, analysis, and
     application-wide singletons that are not hardware-specific:
     - `data/experiment/` — experiment configuration data classes
       (`Experiment`, `FtmwConfig` and subclasses, `RfConfig`,
       `ChirpConfig`, `DigitizerConfig` family,
       `ExperimentObjective`, `ExperimentValidator`).
     - `data/lif/` — LIF-specific data classes (`LifConfig`,
       `LifDigitizerConfig`, `LifStorage`, `LifTrace`).
     - `data/loadout/` — `HardwareLoadout`, `LoadoutManager`,
       `FtmwPreset`, plus the loadout/preset snapshot helpers.
     - `data/storage/` — persistence: `BlackchirpCSV`, the
       `HeaderStorage` tree, `DataStorageBase` and its concrete
       subclasses (`FidStorageBase` and friends, `LifStorage`,
       `OverlayStorage`, `AuxDataStorage`), `SettingsStorage`,
       `WaveformBuffer`, `ApplicationConfigManager`.
     - `data/analysis/` — `FtWorker` (the FT processing worker),
       `Analysis`, `PeakFinder`, `WaveformParser`, `Ft`.
     - `data/processing/parsers/` — overlay/import file parsers
       (`FileParser`, `FileParserRegistry`, `GenericXyParser`,
       `SpcatParser`, `XiamParser`, `CatalogParser`).
     - `data/model/` — Qt item models for chirp/clock/marker/peak
       tables.
     - `data/settings/` — `hardwarekeys.h`, `guikeys.h`.
     - `data/processing/` — overlay processing/operations.
     - `data/presentation/` — `CurveAppearance`.
     - `data/loghandler.h/.cpp`, `data/bcglobals.h/.cpp` — global
       singletons (logging, application-wide constants).
   - `src/gui/` — Qt widgets and the main window:
     - `gui/mainwindow.{cpp,h}` — the application's central window.
     - `gui/dialog/` — modal dialogs (Add Profile, Hardware
       Configuration, FTMW Configuration, etc.).
     - `gui/expsetup/` — the experiment-setup wizard pages.
     - `gui/widget/` — embeddable widgets (FTMW view, chirp
       config, hardware status boxes, …).
     - `gui/plot/` — Qwt-based scientific plots (`ZoomPanPlot`
       base, `MainFtPlot`, `FidPlot`, `ChirpConfigPlot`,
       `PulsePlot`, `TrackingPlot`, plot-curve helpers, custom
       tracker/zoomer).
     - `gui/lif/gui/` — LIF-specific widgets and plots.
     - `gui/overlay/` — overlay configuration widgets.
     - `gui/style/` — theme color management.
     - `gui/util/` — small GUI utilities (numeric formatting).
   - `src/hardware/` — hardware abstraction and implementations:
     - `hardware/core/` — `HardwareObject`, `HardwareManager`,
       `HardwareRegistry`, `HardwareRegistration`,
       `RuntimeHardwareConfig`, `HardwareProfileManager`,
       interface-class subdirectories (`clock/`, `ftmwdigitizer/`,
       `lifdigitizer/`, `liflaser/`), `communication/`.
     - `hardware/optional/` — optional hardware-type interfaces
       and implementations (AWG, pulse generator, flow controller,
       GPIB controller, IO board, pressure controller, temperature
       controller).
     - `hardware/python/` — Python trampoline classes plus
       `python_hw_host.py` and template scripts.
     - `hardware/library/` — `VendorLibrary` and concrete
       subclasses (`LabjackLibrary`, `SpectrumLibrary`).
   - `src/modules/` — optional compile-conditional modules:
     - `modules/cuda/` — `GpuAverager` (CUDA-accelerated FID
       accumulation, gated by `BC_ENABLE_CUDA`).
     - `modules/lif/hardware/` — LIF-specific hw aggregation.
   - `src/main.cpp` — the application entry point; instantiates
     `MainWindow`.
   - `src/resources/` — Qt resource collection (icons, themes).
   - `src/config/` — `version.h.in` and similar configure-time
     templates (if any).

2. **What does each orchestration singleton own, and how do they
   compose?** Establish the cast:

   - `HardwareRegistry` — compile-time catalog: every hardware
     implementation registers itself here at static-init time
     (factories, supported protocols, settings descriptors,
     custom-comm descriptors, library dependencies).
     Singleton; populated before `main()` runs.
   - `HardwareProfileManager` — the profile metadata layer. A
     *profile* is an immutable `(<HardwareType>, <label>,
     <implementation>)` triple with persistent settings and (for
     Python) script path and class name; the
     `<HardwareType>.<label>` pair is the profile's identity.
     CRUD'd by `RuntimeHardwareConfigDialog`. Changing
     implementation requires creating a new profile.
   - `RuntimeHardwareConfig` — the active-selection layer. Records
     which profiles are active in the running session, keyed by
     profile identity. Read by `HardwareManager` to know what to
     instantiate. Validated against `HardwareRegistry`.
   - `LoadoutManager` — named sets of member profiles and the FTMW
     presets that ride on top. A *loadout* captures a complete
     hardware selection (which AWG profile, which digitizer
     profile, which clock profiles) plus its FTMW presets (RF
     chain + chirp + digitizer config, named within the loadout).
   - `HardwareManager` — the runtime owner of live
     `HardwareObject` instances. Lives on its own thread; moves
     threaded hardware objects to per-device threads; mediates
     all cross-thread hardware calls; fans out connection,
     auxiliary-data, and experiment-lifecycle signals.
   - `ClockManager` — owned by `HardwareManager`; maps each
     `RfConfig::ClockType` role to a physical `Clock` device and
     output index; gates the FTMW digitizer during clock
     transitions.
   - `AcquisitionManager` — drives an in-progress experiment:
     waveform processing pipeline, aux/validation timer, backups,
     experiment finalization. Lives on its own thread.
   - `BatchManager` — abstract base; the active concrete subclass
     (`BatchSingle`, `BatchSequence`) lives on the GUI thread and
     coordinates across-experiment state in a batch run.
   - `LogHandler`, `ApplicationConfigManager` — application-wide
     singletons for diagnostic logging and app-level
     configuration (data path, debug logging toggle, vendor
     library paths). Cross-link to existing API pages for
     details.

   A Mermaid diagram (see *Page structure*, below) shows the
   ownership and signal-flow relationships at a glance.

3. **What is the threading model?** Three primary execution
   contexts plus a worker pool:

   - **GUI thread** — the QApplication's main thread. Hosts
     `MainWindow`, every dialog, every plot, the `BatchManager`
     subclass instance.
   - **HardwareManager thread** — created in `MainWindow` before
     the application event loop starts; `HardwareManager` is moved
     onto it; `HardwareManager::initialize` is called from
     `QThread::started`. All `HardwareManager` slots execute on
     this thread.
   - **Per-device threads** — for each `HardwareObject` whose
     `d_threaded` flag is `true`, `HardwareManager` creates a
     dedicated `QThread`, moves the object onto it, then calls
     `bcInitInstrument` via the thread's `started` signal. Threaded
     hardware must not have a `QObject` parent and must construct
     child `QObject`s in `initialize()` (not in the constructor, so
     they are constructed on the device thread).
   - **AcquisitionManager thread** — created in `MainWindow`;
     `AcquisitionManager` is moved onto it. The GUI thread calls
     `AcquisitionManager::beginExperiment` via
     `QMetaObject::invokeMethod`. The drain timer, aux timer, and
     state mutations all execute on this thread.
   - **QtConcurrent thread pool** — used for two things in the AM:
     waveform parse-and-accumulate batches dispatched via
     `QtConcurrent::run` and observed via
     `QFutureWatcher<FtmwProcessingResult>`; periodic experiment
     backups dispatched the same way. The future watcher delivers
     results back to the AM thread, keeping all state mutations
     thread-confined.

   Cross-thread call patterns:

   - **Direct signal → slot connection** crosses thread boundaries
     automatically via `Qt::AutoConnection` (Qt picks queued when
     the connection crosses thread boundaries).
   - **`QMetaObject::invokeMethod`** is used when a non-`QObject`
     caller (or a caller that does not hold a signal) needs to
     invoke a slot on another thread. `Qt::QueuedConnection` for
     fire-and-forget; `Qt::BlockingQueuedConnection` only when the
     caller must wait for the result and the threads are
     guaranteed-different (deadlocks if same-thread).
   - **`QFutureWatcher<T>`** is the canonical pattern for "do
     expensive work on the pool, hop the result back to my
     thread". The AM uses it for waveform processing and backups;
     other AMs in the codebase that wish to dispatch worker
     pools should follow the same pattern rather than inventing
     a new one.
   - **`QSemaphore`** — the digitizer/AM `WaveformBuffer` uses a
     semaphore for low-latency producer→consumer notification
     across the digitizer thread and the AM thread. The full
     story is in `:doc:`/developer_guide/ftmw_acquisition`` (12g).

4. **`MainWindow` as the wiring hub.** Briefly: `MainWindow`
   constructs the singletons (`HardwareManager`,
   `AcquisitionManager`, the batch manager subclass per
   acquisition), wires their signals together
   (`AcquisitionManager::experimentComplete` →
   `BatchManager::experimentComplete`, `HardwareManager::auxData`
   → `AcquisitionManager::processAuxData`, etc.), and dispatches
   user actions to whichever singleton owns them. Subsequent sub-
   pages (12e for hardware-side wiring, 12f for the experiment
   lifecycle) trace the specific signal chains; this page just
   establishes the role.

## Out of scope

- Deep dives into any single subsystem — those are 12d–12n's job.
- The complete signal/slot inventory of any orchestration class —
  the API ref pages already enumerate signals and slots; this page
  cross-links by `:doc:`.
- The build-system layout (where each library is produced) — that
  is 12a's job. A one-paragraph reminder is fine, but do not
  re-cover.
- Any source code change. This page is pure description.

## Sources

### Related source files

- `src/main.cpp` — to confirm what runs at application start.
- `src/gui/mainwindow.{cpp,h}` — to confirm the thread-creation
  sites, the singleton-construction order, and the wiring of
  inter-manager signals. The `MainWindow` constructor is the
  single most informative file for this page.
- `src/hardware/core/hardwaremanager.{cpp,h}` — to confirm the
  per-device threading code (`moveToThread`, `bcInitInstrument`
  dispatch).
- `src/acquisition/acquisitionmanager.{cpp,h}` — to confirm the
  drain timer, the `QFutureWatcher<FtmwProcessingResult>`
  dispatch, and the `auxDataTick` flow.
- `src/hardware/core/hardwareobject.{cpp,h}` — to confirm the
  `d_threaded` semantics and the constructor-vs-`initialize`
  rule.
- The `cmake/Blackchirp*.cmake` modules — only to confirm which
  subsystem produces which library. Do not re-cover the build
  system here.

### Related dev-docs

- `dev-docs/digitizer-data-flow.md` — for the
  producer/consumer/pool background that the *Threading model*
  section needs at a one-paragraph level. Do not link.

### Related user-guide pages

- `doc/source/user_guide/ui_overview.rst` — for terminology
  alignment when the page mentions UI surfaces in passing.

### Related API reference pages

Forward-link, not duplicate:

- `doc/source/classes/hardwaremanager.rst`
- `doc/source/classes/acquisitionmanager.rst`
- `doc/source/classes/batchmanager.rst`
- `doc/source/classes/clockmanager.rst`
- `doc/source/classes/hardwareobject.rst`
- `doc/source/classes/hardwareregistry.rst`
- `doc/source/classes/hardwareprofilemanager.rst`
- `doc/source/classes/runtimehardwareconfig.rst`
- `doc/source/classes/loadoutmanager.rst`
- `doc/source/classes/loghandler.rst`
- `doc/source/classes/applicationconfigmanager.rst`

## Sphinx file deltas

**Created:**

- `doc/source/developer_guide/architecture.rst`.

**Modified:**

- None (toctree already updated by bundle 12).

## Page structure

H1 intro: 2–3 paragraphs framing the chapter's spine — the source
tree, the orchestration singletons, the threading model. Note that
later sub-pages assume this terminology.

H2 sections (`-` underlines):

- *Source tree* — table or bulleted walk through the top-level
  directories under `src/`.
- *Orchestration singletons* — short prose introducing each
  singleton; cross-links to API ref pages.
- *Threading model* — the four execution contexts (GUI, HM,
  per-device, AM) plus the QtConcurrent pool; cross-thread call
  patterns.
- *Diagram* — one Mermaid diagram showing ownership (GUI thread →
  `MainWindow` → singletons; `HardwareManager` → live
  `HardwareObject`s on per-device threads; the major signal flows
  between `HardwareManager`, `AcquisitionManager`, `BatchManager`,
  and the GUI). Use `.. mermaid::` if the Sphinx build supports
  it; otherwise fall back to a `.. code-block:: text` ASCII
  diagram.

## Acceptance criteria

- The source-tree section names every top-level directory under
  `src/` and assigns each a one- or two-sentence responsibility.
- The orchestration-singletons section names every singleton listed
  above, gives each a one-paragraph role description, and
  cross-links its API ref page.
- The threading-model section names the four execution contexts
  and the QtConcurrent pool, describes the cross-thread call
  patterns (`AutoConnection`, `invokeMethod`, `QFutureWatcher`,
  `QSemaphore` mention), and notes the threaded-hardware
  constructor restriction (no parent, no child `QObject` in the
  ctor).
- A diagram (Mermaid or ASCII) shows the singleton ownership and
  major inter-manager signal flow.
- The page reads as a "spine" — a contributor finishing it can
  open any later sub-page (12d–12n) and immediately understand
  the terminology used there.
- No duplication of per-class detail from the API ref.
- No rendered link points into `dev-docs/`.
