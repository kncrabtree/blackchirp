# Bundle 12f — Developer Guide: Experiment Lifecycle

**Status:** complete

<!--
Status log:
- 2026-05-03: not started → complete. Drafted
  doc/source/developer_guide/experiment_lifecycle.rst (~2,400 words)
  covering the cross-manager round-trip — entry-point dialogs,
  startBatch wiring hub, the three-step
  HardwareManager::initializeExperiment, Experiment::initialize on the
  GUI thread, the AM steady-state loop (clocks round-trip via
  MainWindow::clockPrompt, aux/validation, drain timer, backups), the
  LIF parallel path, the four abort flows, the batch-level decision
  tree, and a Mermaid sequence diagram. Forward-references to 12g, 12h,
  and 12n raise expected unknown-document warnings that resolve when
  those bundles land. Two factual corrections vs the original sub-bundle
  scope: Peak Up is documented as an FtmwType setting (not a separate
  entry point), and the newClockSettings chain is documented as routing
  through MainWindow::clockPrompt (not direct AM→HM). Content commit
  c0a9bf843e.
-->

Sub-page of the Developer Guide chapter. The end-to-end coordination
story from the moment the user clicks Start until the batch
completes — across `MainWindow`, `HardwareManager`, `ClockManager`,
`AcquisitionManager`, `BatchManager`, `Experiment`, and the relevant
hardware objects.

The state-machine details for `AcquisitionManager` and
`BatchManager` are already documented on their respective API pages
(`:doc:`/classes/acquisitionmanager`` and
`:doc:`/classes/batchmanager``). This page is the cross-manager
view that neither API page tells in full.

## Scope

Single Sphinx file:
`doc/source/developer_guide/experiment_lifecycle.rst`.

The page should answer the following for a contributor:

1. **Starting a batch.**

   - The user triggers an experiment from one of: the experiment-
     setup wizard (full configuration via
     `ExperimentSetupDialog`), the quick-experiment dialog
     (`QuickExptDialog`), the Peak Up entry point, the
     batch-sequence dialog (`BatchSequenceDialog`).
   - Each entry point constructs a concrete `BatchManager`
     subclass — `BatchSingle` for one-shot experiments,
     `BatchSequence` for repeating-on-interval experiments — and
     hands it to `MainWindow::startBatch`.
   - `MainWindow::startBatch` connects the batch's signals to the
     acquisition layer and the GUI:
     `AcquisitionManager::experimentComplete` (signal) →
     `BatchManager::experimentComplete` (slot, queued connection
     onto the GUI thread); `BatchManager::beginExperiment`
     (signal) → `MainWindow` lambda that calls
     `HardwareManager::initializeExperiment(currentExperiment())`.
   - The first experiment is launched directly by `MainWindow`
     (it does not wait for `BatchManager::beginExperiment`).
     Subsequent experiments in the batch are started by the
     `beginExperiment` signal.

2. **Hardware setup: `HardwareManager::initializeExperiment`.**

   - `HardwareManager::initializeExperiment(exp)` runs on the HM
     thread. Steps in order:
     1. `ClockManager::prepareForExperiment(*exp)` — reads the
        clock map from `exp->ftmwConfig()->d_rfConfig.getClocks()`,
        resolves each `ClockType` role to its `Clock*`, sets
        multiplication factors, calls `setFrequency` on each
        affected output. Achieved frequencies are written back to
        `RfConfig::setCurrentClocks` (cross-link to bundle 12d
        for the loadout-side store).
     2. `hwPrepareForExperiment(*exp)` is invoked on every
        live hardware object via queued connection. The hardware
        validates the experiment config, stages per-experiment
        settings, and returns success/failure. The base wrapper
        `HardwareObject::hwPrepareForExperiment` reattempts a
        connection if disconnected, and (when `d_critical` is set)
        causes `exp->d_hardwareSuccess = false` on persistent
        failure.
     3. The `experimentInitialized(Experiment)` signal is
        emitted carrying the populated experiment.

3. **Experiment registration and AM hand-off.**

   - `MainWindow::experimentInitialized` (slot) handles the
     `HardwareManager::experimentInitialized` signal. It calls
     `Experiment::initialize()` on the GUI thread:
     - `Experiment::initialize` assigns the experiment number
       (querying the data path), creates the on-disk directory,
       writes the initial files (`version.csv`, `header.csv`,
       `objectives.csv`, `hardware.csv`, `chirps.csv`,
       `markers.csv`, `clocks.csv`).
     - For Peak Up without LIF, the experiment is marked
       *dummy* and no files are written.
   - Then `AcquisitionManager::beginExperiment(exp)` is invoked
     across the thread boundary via `QMetaObject::invokeMethod`.
   - From here the AM owns the experiment.

4. **`AcquisitionManager::beginExperiment` and the steady-state
   loop.**

   - Transition to `Acquiring`; emit `statusMessage("Acquiring")`.
   - Register aux-data keys for FTMW
     (`Ftmw/Shots`,
     optionally `Ftmw/ChirpRMS`,
     `Ftmw/ChirpPhaseScore`,
     `Ftmw/ChirpShift`).
   - Start the aux-data interval timer.
   - Emit `newClockSettings` (FTMW only). `HardwareManager`
     receives this and gates the FTMW digitizer during the
     transition; `allClocksReady` returns when settled and is
     connected directly to `AcquisitionManager::clockSettingsComplete`,
     which clears the FTMW processing-paused flag.
   - Emit `beginAcquisition`. `HardwareManager` broadcasts to
     every hardware object's `beginAcquisition` slot — devices
     start triggering, the digitizer's
     `WaveformBuffer` becomes the data conduit.
   - Start the FTMW drain timer. The drain loop is documented in
     `:doc:`/developer_guide/ftmw_acquisition`` (12g) and on
     `:ref:`acquisitionmanager-state-machine``; this page
     references but does not re-cover.
   - For LIF: emit `nextLifPoint(currentDelay, currentLaserPos)`.
     `HardwareManager` routes this to the LIF laser and
     digitizer; `lifSettingsComplete(success)` returns →
     `AcquisitionManager::lifHardwareReady`. From there, LIF shot
     processing follows
     `lifScopeShotAcquired` → `processLifScopeShot` →
     `addWaveform` → `advance` →
     possibly another `nextLifPoint`. Cross-link to
     `:doc:`/developer_guide/lif_acquisition`` (12h).

5. **Aux and validation flow during acquisition.**

   - `auxDataTick` (driven by the aux timer) calls
     `processAuxData`, which writes to `AuxDataStorage` and
     re-emits `auxData(map, timestamp)` for plot widgets. It then
     emits `auxDataSignal()` — `HardwareManager` aggregates
     `bcReadAuxData` calls across all hardware objects, prefixes
     keys with `hwKey`, and re-emits `auxData(map)` →
     `AcquisitionManager::processAuxData`.
   - `processValidationData` checks each incoming key against the
     installed `ValidationMap`; a violation calls `abort`.
   - The `auxData` and `validationData` paths share infrastructure
     but split at the AM: aux data is *recorded*, validation data
     is *checked*. A device declares its validation keys via the
     `validationKeys()` virtual; values for those keys are read by
     the same `readAuxData` call sites.

6. **Backups.**

   - After each FTMW processing batch, AM checks
     `Experiment::canBackup`. When true, a backup is dispatched
     via `QtConcurrent::run` and observed by a `QFutureWatcher`
     that emits `backupComplete` on the AM thread. The
     `FtmwViewWidget` listens to `backupComplete` to refresh its
     list of available backups.

7. **Completion paths.**

   - Normal completion: `Experiment::isComplete` returns true →
     `AcquisitionManager::finishAcquisition` runs:
     - Stop the drain timer.
     - Set the abort-processing atomic; wait for any in-flight
       worker.
     - Emit `endAcquisition` → `HardwareManager` broadcasts to
       every hardware object.
     - Transition to `Idle`.
     - For non-dummy experiments, `Experiment::finalSave` writes
       the closing artifacts to disk.
     - Emit `experimentComplete` (the signal). The connection to
       `BatchManager::experimentComplete` (the slot, GUI thread,
       queued connection) is the central inter-manager handoff.
   - Aborted: `AcquisitionManager::abort` calls `Experiment::abort`
     (which propagates to all `ExperimentObjective`s) and then
     `finishAcquisition`. `AcquisitionManager::finishAcquisition`
     still emits `experimentComplete`, but the experiment's
     `isAborted` flag is set; `BatchManager::experimentComplete`
     reads it and reacts accordingly (typically: terminate the
     batch via `abort` → `writeReport` → `batchComplete(true)`).
   - Hardware-failure abort: a critical-device
     `hardwareFailure` causes `HardwareManager` to emit
     `abortAcquisition`; `AcquisitionManager::abort` is its
     queued slot. The remaining flow is the abort path above.
   - Validation-failure abort: a validation key out of range in
     `processValidationData` calls `abort`. Same flow.

8. **Batch-level loop.**

   - `BatchManager::experimentComplete` (slot, GUI thread) is the
     central post-experiment hook. It logs the experiment result,
     calls `processExperiment()` if init succeeded, evaluates
     `isComplete()` and the abort flag, then either:
     - emits `beginExperiment` for the next experiment (default
       behavior; `BatchSequence` overrides `beginNextExperiment`
     to delay via a `QTimer`); or
     - calls `abort()` (on init failure or experiment abort),
       calls `writeReport()`, and emits
       `batchComplete(aborted)`.
   - `MainWindow` connects to `batchComplete` to update the UI
     (status message, button states) and disconnect the per-batch
     wiring it set up in `startBatch`.

9. **Adding a new experiment mode.** Cross-link forward to bundle
   12n for the recipe (new `FtmwType` subclass and/or new
   `BatchManager` subclass). One paragraph here, not a duplicate.

## Out of scope

- The internal state machines of `AcquisitionManager` and
  `BatchManager` — already documented on their API pages with the
  `acquisitionmanager-state-machine` and `batchmanager-state-machine`
  anchors. Cross-link.
- The waveform processing pipeline inside the AM drain loop —
  bundle 12g.
- LIF-specific acquisition mechanics — bundle 12h.
- The auxiliary-data fan-out plumbing in `HardwareManager` —
  bundle 12e.
- `ClockManager`'s clock-routing model — already on
  `:doc:`/classes/clockmanager``. One-paragraph mention here is
  enough.
- Adding a new experiment objective — bundle 12n.

## Sources

### Related source files

- `src/gui/mainwindow.{cpp,h}` — `startBatch`, the `experimentInitialized`
  slot, the lambda that calls `HardwareManager::initializeExperiment`.
- `src/hardware/core/hardwaremanager.{cpp,h}` —
  `initializeExperiment`, `setClocks`, the abort routing,
  `experimentInitialized` emission.
- `src/hardware/core/clock/clockmanager.{cpp,h}` —
  `prepareForExperiment`, `configureClocks`, `setClocks`.
- `src/acquisition/acquisitionmanager.{cpp,h}` —
  `beginExperiment`, `auxDataTick`, `processAuxData`,
  `processValidationData`, `clockSettingsComplete`,
  `finishAcquisition`, `abort`, `pause`, `resume`.
- `src/acquisition/batch/batchmanager.{cpp,h}` —
  `experimentComplete` slot, `beginNextExperiment`, the abstract
  virtuals.
- `src/acquisition/batch/batchsingle.{cpp,h}` and
  `src/acquisition/batch/batchsequence.{cpp,h}` — concrete
  subclasses.
- `src/data/experiment/experiment.{cpp,h}` — `initialize`,
  `abort`, `finalSave`, `canBackup`, the `d_objectives` set.
- `src/data/experiment/experimentobjective.{cpp,h}` —
  `initialize`, `advance`, `hwReady`, `cleanupAndSave`.
- `src/hardware/core/hardwareobject.{cpp,h}` —
  `hwPrepareForExperiment`, `beginAcquisition`, `endAcquisition`,
  `hardwareFailure`.

### Related dev-docs

None directly. (The old `digitizer-data-flow.md` covers the AM
drain loop, but that material lives in bundle 12g.)

### Related user-guide pages

Forward-link, do not duplicate:

- `doc/source/user_guide/experiment_setup.rst`
- `doc/source/user_guide/ftmw_configuration.rst`
- `doc/source/user_guide/lif.rst`

### Related API reference pages

- `doc/source/classes/acquisitionmanager.rst`
  (use `:ref:`acquisitionmanager-state-machine``)
- `doc/source/classes/batchmanager.rst`
  (use `:ref:`batchmanager-state-machine``)
- `doc/source/classes/hardwaremanager.rst`
- `doc/source/classes/clockmanager.rst`
- `doc/source/classes/experiment.rst`
- `doc/source/classes/hardwareobject.rst`
- `doc/source/classes/ftmwconfig.rst`
- `doc/source/classes/lifconfig.rst`

## Sphinx file deltas

**Created:**

- `doc/source/developer_guide/experiment_lifecycle.rst`.

## Page structure

H1 intro: 2–3 paragraphs. Frame the page as the cross-manager
view; explain that AM/BM API pages cover the per-manager state
machines while this page covers the inter-manager handoffs.

H2 sections (`-` underlines):

- *Starting a batch* — entry points and the wiring `startBatch`
  installs.
- *Hardware setup* — `HardwareManager::initializeExperiment` step
  sequence, including `ClockManager::prepareForExperiment` and
  the `hwPrepareForExperiment` fan-out.
- *Registering and handing off the experiment* —
  `Experiment::initialize` and the AM hand-off.
- *Acquisition steady state* — `beginExperiment`, drain timer
  reference, aux/validation timer, backups.
- *LIF parallel path* — one-paragraph forward-link to 12h.
- *Completion and abort paths* — normal, aborted by user,
  hardware failure, validation failure.
- *Batch-level loop* — `experimentComplete` slot decision tree.
- *Adding a new experiment mode* — one-paragraph forward-link to
  12n.

A Mermaid sequence diagram showing the
`MainWindow → HardwareManager → AcquisitionManager → BatchManager
→ MainWindow` round-trip is the most useful visual for this page;
include it if Mermaid is available.

## Acceptance criteria

- The four entry points are named and the BatchManager subclasses
  are named with their `BatchType`.
- `MainWindow::startBatch` is documented as the wiring hub for
  per-batch signal connections.
- `HardwareManager::initializeExperiment` is documented as a
  three-step sequence (clocks → hwPrepareForExperiment →
  experimentInitialized).
- The thread-crossing patterns are explicit:
  GUI → AM via `invokeMethod`; AM → BM via queued signal/slot;
  HM internal slots run on HM thread.
- The `newClockSettings` → digitizer-gating →
  `clockSettingsComplete` chain is documented.
- The four abort paths are listed (user, hardware-failure,
  validation-failure, and the AM-driven completion path) and
  each ends in `experimentComplete` → `BatchManager::experimentComplete`.
- The `experimentComplete`-signal-vs-slot disambiguation is
  noted (it is the same name on AM and BM, connected via queued
  connection).
- `ExperimentObjective::hwReady` is mentioned and tied to
  `clockSettingsComplete`.
- The page does not re-document the AM and BM state machines;
  it cross-links to them.
- No duplication of per-class API content; cross-links cover
  per-class detail.
- No rendered link points into `dev-docs/`.
