# Bundle 12h — Developer Guide: LIF Acquisition and Visualization

**Status:** complete

<!--
Status log:
- 2026-05-03: not started → complete. Drafted
  doc/source/developer_guide/lif_acquisition.rst covering the LIF
  scan model, LifConfig/LifDigitizerConfig roles, the per-point
  acquisition handshake (nextLifPoint → setLifParameters →
  lifSettingsComplete → lifHardwareReady → lifScopeShotAcquired →
  processLifScopeShot → addWaveform → advance), the laser-fastest
  row-major LifStorage flattening (with a 3×4 grid illustration of
  the visit order under each LifScanOrder), processing-gate
  persistence, the four LIF tab plots and their indexing into the
  flat integrated buffer, the reverse-step axis flip, and the
  configuration-UI split between ExperimentTypePage and
  ExperimentLifConfigPage. Bundle file's *Configuration UI* bullet
  (item 6 in Scope) corrected to match the actual code: the scan
  grid + scan order + randomize + complete mode + disable-flashlamp
  knobs live on ExperimentTypePage (wizard's first page), not on
  ExperimentLifConfigPage; the latter wraps LifControlWidget and
  configures the digitizer, shots-per-point, and processing
  settings. Confirmed `LifPeakUpStorage` does not exist; the page
  documents the absence and notes that live LIF alignment uses
  LifControlWidget against fresh waveformRead shots without a
  LifStorage instance. Doc build clean: 120 warnings, none new and
  none referencing the new page. Content commit
  9d42421170154f69e6859466308ef8f8fb1465f1.
-->

Sub-page of the Developer Guide chapter. Documents the LIF
(laser-induced fluorescence) data pipeline: scan model, acquisition
flow, the 2D delay × laser-position grid storage and its flattening
for slice/spectrogram plots.

LIF is fundamentally simpler than FTMW in many respects (no ring
buffer, signal-based shot path, smaller data) but has its own
non-trivial complexity in the 2D scan grid and the index/flatten
arithmetic that bridges storage to visualization. That complexity
is what this page documents.

## Scope

Single Sphinx file:
`doc/source/developer_guide/lif_acquisition.rst`.

The page should answer the following for a contributor:

1. **The LIF scan model.**

   - A LIF acquisition sweeps a two-dimensional grid:
     - One axis is *delay* — typically the pulse delay between
       a triggering event (e.g., a discharge) and the
       laser shot, in microseconds.
     - The other axis is *laser position* — typically a
       wavelength or frequency, in nanometers or wavenumbers.
   - At each `(delay, laser)` grid point, the digitizer records
     a fluorescence trace; the magnitude inside a configurable
     gate is integrated to produce a single point of the LIF
     spectrum.
   - The traversal is governed by:
     - `LifScanOrder` — `DelayFirst` cycles through every delay
       at one laser position before stepping the laser;
       `LaserFirst` cycles through every laser position at one
       delay before stepping the delay.
     - `d_delayRandom` — when true, the delay axis is randomly
       permuted at the start of each sweep (helps decorrelate
       drifts from the physical delay axis).
     - `LifCompleteMode` — `StopWhenComplete` ends acquisition
       once the grid has been fully covered;
       `ContinueAveraging` continues sweeping for further
       averaging until the user stops manually.

2. **`LifConfig` and `LifDigitizerConfig`.**

   - `LifConfig` (`src/data/lif/lifconfig.{cpp,h}`) is the
     experiment objective. It inherits `ExperimentObjective`
     for the lifecycle interface (`initialize`, `advance`,
     `hwReady`, `cleanupAndSave`) and `HeaderStorage` for
     header serialization.
   - It owns a `LifDigitizerConfig` (accessible via
     `scopeConfig()`) that wraps the digitizer-side parameters,
     including the integration gate.
   - It owns a `LifStorage` instance (cross-link to
     `:doc:`/classes/lifstorage``) that persists raw traces and
     processing-gate settings alongside the experiment.
   - The current `(delay, laser)` cursor is tracked in
     `LifConfig`; `currentDelay()` and `currentLaserPos()`
     return the values needed to drive the next acquisition
     point.

3. **The acquisition flow.** The full per-point loop:

   1. `AcquisitionManager::beginExperiment` (after FTMW setup
      if both objectives are enabled) emits
      `nextLifPoint(currentDelay, currentLaserPos)`.
   2. `HardwareManager` routes the request to the active
      `LifLaser` (set delay, set laser position) and the
      `LifScope` (configure gate / digitizer).
   3. When all hardware is ready,
      `HardwareManager::lifSettingsComplete(success)` is
      emitted. `AcquisitionManager::lifHardwareReady` is the
      slot:
      - On `success == false`: log and `abort`.
      - On success: call `LifConfig::hwReady()`, which clears
        `d_processingPaused`.
   4. The digitizer triggers on the next laser shot;
      `LifScope::shotAcquired` (or the equivalent waveform
      pathway — confirm in source) fires
      `lifScopeShotAcquired(QVector<qint8>)` to
      `HardwareManager`, which forwards to
      `AcquisitionManager::processLifScopeShot`.
   5. `processLifScopeShot` calls
      `LifConfig::addWaveform(trace)` (the integration gate is
      applied inside `LifConfig`), emits `lifPointUpdate` for
      the GUI, then calls `LifConfig::advance()`. If `advance`
      returns true *and* the experiment is not complete, emit
      `nextLifPoint` for the new cursor — go to step 2.
   6. When the grid is exhausted and `LifCompleteMode ==
      StopWhenComplete`, `isComplete()` returns true → AM
      finishes. Under `ContinueAveraging`, sweep continues
      until manually aborted.

   Note that the LIF path is **signal-based** end-to-end (no
   ring buffer). Trace volumes are small (`QVector<qint8>`,
   one trace per grid point with the integration gate already
   reducing per-point data size) and trigger rates are slow
   compared to FTMW, so the per-shot signal overhead is
   acceptable.

4. **Storage: `LifStorage` and the 2D grid.**

   - `LifStorage` (cross-link to
     `:doc:`/classes/lifstorage``) inherits
     `DataStorageBase`. It maintains a 2D grid of `LifTrace`
     objects keyed by `(delayIndex, laserIndex)`.
   - Internally, the grid is stored as a flat array (typical
     for cache-friendly iteration). Two flattening conventions
     coexist:
     - **Storage layout**: row-major or column-major depending
       on `LifScanOrder` — confirm the actual layout in
       `lifstorage.cpp` and document it. The drafter should
       not guess: the visualization arithmetic depends on the
       flattening convention.
     - **Visualization slicing**: `LifSlicePlot` and
       `LifSpectrogramPlot` iterate the grid by delay or by
       laser position. Each plot picks the slice direction it
       needs and indexes accordingly.
   - The LIF storage records the raw traces *plus* the
     processing-gate settings (start/end of the integration
     window, units), so a re-load reconstructs the same
     spectrum without re-asking the user.
   - `LifPeakUpStorage` analog for LIF peak-up — confirm in
     source whether such a thing exists; if not, omit. (LIF
     peak-up may simply use the standard storage with
     `d_number == -1` like FTMW peak-up.)

5. **Visualization.** The LIF tab consumes the storage through
   three plot widgets:

   - `LifTracePlot` — the most recently acquired raw trace
     (one waveform). Drives confirmation that the digitizer
     gate is correct.
   - `LifSlicePlot` — a 1D slice through the spectrogram: at a
     fixed laser position, the integrated signal vs. delay; or
     at a fixed delay, the integrated signal vs. laser
     position. The slice direction is user-selectable.
   - `LifSpectrogramPlot` — the full 2D map, integrated signal
     across `(delay, laser)`. Implemented as a Qwt
     `QwtPlotSpectrogram` over the flattened grid.

   The plots subscribe to AM signals (`lifPointUpdate`,
   `lifShotAcquired`) and refresh by reading from
   `LifStorage`. Like FTMW, the storage mutex coordinates the
   AM writer with GUI readers; the plots take a snapshot copy
   under the mutex and render on the GUI thread.

   Widget hierarchy is obvious from the user guide; the
   developer-relevant pieces are (a) how `LifSlicePlot` and
   `LifSpectrogramPlot` index into the flattened grid, and
   (b) how the integration gate from
   `LifProcessingWidget` flows back to `LifStorage` /
   `LifConfig`.

6. **Configuration UI.**

   - `ExperimentTypePage`
     (`src/gui/expsetup/experimenttypepage.{cpp,h}`) is the
     wizard's first page; its LIF group defines the scan grid
     (delay/laser start/step/points), scan order, randomization,
     complete mode, and the disable-flashlamp option.
     `ExperimentTypePage::apply` writes those onto `LifConfig`.
   - `ExperimentLifConfigPage`
     (`src/gui/lif/gui/experimentlifconfigpage.{cpp,h}`) is the
     per-LIF wizard page that wraps `LifControlWidget` and sets
     the digitizer (`scopeConfig`), shots-per-point, and
     processing-gate settings (`d_procSettings`).
   - `LifControlWidget` (also embedded in the
     **Hardware → LIF Configuration** dialog) hosts the live
     `LifTracePlot`, the digitizer config widget, and
     `LifLaserWidget` for live laser control.
     `LifLaserControlDoubleSpinBox` is the spinbox used inside
     `LifLaserWidget`. `LifProcessingWidget` provides the gate
     and filter controls in both the wizard page and the LIF tab.
   - Forward-link to `:doc:`/user_guide/lif`` for the user-
     facing operation; this page covers the data flow and the
     class wiring.

## Out of scope

- The FTMW pipeline — bundle 12g.
- The state machines of AM and BM — covered on those API
  pages and referenced in bundle 12f.
- The user-facing LIF operation — user guide.
- Peak-up mode for LIF (if it exists) — confirm in source; if
  it differs from the FTMW peak-up, briefly note. Otherwise,
  omit.

## Sources

### Related source files

- `src/data/lif/lifconfig.{cpp,h}` — the principal source.
- `src/data/lif/lifdigitizerconfig.{cpp,h}`.
- `src/data/lif/lifstorage.{cpp,h}` — confirm flattening
  convention.
- `src/data/lif/liftrace.{cpp,h}`.
- `src/hardware/core/lifdigitizer/lifscope.{cpp,h}` — the
  base class.
- `src/hardware/core/liflaser/liflaser.{cpp,h}` — the base
  class.
- `src/acquisition/acquisitionmanager.{cpp,h}` —
  `nextLifPoint`, `lifHardwareReady`, `processLifScopeShot`,
  `lifPointUpdate`, `lifShotAcquired`.
- `src/hardware/core/hardwaremanager.{cpp,h}` —
  `lifSettingsComplete`, `lifLaserPosUpdate`,
  `lifLaserFlashlampUpdate`, `lifScopeShotAcquired`.
- `src/gui/lif/gui/experimentlifconfigpage.{cpp,h}` —
  scan-grid configuration UI.
- `src/gui/lif/gui/lifdisplaywidget.{cpp,h}` — the LIF tab.
- `src/gui/lif/gui/liftraceplot.{cpp,h}`,
  `lifsliceplot.{cpp,h}`,
  `lifspectrogramplot.{cpp,h}` — the three plots.
- `src/gui/lif/gui/lifcontrolwidget.{cpp,h}`,
  `lifprocessingwidget.{cpp,h}` — control and processing UI.

### Related dev-docs

None. (LIF was added before the dev-docs era; the source is
the only ground truth.)

### Related user-guide pages

Forward-link, do not duplicate:

- `doc/source/user_guide/lif.rst` and the `lif/` sub-pages.

### Related API reference pages

- `doc/source/classes/lifconfig.rst`
- `doc/source/classes/lifstorage.rst`
- `doc/source/classes/datastoragebase.rst`
- `doc/source/classes/acquisitionmanager.rst`

The LIF *plot* classes do not currently have dedicated API
pages; if needed, this bundle can request that bundle 13g be
extended (flag for user consideration in the orchestrator
hand-off rather than acting unilaterally).

## Sphinx file deltas

**Created:**

- `doc/source/developer_guide/lif_acquisition.rst`.

## Page structure

H1 intro: 1–2 paragraphs contrasting LIF with FTMW (signal-
based vs. ring-buffered, 2D scan vs. 1D shot accumulation,
small traces vs. large waveforms).

H2 sections (`-` underlines):

- *The LIF scan model* — axes, traversal order, randomization,
  complete mode.
- *LifConfig and LifDigitizerConfig*
- *Acquisition flow* — the per-point loop end-to-end.
- *Storage: the 2D grid and its flattening*
- *Visualization* — trace, slice, spectrogram.
- *Configuration UI*

Diagrams: a simple table or Mermaid diagram of the grid traversal
under each `LifScanOrder` (DelayFirst vs LaserFirst) clarifies the
flattening mental model.

## Acceptance criteria

- The two scan axes (delay, laser position) are named and their
  units are described.
- `LifScanOrder`, `d_delayRandom`, and `LifCompleteMode` are each
  documented with their effect on the traversal.
- The acquisition flow is documented step-by-step from
  `nextLifPoint` through `processLifScopeShot` and back.
- The storage flattening convention is documented and matches the
  actual implementation in `lifstorage.cpp` (drafter must verify,
  not guess).
- The three plots (Trace / Slice / Spectrogram) are each named
  with their purpose.
- The integration-gate flow from `LifProcessingWidget` back into
  `LifStorage` / `LifConfig` is documented.
- The page makes the LIF-vs-FTMW contrast explicit at the data-
  flow level (signal-based vs. ring-buffered, small vs. large
  payloads).
- If `LifPeakUpStorage` does not exist, the page does not
  fabricate it; if it does, the page documents it briefly.
- No duplication of per-class API content; cross-links cover
  per-class detail.
- No rendered link points into `dev-docs/`.
