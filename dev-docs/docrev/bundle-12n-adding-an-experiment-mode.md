# Bundle 12n — Developer Guide: Adding an Experiment Mode

**Status:** complete

<!--
Status log:
- 2026-05-03: not started → complete. Drafted
  doc/source/developer_guide/adding_an_experiment_mode.rst with the two
  five-touch recipes (FtmwType, BatchManager subclass), the
  multi-segment vs. single-segment guidance, the completion-pattern
  taxonomy, the storage-class table, and the persistence note honestly
  flagging report generation as a stub with a recommended batch/ peer
  folder for the future. Added a chapter-intro note pointing wider
  ExperimentObjective-level changes to the Discord/GitHub issues board
  per user request. Content commit 3baba96b.
- (entries appended in reverse chronological order; most recent first)
-->

Sub-page of the Developer Guide chapter. Two recipes, both centered
on extending Blackchirp's experiment-execution behavior:

- **Section A — Adding a new `FtmwType`.** A new FTMW acquisition
  mode (something other than the existing
  `Target_Shots`/`Target_Duration`/`Forever`/`Peak_Up`/`LO_Scan`/
  `DR_Scan`).
- **Section B — Adding a new `BatchManager` subclass.** A new
  batch-execution policy beyond `BatchSingle` and `BatchSequence`.

The two are independent extension points but logically belong on
the same page: both define how the program *runs* an experiment,
and both involve a small enum + a subclass + a wizard wiring.

## Scope

Single Sphinx file:
`doc/source/developer_guide/adding_an_experiment_mode.rst`.

The page should answer the following for a contributor:

### Section A — A new `FtmwType`

1. **Decide whether a new type is necessary.**

   - Existing `FtmwType` enumerators (in `FtmwConfig::FtmwType`):
     `Target_Shots`, `Target_Duration`, `Forever`, `Peak_Up`,
     `LO_Scan`, `DR_Scan`. Each is a *completion criterion*
     and possibly a *segment-traversal pattern*.
   - The threshold for adding a new type vs. parameterizing
     an existing one: if the new mode's completion logic
     can be expressed by tuning `d_objective` on an existing
     subclass, parameterize. If it requires a new
     `_init`/`createStorage`/`isComplete`/`perMilComplete`
     implementation or a new segment-traversal pattern, add
     a new type.

2. **The five touches.**

   1. Add an enumerator to `FtmwConfig::FtmwType` in
      `src/data/experiment/ftmwconfig.h`.
   2. Add a concrete subclass (e.g., `FtmwConfigMyMode`) in
      `src/data/experiment/ftmwconfigtypes.{cpp,h}`. The
      class inherits `FtmwConfig` and overrides:
      - `_init()` — initialize mode-specific state at
        acquisition start. Set up segment counts in
        `RfConfig` for multi-segment modes; record
        wall-clock targets for time-based modes; etc.
      - `_prepareToSave()` and `_loadComplete()` — header
        round-trip serialization for any mode-specific
        scalars.
      - `createStorage(num, path)` — return the right
        `FidStorageBase` subclass. Use
        `FidSingleStorage` for single-segment modes,
        `FidMultiStorage` for multi-segment, and
        `FidPeakUpStorage` for transient peak-up.
      - `isComplete()` — completion predicate.
      - `perMilComplete()` — progress in per-mille
        (`0`–`1000`).
      - `completedShots()` — total shot count for progress
        reporting.
      - Optional: `indefinite()` (return `true` to suppress
        the standard completion check; used by `Forever`),
        `bitShift()` (peak-up uses this to widen the
        rolling-average accumulator), `advance()`
        (multi-segment modes override to step the segment
        cursor).
   3. Update the factory in `ftmwconfigtypes.cpp` —
      `FtmwConfig::create*` or the equivalent dispatch
      that maps `FtmwType` to the concrete class. Confirm
      the actual factory site in source; the convention
      is to switch on `FtmwType` in
      `FtmwConfig::clone`/`fromOther` or in the
      `Experiment::enableFtmw(type)` factory.
   4. Wire the type into the experiment-setup wizard.
      `ExperimentTypePage`
      (`src/gui/expsetup/experimenttypepage.{cpp,h}`) is
      the entry point. Add a radio-button or list entry
      for the new mode; on selection, instruct
      `Experiment::enableFtmw(type)` to construct the new
      subclass. If the mode introduces unique
      configuration parameters (analogous to LO-scan's
      step counts), add a new wizard page in
      `gui/expsetup/` (e.g., `MyModeConfigWidget`) and
      conditionally include it in the wizard's page-
      ordering logic.
   5. Add an API page if the new subclass deserves one
      under `doc/source/classes/<typename>.rst` (cross-
      link to `:doc:`/developer_guide/api_style``). The
      existing `FtmwConfig` page already has
      `.. doxygenclass::` directives for each of the six
      existing subclasses; extend that page to include
      the new one.

3. **Multi-segment vs. single-segment design.**

   - Multi-segment modes (`LO_Scan`, `DR_Scan`) configure
     `RfConfig` with a list of segments in `_init`, then
     drive segment transitions through `advance()`. Each
     segment write its FID data to a separate file
     (`fid/<i>.csv`) under `FidMultiStorage`. The drain
     loop calls `advance()`, which when it returns true
     causes `AcquisitionManager` to emit
     `newClockSettings` for the next segment's clock
     configuration.
   - Single-segment modes (`Target_Shots`, `Target_Duration`,
     `Forever`, `Peak_Up`) keep `advance()` as a no-op
     beyond autosave handling, and `FidSingleStorage` /
     `FidPeakUpStorage` accumulate into a single file.
   - Cross-link to bundle 12g for the drain-loop /
     segment-boundary mechanics, including the
     `setAcquisitionGated` + flush-marker protocol.

4. **Wall-clock vs. shot-based completion.**

   - Shot-based: `isComplete()` checks `completedShots()`
     against the user-supplied `d_objective`.
     `perMilComplete()` is `1000 * completedShots /
     d_objective`.
   - Wall-clock: record start time in `_init`, target time
     in `_prepareToSave` if relevant. `isComplete()`
     compares `QDateTime::currentDateTime()` to the
     target. `Target_Duration` is the existing example.
   - Indefinite: `Forever` returns `true` from
     `indefinite()`, which tells the AM/Experiment
     framework not to fire a completion check.

5. **Storage choice.** A short table listing the existing
   modes' storage classes:
   - `Target_Shots`, `Target_Duration`, `Forever` →
     `FidSingleStorage`.
   - `Peak_Up` → `FidPeakUpStorage` (no disk I/O).
   - `LO_Scan`, `DR_Scan` → `FidMultiStorage`.
   New modes pick whichever fits; if none fit, consider
   whether a new `FidStorageBase` subclass is justified
   (rare; most modes can fit existing storage).

### Section B — A new `BatchManager` subclass

1. **Decide what the batch policy adds.**

   - `BatchSingle` runs one experiment.
   - `BatchSequence` runs an experiment template repeatedly
     on a configurable interval until manually stopped.
   - The threshold for a new subclass: any policy that
     cannot be expressed by varying the *interval* on
     `BatchSequence`. Examples that would justify a new
     subclass: an "until N successful experiments" policy,
     a "scan a parameter through values" policy, an
     externally-triggered "run on cue" policy.

2. **The five touches.**

   1. Add an enumerator to `BatchManager::BatchType` in
      `src/acquisition/batch/batchmanager.h`.
   2. Subclass `BatchManager` (e.g., `BatchScanParameter`)
      in `src/acquisition/batch/<filename>.{cpp,h}`.
      Implement the five pure virtuals:
      - `currentExperiment()` — return the active
        experiment shared pointer; never null while the
        batch is running.
      - `isComplete()` — return true when no further
        experiments remain.
      - `abort()` — mark the batch as complete; release
        any pending timers.
      - `processExperiment()` — post-acquisition
        bookkeeping (no-op is allowed).
      - `writeReport()` — write any per-batch summary
        report (file or log entry; no-op is allowed).
      - Optional: `beginNextExperiment()` — override
        when the default behavior (emit
        `beginExperiment()` immediately) is wrong.
        `BatchSequence` overrides this to wait on a
        `QTimer`.
   3. Wire a configuration dialog. For sequence-style
      batches, model on `BatchSequenceDialog`; the dialog
      collects parameters and constructs the concrete
      subclass.
   4. Add a menu / button entry in `MainWindow` that
      opens the new dialog and, on accept, calls
      `MainWindow::startBatch(batchManager)`.
   5. Update the API ref page for `BatchManager` to add
     `.. doxygenclass::` directives for the new subclass,
     and (if useful) a paragraph in the *Subclassing
     guide* section.

3. **Build a representative wiring.** A worked example,
   abstract enough to apply to any new subclass. Key
   points:

   - The subclass owns the *next experiment* — it
     constructs each `Experiment` (or clones a template)
     as the batch progresses. `currentExperiment()`
     returns the in-flight one.
   - `processExperiment()` is the place to inspect the
     just-completed experiment's data (validation pass,
     numeric outputs, calculated derivatives) and
     store/aggregate.
   - `writeReport()` runs once at batch end. Typical
     content: a summary CSV with per-experiment
     statistics, a markdown log entry, or both.
   - `beginNextExperiment()` defaults to immediate emit;
     override when:
     - waiting on a timer (sequence model);
     - waiting on an external trigger;
     - waiting on user confirmation (interactive batch).
   - The batch's connection to
     `AcquisitionManager::experimentComplete`
     (signal/slot pair) is set up by `MainWindow::startBatch`
     and uses a queued connection from the AM thread to
     the GUI thread. The subclass does not manage that
     connection.

4. **Coordination with the AM.**

   - The AM does not know which batch type is running.
     `AcquisitionManager::experimentComplete` (signal) is
     emitted unconditionally at the end of each
     experiment; the connection to
     `BatchManager::experimentComplete` (slot) is what
     advances the batch.
   - Cross-link to bundle 12f for the cross-manager flow
     and to `:doc:`/classes/batchmanager`` for the slot's
     internal decision tree.

5. **Persistence.**

   - The batch's *configuration* (parameters chosen in the
     dialog) is typically persisted via `SettingsStorage`
     under a `BC::Store::<MyBatch>` namespace, so the
     dialog can recall the user's last choices.
   - **Reports are not yet generated by any existing batch
     type.** `BatchSingle::writeReport` and
     `BatchSequence::writeReport` are no-ops at the time
     this guide is written, and there is no on-disk
     convention for where a batch report would live.
     **Recommendation:** when a future batch type
     introduces report generation, add a `batch/` top-level
     folder at the Data Storage Location (peer to the
     experiment number directories) and wire it through
     the data-storage-location creation flow and the
     change-of-DSL flow in
     `BcSavePathDialog`/`BcSavePathWidget` plus
     `ApplicationConfigManager`. The drafter should
     **reflect this honestly**: state that report
     generation is currently a stub, name the proposed
     `batch/` convention as the recommended layout, and
     forward-link to the data-storage-location
     documentation rather than implying that the path
     already exists.

## Out of scope

- Adding a new hardware type or new driver — bundles 12m,
  12l.
- The AM and BM internal state machines — already on
  `:doc:`/classes/acquisitionmanager`` and
  `:doc:`/classes/batchmanager``.
- The cross-manager experiment-lifecycle flow — bundle
  12f.
- Storage class authoring (a new `FidStorageBase`
  subclass) — bundle 12i covers `DataStorageBase`; a new
  storage class would expand bundle 12g's pipeline page.

## Sources

### Related source files

- `src/data/experiment/ftmwconfig.{cpp,h}` — base class
  and the `FtmwType` enum.
- `src/data/experiment/ftmwconfigtypes.{cpp,h}` — the
  six concrete subclasses; the factory dispatch.
- `src/data/storage/fidstoragebase.{cpp,h}` and the three
  `Fid*Storage` concretes — for the storage-choice table.
- `src/gui/expsetup/experimenttypepage.{cpp,h}` —
  experiment-type selection.
- `src/gui/expsetup/loscanconfigwidget.{cpp,h}`,
  `drscanconfigwidget.{cpp,h}` — examples of mode-
  specific wizard pages.
- `src/gui/expsetup/experimentsetupdialog.{cpp,h}` —
  page-ordering logic.
- `src/acquisition/batch/batchmanager.{cpp,h}` — base.
- `src/acquisition/batch/batchsingle.{cpp,h}` — example
  of the simplest concrete subclass.
- `src/acquisition/batch/batchsequence.{cpp,h}` — example
  with `beginNextExperiment` override and timer-based
  scheduling.
- `src/gui/dialog/batchsequencedialog.{cpp,h}` —
  example batch-config dialog.
- `src/gui/mainwindow.{cpp,h}` — batch-launch sites
  (`startBatch`).

### Related dev-docs

None directly.

### Related user-guide pages

Forward-link, do not duplicate:

- `doc/source/user_guide/experiment_setup.rst`.

### Related API reference pages

- `doc/source/classes/ftmwconfig.rst` (covers all six
  current subclasses; new subclasses extend the page).
- `doc/source/classes/batchmanager.rst` (covers the base
  and the two existing concretes).
- `doc/source/classes/experiment.rst`
- `doc/source/classes/fidstoragebase.rst`
- `doc/source/classes/acquisitionmanager.rst`
- `doc/source/classes/rfconfig.rst` (for multi-segment
  setup).

## Sphinx file deltas

**Created:**

- `doc/source/developer_guide/adding_an_experiment_mode.rst`.

## Page structure

H1 intro: 1 paragraph framing the page as covering two related
extension points (FtmwType, BatchManager subclass).

H2 sections (`-` underlines):

- *A new FtmwType*

  H3 (`'` underlines):

  - *When to add a new type*
  - *The touches: enum, subclass, factory, wizard, API page*
  - *Multi-segment vs. single-segment design*
  - *Completion: shot-based, wall-clock, indefinite*
  - *Storage choice*

- *A new BatchManager subclass*

  H3:

  - *When to add a new subclass*
  - *The touches: enum, subclass, dialog, MainWindow entry,
    API page*
  - *Building the wiring*
  - *Coordination with the AcquisitionManager*
  - *Persistence*

## Acceptance criteria

- The page is split into two clearly-labeled sections (A
  for FtmwType, B for BatchManager).
- Section A enumerates the six existing FtmwType values
  and gives a "when to add a new type" criterion.
- Section A's five-touch recipe (enum, subclass, factory,
  wizard, API page) is explicit, with the FtmwConfig
  virtuals named.
- The multi-segment vs. single-segment guidance points to
  the right storage class for each.
- Section B enumerates the two existing BatchManager
  subclasses with their `BatchType` value and gives a
  "when to add" criterion.
- Section B's five-touch recipe (enum, subclass, dialog,
  MainWindow entry, API page) is explicit, with the
  BatchManager virtuals named.
- The role of `beginNextExperiment` is explained as the
  optional override for non-immediate scheduling.
- The cross-manager handoff (AM → BM via queued
  signal/slot) is named without duplicating bundle 12f.
- Persistence guidance covers the `SettingsStorage`
  namespace convention for batch configuration, and
  honestly documents that report generation is currently a
  stub. The proposed `batch/` top-level folder at the Data
  Storage Location is named as the recommended future
  layout but is not described as already existing.
- No duplication of per-class API content; cross-links
  cover per-class detail.
- No rendered link points into `dev-docs/`.
