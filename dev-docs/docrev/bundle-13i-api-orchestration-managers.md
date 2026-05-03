# Bundle 13i — API Reference: Orchestration Managers

**Status:** complete

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
- 2026-05-02: drafted → complete. Content commit 2f50d947. Adds
  Sphinx pages and Doxygen refreshes for HardwareManager,
  AcquisitionManager, BatchManager, and ClockManager. Also removes
  three pieces of dead code surfaced during the documentation pass:
  AcquisitionManager::motorRest() (missed by 2021 motor removal),
  AcquisitionManager::takeSnapshot() + doFinalSave() (orphaned by
  the 2021 SnapWorker → QtConcurrent refactor), and the
  motorscan.svg icon entry re-added by accident during a 2025
  PNG → SVG migration.
- 2026-05-02: not started → drafted. Two parallel drafter/verifier
  cycles (Drafter A: HardwareManager; Drafter B: AcquisitionManager
  + BatchManager). Mid-flight scope extension added ClockManager via
  Drafter C. Each verifier returned one to three load-bearing items;
  all addressed in single revision passes. Working tree carries
  four new .rst files under doc/source/classes/ and Doxygen
  refreshes of the four target headers (including removal of
  Phase/Task temporal markers from hardwaremanager.h code comments
  and source-evolution comments from clockmanager.cpp).
-->

Adds API reference pages for the three top-level orchestration
classes that mediate between the GUI and the hardware /
acquisition / batch subsystems. None are abstract base classes,
but each is a load-bearing collaborator that a developer
integrating new hardware, a new acquisition mode, or a new batch
type must understand. Without these pages, the existing API
reference shows the leaf classes (`HardwareObject`,
`Experiment`, etc.) without the surrounding orchestration layer
that drives them.

## Scope

New pages under `doc/source/classes/`:

- `hardwaremanager.rst` ←
  `src/hardware/core/hardwaremanager.h`. Owns the live
  `HardwareObject` instances for the active loadout, marshals
  cross-thread calls, fans out connection / settings-change
  notifications, and exposes the public surface the GUI uses to
  initiate connection, scan, and shutdown sequences.
- `acquisitionmanager.rst` ←
  `src/acquisition/acquisitionmanager.h`. Drives a single
  `Experiment` from `prepareForExperiment` through
  acquisition-loop completion, owns the per-experiment
  `DataStorage*` tree, mediates FID / aux / rolling data fan-in,
  and emits the progress and completion signals consumed by the
  main window and the batch manager.
- `batchmanager.rst` ←
  `src/acquisition/batch/batchmanager.h`. Iterates a list of
  configured experiments (or a single experiment for non-batch
  runs), advances `AcquisitionManager` between experiments,
  handles auto-save / continue-on-error policy, and is the
  object that the main window connects to when starting any
  acquisition (single or batch).

## Out of scope

- Concrete batch-type subclasses (`BatchSingle`, `BatchSequence`,
  etc.) — these are leaf classes whose public surface is
  trivial; cross-link from `batchmanager.rst` rather than
  documenting separately.
- The thread / `QThread` plumbing in `main.cpp` that wires the
  three managers together. Mention it in the orientation prose
  on each page (which thread the object lives on, which signals
  cross thread boundaries) but do not document `main.cpp` itself.
- Per-hardware-type signal/slot menagerie. The
  `hardwaremanager.rst` page describes the *pattern* by which
  hardware-type-specific signals are exposed but does not
  enumerate every signal.

## Sources

- The three header files above.
- `dev-docs/` — check for any existing notes on the manager
  threading model or acquisition-state machine that should be
  cited rather than re-derived.
- The user-guide pages on experiment workflow
  (`doc/source/user_guide/experiment_setup.rst`) and hardware
  configuration (`doc/source/user_guide/hardware_config.rst`)
  for cross-link targets.
- Existing API pages for `Experiment`, `HardwareObject`,
  `HardwareLoadout`, `LoadoutManager`, `DataStorageBase` for
  collaborator cross-references.

## Sphinx file deltas

**Created:** one `.rst` per class above (three files).

**Possibly modified (Doxygen comment refresh):**
- All three headers listed above. The drafter may also touch
  the corresponding `.cpp` files only if a header `\brief` needs
  to reference an implementation detail that lives in the `.cpp`
  (rare; default to header-only).

`doc/source/classes.rst` uses `:glob:` so no toctree edits are
required.

## Style guidance specific to this bundle

These three classes are larger and more interconnected than
typical leaf classes. Apply the following:

- **Thread/ownership note.** Each page's orientation prose
  states which thread the object lives on, who owns its
  lifetime, and which signals cross thread boundaries. This is
  the single most useful piece of context for a developer
  integrating a new hardware or acquisition type.
- **Collaborator diagram in prose.** Each page names the
  immediate collaborators (e.g., for `AcquisitionManager`:
  `HardwareManager`, `Experiment`, `FidStorageBase`,
  `AuxDataStorage`, `BatchManager`) and the role each plays in
  one or two sentences — without re-documenting them. Use
  `:doc:` cross-references.
- **State-machine notes.** `AcquisitionManager` and
  `BatchManager` both walk through phased state. Where the code
  exposes named phases (enums, signal names like
  `experimentInitialized`, `beginAcquisition`, etc.), enumerate
  them in a short prose paragraph or a small table so a
  developer can map "what runs when" without reading the `.cpp`.
- **Per-method `///` docs.** Apply the api_style.rst rule:
  rich per-method documentation in the header, tight
  class-level `\brief`. Public signals get `\brief` blocks too.
- The `ParseSettings` / `ParsePreview` pattern from bundle 13h
  (nested types documented inline via `:cpp:class:` or literal
  back-tick references) applies here for any nested enum/struct
  these managers expose.

## Acceptance criteria

- Each page documents the manager's role, the thread it lives
  on, and its primary collaborators with `:doc:` cross-links.
- Each page enumerates the public signal/slot surface a
  consumer (typically the main window or another manager) would
  connect to, grouped by purpose (lifecycle, data fan-in,
  status, error).
- `acquisitionmanager.rst` and `batchmanager.rst` describe the
  state machine they drive in enough detail that a developer
  adding a new acquisition mode or batch type knows which
  signals/slots they must hook into.
- `hardwaremanager.rst` describes how new hardware types are
  surfaced through the manager's public API (the pattern, not
  every type-specific signal).
- All three headers conform to the api_style.rst convention:
  single-sentence class-level `\brief`, rich per-method `///`
  blocks, every public/protected member documented, no
  `QStringLiteral` introduced.
- Cross-links to the user-guide chapters that describe the
  surface visible to end users (experiment setup, hardware
  configuration).
