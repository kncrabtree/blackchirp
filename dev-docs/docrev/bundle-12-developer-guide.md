# Bundle 12 — Developer Guide

Builds the Developer Guide chapter from scratch in the
`doc/source/developer_guide/` folder created by bundle 00. This
bundle is independent of the user-guide track and can be worked in
parallel.

## Scope

The chapter is organised into self-contained sub-pages so a
contributor can land on a single topic without reading the rest.

- `developer_guide.rst` — chapter intro + toctree.
- `developer_guide/build_system.rst` — CMake layout
  (`CMakeLists.txt`, `cmake/*.cmake`), `BuildConfig.cmake` user
  options, hardware enable flags, debug vs. release, the docs
  target, the tests target, packaging via CPack at a developer's
  level. Sources: `dev-docs/packaging-and-ci.md`, `CLAUDE.md`.
- `developer_guide/architecture.rst` — module overview (acquisition
  / data / gui / hardware / modules), the major
  thread-of-control diagram (HardwareManager thread per device,
  AcquisitionManager event loop, GUI thread, QtConcurrent worker
  pool), where the data flows from FtmwScope through the buffer to
  AcquisitionManager. Reuses `digitizer-data-flow.md` content.
- `developer_guide/code_style.rst` — string usage, container
  conventions, key-declaration patterns (Pattern A/B/C), function
  signature policy, naming conventions, indentation. Sources:
  `dev-docs/string-usage.md`, `CLAUDE.md`.
- `developer_guide/logging.rst` — `LogHandler` global singleton,
  `bcLog`/`bcDebug`/`bcWarn`/`bcError`/`bcHighlight`, the `hw*`
  helpers in `HardwareObject`, severity guidelines, runtime debug
  toggle, on-disk format. Source: `dev-docs/string-usage.md`
  (Logging section).
- `developer_guide/settings_storage.rst` — `SettingsStorage`
  concepts, key namespaces in `bcglobals.h` and `hardwarekeys.h`,
  protected `set` vs. public `get`, Hardware Settings Registry
  (`REGISTER_HARDWARE_SETTINGS`, `REGISTER_HARDWARE_BASE`,
  priority levels). Source: `dev-docs/settings-registry.md`.
- `developer_guide/hardware_registration.rst` — the trio of
  `REGISTER_HARDWARE_META`, `REGISTER_HARDWARE_PROTOCOLS`,
  `REGISTER_HARDWARE_SETTINGS`, base-class inheritance,
  base-array placeholders, runtime hardware configuration flow.
- `developer_guide/communication_protocols.rst` —
  `CommunicationProtocol` hierarchy, runtime protocol selection,
  protocol-specific widgets, `CustomInstrument` for
  hardware-specific transports.
- `developer_guide/experiment_lifecycle.rst` — `Experiment`
  construction, `prepareForExperiment` chain, `beginAcquisition`,
  `endAcquisition`, completion conditions, batch experiments.
- `developer_guide/digitizer_data_flow.rst` — `WaveformBuffer`,
  drop-newest overflow, backpressure-triggered pre-accumulation,
  parallel batch parse, AcquisitionManager consumer loop. Source:
  `dev-docs/digitizer-data-flow.md`.
- `developer_guide/python_hardware.rst` — full architecture for
  developers (PythonProcess, PythonHardwareBase, IPC protocol,
  proxy injection, scope push, trampoline contract for adding a
  new `Python*` class). Source: `dev-docs/python-hardware.md`,
  `dev-docs/python-process-push-refactor.md`.
- `developer_guide/vendor_libraries.rst` — `VendorLibrary` base
  class, dynamic-loading pattern, `LibraryStatusWidget`,
  per-platform back ends (LabJack exo/UD split, Spectrum). Source:
  `dev-docs/labjack-cross-platform-support.md`.
- `developer_guide/adding_hardware_implementation.rst` — adding a
  new implementation of an existing hardware type (the most
  common task). Step-by-step.
- `developer_guide/adding_hardware_type.rst` — adding a new
  hardware type entirely (the rarer task): `HardwareObject`
  inheritance, virtual methods, virtual instance, optional Python
  template, `HeaderStorage`, rolling/aux data, validation keys,
  status box, control widget, optional config object.
- `developer_guide/adding_experiment_objective.rst` — adding a
  new `FtmwConfigType` (target shots / duration / forever / peak
  up / LO scan / DR scan are the existing ones) or a new batch
  experiment.

## Out of scope

- API class reference pages (bundles 13a–13h). The developer
  guide may cross-link to API pages but does not duplicate
  Doxygen content.
- Hardware-implementation specifics — the implementer fills in
  device-specific details when they actually add hardware.

## Sources

Listed per sub-page above. The principal dev-doc reuses are:

- `dev-docs/string-usage.md`
- `dev-docs/settings-registry.md`
- `dev-docs/loadout-system.md`
- `dev-docs/python-hardware.md`
- `dev-docs/python-process-push-refactor.md`
- `dev-docs/python-script-reload.md`
- `dev-docs/python-env-support.md`
- `dev-docs/digitizer-data-flow.md`
- `dev-docs/awg-marker-system.md`
- `dev-docs/labjack-cross-platform-support.md`
- `dev-docs/packaging-and-ci.md`

The `CLAUDE.md` at the project root is the single source of truth
for build commands and directory navigation; quote it where
applicable.

## Sphinx file deltas

**Modified:**
- `doc/source/developer_guide.rst` — populated from scaffold.

**Created:**
- `doc/source/developer_guide/build_system.rst`
- `doc/source/developer_guide/architecture.rst`
- `doc/source/developer_guide/code_style.rst`
- `doc/source/developer_guide/logging.rst`
- `doc/source/developer_guide/settings_storage.rst`
- `doc/source/developer_guide/hardware_registration.rst`
- `doc/source/developer_guide/communication_protocols.rst`
- `doc/source/developer_guide/experiment_lifecycle.rst`
- `doc/source/developer_guide/digitizer_data_flow.rst`
- `doc/source/developer_guide/python_hardware.rst`
- `doc/source/developer_guide/vendor_libraries.rst`
- `doc/source/developer_guide/adding_hardware_implementation.rst`
- `doc/source/developer_guide/adding_hardware_type.rst`
- `doc/source/developer_guide/adding_experiment_objective.rst`

## Toctree delta

In `developer_guide.rst`:

```rst
.. toctree::
   :hidden:

   developer_guide/build_system
   developer_guide/architecture
   developer_guide/code_style
   developer_guide/logging
   developer_guide/settings_storage
   developer_guide/hardware_registration
   developer_guide/communication_protocols
   developer_guide/experiment_lifecycle
   developer_guide/digitizer_data_flow
   developer_guide/python_hardware
   developer_guide/vendor_libraries
   developer_guide/adding_hardware_implementation
   developer_guide/adding_hardware_type
   developer_guide/adding_experiment_objective
```

## Screenshots

None for the developer guide; ASCII / Mermaid diagrams are
acceptable inline if helpful.

## Acceptance criteria

- A new contributor with strong C++/Qt skills but no project
  history can read `developer_guide.rst` plus
  `architecture.rst` and gain enough context to understand a
  pull request that touches any major subsystem.
- Each "adding" sub-page (implementation, type, experiment
  objective) walks through a concrete example: which files to
  create, which macros to invoke, which virtuals to override,
  and how to wire the result into the build.
- The developer guide cross-references the API reference
  (bundles 13a–13h) for class-level details rather than
  duplicating Doxygen content.
- The chapter does not duplicate dev-docs verbatim; it extracts
  and shapes them for an external audience.
