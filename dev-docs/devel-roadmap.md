# Development Roadmap

Projects sorted by estimated complexity (smallest first). All are largely independent.

## Small

- Address TODO item on line 106 of quickexptdialog.cpp - optional HW from HardwareRegistry?
- Remove dead configParams code from HardwareRegistry (and any callers)
- Move most settings registrations to base classes; subclasses override base class defaults. See
  hardware json files removed in commit 67bd00442be2c41da79ad51a686439520453857d for
  a convenient summary of implementation values to decide on sensible defaults for base
  classes. Remove registrations from derived classes unless they differ from base class
  defaults. (Mostly a mechanical refactor even though there are many files involved)

## Medium

### Labjack Cross-Platform Support

**Phase 1 (complete):** Removed compile-time dependency on the LabJack exodriver
vendor header (`labjackusb.h`) from `u3.h`. The `LabjackLibrary` class dynamically
loads the vendor library at runtime, so the header was unnecessary. Blackchirp now
compiles on all platforms without the exodriver installed. The exodriver works on
both Linux and macOS with the same API, so both platforms are fully supported at
runtime.

**Phase 2 (future — Windows support):** The LabJack U3 uses the UD Library
(`LabJackUD.dll`) on Windows instead of the exodriver. The UD library has a
different API (e.g., `eAIN` has different parameters — no calibration struct, no
ConfigIO flag — because UD manages state internally). Three approaches to evaluate:

1. Use UD library's `RAW_IN`/`RAW_OUT` IOTypes to send raw USB packets, allowing
   reuse of existing `u3.cpp` packet-building code
2. Use `libusb-1.0` directly on all platforms (exodriver is a thin wrapper)
3. Platform-conditional code paths calling UD easy functions on Windows
Note: LJM library does NOT support U3 (T-series only).

## Large

### ESD Scan Page Consolidation

**Prerequisite:** loadout system Phase E (creates `ExperimentFtmwConfigPage`).

`ExperimentLOScanConfigPage` and `ExperimentDRScanConfigPage` are separate ESD
navigation nodes today. The clock-range dependency (both pages call
`rfConfig.clockRange()` to constrain spinbox ranges) is a soft dependency: the
controls can exist without live constraints, and out-of-range values are caught
during validation. The range constraints could be applied lazily via
`ExperimentSetupDialog::pageChanged` or simply left to validation.

**Placement decision to resolve before implementation:**

- **Option A — on `ExperimentFtmwConfigPage`:** scan parameters sit adjacent
  to the RF/Chirp/Digi settings they derive from. Range constraints are natural
  to apply here. ESD tree loses two nodes but the FTMW page grows further.
- **Option B — on `ExperimentSetupPage`:** scan type (Single / LO Scan / DR
  Scan / Target Shots / etc.) and all scan parameters live together on the
  first page where the user already chooses the experiment type. This is more
  intuitive since the non-frequency scan settings (shots per step, target
  sweeps) already sit there. Range constraints would need to be applied during
  validation or via `pageChanged` signaling from the FTMW page.

Option B is likely the better UX because the user selects experiment type and
all associated settings in one place. The clock-range constraint is the only
technical reason those pages are separate today, and that can be handled
without co-location.

**Proposed shape (Option B):**

- Extract `LOScanConfigWidget` and `DRScanConfigWidget` from the existing page
  classes (own `.{h,cpp}` files, inherit `QWidget` + `SettingsStorage`).
- `ExperimentSetupPage` gains a `QStackedWidget` that shows the appropriate
  scan widget when scan type changes. Shots/sweeps settings already on that
  page merge into the scan widgets.
- Range constraints: `ExperimentSetupDialog::pageChanged` applies
  `initialize(rfConfig)` to the active scan widget when leaving the FTMW page,
  or validation emits an error if values are out of range.
- Delete `ExperimentLOScanConfigPage` and `ExperimentDRScanConfigPage`; remove
  their tree nodes from `ExperimentSetupDialog`.

### [Hardware Loadout System](loadout-system.md) **IN PROGRESS**

Named loadouts bundling hardware map + RF config + chirp config. Loadouts are edited in
the Hardware Configuration dialog (with new RF/Chirp tabs); experiment setup defaults to
the active loadout with a "Reset to Loadout Defaults" button. Includes loadout selection
UI, save prompts, and experiment initialization priority logic.

## Pre-Release

### [Documentation Revision](documentation-revision.md)

The sphinx/breathe documentation is outdated and needs to be updated for the
`cmakemigration` branch. The goals are:

- Improve the readme and program summary for the landing page and Github
- Update the user guide to provide a walkthrough of major program features and use-cases
- Maintain a hardware catalog of C++ drivers/capabilities
- Create a developer's guide to explain the overall code structure, conventions, major
data classes, and guides for adding new hardware and implementations
- Provide an API reference for the most important classes for developers. Specifically,
these should be classes like SettingsStorage, HardwareObject, etc that are used
throughout the code. These classes should have Doxygen-style annotations in headers for
autogeneration with breathe.

### Packaging and Binary Generation (Github Actions)

Ensure that cmake packaging instructions (cmake/Packaging.cmake) are compatible with
Github Actions runners for binary compilation for Windows, MacOS, and Linux (rpm and
deb). Binaries should be generated only on tagged releases, not on every push.
