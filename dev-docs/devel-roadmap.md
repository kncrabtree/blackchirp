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

`ExperimentLOScanConfigPage` and `ExperimentDRScanConfigPage` are separate ESD
navigation nodes today. The clock-range dependency (both pages call
`rfConfig.clockRange()` to constrain spinbox ranges) is a soft dependency: the
controls can exist without live constraints, and out-of-range values are caught
during validation. The range constraints could be applied lazily via
`ExperimentSetupDialog::pageChanged` or simply left to validation.

Controls will move to `ExperimentSetupPage`:** scan type (Single / LO Scan / DR   Scan / Target Shots / etc.) and all scan parameters live together on the  first page where the user already chooses the experiment type. Range constraints will need to be applied during validation and via `pageChanged` signaling from the FTMW page. Consider making a special `FtmwViewWidget::clockHwChanged` signal that can trigger validaton/constraint rechecking, as the dependency is based on the actual clock hardware which should rarely change from the ESD itself.

**Proposed shape**

- Extract `LOScanConfigWidget` and `DRScanConfigWidget` from the existing page classes (own `.{h,cpp}` files, inherit `QWidget` + `SettingsStorage`).
- `ExperimentSetupPage` gains a `QStackedWidget` that shows the appropriate
  scan widget when scan type changes. The simpler Ftmw modes (Target shots, target time, forever, Peak Up), when selected, can surface a simpler wrapper widget (managed directly in `ExperimentSetupPage`) in the stacked widget with the necessary settings as needed (number of shots for Target shots and peak up, duration and Est. completion label for target time [which should be adjusted to use themecolors!], nothing for Forever). The `QStackedWidget` is placed between the type selection box and the Phase correction checkbox. Where necessary, the layouts should be reconsidered to be taller than they are wide.
- Range constraints: trigger reapplication of constraints and validation on new `FtmwViewWidget::clockHwChanged`; not necessary to re-validate and apply constraints on any page change.
- Delete `ExperimentLOScanConfigPage` and `ExperimentDRScanConfigPage`; remove their tree nodes from `ExperimentSetupDialog`.

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
