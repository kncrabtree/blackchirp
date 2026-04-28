# Development Roadmap

Projects sorted by estimated complexity (smallest first). All are largely independent.

## Small

None

## Medium

### [Labjack Cross-Platform Support](labjack-cross-platform-support.md)

Linux and macOS already work via dynamic loading of the LabJack exodriver
(`liblabjackusb`). Windows uses a different vendor library — the UD library
(`LabJackUD.dll`) — with a different API shape (calibration is internal,
handle type and calling convention differ, easy-function parameters differ).

The plan adds Windows support by introducing a thin `BC::Labjack` operational
facade with per-device factories (`openU3`, with `openU6` slot ready for
future use) and an opaque `DeviceHandle`. Two backend translation units
implement the facade — one wrapping the existing exodriver helpers on
Linux/macOS, one wrapping UD easy functions on Windows. `LabjackLibrary`
remains the loader, with its symbol set conditionally selected per platform.
The full facade surface (eAIN/eDI/eDO/eDAC/eTCConfig/eTCValues) is exposed
up front so future ioboard features have everything they need, and the U6
can be added later as a strictly additive change.

## Large

None.

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
deb). Binaries should be generated only on demand, not on every push.
