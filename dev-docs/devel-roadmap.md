# Development Roadmap

Projects sorted by estimated complexity (smallest first). All are largely independent.

## Small

None.


## Medium

### [Digitizer Data Flow Optimization](digitizer-data-flow.md) **COMPLETE**
Replace per-shot Qt signal emission (Digitizer -> HardwareManager ->
AcquisitionManager) with a bounded SPSC ring buffer shared between the digitizer
and AcquisitionManager threads. Eliminates unbounded event queue growth and
unnecessary HardwareManager relay. The FtmwScope base class handles all buffer
management; digitizer implementations call `emitShot()` as before. Includes
optional producer-side pre-accumulation for backpressure handling. LifScope
stays signal-based. Python digitizers addressed at design level only.

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

### [Generalized AWG Marker System](awg-marker-system.md) **COMPLETE**
Replace the hardcoded 2-marker (protection/gate) system with a flexible N-marker
architecture. Users define named marker channels with roles (Protection, Gate, Trigger,
Custom) and chirp-relative timing. AWGs report `markerCount` and pack their own bitfields.
Phase 2 adds absolute timing and per-chirp marker overrides, but is deferred for a future
release.

### [Hardware Settings Registry](settings-registry.md) **COMPLETE**
Unified settings registration system with metadata (labels, descriptions, priority
levels). Hardware classes declare settings via static macros; settings are available
before construction and presented to the user at profile creation time with
human-readable labels, tooltips, and priority-based grouping. Replaces the
`HwConfigParam` system and raw key/value tree in the hardware settings dialog.

## Large

### [String Usage](string-usage.md) **COMPLETE**
Reference document for string literals, key declaration patterns, function signature
policy, container policy, and the logging API (`bcLog`/`bcWarn`/etc.).

### [Python Hardware Implementations](python-hardware.md) **COMPLETE**
User-editable Python scripts as hardware drivers via JSON IPC.
A C++ trampoline class per hardware type dispatches virtual methods to a user's Python
class. Users can use Blackchirp's CommunicationProtocol or bring their own vendor
libraries. Phased rollout: simple polling types first, then pulse generator and AWG,
then digitizers with tunable poll frequency.

### [Hardware Loadout System](loadout-system.md)
Named loadouts bundling hardware map + RF config + chirp config. Loadouts are edited in
the Hardware Configuration dialog (with new RF/Chirp tabs); experiment setup defaults to
the active loadout with a "Reset to Loadout Defaults" button. Includes loadout selection
UI, save prompts, and experiment initialization priority logic.

## Pre-Release

### Documentation Revision
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
