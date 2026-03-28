# Development Roadmap

Projects sorted by estimated complexity (smallest first). All are largely independent.

## Medium

### Labjack Cross-Platform Support
Currently, Blackchirp will not compile on a system that does not have the LabJack
exodriver package installed, which is a Linux-only driver. This breaks 2 desired
features: cross-platform support AND runtime library configuration rather than
compile-time. The issue is the "labjackusb.h" header inclusion in
src/hardware/optional/ioboard/u3.h. Needs investigation for how to get the correct
library on MacOS and Windows and how to enable compilation without library.
First step is to research and create a new labjack-cross-platform.md file with a
plan, and then reassess whether the scope is small, medium, or large, updating
this entry accordingly.

### [Generalized AWG Marker System](awg-marker-system.md)
Replace the hardcoded 2-marker (protection/gate) system with a flexible N-marker
architecture. Users define named marker channels with roles (Protection, Gate, Trigger,
Custom) and chirp-relative timing. AWGs report `markerCount` and pack their own bitfields.
Phase 2 adds absolute timing and per-chirp marker overrides.

## Large

### [Python Hardware Implementations](python-hardware.md)
User-editable Python scripts as hardware drivers via pybind11 embedded interpreter.
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

### [Logging and Debug Message Cleanup](logging-cleanup.md)
Review and rationalize all qDebug() (~41 calls) and logMessage() (~445 calls) output.
Eliminate qDebug in favor of the log system, downgrade diagnostic traces from Error/Normal
to Debug, and remove development scaffolding. Bulk of work is in FTMW digitizer files
(~285 calls) and HardwareManager (~74 calls). Should be one of the last tasks before
documentation revision for 2.0.0.
