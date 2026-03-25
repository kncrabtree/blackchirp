# Development Roadmap

Projects sorted by estimated complexity (smallest first). All are largely independent.

## Small

### [Hardware Threading Configuration](hardware-threading.md)
Restore per-object threading and make it user-configurable via RuntimeHardwareConfig.
Type-level defaults in intermediate classes; per-instance override persisted in
HardwareSelection and applied during HardwareManager object creation.

### [Widget Settings Cleanup on Profile Removal](widget-settings-cleanup.md)
When a hardware profile is purged, also purge associated widget settings (e.g.,
`PulseWidget.PulseGenerator.main`). Currently only the hardware object's own settings
group is cleaned up, leaving stale widget settings behind.

## Medium

### [Unified Application Configuration Dialog](app-config-dialog.md)
Consolidate scattered application settings (font, save path, LIF toggle, debug logging)
into a single dialog with a declarative option registry. Also serves as the first-run
onboarding experience. CUDA remains hidden/disabled until revived.

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
