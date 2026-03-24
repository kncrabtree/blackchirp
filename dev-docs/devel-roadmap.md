# Development Roadmap

Projects sorted by estimated complexity (smallest first). All are largely independent.

## Small

### Restore d_threaded defaults from devel branch
The cmakemigration constructor refactor lost the per-type `d_threaded` defaults.
On `devel`, the base class defaults to `threaded = true`; FtmwScope, Clock,
IOBoard, LifScope, LifLaser, and GpibController all run in their own QThread.
Currently everything on cmakemigration is non-threaded.

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
