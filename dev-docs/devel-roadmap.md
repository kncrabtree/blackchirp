# Development Roadmap

Projects sorted by estimated complexity (smallest first). All are largely independent.

## Small

### FtmwViewWidget Performance Under High Data Rates
When multi-record mode uses many large records (e.g., 750k+ points × 10+ records),
the live display's per-frame averaging can cause visible flicker at the default 500ms
refresh interval. The issue is CPU-bound: averaging and FT processing compete with
the acquisition worker for cores. Assess whether the AcquisitionManager's async 
processing can use a thread pool to distribute accumulation across available
cores, or whether the FtmwViewWidget averaging should be moved to a persistent worker to avoid
overlapping computations between refresh ticks.

### [Python Script Hot-Reload](python-script-reload.md)
Reload Python hardware scripts without restarting Blackchirp. Adds a control
widget to HWDialog with Open in Editor, Reload, and status display. Uses the
existing PythonProcess stop/start cycle — no HardwareObject reconstruction
needed.

### [Python Environment Support](python-env-support.md)
Per-profile venv/conda environment path for Python hardware. Replaces the
hardcoded `python3` executable with a configurable interpreter resolved from
the environment directory. Stored in `HardwareProfileManager` alongside the
existing script path and class name fields.

## Medium

### [Digitizer Data Flow Optimization](digitizer-data-flow.md)
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

### [Generalized AWG Marker System](awg-marker-system.md)
Replace the hardcoded 2-marker (protection/gate) system with a flexible N-marker
architecture. Users define named marker channels with roles (Protection, Gate, Trigger,
Custom) and chirp-relative timing. AWGs report `markerCount` and pack their own bitfields.
Phase 2 adds absolute timing and per-chirp marker overrides.

### [Hardware Settings Registry](settings-registry.md)
Unified settings registration system with metadata (labels, descriptions,
priority levels). Hardware classes declare settings via static macros;
settings are available before construction and presented at profile creation
time. Replaces the `HwConfigParam` system and eliminates the gap where users
must manually discover and configure settings after profile creation.

## Large

### [Python Hardware Implementations](python-hardware.md)
**In progress** User-editable Python scripts as hardware drivers via JSON IPC.
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
