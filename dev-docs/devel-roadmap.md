We are in the middle of a hardware settings registry migration for the Blackchirp project
(cmakemigration branch). The goal is to replace runtime setDefault() calls in hardware
constructors with static REGISTER_HARDWARE_SETTINGS macros so that settings are available
before object construction.

Start by reading these planning documents in full:
  - dev-docs/settings-registry.md — overall design and current status (the Step 3 status
    tables show what is done and what remains)
  - dev-docs/phase4-settings/agent-instructions.md — the instructions given to each Haiku
    agent that performs a migration
  - dev-docs/phase4-settings/baseclass.toml — approved settings for hardware base classes
  - The per-type TOML file in dev-docs/phase4-settings/ for the type you are about to work 
on

Your role is orchestrator. Choose 1 hardware type to migrate, then spawn a Haiku subagent
(using the Agent tool with model: "haiku") to carry out the migration. The agent
instructions file explains exactly what each agent must do. Use the status tables in
settings-registry.md to determine what remains and update the table when a type is
complete. After the agent finishes, stop and wait — do not pre-select or queue the next
type. The user will test, confirm, and prompt you to commit changes before moving on.

Key constraints to pass on to each agent (and follow yourself):
  - No worktrees — agents edit files in place
  - No building — the user builds and reviews after the agent completes
  - Agents may use codebase-memory-mcp (project: home-kncrabtree-github-blackchirp-src) if
    they need to explore code beyond the files listed in the instructions

The per-type research (reading source files, finding setDefault calls, checking for
configParams) belongs in the agent, not the orchestrator. The agent is already instructed
to read the TOML and reference files; give it the list of source files to modify and let
it do the archaeology. Your job before spawning is narrower: glance at the source files
just enough to identify any non-obvious situations that the agent might not infer from the
TOML alone (e.g., a constructor lambda that reads a setting before the base class is
constructed, or a two-class file that requires a refactor). Flag those in the brief; do
not pre-compute the macro content.

Base class registrations (HardwareObject, Clock, IOBoard, TemperatureController — see
baseclass.toml) are not covered by any per-type TOML. Handle them as a standalone agent
or fold them into the first agent for the corresponding hardware type.

The FlowController migration (commit ba485294) is the reference for how a completed
migration looks.


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

### [Generalized AWG Marker System](awg-marker-system.md)
Replace the hardcoded 2-marker (protection/gate) system with a flexible N-marker
architecture. Users define named marker channels with roles (Protection, Gate, Trigger,
Custom) and chirp-relative timing. AWGs report `markerCount` and pack their own bitfields.
Phase 2 adds absolute timing and per-chirp marker overrides.

### [Hardware Settings Registry](settings-registry.md)
**In Progress** Unified settings registration system with metadata (labels, descriptions,
priority levels). Hardware classes declare settings via static macros;
settings are available before construction and presented at profile creation
time. Replaces the `HwConfigParam` system and eliminates the gap where users
must manually discover and configure settings after profile creation.
**Progress:** Phases 1-2 complete (infrastructure + 4 pilot classes).
Next: Phase 3 (creation-time UI) after dialog refactoring.

## Large

### [String Usage](string-usage.md)
QString is used extensively throughout the codebase, which has performance and
memory consequences. Evaluate the usages of QString across Blackchirp, considering
whether QAnyStringView should replace QString in functions and whether settings/header
keys etc. should be replaced with inline constexpr auto x = "string"_s, available
in the Qt::StringLiterals namespace. A preliminary analysis of the performance and
memory tradeoffs by Gemini is provided.

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

### [Logging and Debug Message Cleanup](logging-cleanup.md)
Review and rationalize all qDebug() (~41 calls) and logMessage() (~445 calls) output.
Eliminate qDebug in favor of the log system, downgrade diagnostic traces from Error/Normal
to Debug, and remove development scaffolding. Bulk of work is in FTMW digitizer files
(~285 calls) and HardwareManager (~74 calls). Should be one of the last tasks before
documentation revision for 2.0.0.

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
