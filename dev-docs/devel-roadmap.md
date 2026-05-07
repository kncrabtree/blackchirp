# Development Roadmap

Projects sorted by estimated complexity (smallest first). All are largely independent.

## Medium

### Sirah Cobra integration refresh

A new Sirah Cobra dye laser is coming online in late May / early June
2026. Use that hardware as the trigger for revisiting the
`SirahCobra` driver: the existing TODO in
`src/hardware/core/liflaser/sirahcobra.cpp:112` flags that the
external-stage communication settings need a different solution
(separate baud / read terminator from the laser comm port). Today the
driver works around it by ad-hoc instantiating a second
`Rs232Instrument` alongside the inherited `p_comm`; this is the only
multi-port driver in the tree.

**Direction (chosen 2026-05-06):** Approach A — single
`HardwareObject` with multiple managed `CommunicationProtocol`
objects, formalized into reusable infrastructure. Approach B (a
composite manager over multiple `HardwareObject` subsystems) was
rejected for this device because wavelength, doubling crystal, and
compensator share calibration polynomials and move-direction state
that don't survive a thread boundary cleanly. A genuinely independent
device (the pump laser) should land as a sibling `HardwareObject` in
the loadout rather than as a child subsystem.

Implementation plan:

1. **Driver-declared aux ports.** Add a `REGISTER_HARDWARE_AUX_PORT`
   macro alongside `REGISTER_HARDWARE_PROTOCOLS`, declaring each
   secondary port's name and supported communication protocols. Adds a
   base-class hook (`auxPorts()` or similar) for the lifecycle to
   iterate.
2. **`HardwareObject` lifecycle plumbing.** Before the driver's
   `initialize()` runs, the base class builds each declared port's
   `CommunicationProtocol` from settings, wires its
   `hardwareFailure()` into the device's, and exposes it as
   `auxPort(name)`. Symmetric teardown on destruction.
3. **Comm-config UI.** Extend the existing comm-config dialog so it
   shows one tab per port (primary + each aux). The per-protocol
   widgets are reused unchanged.
4. **Settings hierarchy.** Aux-port settings nest under the device
   key: e.g. `LifLaser.sirah/extStage/rs232/baud`. The existing
   `BC::Key::Comm::*` constants stay; per-port nesting is one extra
   level.
5. **Sirah migration.** `p_extStagePort` becomes
   `auxPort("extStage")`. The `hasExtStage`, `extStagePort`,
   `extStageBaud` ad-hoc settings collapse into the auto-managed comm
   subgroup. The line-112 TODO (read options on the secondary port)
   becomes a property on the declared port descriptor.

Rough scope: ~200–400 LOC in `HardwareObject` / `buildCommunication`
/ comm-config dialog plus the macro, and a small Sirah migration on
top. Plan the dev-doc draft (settings layout, dialog mockups, macro
signature) when the new instrument is on the bench and after the
2.0.0-alpha packaging work is finished.

## Large

None.

## Pre-Release

### [Crash Reporting](crash-reporting.md)

End-user crash diagnostics for stripped release builds. POSIX (Linux +
macOS) installs `sigaction` handlers that capture a `std::stacktrace`
and write a text crash log under `<savePath>/log/crashes/`; Windows
installs `SetUnhandledExceptionFilter` and emits a minidump via
`MiniDumpWriteDump`. Symbols (`.debug`, `.dSYM`, `.pdb`) are kept
developer-side as 90-day GitHub Actions workflow artifacts — never
shipped to users — and resolved against a crash log's embedded git
SHA via `addr2line` / `atos` / WinDbg. Builds at the application
side: ~60 lines of POSIX boilerplate, a similar Windows path, and a
symbol-capture step in `.github/workflows/release.yml`. Lands after
the packaging-and-ci verification settles since it edits the same
workflow file. See `crash-reporting.md` for file layout, phasing, and
the developer triage runbook.

### [Packaging and Binary Generation (Github Actions)](packaging-and-ci.md)

CMake-side work is complete: `cmake/Packaging.cmake` produces release-only DEB,
RPM, DMG, NSIS, and TGZ/ZIP packages via CPack; `cmake/QtDeployment.cmake` wires
`windeployqt`/`macdeployqt` as install hooks; and `.github/workflows/release.yml`
defines five jobs (linux-deb, linux-rpm, linux-appimage, macos-dmg, windows-nsis)
triggered by `workflow_dispatch` and `release: published`. **Remaining work is
testing and verification**: dispatch the workflow per-platform, iterate on
first-run issues (likely candidates: linuxdeploy library paths, macdeployqt's
Qwt resolution, windeployqt's runtime DLLs), and confirm the resulting packages
launch on clean VMs. See `packaging-and-ci.md` for the full strategy reference,
file map, and acceptance criteria.
