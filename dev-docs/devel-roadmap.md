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

### Async PythonProcess + hardware base contracts

Refactor `PythonProcess::sendRequest` from its current
synchronous-with-nested-`QEventLoop` shape into a true async API
(`QFuture<QJsonObject>` or callback-style), and propagate the change
through every Python-driver-facing hardware base class
(`FlowController`, `PressureController`, `TemperatureController`,
`IOBoard`, `LifLaser`, `LifDigitizer`, `FtmwDigitizer`, `Clock`,
`ChirpSource`/AWG, `PulseGenerator`, `GpibController`).

Motivation: the nested event loop in `sendRequest` is the structural
source of a destruction race observed at app shutdown — the loop
processes events that can free `this` mid-call, so the post-loop
member accesses dereference a corpse. The shutdown ordering fix in
`python-process-shutdown-fix.md` (B + QPointer guard) treats the
observed trigger and one defensive case but does not eliminate the
class of bugs. Any new caller that initiates a `sendRequest` during
a destructible sequence is still re-entrant.

Why it cannot be hidden inside `PythonProcess`: relay requests from
the Python script (`self.comm.write`, `self.settings.set`,
`self.log`, scope waveform pushes) need to be serviced *on the
hardware thread* while a Python method is in flight. Blocking the
hardware thread on a semaphore while a separate dispatcher services
relays deadlocks on the `BlockingQueuedConnection` back into the
blocked thread. So either the nested loop stays (current) or the
contract changes all the way out to the per-driver virtual.

Rough scope: ~500–1000 LOC across ~50 files; 2–3 days of focused
work plus a per-driver-type testing pass and updates to
`developer_guide/adding_a_driver.rst`. Trigger for picking it up:
the next major hardware-contract change (e.g., the Sirah aux-port
work, or a new "remote hardware proxy" driver type that genuinely
needs async), or evidence in production that the QPointer guard in
`sendRequest` is being hit.

## Pre-Release

### [Crash Reporting](crash-reporting.md)

End-user crash diagnostics for stripped release builds. POSIX (Linux +
macOS) installs `sigaction` handlers that capture a `std::stacktrace`
and write a text crash log under `<savePath>/log/crashes/`; Windows
installs `SetUnhandledExceptionFilter` and emits a minidump via
`MiniDumpWriteDump`. Symbols (`.debug`, `.dSYM`, `.pdb`) are kept
developer-side as 90-day GitHub Actions workflow artifacts — never
shipped to users — and resolved against a crash log's embedded git
SHA via `addr2line` / `atos` / WinDbg.

In-process pieces have landed: `src/data/crashhandler.{h,cpp}` plus
the `_unix.cpp` and `_win.cpp` per-platform implementations; install
+ reopen + setActiveExperiment wired through `main.cpp`,
`BCSavePathWidget`, and `AcquisitionManager`; startup detection of
prior crashes via `CrashReportDialog`; and Sphinx user-guide and
developer-triage pages (`user_guide/crash_reports.rst`,
`developer_guide/crash_handling.rst`). Release builds now keep `-g`
/ `/Zi` so the captured frames resolve. Verified on Linux against a
null-pointer deref triggered from the About dialog.

The remaining piece is the CI symbol-capture step in
`.github/workflows/release.yml` (Phase 3 of `crash-reporting.md`).
That step lands after the packaging-and-ci verification settles since
it edits the same workflow file. See `crash-reporting.md` for the
phase table and triage runbook.

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

## Cleanups

Low-priority code-debt items, none release-blocking. Each is gated on
an external trigger; revisit when the trigger fires.

### Drop the `QAnyStringView` -> `QString` workaround in `hwLog/hwWarn/hwError/hwDebug`

`src/hardware/core/hardwareobject.h` calls `text.toString()` on the
`QAnyStringView` parameter before passing it to
`QString::arg(d_key, ...)` (the four `hwLog`-family one-liners around
line 403). Qt 6.4's `QString::arg` variadic-template trait
(`is_convertible_to_view_or_qstring`) does not accept
`QAnyStringView`; Qt 6.5 added it. The deb job's Ubuntu runner pins
to apt's `qt6-base-dev`, which on the current `ubuntu-latest`
(noble, 6.4.2) is the version that forces the workaround.

When the GitHub-hosted `ubuntu-latest` image rolls forward to an
Ubuntu release whose `qt6-base-dev` is >= 6.5, drop the
`.toString()` calls in those four lines and remove the explanatory
comment. No other call site is affected — `loghandler.cpp` already
pins its `QAnyStringView` entry point through a
`text.toString()` conversion before any `arg()` reaches it, and no
multi-arg `arg()` elsewhere in the tree consumes a `QAnyStringView`.

### Drop the `QStringView` -> `QString` workaround in `lifdisplaywidget.cpp`

`src/gui/lif/gui/lifdisplaywidget.cpp:102` concatenates `QString` with
`BC::Unit::us` (a `QStringView`) via `operator+`. Qt 6.4's `QString`
has no `operator+` overload accepting `QStringView`; Qt 6.5 added one.
The same Ubuntu-noble apt-Qt 6.4.2 ceiling that forces the
`hwLog`-family workaround is what forces this one. When the deb-job
Qt rolls forward to >= 6.5, drop the `.toString()` call and remove
the inline comment.

### MSVC cosmetic warnings

The Windows release build emits ~50 unique non-vendor warnings.
The three warnings with cross-platform behavior or correctness implications
(C4701 uninitialized `ChirpSegment`, C4702 dead `return`, C4804
`bool > 0`) are fixed. The remaining warnings produce identical,
well-defined behavior on MSVC and GCC; they're left as a future
clean-up pass when there's appetite for `-Wall`/MSVC-W4 hygiene work.

Categories, all platform-consistent (no Linux-vs-Windows divergence):

- **C4267** (size_t -> int truncation) — ~11 sites in
  `digitizerconfig.cpp`, `overlaystorage.cpp`, `settingsstorage.cpp`,
  `clock.cpp`, the AWG drivers, `temperaturecontrollerconfig.h`,
  `pulsestatusbox.cpp`. Channel/segment/array sizes that comfortably
  fit in `int`. Fix by switching the receiving locals to
  `qsizetype` / `std::size_t` or adding a `static_cast<int>` at the
  use site.
- **C4456 / C4457 / C4458 / C4459** (name shadowing) — ~17 sites,
  mostly inner `key` / `i` / `obj` locals shadowing globals or outer
  scopes. Inner scope wins identically on both compilers; rename
  the inner locals to silence.
- **C4101** (unused `e` in `catch`) — `xiamparser.cpp:371,497`,
  `catalogoverlaywidget.cpp:547`. Drop the binding (`catch (...)`)
  or `[[maybe_unused]]` it.
- **C4334** `ftworker.cpp:475` — `1 << zeroPadFactor` where
  `zeroPadFactor <= 2`; the int-shift result is then promoted to
  `size_t` for the surrounding multiplication. Cast the `1` to
  `size_t` to silence.
- **C4305** `ftmwconfig.cpp:429` — `float thresh = 1.15;` literal is
  `double`. Append `f` (`1.15f`) to silence.
- **C4309** `tst_waveformbuffertest.cpp:708` — `QByteArray(size, 0xAB)`
  truncates to `signed char`. `static_cast<char>(0xAB)` silences.
- **C4005** `crashhandler_win.cpp:10` —
  `WIN32_LEAN_AND_MEAN` already defined by Qt headers; guard the
  redefinition with `#ifndef`.
- **C4996** `main.cpp:94` — MSVC's "use `sprintf_s`" deprecation
  notice. Switch to `snprintf` (already available on both platforms)
  to silence without per-platform code.
