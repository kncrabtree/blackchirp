# Bundle 12j — Developer Guide: Python Hardware

**Status:** complete

<!--
Status log:
- 2026-05-03: not started → complete. New page
  doc/source/developer_guide/python_hardware.rst landed with the
  scope's nine sections (subprocess+IPC rationale, push model,
  proxy injection, mixin members, three state-management
  patterns A/B/C, trampoline recipe, QSettings key-path gotcha,
  hot-reload, per-profile env). The trampoline→pattern table
  was forwarded to the user-guide companion at
  python-hardware-trampoline-overview rather than duplicated.
  Recipe step 4 dropped its `forbiddenKeys()` /
  `pythonForbiddenKeys()` clause: neither helper exists in the
  current source tree (verified by grep across src/hardware/);
  this sub-bundle file and dev-docs/python-hardware.md were
  edited to remove the obsolete reference. Content commit
  b7e3729b.
-->

Sub-page of the Developer Guide chapter. Documents the Python
hardware subsystem from a developer's perspective: the IPC
architecture, proxy injection, the three base-class state-management
patterns, the trampoline implementation contract, and the recipes
for adding a new trampoline class and a new push-style proxy.

The user-facing perspective (writing a `.py` driver script,
selecting a profile, hot-reload from the UI) is in
`:doc:`/user_guide/python_hardware`` and its sub-pages. This page
documents what a contributor needs when *adding to the C++ side* of
the Python hardware system.

## Scope

Single Sphinx file:
`doc/source/developer_guide/python_hardware.rst`.

The page should answer the following for a contributor:

1. **Why a subprocess + JSON IPC.** Brief rationale:

   - Earlier work attempted an embedded Python interpreter via
     pybind11. The pybind11 + Python 3.13 + mimalloc combination
     produced persistent heap corruption in multi-threaded Qt
     applications, and pybind11 is tightly version-coupled.
   - The subprocess approach gives complete memory isolation
     (Python heap is in a separate process; crashes cannot
     corrupt Qt state), no GIL concerns (the GIL is entirely
     within the subprocess), Python-version independence (the
     C++ side only uses `QProcess`), and crash resilience
     (Python exceptions surface as clean errors on the C++
     side).
   - The cost is IPC overhead (~1 ms per round-trip), which is
     negligible compared to typical instrument I/O latencies
     (10–100 ms).

   Keep this section tight (one paragraph). The page is about
   *how the system works now*, not the history.

2. **IPC protocol.** Brief reference, not exhaustive:

   - Transport: `stdin`/`stdout` pipes via `QProcess`. Format:
     JSON-lines (one JSON object per line, compact).
   - Direction C++ → Python: synchronous `sendRequest()` —
     write a method-call JSON object, run a nested `QEventLoop`
     until the matching `id`-tagged response arrives.
     `QFutureWatcher`-style asynchronous calls are not used;
     `sendRequest` blocks the caller's thread.
   - Direction Python → C++: three message types:
     - **Relay** (synchronous from Python's POV): `self.comm.query`
       and `self.settings.get`/`set` need to call C++. Python
       writes a relay request, C++ services it
       (`p_comm->queryCmd`, `SettingsStorage::get`, etc.), and
       writes a relay response. The C++ `sendRequest` read loop
       handles interleaved relay requests while waiting for the
       method response.
     - **Log message** (unsolicited): `self.log.log(msg)` etc.
       Python writes a log record; C++ forwards to `bcLog`
       with the appropriate severity.
     - **Waveform push** (unsolicited): `self.scope.emit_shot()`
       writes a base64-encoded waveform record; C++ emits
       `waveformReceived(QByteArray, quint64)` on the
       `PythonProcess`.
   - Cross-link to `:doc:`/classes/pythonprocess`` for the
     class-level API.

3. **`PythonProcess` push model.**

   - `QProcess::readyReadStandardOutput` is connected to
     `onReadyRead`, which accumulates partial-line data in
     `d_readBuf` and parses each `\n`-terminated JSON line. A
     dispatch table routes by message type (`log`, `waveform`,
     `relay`, `id`).
   - `sendRequest` uses a nested `QEventLoop`:
     ```cpp
     QEventLoop loop;
     connect(this, &PythonProcess::responseReady, &loop, &QEventLoop::quit);
     connect(p_process, &QProcess::finished, &loop, [&]{
         onReadyRead();   // drain remaining stdout
         loop.quit();
     });
     QTimer::singleShot(d_timeoutMs, &loop, &QEventLoop::quit);
     loop.exec();
     ```
     This lets relay requests and waveform pushes be processed
     while `sendRequest` is waiting.
   - **Reentrancy:** A waveform arriving during `sendRequest`
     fires `waveformReceived` from inside the nested loop. The
     trampoline's `onWaveformReceived` slot only writes to the
     `WaveformBuffer` and never re-enters `sendRequest`; this is
     the contract that keeps the model safe.

4. **Proxy injection.**

   - The Python host script (`python_hw_host.py`) injects three
     standard proxies onto the user object on every subprocess
     start: `self.comm`, `self.settings`, `self.log`.
   - Optional, hardware-type-specific proxies (currently
     `self.scope` for digitizer push) are gated by
     `setEnabledProxies()`. The trampoline calls
     `pu_process->setEnabledProxies({"scope"})` between
     `initPythonProcess()` and the first `sendRequest()`; the
     C++ `_init` message carries the proxy list, and the host
     script's factory map (`_OPTIONAL_PROXY_FACTORIES`)
     instantiates only the requested ones.
   - **Adding a new push-style proxy** (extension recipe):
     1. Implement the proxy class on the Python side in
        `python_hw_host.py`.
     2. Add an entry to `_OPTIONAL_PROXY_FACTORIES`.
     3. Define the corresponding C++ message type in
        `PythonProcess::onReadyRead` (signal emission, payload
        parsing).
     4. Call `pu_process->setEnabledProxies({"yourproxy"})`
        in your trampoline's `initialize()` after
        `initPythonProcess()`.
     5. Connect `PythonProcess`'s new signal to the
        trampoline's handler slot.

5. **`PythonHardwareBase` mixin.**

   - The C++ trampoline inherits *both* its hardware base class
     (`AWG`, `Clock`, `FtmwDigitizer`, …) *and*
     `PythonHardwareBase` via multiple inheritance. The
     hardware base supplies the slot/signal API the rest of
     Blackchirp consumes; the mixin owns the subprocess and
     the IPC plumbing.
   - The mixin's constructor takes `(d_key, d_model)` strings;
     it does not need a back-pointer to `HardwareObject`.
   - Members the mixin owns / provides:
     - `pu_process` — the `PythonProcess` instance.
     - `initPythonProcess(comm, getter, setter)` — creates the
       process and wires settings get/set callbacks.
     - `testPythonConnection(comm)` — lazily starts the
       subprocess on first invocation, sends `test_connection`,
       returns success.
     - `startPythonProcess()` — looks up `pythonScriptPath` and
       `pythonClassName` from `HardwareProfileManager`; refuses
       to start if either is empty.
     - `findHostScript()` — locates `python_hw_host.py` (app
       dir, `share/blackchirp/`).
     - `resolvePythonExecutable()` — probes the configured env
       directory for `bin/python3` / `bin/python` /
       `Scripts/python.exe`; falls back to system `python3`.
     - `pythonSleep(b)`, `pythonReadSettings()` — common IPC
       dispatches the trampoline's overrides delegate to.
     - `pythonErrorString()` — exposes the human-readable error
       from the most recent failed `startPythonProcess` /
       `testPythonConnection`; trampolines copy it into
       `d_errorString` on failure.
     - The destructor stops `pu_process` if running.
   - Cross-link to `:doc:`/classes/pythonhardwarebase``.

6. **The three base-class state-management patterns.**

   The trampoline implementation depends on how the hardware
   base class manages its config state. There are three
   patterns:

   - **Pattern A — Bulk Configure.** The base class *inherits*
     from a complex config class (`DigitizerConfig`) and is
     reconfigured wholesale via a `configure()` virtual. The
     trampoline serializes the full config to JSON, sends a
     `configure` IPC call, parses the validated config back.
     Examples: `IOBoard` / `IOBoardConfig`,
     `LifScope` / `LifDigitizerConfig`. Note that `FtmwScope`
     does **not** have a `configure()` virtual; each subclass
     overrides `prepareForExperiment` directly. The Python
     trampoline `PythonFtmwScope` overrides
     `prepareForExperiment` to serialize via JSON IPC, and the
     `WaveformBuffer` is created automatically afterward by
     `FtmwScope::hwPrepareForExperiment`.
   - **Pattern B — Granular Methods.** The base class *contains*
     a config object and exposes per-channel /per-parameter
     getter/setter slots, each of which delegates to a `hw*`
     pure virtual. The base class owns sequencing (e.g.,
     `FlowController::poll` cycles channels). The trampoline
     implements each `hw*` virtual as a simple IPC call;
     trampolines do not override the polling sequence.
     Examples: `FlowController`, `PulseGenerator`,
     `TemperatureController`, `PressureController`.
   - **Pattern C — Stateless / Pass-Through.** The base class
     has no complex internal config. At
     `prepareForExperiment` time the trampoline serializes the
     experiment-supplied data (chirp config + markers for AWG;
     frequency assignments for Clock) and sends it.
     Examples: `AWG`, `Clock`.

   A short table mapping each `Python*` trampoline class to
   its pattern is helpful here. Cross-link to the relevant
   API pages for class-level detail.

7. **Trampoline implementation contract.** A condensed recipe
   for adding a new `PythonXxx` trampoline:

   1. Inherit from both the hardware base class and
      `PythonHardwareBase`. Initialize
      `PythonHardwareBase(d_key, d_model)` in the constructor.
   2. In `initialize()` (or the type-specific helper virtual,
      e.g., `fcInitialize` for `FlowController`), call
      `initPythonProcess(p_comm, getter, setter)` and connect
      `pu_process->logMessage` to `this->logMessage`.
   3. In `testConnection()` (or the type-specific helper,
      e.g., `fcTestConnection`), call
      `testPythonConnection(p_comm)`.
   4. Delegate `sleep()` to `pythonSleep()` and
      `readSettings()` to `pythonReadSettings()`.
   5. Implement hardware-specific virtuals as
      `pu_process->sendRequest(...)` dispatches. Pattern A
      classes implement `configure(...)`; Pattern B classes
      implement each `hw*` pure virtual; Pattern C classes
      typically only override `prepareForExperiment`.
   6. For push-style hardware (currently digitizers): call
      `pu_process->setEnabledProxies({"scope"})` after
      `initPythonProcess()` and connect
      `pu_process->waveformReceived` to a handler slot that
      writes into the `WaveformBuffer`.
   7. Register with `REGISTER_HARDWARE_META`,
     `REGISTER_HARDWARE_PROTOCOLS` (typically
     `Rs232 + Tcp + Gpib + Custom + Virtual`; omit any not
     applicable — `PythonGpibController` omits `Gpib` because
     it *is* the GPIB controller). `Custom` is the explicit
     "comm is in the .py script" indicator.
   8. Provide a template script
     (`python_<type>_template.py`) with the canonical class
     name (`AwgDriver`, `FtmwScopeDriver`, etc.). Templates
     must work out of the box with the Virtual protocol and
     document each method.
   9. The CMake glob in `BlackchirpHardware.cmake` picks up
     `*.cpp`, `*.h`, and `python_*_template.py` automatically;
     no manual cmake edit is needed. Confirm that the
     trampoline header is appended to
     `HARDWARE_IMPLEMENTATION_HEADERS` so AUTOMOC pulls in
     the registration initializers.

8. **QSettings key paths.** Brief reminder, not a recipe:

   - All persistent settings for a hardware object — the
     `commType` chosen at profile creation, the registry-
     declared Required/Important/Optional values, and any
     custom-comm parameters — live directly under the
     `<hwType:label>` `QSettings` group. That group is the
     `SettingsStorage` root for the `HardwareObject`.
   - Required parameters that used to flow through the
     standalone `HwConfigParam` system (the older
     `configParams` mechanism) now live in the same group as
     ordinary settings, declared via
     `REGISTER_HARDWARE_SETTINGS` with
     `HwSettingPriority::Required`. The settings registry is
     the single source of truth — see
     `:doc:`/developer_guide/hardware_configuration`` for
     the full registration model.
   - **Note on dev-docs research:** the
     `dev-docs/python-hardware.md` *QSettings Key Paths*
     subsection describes the older
     `<hwType:label>/<impl>/...` layered model and the
     `HwConfigParam` system. Both have been superseded by
     the settings registry. Read
     `dev-docs/settings-registry.md` for the current model
     when this section is drafted, and do not reproduce the
     outdated layered-path or `configParams` content from
     `dev-docs/python-hardware.md`.

9. **Hot-reload entry point.** The user reloads a script via
   the Python control widget; bundle 12e covers the
   `HardwareManager::reloadPythonScript` slot. From this
   page's perspective: the reload simply does
   `PythonProcess::stop()` followed by
   `startPythonProcess()`, which re-runs `_init` →
   `initialize` → `test_connection`. The C++ `HardwareObject`
   state (settings cache, comm protocol, signal connections,
   thread) is untouched; only the subprocess is replaced.

10. **Per-profile Python environment.**

    - `pythonEnvPath` (per-profile, in
      `HardwareProfileManager`) optionally points at a venv or
      conda environment directory. `resolvePythonExecutable`
      probes the standard layouts (`bin/python3`,
      `bin/python`, `Scripts/python.exe`); empty path falls
      back to system `python3` on PATH.
    - Per-profile env support means a script using a vendor
      Python SDK can ship with the SDK installed in a
      dedicated env without polluting the system Python.

## Out of scope

- The user-facing `.py` script API (`self.comm.query`,
  `self.settings.get`, the lifecycle methods) — already in
  `:doc:`/user_guide/python_hardware/writing_a_driver``.
- The user-facing template-copy / class-name-picker dialog —
  user guide.
- The hot-reload UI surface — bundle 12e (control widget)
  and the user guide.
- Detailed digitizer `WaveformBuffer` mechanics — bundle 12g.

## Sources

### Related source files

- `src/hardware/python/pythonhardwarebase.{cpp,h}`.
- `src/hardware/python/pythonprocess.{cpp,h}`.
- `src/hardware/python/python_hw_host.py`.
- Each `src/hardware/python/python<type>.{cpp,h}` — read at
  least one example per pattern (e.g., `pythonflowcontroller`
  for Pattern B, `pythonawg` for Pattern C, `pythonioboard`
  for Pattern A, `pythonftmwscope` for the digitizer push
  case).
- The matching `python_<type>_template.py` files for the
  template-script convention.
- `src/hardware/core/hardwareprofilemanager.{cpp,h}` — for
  `pythonScriptPath`, `pythonClassName`, `pythonEnvPath`
  accessors.
- `src/cmake/BlackchirpHardware.cmake` — confirm that the
  Python hardware globs and the
  `HARDWARE_IMPLEMENTATION_HEADERS` append are still in place.

### Related dev-docs

Research material; do not link:

- `dev-docs/python-hardware.md` — the principal architecture
  document (motivation, IPC, contract, the three patterns,
  template scripts, status notes).
- `dev-docs/python-process-push-refactor.md` — the push model
  refactor, proxy injection.
- `dev-docs/python-script-reload.md` — hot-reload design.
- `dev-docs/python-env-support.md` — per-profile env path.

### Related user-guide pages

Forward-link, do not duplicate:

- `doc/source/user_guide/python_hardware.rst` and
  `python_hardware/` sub-pages.

### Related API reference pages

- `doc/source/classes/pythonhardwarebase.rst`
- `doc/source/classes/pythonprocess.rst`
- `doc/source/classes/hardwareobject.rst`
- `doc/source/classes/hardwareprofilemanager.rst`
- `doc/source/classes/communicationprotocol.rst`
- `doc/source/classes/custominstrument.rst`
- `doc/source/classes/waveformbuffer.rst` (for the digitizer
  push case)

## Sphinx file deltas

**Created:**

- `doc/source/developer_guide/python_hardware.rst`.

## Page structure

H1 intro: 2 paragraphs framing the developer view (subprocess
isolation, IPC, mixin pattern) and pointing the user-guide
audience back to the user-guide chapter.

H2 sections (`-` underlines):

- *Subprocess and JSON IPC*
- *PythonProcess push model*
- *Proxy injection*
- *PythonHardwareBase mixin*
- *Three state-management patterns* — A/B/C with examples and
  a small table.
- *Trampoline implementation contract* — recipe.
- *QSettings key paths* — the gotcha.
- *Hot-reload* — one paragraph.
- *Python environments* — one paragraph.

A short table mapping each `Python*` trampoline class to its
pattern and template-script class name is the most useful
single visual.

## Acceptance criteria

- The subprocess+IPC rationale is one paragraph and timeless
  (no "we previously tried pybind11 and it failed" temporal
  framing — describe the current architecture and note in
  passing that an embedded interpreter is not used).
- The IPC protocol is documented at the message-type level
  (relay, log, waveform push, method call/response).
- The push-model nested-`QEventLoop` pattern is documented
  with the reentrancy contract.
- The proxy injection model and the
  `_OPTIONAL_PROXY_FACTORIES` extension recipe are
  documented.
- The `PythonHardwareBase` mixin's owned members are listed.
- The three state-management patterns (A/B/C) are each
  documented with at least one example each, and a class →
  pattern table is included.
- The trampoline-implementation recipe is a numbered list.
- The QSettings-key-path section states that all
  persistent settings live directly under
  `<hwType:label>` and that Required parameters are now
  declared via the settings registry (no `configParams` /
  layered subgroup model). Outdated content from
  `dev-docs/python-hardware.md` is not reproduced; the
  drafter consults `dev-docs/settings-registry.md` for the
  current model.
- Hot-reload and per-profile env are each one-paragraph
  pointers.
- No duplication of per-class API content; cross-links cover
  per-class detail.
- No rendered link points into `dev-docs/`.
