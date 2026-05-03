# Bundle 12e — Developer Guide: Hardware Runtime

**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
-->

Sub-page of the Developer Guide chapter. Documents the runtime half
of the hardware story: how `HardwareManager` brings a configured
hardware map to life, the threading rules, the communication-protocol
selection mechanism, the auxiliary-data fan-out, and the GUI
touchpoints in `CommunicationDialog` and `HwDialog`.

This page assumes the configuration model from bundle 12d. Together
the two pages cover hardware end-to-end without duplicating the API
reference.

## Scope

Single Sphinx file:
`doc/source/developer_guide/hardware_runtime.rst`.

The page should answer the following for a contributor:

1. **What `HardwareManager` owns.** Establish the role:

   - The runtime owner of every live `HardwareObject` instance for
     the active loadout.
   - Lives on a dedicated `QThread` created by `MainWindow` before
     the application event loop starts.
     `HardwareManager::initialize` is called from the thread's
     `started` signal; thereafter every public slot executes on
     this thread.
   - Owns `ClockManager` via `std::unique_ptr<ClockManager>
     pu_clockManager` (cross-link to
     `:doc:`/classes/clockmanager``).
   - Holds the hardware map under a read-write lock
     (`d_hardwareMapLock`) so multiple readers can query the map
     concurrently. Connection-state mutations use a separate mutex.
     Static `constInstance` provides read-only access from threads
     that cannot hold a direct reference.

2. **Hardware lifecycle: `syncWithRuntimeConfig` and friends.**

   - On `initialize`, `HardwareManager` calls
     `syncWithRuntimeConfig`, which reads `RuntimeHardwareConfig`
     and creates a `HardwareObject` for each active profile via
     `HardwareRegistry::createHardware`. The factory captured at
     `REGISTER_HARDWARE_META` time is invoked with the user-
     supplied label.
   - The `d_threaded` flag (set in the constructor of the interface
     class — e.g., `AWG::AWG` sets `d_threaded = true`) determines
     whether the object is moved to a dedicated thread. The
     per-profile threading override stored by
     `RuntimeHardwareConfig::setThreaded` takes precedence when
     present; describe the override flow.
   - For threaded objects: `HardwareManager` creates a `QThread`,
     names it `<hwKey>Thread`, calls `moveToThread`, wires
     `QThread::started` to `HardwareObject::bcInitInstrument`, and
     starts the thread. **Threaded hardware must not have a
     `QObject` parent and must construct child `QObject`s in
     `initialize()` rather than the constructor**, so they are
     constructed on the device thread; this is enforced socially,
     not by code, but failure to follow it produces hard-to-debug
     cross-thread parent issues.
   - For non-threaded objects: `HardwareManager` becomes the
     parent and dispatches `bcInitInstrument` on its own thread.
   - Connection testing is deferred until all hardware changes are
     complete — important when GPIB controllers must resolve their
     children. After the sync settles, `HardwareManager` calls
     `testAll`.
   - At runtime, `applyHardwareMap` and `syncWithRuntimeConfig`
     route every change through
     `addHardwareInternal` /
     `removeHardwareInternal` /
     `replaceHardwareInternal`, which handle thread teardown
     (`thread->quit()` then `wait` with a timeout, then
     `terminate` as last resort) and connection cleanup.

3. **`bcInitInstrument` and `bcTestConnection`.**

   - `bcInitInstrument` builds the `CommunicationProtocol` via
     `buildCommunication()` (the comm type is loaded from
     `QSettings` under the profile group), calls the comm's
     `initialize()`, calls the driver's `initialize()` override,
     and wires `hardwareFailure → clear d_isConnected`.
   - `bcTestConnection` reloads settings from disk (so the test
     sees up-to-date settings), calls
     `CommunicationProtocol::testConnection` (which exercises the
     underlying `QIODevice`), then calls the driver's
     `testConnection()` override. The result is stored in
     `d_isConnected` and reported via the `connected` signal.
   - The driver's `initialize()` is the right place for one-shot
     setup that does not need a live connection; the driver's
     `testConnection()` is the right place for a cheap interaction
     with the device (typically `*IDN?`) plus an assertion that the
     responding device is the expected model.
   - Cross-link to `:doc:`/classes/hardwareobject`` for the full
     virtual surface (sleep, prepareForExperiment,
     beginAcquisition, endAcquisition, readAuxData,
     readValidationData, readSettings).

4. **Communication protocols.**

   - The `CommunicationProtocol` hierarchy (`Rs232Instrument`,
     `TcpInstrument`, `GpibInstrument`, `CustomInstrument`,
     `VirtualInstrument`) wraps the OS-level I/O. Each driver
     declares which protocols it supports via
     `REGISTER_HARDWARE_PROTOCOLS`.
   - At profile-creation time, the user picks one of the supported
     protocols. The selection is stored in the profile's QSettings
     group; `bcInitInstrument` reads it back and constructs the
     matching protocol instance.
   - **`Custom` protocol** is special: it indicates that the
     driver's communication is handled outside the standard
     `QIODevice` abstractions. Drivers register
     `CustomCommDef` descriptors via `REGISTER_CUSTOM_COMM` (or
     `REGISTER_CUSTOM_COMM_BASE` for shared base-class
     parameters); the descriptors drive
     `CustomProtocolWidget` to render the right input widgets at
     profile creation. For Python-backed drivers, `Custom` is the
     explicit "comm is handled by the .py script" indicator —
     connection parameters live as constants in the script. See
     bundle 12j for the Python angle.
   - GPIB controllers have an extra layer: a `GpibController`
     hardware object owns the actual GPIB bus and resolves child
     `GpibInstrument` queries to its bus. This is why connection
     testing is deferred until after the full hardware sync — the
     controller must exist before its children can talk through
     it.
   - Cross-link to `:doc:`/classes/communicationprotocol`` and
     `:doc:`/classes/custominstrument``.

5. **Communication settings UI: `CommunicationDialog`.**

   - `CommunicationDialog`
     (`gui/dialog/communicationdialog.{cpp,h}` plus the `.ui`)
     is the single UI surface for changing a hardware object's
     communication protocol or connection parameters at runtime.
   - It requests current state via
     `HardwareManager::getHardwareCommunicationInfo(hwKey)`, which
     responds with
     `hardwareCommunicationInfoReady(hwKey, currentProtocol,
     supportedProtocols, connected)`. The dialog populates a
     protocol combo from `supportedProtocols`, hosts the
     transport-specific widget (`Rs232ProtocolWidget`,
     `TcpProtocolWidget`, `GpibProtocolWidget`,
     `CustomProtocolWidget`), and on accept calls
     `HardwareManager::setHardwareProtocol(hwKey, newProtocol,
     params)`. The result returns via
     `protocolSetResult(hwKey, success, msg)`.
   - For GPIB, the dialog populates the controller dropdown via
     `HardwareManager::getActiveGpibControllers()` →
     `gpibControllersAvailable(controllerKeys)`.

6. **Hardware settings and Control UI: `HwDialog`.**

   - `HwDialog` (`gui/dialog/hwdialog.{cpp,h}`) hosts two tabs
     when the hardware type provides them: a Settings tab
     containing an `HwSettingsWidget` in Edit mode (registry-driven
     fields, with Required settings shown read-only post-creation)
     and a Control tab containing a hardware-type-specific control
     widget (e.g., `GasControlWidget` for `FlowController`,
     `PulseGenChannelTable` for `PulseGenerator`).
   - On accept, `HwSettingsWidget` writes to QSettings;
     `HardwareManager` then dispatches `bcReadSettings` to the
     hardware object so the driver refreshes its cached state. For
     Python-backed drivers, `bcReadSettings` triggers an IPC
     `read_settings` message rather than restarting the
     subprocess (cross-link to bundle 12j).
   - The Control tab interacts with the live hardware object via
     `HardwareManager`-mediated slots; control widgets cannot
     touch `HardwareObject` directly because of the threading
     rules — every interaction is queued.

7. **Connection state and signal fan-out.**

   - `connectionResult(hwKey, success, msg)` is the unified
     signal for every connection-state change: a successful
     test, a failed test, a runtime hardware failure, a
     hardware removal. One subscription gives a consumer the
     full state picture.
   - `allHardwareConnected(bool)` fires after every
     `testAll` round; it indicates whether *every critical
     device* is connected. Non-critical devices' connection
     status is reflected only in `connectionResult`.

8. **Auxiliary, validation, and rolling data fan-out.**

   - Each `HardwareObject` may override `readAuxData()` and
     `readValidationData()` to return an
     `AuxDataStorage::AuxDataMap`. Aux data appears on the Aux
     and Rolling tabs and persists in `AuxDataStorage` as
     time-series; validation data is range-checked and aborts
     the experiment on failure.
   - `HardwareObject` also runs a rolling-data timer (interval
     loaded from settings) that calls `readAuxData()` outside
     the experiment context and emits `rollingDataRead`.
   - `HardwareManager` aggregates: it prefixes every map key with
     the source object's `hwKey` and re-emits as
     `auxData(map)`, `validationData(map)`, and
     `rollingData(map, timestamp)`. The hwKey prefix lets a
     consumer disambiguate readings from multiple devices of the
     same type without enumerating them in advance.
   - `AcquisitionManager` consumes `auxData` (writes to
     `AuxDataStorage`) and `validationData` (range-check; abort
     on violation). The Rolling and Aux tabs in the GUI consume
     `rollingData` directly. Cross-link to bundle 12f for the
     experiment-context details and to
     `:doc:`/classes/acquisitionmanager`` for the consumer
     state.

9. **Type-specific signal patterns.** Briefly: for each optional
   hardware category (pulse generator, flow controller, pressure
   controller, temperature controller), `HardwareManager` exposes
   a set of signals that all carry the source `hwKey` as their
   first argument. The GUI wires per-device widgets dynamically
   based on which `hwKey`s appear, rather than enumerating
   possible devices. Cross-link to
   `:doc:`/classes/hardwaremanager`` for the full signal list;
   note the *pattern*, not every signal.

10. **Python script reload entry point.** Briefly: the user
    triggers a script hot-reload via the Python control widget,
    which calls `HardwareManager::reloadPythonScript(hwKey)`.
    `HardwareManager` stops the subprocess, restarts it (which
    re-runs `_init` → `initialize` → `test_connection`), and
    reports the outcome via
    `pythonScriptReloadResult(hwKey, success, msg)`. Full
    architecture in bundle 12j.

## Out of scope

- The configuration model (registry, profiles, runtime config,
  loadouts) — bundle 12d.
- Vendor library integration — bundle 12k.
- The Python trampoline contract and IPC details — bundle 12j.
- Adding a new hardware implementation or new hardware type —
  12l, 12m.
- The experiment lifecycle as a whole — bundle 12f. This page
  covers the per-device side; the experiment-wide coordination
  story belongs to 12f.
- Detailed control-widget walkthroughs — the user guide and the
  per-class API pages cover those; this page only mentions the
  Control-tab pattern.

## Sources

### Related source files

- `src/hardware/core/hardwaremanager.{cpp,h}` — the principal
  source.
- `src/hardware/core/hardwareobject.{cpp,h}` — `bcInitInstrument`,
  `bcTestConnection`, `bcReadAuxData`, `bcReadSettings`, the
  `d_threaded`/`d_critical`/`d_commType` flags, the rolling-data
  timer, the virtual surface.
- `src/hardware/core/communication/*.{cpp,h}` — concrete
  `CommunicationProtocol` subclasses.
- `src/hardware/core/clock/clockmanager.{cpp,h}` — for the
  `HardwareManager` ↔ `ClockManager` ownership; bundle 12f covers
  the experiment-time clock-routing flow.
- `src/gui/dialog/communicationdialog.{cpp,h}` and
  `communicationdialog.ui`.
- `src/gui/dialog/hwdialog.{cpp,h}`.
- `src/gui/widget/customprotocolwidget.{cpp,h}`,
  `gpibprotocolwidget.{cpp,h}`, etc. — protocol-specific widgets.
- `src/gui/mainwindow.{cpp,h}` — to confirm the wiring of
  `HardwareManager`'s signals into the GUI thread.
- `src/data/storage/auxdatastorage.{cpp,h}` — for the
  AuxDataMap key conventions.

### Related dev-docs

- `dev-docs/python-script-reload.md` — research material for the
  reload entry point. Do not link.

### Related user-guide pages

Forward-link, do not duplicate:

- `doc/source/user_guide/hardware_menu.rst` — Hardware menu and
  Hardware-status panel.
- `doc/source/user_guide/hwdialog.rst` — the Hardware Settings
  dialog from the user's perspective.
- `doc/source/user_guide/library_status.rst` — vendor library
  status surface (touched here only at one-paragraph depth).

### Related API reference pages

- `doc/source/classes/hardwaremanager.rst`
- `doc/source/classes/hardwareobject.rst`
- `doc/source/classes/communicationprotocol.rst`
- `doc/source/classes/custominstrument.rst`
- `doc/source/classes/clockmanager.rst`
- `doc/source/classes/auxdatastorage.rst`
- `doc/source/classes/applicationconfigmanager.rst` (for the
  rolling-interval setting source)

## Sphinx file deltas

**Created:**

- `doc/source/developer_guide/hardware_runtime.rst`.

## Page structure

H1 intro: 1–2 paragraphs framing this page as the runtime
companion to the configuration page.

H2 sections (`-` underlines):

- *HardwareManager: ownership and threading*
- *Bringing a hardware map online*
- *Per-object lifecycle: bcInitInstrument and bcTestConnection*
- *Communication protocols and Custom*
- *CommunicationDialog: changing protocol at runtime*
- *HwDialog: settings and control*
- *Connection state and signal fan-out*
- *Auxiliary, validation, and rolling data*
- *Python script reload*

A short Mermaid sequence diagram showing the lifecycle from
`MainWindow::startup` → `HardwareManager::syncWithRuntimeConfig`
→ per-device thread start → `bcInitInstrument` →
`bcTestConnection` is helpful but optional.

## Acceptance criteria

- The `HardwareManager` ownership story is clear: lives on its
  own thread, owns the hardware map, owns `ClockManager`,
  mediates all cross-thread hardware calls.
- The threaded-hardware constructor restriction (no parent, no
  child `QObject` in ctor) is documented.
- The deferred-connection-testing rationale is documented (GPIB
  controllers must exist before their children).
- `bcInitInstrument` and `bcTestConnection` are each documented
  with their step sequence.
- The `Custom` protocol is documented as the explicit "driver
  handles its own comm" indicator.
- `CommunicationDialog` and `HwDialog` are each mapped to the
  `HardwareManager` slot and signal pair they use.
- The connection-state signal pattern (`connectionResult`,
  `allHardwareConnected`) is documented.
- Aux/validation/rolling fan-out is described as
  `HardwareManager` prefixing keys with `hwKey` so consumers can
  disambiguate.
- The Python reload entry point is mentioned as a one-paragraph
  pointer to bundle 12j.
- No duplication of per-method API content; cross-links cover
  per-class detail.
- No rendered link points into `dev-docs/`.
