.. index::
   single: hardware runtime
   single: HardwareManager; runtime
   single: HardwareObject; lifecycle
   single: bcInitInstrument
   single: bcTestConnection
   single: bcReadSettings
   single: CommunicationProtocol; runtime selection
   single: CustomInstrument; protocol selection
   single: GpibController; resolution
   single: CommunicationDialog
   single: HWDialog
   single: hardware threading; per-device QThread
   single: connectionResult
   single: allHardwareConnected
   single: auxiliary data; key prefixing
   single: rolling data; HardwareObject timer
   single: Python hardware; script reload

Hardware Runtime
================

This page is the runtime companion to
:doc:`/developer_guide/hardware_configuration`. The configuration page
describes the four singletons that decide *which* hardware exists in
the active loadout, *which* driver backs each profile, and
*which* persisted settings each profile carries. This page picks up at
the moment :cpp:func:`HardwareManager::initialize` runs: how the
manager turns that configuration into a live set of
:cpp:class:`HardwareObject` instances, places each one on the right
thread, opens its communication channel, fans connection-state and
sensor data out to the rest of the program, and reacts when the user
edits hardware from the GUI.

The two cross-system surfaces a contributor most often touches when
working in this area are the :cpp:class:`CommunicationDialog` (for
changing a device's protocol or connection parameters at runtime) and
the :cpp:class:`HWDialog` (for editing a device's persistent settings
and exercising its live controls). Both are mediated through
:cpp:class:`HardwareManager`; nothing in the GUI ever touches a
:cpp:class:`HardwareObject` directly, because the per-device threading
rules require all interaction to be queued through the manager's slot
and signal surface.

HardwareManager: ownership and threading
----------------------------------------

:cpp:class:`HardwareManager` is the runtime owner of every live
:cpp:class:`HardwareObject` instance for the active loadout. It is
constructed by :cpp:class:`MainWindow` before the application's event
loop begins serving widgets, immediately moved onto a dedicated
``QThread`` named ``HardwareManagerThread``, and started by wiring the
thread's :cpp:func:`QThread::started` signal to
:cpp:func:`HardwareManager::initialize`:

.. code-block:: cpp

   QThread *hwmThread = new QThread(this);
   hwmThread->setObjectName("HardwareManagerThread");
   connect(hwmThread, &QThread::started, p_hwm, &HardwareManager::initialize);
   p_hwm->moveToThread(hwmThread);

After this point every public slot on :cpp:class:`HardwareManager`
executes on the manager's thread. Callers in the GUI thread, in
:cpp:class:`AcquisitionManager`, or anywhere else, reach the manager
through queued connections (or
:cpp:func:`QMetaObject::invokeMethod` for direct dispatch).

The manager owns three things outright:

- The **hardware map**, ``d_hardwareMap``, a
  ``std::map<QString, HardwareObject*>`` keyed by ``"<Type>.<label>"``
  (for example ``"FtmwScope.frontPanel"``). Read access is concurrent —
  any thread can take a read lock on ``d_hardwareMapLock`` and look up
  a device — while writes are serialized. The static accessor
  :cpp:func:`HardwareManager::constInstance` returns a const reference
  to the singleton so code that cannot hold a manager reference
  (typically a :cpp:class:`HardwareObject` resolving its
  :cpp:class:`GpibController`) can still query the map under that read
  lock.
- The **connection-state lock**, ``d_connectionStateLock``, a separate
  ``QMutex`` that protects the per-test-round response counter. The
  two locks are deliberately distinct so a long-running connection
  attempt does not block readers of the hardware map; when both must
  be held, the hardware-map lock is always acquired first.
- :cpp:class:`ClockManager`, held by
  ``std::unique_ptr<ClockManager> pu_clockManager``. The clock
  subsystem lives on the same thread as the manager and is rebuilt
  whenever the active set of clock hardware changes (see
  :doc:`/classes/clockmanager`).

Per-device hardware objects do *not* live on the manager's thread by
default. Each :cpp:class:`HardwareObject` whose interface class sets
``d_threaded = true`` in its constructor — interface classes such as
``AWG`` and ``FtmwDigitizer`` enable threading because their I/O is
expensive enough to deserve its own thread of execution — is moved
onto a dedicated ``QThread`` whose ``objectName`` is
``"<hwKey>Thread"``. The remaining devices use the manager's own
thread as parent. Either way, the manager mediates all access; cross-
thread calls go through queued connections.

The threading-override mechanism is per-profile. The default for a
hardware type is set in its interface-class constructor; the user can
override that default at profile creation time, persisted on the
:cpp:class:`RuntimeHardwareConfig` side and applied by the manager at
construction time:

.. code-block:: cpp

   auto threadedOverride = RuntimeHardwareConfig::constInstance().getThreaded(hwKey);
   if (threadedOverride.has_value())
       hwObj->d_threaded = *threadedOverride;

So a contributor who needs to force a normally-in-thread driver onto
the manager's thread (or vice versa) does it by editing the override
on the profile, never by patching the interface class.

A consequence of the per-device threading model is the
**threaded-hardware constructor restriction**: a threaded
:cpp:class:`HardwareObject` must not have a ``QObject`` parent at
construction time, and must not construct any child ``QObject`` in
its own constructor. The base class is constructed (along with the
driver) before the move-to-thread step runs; any child ``QObject``
created in the constructor would be parented on the wrong thread,
yielding the kind of cross-thread parent error that is hard to debug
because nothing crashes immediately. Construct child ``QObject``\ s
inside :cpp:func:`HardwareObject::initialize` instead, which the
manager invokes after the move-to-thread step has completed. This
restriction is enforced socially, not by code.

Bringing a hardware map online
------------------------------

On :cpp:func:`HardwareManager::initialize` the manager:

1. Calls :cpp:func:`HardwareProfileManager::ensureSystemProfiles` and
   :cpp:func:`RuntimeHardwareConfig::activateMissingSystemProfiles` so
   every required hardware type (``FtmwScope``, ``Clock``, plus the
   LIF types when LIF is enabled) has at least one active profile —
   the virtual driver when no real device is configured.
2. Calls :cpp:func:`HardwareManager::syncWithRuntimeConfig` to
   reconcile the empty hardware map against the runtime configuration
   and bring up every active profile.
3. Iterates the populated map and emits a ``bcWarn`` for every
   instrument whose ``d_commType`` resolves to
   :cpp:enumerator:`CommunicationProtocol::Virtual`, so the user is
   alerted that some readings will be simulated.
4. Emits :cpp:func:`HardwareManager::hwInitializationComplete`.

:cpp:func:`HardwareManager::syncWithRuntimeConfig` is also the slot
used at runtime when the user activates a different loadout, edits
the active profile set, or changes a profile's threading override.
Its job is to compute the difference between the current
``d_hardwareMap`` and the target map returned by
:cpp:func:`RuntimeHardwareConfig::getCurrentHardware`, then apply
that difference in a deliberate sequence:

.. code-block:: text

   1. Compute (toRemove, toAdd, toReplace) under a read lock.
   2. Augment the lists with hardware that depends on a vendor library
      whose configuration changed (so the library can be reloaded
      cleanly with no live consumers).
   3. Tear down everything in toRemove and toReplace.
   4. Apply pending vendor-library configuration changes — safe now
      that no live object can be holding a library handle.
   5. Re-create everything in toReplace.
   6. Create everything in toAdd.
   7. Resolve GPIB controllers for the surviving GPIB instruments.
   8. Update ClockManager with the new clock list.
   9. Run testAll() — the deferred connection sweep.

Each individual change goes through one of three internal helpers:

- :cpp:func:`HardwareManager::addHardwareInternal` constructs the
  driver via :cpp:func:`HardwareRegistry::createHardware`, validates
  the resulting object's ``d_key``, takes the per-profile threading
  override into account, wires the common signal connections (next
  section), and starts the per-device thread (or attaches the object
  to the manager's thread when not threaded).
- :cpp:func:`HardwareManager::removeHardwareInternal` tears the
  hardware out of the map under the write lock, disconnects every
  stored Qt connection handle, and joins the per-device thread:
  ``thread->quit()`` followed by a 5-second ``wait``; if that times
  out, ``terminate`` is the last resort. When the corresponding
  profile no longer exists in :cpp:class:`HardwareProfileManager`
  (the loadout edit was a permanent delete, not a deactivation), the
  manager additionally calls
  :cpp:func:`HardwareObject::purgeSettings` so the settings are
  scrubbed from disk and emits
  :cpp:func:`HardwareManager::profileDeleted`.
- :cpp:func:`HardwareManager::replaceHardwareInternal` is composed of
  the two above, in that order.

**Connection testing is deferred** until every add, remove, and
replace has settled. This is non-obvious but important:
GPIB-attached instruments require a live :cpp:class:`GpibController`
to talk through, and the controller is itself a hardware object that
might be added in the same sync cycle. Running the connection sweep
device-by-device would race the controller against its children. By
deferring to a final
:cpp:func:`HardwareManager::resolveGpibControllersForInstruments`
followed by :cpp:func:`HardwareManager::testAll`, every controller is
guaranteed to exist by the time its children try to use it.

Per-object lifecycle: bcInitInstrument and bcTestConnection
-----------------------------------------------------------

Every newly-added hardware object follows the same two-step bring-up:
:cpp:func:`HardwareObject::bcInitInstrument` runs once, on the
device's own thread, immediately after that thread starts (or
immediately, in the manager's thread, when ``d_threaded`` is false).
:cpp:func:`HardwareObject::bcTestConnection` runs once after the full
sync settles and again every time the user requests a connection
test or the manager runs :cpp:func:`HardwareManager::testAll`.

``bcInitInstrument`` does the following, in order:

1. Calls :cpp:func:`SettingsStorage::readAll` so the in-memory
   settings cache reflects what is on disk.
2. Calls :cpp:func:`HardwareObject::buildCommunication` with no GPIB
   controller. Build creates the
   :cpp:class:`CommunicationProtocol` subclass that matches
   ``d_commType`` and stores it in ``p_comm``. For GPIB instruments
   the controller pointer is filled in later by
   :cpp:func:`HardwareManager::resolveGpibControllersForInstruments`,
   which calls ``buildCommunication`` again with the resolved
   controller.
3. Moves ``p_comm`` to the device's thread (if it isn't there
   already) and calls its :cpp:func:`CommunicationProtocol::initialize`,
   which constructs any underlying ``QIODevice`` (a ``QSerialPort``
   for RS-232, a ``QTcpSocket`` for TCP, none for the Custom and
   Virtual variants).
4. Calls the driver's pure-virtual
   :cpp:func:`HardwareObject::initialize`. **This is the right place
   for one-shot setup** — constructing child ``QObject``\ s,
   pre-allocating buffers, etc. — *especially* for
   threaded drivers, which cannot do that work in the constructor.
   Per-connection work belongs in
   :cpp:func:`HardwareObject::testConnection` instead, which runs on
   every test round.
5. Wires :cpp:func:`HardwareObject::hardwareFailure` to a small
   lambda that clears ``d_isConnected`` and writes ``connected =
   false`` to settings, so a failure reflected anywhere in the
   driver is immediately visible to anything that reads the cached
   state.

``bcTestConnection`` is the dispatch the manager (and the GUI)
trigger to verify a device is responsive:

1. Pre-clears ``d_isConnected``.
2. Calls :cpp:func:`HardwareObject::bcReadSettings`, which reloads
   the persisted settings, refreshes ``d_critical`` and
   ``d_commType`` (so a protocol change made in
   :cpp:class:`CommunicationDialog` while the test was queued is
   honored), restarts the rolling-data timer to the current
   ``BC::Key::HW::rInterval`` value (described in
   :doc:`/classes/applicationconfigmanager` and the Aux/Rolling
   section below), and finally dispatches to
   :cpp:func:`HardwareObject::hwReadSettings` so the driver — or its
   intermediate base class via the NVI hook described in
   :doc:`/developer_guide/adding_a_hardware_type` — can refresh its
   own cached state.
3. Calls ``p_comm->bcTestConnection``, which exercises the
   underlying ``QIODevice`` (open the serial port, connect the
   socket, …). On failure the driver short-circuits and reports
   disconnected.
4. Calls the driver's pure-virtual
   :cpp:func:`HardwareObject::testConnection`. **This is the right
   place for a cheap interaction with the device**, typically an
   ``*IDN?`` query, plus an assertion that the responding device is
   the expected model. Drivers that detect a model mismatch should
   write the diagnostic into ``d_errorString`` and return ``false``.
5. Stores the result in ``d_isConnected``, persists ``connected`` to
   settings, and emits :cpp:func:`HardwareObject::connected`.

The manager listens on ``connected`` per-device; the
``setupHardwareObjectWithTracking`` helper installs the lambda that
forwards to :cpp:func:`HardwareManager::handleConnectionResult` (see
the connection-state section below).

For the full virtual surface a driver can override —
:cpp:func:`HardwareObject::sleep`,
:cpp:func:`HardwareObject::beginAcquisition`,
:cpp:func:`HardwareObject::endAcquisition`,
:cpp:func:`HardwareObject::hwPrepareForExperiment`,
:cpp:func:`HardwareObject::readAuxData`,
:cpp:func:`HardwareObject::readValidationData`,
:cpp:func:`HardwareObject::hwReadSettings` — see
:doc:`/classes/hardwareobject`. The experiment-context hooks are
covered cross-system in
:doc:`/developer_guide/experiment_lifecycle`.

Communication protocols and Custom
----------------------------------

The :cpp:class:`CommunicationProtocol` hierarchy is a thin wrapper
around the OS-level I/O facilities. Each subclass provides a uniform
``writeCmd`` / ``writeBinary`` / ``queryCmd`` / ``readBytes`` API and
exposes the underlying ``QIODevice`` (when there is one) through the
``device<T>()`` template:

- :cpp:class:`Rs232Instrument` wraps a ``QSerialPort``.
- :cpp:class:`TcpInstrument` wraps a ``QTcpSocket``.
- :cpp:class:`GpibInstrument` proxies through a
  :cpp:class:`GpibController` hardware object.
- :cpp:class:`VirtualInstrument` and :cpp:class:`CustomInstrument`
  carry no ``QIODevice`` at all.

Each driver declares the protocols it supports at static-
initialization time via ``REGISTER_HARDWARE_PROTOCOLS`` (see
:doc:`/developer_guide/hardware_configuration`). The user picks one of
those at profile creation time, the choice is persisted in the
profile's ``QSettings`` group under ``BC::Key::HW::commType``, and
:cpp:func:`HardwareObject::buildCommunication` reads it back to
construct the matching protocol instance. Read behavior — timeout
and termination characters — is shared across transports and is
loaded by :cpp:func:`CommunicationProtocol::loadCommReadOptions` from
the same profile group. See :doc:`/classes/communicationprotocol`
for the per-method API.

The **Custom** protocol is the explicit indicator that the driver
handles its own communication outside the standard ``QIODevice``
abstractions. :cpp:class:`CustomInstrument` keeps its device pointer
``nullptr``, its ``initialize()`` and ``testConnection()`` are
no-ops, and the driver's own ``testConnection()`` does whatever
vendor-specific handshake is required. What makes this useful is the
companion convention for collecting connection parameters from the
user *without* instantiating the driver. Drivers register
:cpp:struct:`CustomCommDef` descriptors via the
``REGISTER_CUSTOM_COMM`` family of macros (or
``REGISTER_CUSTOM_COMM_BASE`` for parameters shared across an
inheritance chain). Each descriptor specifies the settings key,
user-facing label, description, type (``String``, ``Int``, or
``FilePath``), and optional bounds. :cpp:class:`HardwareRegistry`
makes those descriptors available to the GUI before any object is
constructed, so :cpp:class:`CustomProtocolWidget` can render the
right inputs. The driver reads the user-supplied values back from
the ``BC::Key::Comm::custom`` group of its
:cpp:class:`SettingsStorage` inside ``testConnection()``. See
:doc:`/classes/custominstrument` for the descriptor reference. For
Python-backed drivers, ``Custom`` is the explicit "communication is
handled by the ``.py`` script" indicator — the script's connection
parameters live as constants in the script. The Python side is on
:doc:`/developer_guide/python_hardware`.

GPIB has an extra layer worth calling out. A
:cpp:class:`GpibController` is itself a :cpp:class:`HardwareObject`
that owns the actual GPIB bus (a Prologix GPIB-LAN bridge in the
supported configuration); a :cpp:class:`GpibInstrument` resolves
queries to its controller, not directly to the bus. This is why
connection testing is deferred until after the full hardware sync —
the controller must exist before its children can talk through it.
:cpp:func:`HardwareManager::resolveGpibControllersForInstruments`
walks the post-sync hardware map, looks up each GPIB instrument's
controller key from settings, and re-runs ``buildCommunication`` on
the instrument with the resolved pointer.

CommunicationDialog: changing protocol at runtime
-------------------------------------------------

:cpp:class:`CommunicationDialog`
(``gui/dialog/communicationdialog.{cpp,h}`` plus the ``.ui`` file)
is the single GUI surface for changing a hardware object's
communication protocol or its connection parameters at runtime. It
is a master-detail layout: the left panel lists every device in the
hardware map with a connection-status indicator, the right panel
shows the protocol selector, the protocol-specific input widget, and
the shared read-options group (timeout, termination character).

The dialog talks to :cpp:class:`HardwareManager` through three
slot/signal pairs:

- :cpp:func:`HardwareManager::getHardwareCommunicationInfo` →
  :cpp:func:`HardwareManager::hardwareCommunicationInfoReady`
  ``(hwKey, currentProtocol, supportedProtocols, connected)``.
  The dialog calls the slot when the user selects a device in the
  left panel; the response populates the protocol combo box and
  drives which protocol-specific widget the
  :cpp:class:`QStackedWidget` shows.
- :cpp:func:`HardwareManager::setHardwareProtocol` →
  :cpp:func:`HardwareManager::protocolSetResult`
  ``(hwKey, success, msg)``.
  Called when the user accepts a new protocol or new connection
  parameters. The manager validates that the requested protocol is
  in the device's :cpp:func:`HardwareObject::supportedProtocols`,
  resolves the GPIB controller (if applicable, via the same callback
  path as the post-sync resolution), then dispatches
  :cpp:func:`HardwareObject::setCommProtocol` on the device's
  thread — using a blocking queued invocation so the result is
  delivered synchronously to the manager's slot.
- :cpp:func:`HardwareManager::getActiveGpibControllers` →
  :cpp:func:`HardwareManager::gpibControllersAvailable`
  ``(controllerKeys)``. Used to populate the GPIB controller
  drop-down in :cpp:class:`GpibProtocolWidget`.

The protocol-specific widgets — :cpp:class:`Rs232ProtocolWidget`,
:cpp:class:`TcpProtocolWidget`, :cpp:class:`GpibProtocolWidget`,
:cpp:class:`CustomProtocolWidget` — share a
:cpp:class:`ProtocolWidget` base; the dialog instantiates one per
``(deviceKey, protocolType)`` pair on demand and caches them in a
``QMap``. The user-facing walkthrough is on
:doc:`/user_guide/hardware_menu` (Communication section).

HwDialog: settings and control
------------------------------

:cpp:class:`HWDialog` (``gui/dialog/hwdialog.{cpp,h}``) is the
per-device dialog opened from each hardware-key entry in the
**Hardware** menu. It is tabbed:

- A **Settings** tab containing an
  :cpp:class:`HwSettingsWidget` constructed in
  :cpp:enumerator:`HwSettingsMode::Edit`. Required settings render
  as read-only text in this mode (changing a Required value would
  invalidate the constructor's view of the device, so the user must
  delete and recreate the profile instead — see
  :doc:`/developer_guide/hardware_configuration`).
  :cpp:func:`HwSettingsWidget::saveToStorage` runs from
  :cpp:func:`HWDialog::accept` to write all edited values back to
  ``QSettings`` synchronously.
- A **Control** tab, present only when the calling code passes a
  control widget into the dialog. The control widget is hardware-
  type-specific:
  :cpp:class:`GasControlWidget` for :cpp:class:`FlowController`,
  :cpp:class:`PulseGenChannelTable` for :cpp:class:`PulseGenerator`,
  :cpp:class:`PythonHardwareControlWidget` for any Python-backed
  device (composed with the per-type widget when both apply), and
  so on. **Control-tab interactions take effect immediately** — the
  user does not have to accept the dialog for a control change to
  reach the hardware; they cannot be rolled back by dismissing the
  dialog.

Below the tabs the dialog hosts a **Test Connection** button (which
emits :cpp:func:`HWDialog::requestTestConnection`, wired by
:cpp:class:`MainWindow` to
:cpp:func:`HardwareManager::testObjectConnection`) and a
**Communication Settings…** button that opens
:cpp:class:`CommunicationDialog` pre-selected to this device.

The dispatch to the hardware object on accept goes through the
manager:

.. code-block:: cpp

   connect(out, &HWDialog::accepted, [this, key]() {
       QMetaObject::invokeMethod(p_hwm, [this, key]() {
           p_hwm->updateObjectSettings(key);
       });
   });

:cpp:func:`HardwareManager::updateObjectSettings` looks up the device
in ``d_hardwareMap`` and dispatches
:cpp:func:`HardwareObject::bcReadSettings` on its thread, which
reloads the persisted settings, refreshes ``d_critical`` and
``d_commType``, restarts the rolling-data timer, and dispatches to
:cpp:func:`HardwareObject::hwReadSettings` (or, for the intermediate
bases that ``final``-override it, the per-base hook —
``fcReadSettings``, ``pcReadSettings``, ``tcReadSettings``,
``pgReadSettings``, ``awgReadSettings``, ``clockReadSettings``,
``ftmwReadSettings``, ``gpibReadSettings``, ``ioReadSettings``,
``lifLaserReadSettings``, or ``lifScopeReadSettings`` — described in
:doc:`/developer_guide/adding_a_hardware_type`). For Python-backed
drivers this triggers an IPC ``read_settings`` message rather than
restarting the subprocess; that path is on
:doc:`/developer_guide/python_hardware`.

Control widgets cannot reach a :cpp:class:`HardwareObject` directly
because of the threading rules: every interaction has to be queued
through a slot on :cpp:class:`HardwareManager` (for example
:cpp:func:`HardwareManager::setPGenSetting` or
:cpp:func:`HardwareManager::setFlowSetpoint`) so the call lands on
the device's thread. This is why each per-type control widget takes
a manager pointer in its constructor instead of a hardware-object
pointer. The user-facing walkthrough is on
:doc:`/user_guide/hwdialog`.

Connection state and signal fan-out
-----------------------------------

:cpp:class:`HardwareManager` exposes a unified view of connection
state through two signals.

:cpp:func:`HardwareManager::connectionResult` ``(hwKey, success,
msg)`` fires for **every** connection-state change a device can
undergo: a successful or failed test
(:cpp:func:`HardwareManager::handleConnectionResult` forwards from
:cpp:func:`HardwareObject::connected`), a runtime hardware failure
(:cpp:func:`HardwareManager::hardwareFailure` forwards from
:cpp:func:`HardwareObject::hardwareFailure`), or hardware removal
(:cpp:func:`HardwareManager::removeHardwareInternal` emits with
``success = false`` and ``msg = "Hardware removed"``). One
subscription gives a consumer the complete connection-state picture
without per-device wiring; that is how :cpp:class:`MainWindow`
maintains its ``d_hardwareConnectionState`` map.

:cpp:func:`HardwareManager::allHardwareConnected` ``(bool)`` fires at
the end of every :cpp:func:`HardwareManager::testAll` round and
indicates whether **every critical device** is connected. A device's
``d_critical`` flag (default ``true``) controls whether a failure on
that device should disable the *Start Experiment* state machine;
non-critical devices' connection status is reflected only in
``connectionResult``. The manager uses an atomic response counter
(``ConnectionTestState`` in the header) to know when every device
has reported back, so the final ``allHardwareConnected`` is emitted
exactly once per round even when individual devices report at very
different latencies.

When a previously-connected device emits
:cpp:func:`HardwareObject::hardwareFailure` mid-experiment, the
manager disconnects the failure handler (so a single failure is not
reprocessed on every retry), emits ``connectionResult`` with
``success = false``, and — if the device is critical — emits
:cpp:func:`HardwareManager::abortAcquisition` to terminate the
in-progress experiment. The cross-system view of how
:cpp:class:`AcquisitionManager` reacts to that abort is on
:doc:`/developer_guide/experiment_lifecycle`.

For each optional hardware category (pulse generator, flow
controller, pressure controller, temperature controller),
:cpp:class:`HardwareManager` exposes a set of update signals that
all carry the source ``hwKey`` as their first argument — for
example :cpp:func:`HardwareManager::flowUpdate` ``(hwKey, channel,
value)`` and :cpp:func:`HardwareManager::pGenConfigUpdate` ``(hwKey,
config)``. The GUI wires per-device widgets dynamically based on
which keys appear in the active hardware map; nothing in the manager
or the GUI enumerates "the possible flow controllers" in advance.
The full type-specific signal list is on
:doc:`/classes/hardwaremanager`; the pattern matters more than the
individual signals.

Auxiliary, validation, and rolling data
---------------------------------------

Each :cpp:class:`HardwareObject` may override
:cpp:func:`HardwareObject::readAuxData` and
:cpp:func:`HardwareObject::readValidationData` to return an
:cpp:type:`AuxDataStorage::AuxDataMap`. The two are dispatched by
:cpp:func:`HardwareObject::bcReadAuxData`, which the manager calls
on every device when something upstream wants a fresh aux/
validation snapshot — for example, the per-acquisition signal
:cpp:class:`AcquisitionManager` emits when it needs to record a
time point, fanned out via :cpp:func:`HardwareManager::getAuxData`.
Aux data lands on the **Aux** and **Rolling** tabs of the main
window and is persisted as time-series in
:cpp:class:`AuxDataStorage` (see :doc:`/classes/auxdatastorage`).
Validation data is range-checked against the experiment's expected
ranges and aborts the experiment on violation.

In addition, :cpp:class:`HardwareObject` runs a **rolling-data
timer** outside the experiment context. The interval is loaded from
``BC::Key::HW::rInterval`` (in seconds) by
:cpp:func:`HardwareObject::bcReadSettings`. The timer is started in
``bcReadSettings`` and re-started every time settings are reloaded;
on each tick (handled by
:cpp:func:`HardwareObject::timerEvent`) the driver's
``readAuxData`` is called and the result is emitted as
:cpp:func:`HardwareObject::rollingDataRead`. Zero or negative
intervals disable the timer, and the timer fires only while the
device is connected.

The manager aggregates all three streams. The
``setupHardwareObjectWithTracking`` helper installs three lambdas
that prefix every key in the per-device map with the source
object's ``hwKey`` (using
:cpp:func:`AuxDataStorage::makeKey`) before re-emitting:

- :cpp:func:`HardwareObject::auxDataRead` →
  :cpp:func:`HardwareManager::auxData`
- :cpp:func:`HardwareObject::auxDataRead` →
  :cpp:func:`HardwareManager::validationData`
  (the same per-device signal feeds both fan-outs; downstream
  consumers split on map content)
- :cpp:func:`HardwareObject::rollingDataRead` →
  :cpp:func:`HardwareManager::rollingData` ``(map,
  QDateTime::currentDateTime())``

The hwKey prefix is what lets a consumer disambiguate readings from
multiple devices of the same type (two flow controllers, three
temperature channels) without having to enumerate the devices in
advance. :cpp:class:`AcquisitionManager` consumes ``auxData`` and
``validationData`` (writing to :cpp:class:`AuxDataStorage` and
range-checking, respectively); the **Rolling** and **Aux** tabs in
the GUI consume ``rollingData`` directly. The experiment-context
side of the same flow is on
:doc:`/developer_guide/experiment_lifecycle`.

Python script reload
--------------------

Python-backed hardware uses the ``Custom`` communication protocol
and runs the user's ``.py`` script in a separate subprocess managed
by :cpp:class:`PythonProcess`. The user can edit the script and
trigger a hot-reload from the **Control** tab of the
:cpp:class:`HWDialog`, which (through
:cpp:class:`PythonHardwareControlWidget`) calls
:cpp:func:`HardwareManager::reloadPythonScript`. The manager looks
up the device, verifies it is a :cpp:class:`PythonHardwareBase`
subclass, and dispatches a single lambda to the device's thread
that stops the subprocess and immediately calls
:cpp:func:`HardwareObject::bcTestConnection`. Because
``bcTestConnection`` ultimately calls back through
``startPythonProcess`` to launch a fresh subprocess, the reload is
expressed as a stop-then-test rather than as an explicit restart;
the C++ object's threading, signal connections, and settings
storage are unaffected. The outcome (and any Python traceback) is
reported via :cpp:func:`HardwareManager::pythonScriptReloadResult`
``(hwKey, success, msg)``. The full Python-side architecture is on
:doc:`/developer_guide/python_hardware`.

Lifecycle at a glance
---------------------

The end-to-end startup sequence, from ``MainWindow`` constructing
the manager to the first round of connection results reaching the
GUI:

.. mermaid::

   sequenceDiagram
       autonumber
       participant MW as MainWindow
       participant HMT as HardwareManagerThread
       participant HM as HardwareManager
       participant RC as RuntimeHardwareConfig
       participant HR as HardwareRegistry
       participant DT as hwKeyThread
       participant HO as HardwareObject

       MW->>HM: new HardwareManager
       MW->>HMT: new QThread
       MW->>HM: moveToThread(HMT)
       MW->>HMT: started -> HM::initialize
       HMT->>HM: initialize()
       HM->>RC: getCurrentHardware()
       HM->>HR: createHardware(type, impl, label)
       HR-->>HM: HardwareObject ptr
       alt threaded driver
           HM->>DT: new QThread named hwKeyThread
           HM->>HO: moveToThread(DT)
           HM->>DT: started -> HO::bcInitInstrument
           DT->>HO: bcInitInstrument()
       else in-thread driver
           HM->>HO: setParent(HM), invokeMethod(bcInitInstrument)
           HM->>HO: bcInitInstrument()
       end
       HO->>HO: buildCommunication() then initialize()
       HM->>HM: resolveGpibControllersForInstruments()
       HM->>HO: testAll() then bcTestConnection()
       HO-->>HM: connected(success, msg)
       HM-->>MW: connectionResult(hwKey, ...)
       HM-->>MW: allHardwareConnected(bool)
