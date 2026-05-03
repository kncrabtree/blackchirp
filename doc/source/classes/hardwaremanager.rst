.. index::
   single: HardwareManager
   single: hardware; orchestration
   single: hardware; connection testing
   single: hardware; dynamic synchronization
   single: hardware; thread management

HardwareManager
===============

``HardwareManager`` is the central orchestration layer that owns all live
:cpp:class:`HardwareObject` instances for the active hardware loadout. It
marshals cross-thread hardware calls, fans out connection and sensor-data
notifications to the GUI and acquisition subsystem, and exposes the public
slot surface through which callers request hardware operations without
knowing which thread a given device lives on.

The manager lives on a dedicated ``QThread`` created by ``MainWindow``
before the application event loop starts. ``MainWindow`` calls
``initialize()`` via the thread's ``started()`` signal; thereafter, the
manager and all its public slots run on that thread. Individual
:cpp:class:`HardwareObject` instances may run on further per-device threads;
``HardwareManager`` mediates all interaction with them through
``QMetaObject::invokeMethod`` (blocking or queued as appropriate) so that
callers on any thread can request hardware operations safely. The static
:cpp:func:`HardwareManager::constInstance` accessor provides read-only access
to the hardware map from threads that cannot hold a direct reference.

Primary collaborators are :doc:`hardwareobject` (the managed devices),
:doc:`runtimehardwareconfig` (the source of truth for which profiles are
active), :doc:`loadoutmanager` (named hardware maps and FTMW presets),
:cpp:class:`ClockManager` (RF clock subsystem, owned by this manager),
:doc:`experiment` (populated and handed off via ``experimentInitialized()``),
and ``AcquisitionManager`` (consumes the experiment lifecycle signals). The
user-facing hardware configuration workflow is described in
:doc:`/user_guide/hardware_config` and its sub-pages; Hardware-menu actions
are covered in :doc:`/user_guide/hardware_menu`.

Hardware lifecycle
------------------

On startup ``initialize()`` calls ``syncWithRuntimeConfig()``, which
consults :cpp:class:`RuntimeHardwareConfig` and creates a
:cpp:class:`HardwareObject` for each active profile via
:cpp:class:`HardwareRegistry`. Objects whose ``d_threaded`` flag is set are
moved to their own ``QThread`` before initialization begins. At runtime the
loadout-switching flow calls ``applyHardwareMap()`` or
``syncWithRuntimeConfig()`` to add, remove, or replace hardware objects
without restarting the application; each change is routed through the
internal ``addHardwareInternal`` / ``removeHardwareInternal`` /
``replaceHardwareInternal`` helpers, which handle thread teardown and
connection cleanup automatically.

Signal surface grouped by purpose
---------------------------------

**Lifecycle and connection status**

``hwInitializationComplete()`` fires once, when the hardware map is first
populated and threads are started. ``allHardwareConnected(bool)`` fires after
every connection-test round (see ``testAll()`` and ``testObjectConnection()``),
reporting whether all critical devices are connected. The unified
``connectionResult(hwKey, success, msg)`` signal fires for every individual
status change — successful or failed connection test, runtime hardware
failure, hardware removal — so a single connection lets a consumer maintain a
complete picture of the current connection state without tracking each device
separately.

**Experiment lifecycle**

The acquisition subsystem connects to ``experimentInitialized(Experiment)``,
which carries the fully prepared experiment after ``initializeExperiment()``
has configured clocks and called ``hwPrepareForExperiment()`` on each object.
``beginAcquisition()`` and ``endAcquisition()`` are broadcast to all hardware
objects at acquisition start and end; ``abortAcquisition()`` is emitted when
a critical device reports failure during a run.

**Auxiliary data fan-out**

``auxData(AuxDataMap)``, ``validationData(AuxDataMap)``, and
``rollingData(AuxDataMap, QDateTime)`` aggregate sensor readings from all
hardware objects. Each hardware object emits keyed scalar readings;
``HardwareManager`` prefixes each key with the object's hardware key before
re-emitting, so consumers receive a flat map that is unambiguous across
multiple devices of the same type.

**Hardware-type-specific data and control**

For each optional hardware category, ``HardwareManager`` exposes a set of
type-specific signals carrying the hardware key of the source device as their
first argument. This pattern — one signal per reading type, distinguished by
``hwKey`` — lets the GUI wire per-device widgets dynamically based on which
keys are present in the map, without enumerating every possible device. The
categories and representative signals are:

- **Pulse generators:** ``pGenSettingUpdate(hwKey, channel, setting, value)``
  and ``pGenConfigUpdate(hwKey, config)`` for per-channel and whole-config
  updates.
- **Flow controllers:** ``flowUpdate``, ``flowSetpointUpdate``,
  ``gasPressureUpdate``, ``gasPressureSetpointUpdate``, and
  ``gasPressureControlMode``, all carrying the controller's hardware key.
- **Pressure controllers:** ``pressureUpdate``, ``pressureSetpointUpdate``,
  and ``pressureControlMode``.
- **Temperature controllers:** ``temperatureUpdate`` and
  ``temperatureEnableUpdate``, each carrying a channel index.

**Clocks**

``clockFrequencyUpdate(type, freqMHz)`` and ``clockHardwareUpdate(type,
hwKey, output)`` propagate incremental changes from
:cpp:class:`ClockManager`. ``allClocksReady(clocks)`` fires after
``setClocks()`` completes the digitizer-gated frequency transition and is the
signal ``AcquisitionManager`` waits for before resuming the acquisition loop.

**LIF**

``lifSettingsComplete(success)`` confirms that laser position and pulse delay
have been applied atomically. ``lifScopeShotAcquired(data)`` delivers raw
waveform samples from the LIF digitizer during configuration acquisition.
``lifLaserPosUpdate(pos)`` and ``lifLaserFlashlampUpdate(enabled)`` relay
device state changes from the active LIF laser.

**Communication protocol management**

``hardwareCommunicationInfoReady(hwKey, currentProtocol,
supportedProtocols, connected)`` answers a ``getHardwareCommunicationInfo()``
request and drives the communication-settings UI described in
:doc:`/user_guide/hardware_menu`. ``protocolSetResult(hwKey, success, msg)``
confirms the outcome of ``setHardwareProtocol()``.
``gpibControllersAvailable(controllerKeys)`` lists active GPIB controllers in
response to ``getActiveGpibControllers()``.

**Python script hot-reload**

``pythonScriptReloadResult(hwKey, success, msg)`` reports the outcome of a
``reloadPythonScript()`` request. Python-backed hardware and the hot-reload
workflow are described in :doc:`/user_guide/python_hardware`.

Threading and synchronization
-----------------------------

``HardwareManager`` uses a multi-lock architecture: a read-write lock
(``d_hardwareMapLock``) protects the hardware map and allows concurrent
readers, while a separate mutex (``d_connectionStateLock``) protects
connection-test state without serializing on a single global lock. When
both locks must be held, the hardware-map lock is always acquired first.
Consult the per-member ``\brief`` blocks in the API reference below for
which routines acquire which lock and what each lock protects.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: HardwareManager
   :members:
   :protected-members:
   :undoc-members:
