.. index::
   single: adding a driver
   single: hardware driver; recipe
   single: REGISTER_HARDWARE_META; driver
   single: REGISTER_HARDWARE_PROTOCOLS; driver
   single: REGISTER_HARDWARE_SETTINGS; driver
   single: REGISTER_CUSTOM_COMM; driver
   single: REGISTER_LIBRARY; driver
   single: state-management patterns; C++ drivers
   single: virtual sibling; driver
   single: HardwareObject; new driver
   single: testConnection; new driver
   single: initialize; new driver

Adding a New Hardware Driver
============================

Adding a new C++ driver of an *existing* hardware type — a new
AWG model, a new mass flow controller, a new GPIB synthesizer — is the
single most common contributor task in Blackchirp. This page is the
canonical recipe. It walks through picking the right interface class,
the five files a driver consists of, the registration macros, the three
state-management patterns the existing drivers fall into (with one
worked example each), and the smoke-testing checklist before you
declare the driver done.

This page assumes the *type* already exists. If no existing interface
class matches your hardware — that is, you are adding a new abstract
base alongside :cpp:class:`AWG`, :cpp:class:`FlowController`, and
friends — see :doc:`/developer_guide/adding_a_hardware_type`. If you
want to drive your hardware from a Python script rather than a C++
class, see :doc:`/developer_guide/python_hardware` for the trampoline
architecture; this page covers the C++ side that any new Python
trampoline still has to coexist with.

Picking the right base class
----------------------------

Every driver inherits from a hardware-type interface class. There are
eleven of them. Pick the one that matches the role your hardware will
play:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Interface class
     - Domain
     - Source directory
   * - :cpp:class:`AWG`
     - Chirp generation: arbitrary-waveform generator or DDS-based
       ramp generator that drives the CP-FTMW excitation.
     - ``src/hardware/optional/chirpsource/``
   * - :cpp:class:`Clock`
     - RF/microwave synthesizer used as a tunable LO, AWG reference,
       or DR clock.
     - ``src/hardware/core/clock/``
   * - :cpp:class:`FtmwDigitizer`
     - FTMW digitizer: the high-bandwidth oscilloscope or transient
       recorder that captures FIDs.
     - ``src/hardware/core/ftmwdigitizer/``
   * - :cpp:class:`LifDigitizer`
     - LIF digitizer: the slower oscilloscope that records
       laser-induced-fluorescence transients.
     - ``src/hardware/core/lifdigitizer/``
   * - :cpp:class:`LifLaser`
     - Tunable laser source for the LIF module (wavelength setpoint,
       optional flashlamp control).
     - ``src/hardware/core/liflaser/``
   * - :cpp:class:`FlowController`
     - Multichannel mass flow controller, optionally with a
       chamber-pressure setpoint.
     - ``src/hardware/optional/flowcontroller/``
   * - :cpp:class:`GpibController`
     - GPIB bus bridge (Prologix GPIB-LAN, GPIB-USB) that other
       :cpp:class:`HardwareObject`\ s talk through.
     - ``src/hardware/optional/gpibcontroller/``
   * - :cpp:class:`IOBoard`
     - General-purpose analog/digital I/O board for auxiliary
       readbacks, gate signals, and so on.
     - ``src/hardware/optional/ioboard/``
   * - :cpp:class:`PressureController`
     - Chamber pressure gauge / pressure controller, optionally with
       a gate valve and pressure-control mode.
     - ``src/hardware/optional/pressurecontroller/``
   * - :cpp:class:`PulseGenerator`
     - Multichannel pulse/delay generator that sequences gas, laser,
       AWG-trigger, and protection pulses.
     - ``src/hardware/optional/pulsegenerator/``
   * - :cpp:class:`TemperatureController`
     - Multichannel temperature readout (LakeShore-style cryogenic
       monitor, etc.).
     - ``src/hardware/optional/tempcontroller/``

The ``core/`` vs ``optional/`` split corresponds to whether the type is
required to run an FTMW experiment (``core``) or genuinely optional
(``optional``). The split is structural rather than user-facing —
:cpp:class:`HardwareManager` ensures every required type has at least a
virtual profile so the application always has something to talk to —
but it is the directory layout you have to put new files into. The
directory layout is described on
:doc:`/developer_guide/architecture` (source tree section).

If none of these match your hardware, the right move is almost always
to add a new abstract interface class first; see
:doc:`/developer_guide/adding_a_hardware_type`.

Files you will create
---------------------

Every driver consists of the same touches:

1. ``src/hardware/<core|optional>/<type>/<driver>.h`` — the class
   declaration. Inherits from the type's interface class; declares
   ``Q_OBJECT``; declares the constructor with the
   ``(label, parent=nullptr)`` signature (see below); overrides the
   pure virtuals the interface class requires.
2. ``src/hardware/<core|optional>/<type>/<driver>.cpp`` — the
   implementation. Carries the registration macros at file scope and
   the constructor and method bodies.
3. ``src/data/settings/hardwarekeys.h`` — only if the driver introduces
   *new* setting keys beyond what the interface class already declares.
   Add them to the appropriate ``BC::Key::<Domain>`` namespace, or
   declare a per-driver namespace inside the driver's header (see
   :cpp:any:`BC::Key::AWG::awg70002a` for the convention used by
   most drivers today).
4. *(Optional)* ``REGISTER_LIBRARY`` invocation pointing at a
   :cpp:class:`VendorLibrary` subclass, if the driver depends on a
   closed-source SDK. Authoring the library subclass itself is the
   topic of :doc:`/developer_guide/vendor_libraries`.
5. *(Optional)* a ``Virtual<Driver>`` sibling that synthesizes plausible
   readings without a real instrument. Most existing drivers ship with
   one; see *Virtual sibling* below.

**No CMake edits are required**, in the typical case. The hardware
glob in ``cmake/BlackchirpHardware.cmake`` discovers source files by
filename pattern: dropping ``vendormodel.cpp`` / ``vendormodel.h`` into
the right hardware-type directory under one of the recognized prefixes
is enough. The recognized prefixes per directory and the AUTOMOC
linkage that keeps the static-initialization registrations from being
dropped at link time are documented on
:doc:`/developer_guide/build_system` (*Hardware aggregator headers*
and *Glob-based source discovery*). If your vendor prefix is not yet
in the list, that is the only edit required, and it is one line in
each of two parallel globs. After adding files (or a new prefix) you
must re-run ``cmake`` so the globs are re-evaluated; ``cmake --build``
alone will not pick up new files.

Constructor and registration macros
-----------------------------------

Every driver follows the same three-macro registration pattern at file
scope in the ``.cpp``, plus a small constructor:

.. code-block:: cpp

   // myawg.h
   #include <hardware/optional/chirpsource/awg.h>

   namespace BC::Key::AWG {
   inline constexpr QLatin1StringView myawg{"myawg"};
   inline const QString myawgName{"Vendor Model 1234 AWG"};
   }

   class MyAwg : public AWG
   {
       Q_OBJECT
   public:
       explicit MyAwg(const QString& label, QObject *parent = nullptr);
       ~MyAwg() override = default;

   public slots:
       bool prepareForExperiment(Experiment &exp) override;
       void beginAcquisition() override;
       void endAcquisition() override;

   protected:
       bool testConnection() override;
       void initialize() override;
   };

.. code-block:: cpp

   // myawg.cpp
   #include "myawg.h"
   #include <hardware/core/hardwareregistration.h>

   REGISTER_HARDWARE_META(MyAwg, "Vendor Model 1234 high-performance AWG")
   REGISTER_HARDWARE_PROTOCOLS(MyAwg, CommunicationProtocol::Tcp,
                                       CommunicationProtocol::Rs232)
   REGISTER_HARDWARE_SETTINGS(MyAwg,
       {BC::Key::AWG::markerCount, "Marker Count",
        "Number of physical marker output channels",
        2, 0, QVariant{}, HwSettingPriority::Required}
   )

   MyAwg::MyAwg(const QString& label, QObject *parent) :
       AWG(QString(MyAwg::staticMetaObject.className()), label, parent)
   {
       setDefault(BC::Key::Comm::timeout, 10000);
       setDefault(BC::Key::Comm::termChar, QString("\n"));
       save();
   }

A few non-obvious points:

- **Driver key.** The base class constructor takes an
  ``impl`` string as its first argument. Pass
  ``QString(MyAwg::staticMetaObject.className())`` so the driver
  key tracks the class name automatically — renaming the class renames
  the registry key for free, with no parallel string table to update.
  The base class's ``hwType`` is filled in similarly inside the
  interface-class constructor. The two together combine into the
  instance's ``d_key`` (``"<HardwareType>.<label>"``), which is also the
  ``QSettings`` group root.
- **No child** ``QObject``. If the interface class sets
  ``d_threaded = true`` in its constructor (most do — :cpp:class:`AWG`,
  :cpp:class:`FtmwDigitizer`, and others enable threading because their
  I/O is expensive), the driver constructor must not have a
  ``QObject`` parent that lives on a different thread, and must not
  construct child ``QObject``\ s. Construct children inside
  :cpp:func:`HardwareObject::initialize` instead, which the manager
  invokes after the move-to-thread step. The threaded-hardware
  constructor restriction is enforced socially, not by code, so a
  buggy driver compiles fine but yields hard-to-debug cross-thread
  parent errors at runtime. See
  :doc:`/developer_guide/hardware_runtime` for the move-to-thread
  sequence.
- **Communication defaults.** Set per-protocol defaults
  (``BC::Key::Comm::timeout``, ``BC::Key::Comm::termChar``,
  per-protocol baud rates, and so on) in the constructor with
  :cpp:func:`SettingsStorage::setDefault`, then call
  :cpp:func:`SettingsStorage::save`. ``setDefault`` only writes a key
  that does not already exist on disk, so a user-supplied override
  survives subsequent constructions of the same profile.
- **Settings precedence.** ``REGISTER_HARDWARE_SETTINGS`` re-registers
  one key from the base class to override its default, bounds, or
  priority for this driver; you do *not* need to copy the
  whole base set. The ``AWG70002a`` example above re-registers only
  ``markerCount`` (the AWG70002A has two markers, the base class
  declares four as the generic default). The base/driver
  override pattern is described in
  :doc:`/developer_guide/hardware_configuration` (*Base / driver
  override pattern*); the macro reference and the
  :cpp:struct:`HwSettingDef` field list are on
  :doc:`/classes/hardwareregistry`.

State-management patterns
-------------------------

How a driver structures its hardware-specific overrides depends on how
the interface class manages the per-experiment configuration. The
existing drivers fall into three patterns. Pick the one that matches
the *interface class's* contract; you do not get to choose the pattern
on a per-driver basis.

The same three patterns appear on the Python side
(:doc:`/developer_guide/python_hardware`), one driver type at a time.
This page covers the C++ side; the Python documentation maps each
trampoline to its pattern with the same A/B/C labels.

**Pattern A — Bulk Configure.** The interface class inherits from a
complex config object — a :cpp:class:`DigitizerConfig` with channel
maps, trigger settings, sample rates, multi-record state, and so on —
and exposes a ``configure(config&)`` virtual. The experiment hands the
driver a desired config, and the driver applies it in one shot.
:cpp:class:`IOBoard` and :cpp:class:`LifDigitizer` follow this pattern.
:cpp:class:`FtmwDigitizer` is shaped similarly but exposes the per-experiment
hook directly as :cpp:func:`HardwareObject::prepareForExperiment` rather
than a separate ``configure`` virtual.

**Pattern B — Granular methods.** The interface class contains a
config object as a member and exposes per-channel or per-parameter
``hw*`` pure virtuals. The driver implements each ``hw*`` and sees only
one value per call; the interface class owns the polling sequence and
the validity checks. :cpp:class:`FlowController`,
:cpp:class:`PulseGenerator`, :cpp:class:`PressureController`, and
:cpp:class:`TemperatureController` follow this pattern.

**Pattern C — Stateless / pass-through.** The interface class has no
internal config object to manage. The driver receives experiment data
(chirp segments and markers for an :cpp:class:`AWG`; frequency
assignments for a :cpp:class:`Clock`) at
:cpp:func:`HardwareObject::prepareForExperiment` time, programs the
hardware to match, and returns. There is no bulk read-back: the
experiment data is the truth.

The fastest way to identify the pattern for a new driver is to read
the interface class header. A ``configure(...)`` virtual is a Pattern
A hint; one or more ``hw*``-style pure virtuals (``hwSetFlow``,
``hwReadPressure``, ``setHwFrequency``) are Pattern B hints; an
interface that final-overrides ``hwPrepareForExperiment`` and exposes
``prepareForExperiment`` for the driver to override is the Pattern C
shape. The three worked examples below show one driver per pattern.

Worked example A — Pattern A (IOBoard)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:cpp:class:`IOBoard` exposes a pure-virtual ``configure(IOBoardConfig&)``
and two pure-virtual readers. The driver applies the experiment's
analog and digital channel selections in ``configure``, then services
``readAnalogChannels()`` / ``readDigitalChannels()`` per call:

.. code-block:: cpp

   // myioboard.h (sketch)
   class MyIOBoard : public IOBoard
   {
       Q_OBJECT
   public:
       explicit MyIOBoard(const QString& label, QObject *parent = nullptr);

   protected:
       bool testConnection() override;
       void initialize() override;
       bool configure(IOBoardConfig &config) override;
       std::map<int,double> readAnalogChannels() override;
       std::map<int,bool>   readDigitalChannels() override;
   };

   bool MyIOBoard::configure(IOBoardConfig &config)
   {
       for (auto &[k, ch] : config.d_analogChannels)
       {
           if (!ch.enabled) continue;
           // Apply ch.range / ch.coupling to physical channel k.
           // Read back actuals; clamp ch.range, ch.coupling if needed.
       }
       for (auto &[k, ch] : config.d_digitalChannels)
       {
           if (!ch.enabled) continue;
           // Apply digital channel k's direction, level, etc.
       }
       return true; // base copies modified config into Experiment
   }

The ``config`` argument is mutable: the driver should write back any
clamped or coerced values so the experiment record reflects what the
hardware actually applied. The base :cpp:class:`IOBoard` copies the
modified config into the experiment record on success.

The two read methods receive no channel selection: each call should
walk the driver's own ``d_analogChannels`` / ``d_digitalChannels``
state (inherited from :cpp:class:`IOBoardConfig`) and return readings
only for the channels currently enabled. The
``src/hardware/optional/ioboard/labjacku3.cpp`` driver is the canonical
ground-truth driver; ``virtualioboard.cpp`` is the non-vendor
sibling.

Worked example B — Pattern B (FlowController)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:cpp:class:`FlowController` final-overrides
:cpp:func:`HardwareObject::initialize`,
:cpp:func:`HardwareObject::testConnection`, and
:cpp:func:`HardwareObject::prepareForExperiment`. A driver does not
override any of those; instead it implements the per-channel/
per-parameter ``hw*`` virtuals plus the small ``fcInitialize`` /
``fcTestConnection`` hooks the base class calls from its own
``initialize`` / ``testConnection``:

.. code-block:: cpp

   // myflow.h (sketch)
   class MyFlow : public FlowController
   {
       Q_OBJECT
   public:
       explicit MyFlow(const QString& label, QObject *parent = nullptr);

   public slots:
       void   hwSetFlowSetpoint(int ch, double val) override;
       double hwReadFlow(int ch) override;
       double hwReadFlowSetpoint(int ch) override;
       void   hwSetPressureSetpoint(double val) override;
       double hwReadPressure() override;
       double hwReadPressureSetpoint() override;
       void   hwSetPressureControlMode(bool enabled) override;
       int    hwReadPressureControlMode() override;
       // Optional: override only if the hardware can enable/disable
       // individual channels. The base default is a no-op.
       void   hwSetChannelEnabled(int ch, bool en) override;

   protected:
       void fcInitialize() override;
       bool fcTestConnection() override;
   };

Each ``hw*`` issues a few SCPI / serial commands through ``p_comm``
(the :cpp:class:`CommunicationProtocol` instance the base class built
in :cpp:func:`HardwareObject::bcInitInstrument`), parses the response,
and returns the value:

.. code-block:: cpp

   double MyFlow::hwReadFlow(int ch)
   {
       QByteArray resp = p_comm->queryCmd(u"FLOW? %1\n"_s.arg(ch+1));
       if (resp.isEmpty())
       {
           emit hardwareFailure();
           hwError(u"No response to flow query for channel %1."_s.arg(ch+1));
           return -1.0;
       }
       bool ok = false;
       double f = resp.trimmed().toDouble(&ok);
       if (!ok)
       {
           emit hardwareFailure();
           hwError(u"Could not parse flow response: %1"_s.arg(QString(resp)));
           return -1.0;
       }
       return f;
   }

The base :cpp:class:`FlowController` owns the polling timer, the
round-robin channel sequencing inside :cpp:func:`FlowController::poll`,
the :cpp:func:`FlowController::readAll` helper, the
``flowUpdate`` / ``pressureUpdate`` signal emission, and the per-experiment
validation/aux-data dispatch. The driver only sees one value at a time
and never has to worry about the cadence.

A representative ground-truth driver lives at
``src/hardware/optional/flowcontroller/mks647c.cpp`` (an MKS 647C
mass flow controller over RS-232 with a ``mksQueryCmd`` retry helper
that compensates for an idiosyncratic firmware bug). The same
directory carries ``virtualflowcontroller.cpp``, which synthesizes
plausible flow and pressure readings via :cpp:any:`QRandomGenerator`
and serves both as the user's no-hardware fallback and as the test
fixture.

Worked example C — Pattern C (AWG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:cpp:class:`AWG` declares no ``configure`` virtual and no ``hw*`` per-
parameter accessors. A driver overrides
:cpp:func:`HardwareObject::prepareForExperiment` directly: read the
chirp definition out of the :cpp:class:`Experiment`, compute (or
upload) the waveform, program any markers, and return.

.. code-block:: cpp

   bool MyAwg::prepareForExperiment(Experiment &exp)
   {
       d_enabledForExperiment = exp.ftmwEnabled();
       if (!d_enabledForExperiment)
           return true;

       const ChirpConfig &cc = exp.ftmwConfig()->d_rfConfig.d_chirpConfig;

       QVector<QPointF>  samples       = cc.getChirpMicroseconds();
       QVector<quint32>  packedMarkers = cc.getPackedMarkerData();

       // Upload waveform via vendor-specific commands; remap markers to
       // the device's bit positions; verify with *OPC?, etc.
       if (!writeWaveform(samples, packedMarkers))
       {
           exp.d_errorString = u"AWG waveform upload failed."_s;
           emit hardwareFailure();
           return false;
       }

       p_comm->writeCmd(u"Source1:RMode Triggered\n"_s);
       p_comm->writeCmd(u"Source1:TINPut ATRigger\n"_s);
       p_comm->writeCmd(u"TRIGger:MODE SYNChronous\n"_s);
       return true;
   }

Two things to call out:

- :cpp:func:`ChirpConfig::getPackedMarkerData` returns the marker data
  packed one channel per bit, indexed by *logical* marker channel. Each
  AWG vendor uses a different physical bit layout for its marker
  outputs; the driver remaps logical bits to physical bits before
  uploading. The Tektronix AWG70002A in
  ``src/hardware/optional/chirpsource/awg70002a.cpp`` puts logical
  channel 0 on bit 6 and channel 1 on bit 7 of an 8-bit marker byte,
  for instance; other vendors pack two channels into a 32-bit word
  upper bits, and so on.
- :cpp:class:`AWG` registers its scalar settings (``sampleRate``,
  ``maxSamples``, ``minFreq``, ``maxFreq``, ``markerCount``,
  ``rampOnly``, ``triggered``) via ``REGISTER_HARDWARE_BASE`` so they
  appear automatically on every driver. Re-register a key with
  ``REGISTER_HARDWARE_SETTINGS`` only when the driver's value
  is fixed by the model — e.g., ``markerCount = 2`` for the
  AWG70002A — or when the bounds need tightening.

Pattern C also covers :cpp:class:`Clock` drivers, which override the
``setHwFrequency`` / ``readHwFrequency`` per-output virtuals plus an
optional ``prepareClock`` for one-shot reference and lock setup. See
``src/hardware/core/clock/valon5009.cpp`` for the canonical Pattern C
clock driver.

initialize() and testConnection()
---------------------------------

Every driver implements two more pure virtuals from
:cpp:class:`HardwareObject`. The split between them is not arbitrary
and is the source of more contributor confusion than any other point
on this page:

- :cpp:func:`HardwareObject::initialize` runs once, on the device's own
  thread, immediately after construction and the move-to-thread step.
  It is the place to construct child :cpp:class:`QObject`\ s, allocate
  device-side buffers, register :cpp:class:`QTimer`\ s — anything
  that must happen exactly once per instance lifetime. **Do not attempt
  vendor I/O here.** The :cpp:class:`CommunicationProtocol` has been
  built and initialized at this point but no successful connection has
  been established yet; vendor I/O will fail in any reasonable
  no-hardware test environment.
- :cpp:func:`HardwareObject::testConnection` runs on every connection
  test — the deferred sweep at the end of the hardware-manager sync,
  the user clicking *Test Connection* in :cpp:class:`HWDialog`, the
  retry triggered by a hot-reload of a Python script. It is the place
  for the cheap interaction with the device: typically an ``*IDN?``
  query plus an assertion that the responding device is the model the
  driver expects. On failure, store a descriptive message in
  ``d_errorString`` and return ``false``; the wrapper
  :cpp:func:`HardwareObject::bcTestConnection` will report
  disconnected. On success, return ``true``.

A few interface classes (``Clock``, ``FlowController``, ``IOBoard``,
``PressureController``) final-override
:cpp:func:`HardwareObject::initialize` and
:cpp:func:`HardwareObject::testConnection` themselves to do shared
setup, and call into a smaller per-driver hook
(``initializeClock`` / ``testClockConnection``,
``fcInitialize`` / ``fcTestConnection``,
``pcInitialize`` / ``pcTestConnection``). Implement those instead;
the rule is "look at the interface header and override what is pure
virtual."

The full lifecycle from construction through the first
:cpp:func:`HardwareObject::connected` emission — the move-to-thread
step, ``buildCommunication``, the ``hardwareFailure`` lambda the
manager wires up — is documented on
:doc:`/developer_guide/hardware_runtime` (*Per-object lifecycle:
bcInitInstrument and bcTestConnection*).

Auxiliary and validation data
-----------------------------

A driver may optionally override two more virtuals to participate in
the auxiliary-data pipeline:

- :cpp:func:`HardwareObject::readAuxData` returns an
  :cpp:type:`AuxDataStorage::AuxDataMap` of per-experiment readings —
  flow values, pressures, temperatures, anything worth plotting on the
  *Aux* and *Rolling* tabs and persisting in
  :cpp:class:`AuxDataStorage`. The interface classes for the
  Pattern B types (``FlowController``, ``PressureController``,
  ``TemperatureController``) implement this for you out of the
  config-object state, so a typical driver of those types does not
  need its own override; for AWG, FtmwDigitizer, IOBoard, or a custom
  type, override it when there are device-specific readings worth
  recording. The default returns an empty map.
- :cpp:func:`HardwareObject::readValidationData` returns the subset of
  readings the experiment validator should range-check during
  acquisition. The keys returned must be a subset of the values
  returned by :cpp:func:`HardwareObject::validationKeys` — see
  :doc:`/classes/hardwareobject`.

Both run from the wrapper :cpp:func:`HardwareObject::bcReadAuxData`,
which also emits the corresponding signals, and the
:cpp:class:`HardwareManager` prefixes the per-device map keys with the
source object's ``hwKey`` before fanning the data out to consumers.
The full fan-out plumbing is documented on
:doc:`/developer_guide/hardware_runtime` (*Auxiliary, validation, and
rolling data*); the persistence side is on
:doc:`/classes/auxdatastorage`.

Virtual siblings
----------------

Every hardware *type* in Blackchirp ships with a ``Virtual<Type>``
driver that synthesizes plausible readings without a real
instrument — :cpp:class:`VirtualAwg`,
:cpp:class:`VirtualFlowController`, :cpp:class:`VirtualIOBoard`, and
so on. The virtual driver backs the system profiles that
guarantee a required type always has something for
:cpp:class:`HardwareManager` to talk to, and it is the canonical
fixture for the hardware unit tests under ``tests/``.

Adding a new *driver* of an existing type does **not** require a
parallel ``Virtual<Driver>``: the per-type virtual already covers the
fall-back and test-fixture roles for every driver of that
type. If your driver's behavior diverges from the type's virtual
sibling enough that the existing fixture no longer represents it
faithfully, that usually indicates the *type* itself needs an
expanded contract (new virtuals, new aux-data keys, …) — which is a
larger blast-radius change than adding a driver. See
:doc:`/developer_guide/adding_a_hardware_type`, which covers virtual
sibling authoring alongside the rest of the type-level surface.

Custom protocol parameters
--------------------------

Drivers that talk to their hardware outside the standard
:cpp:class:`Rs232Instrument` / :cpp:class:`TcpInstrument` /
:cpp:class:`GpibInstrument` abstractions register
:cpp:enumerator:`CommunicationProtocol::Custom` in their
``REGISTER_HARDWARE_PROTOCOLS`` invocation. :cpp:class:`CustomInstrument`
keeps its underlying :cpp:any:`QIODevice` ``nullptr`` and its
``initialize()`` and ``testConnection()`` are no-ops; the driver's own
``testConnection()`` does whatever vendor-specific handshake is
required.

The complementary problem is collecting connection parameters from the
user *without* instantiating the driver — the
:cpp:class:`AddProfileDialog` and :cpp:class:`CommunicationDialog`
need to render the right input widgets before any object exists. That
is what ``REGISTER_CUSTOM_COMM`` is for:

.. code-block:: cpp

   REGISTER_CUSTOM_COMM(MyDriver,
       {"devPath"_L1, "Device Path",
        "Path to the device node (e.g. /dev/spcm0)",
        CustomCommType::String, 260, QVariant{}},
       {"serialNo"_L1, "Serial Number",
        "USB serial number",
        CustomCommType::Int, 0, INT_MAX})

Each :cpp:struct:`CustomCommDef` carries a settings key, a user-facing
label, a description, a :cpp:enum:`CustomCommType`
(``String``, ``Int``, or ``FilePath``), and type-dependent bound
fields. The descriptors are read out of
:cpp:class:`HardwareRegistry` by :cpp:class:`CustomProtocolWidget` at
profile-creation time. The driver reads the resulting user-supplied
values back from the ``BC::Key::Comm::custom`` settings group
inside ``testConnection()``:

.. code-block:: cpp

   bool MyDriver::testConnection()
   {
       auto path = getGroupValue<QString>(BC::Key::Comm::custom,
                                          "devPath"_L1,
                                          QString("/dev/spcm0"));
       d_serialNo = getGroupValue<int>(BC::Key::Comm::custom,
                                       "serialNo"_L1,
                                       0);
       // ...vendor-specific open() / *IDN? / ...
   }

The full descriptor reference is on :doc:`/classes/custominstrument`;
the runtime side of the protocol selector is on
:doc:`/developer_guide/hardware_runtime`. Python-backed drivers also
use ``Custom`` as the explicit "communication is handled by the ``.py``
script" indicator — see :doc:`/developer_guide/python_hardware`.

Vendor library dependency
-------------------------

If the driver depends on a closed-source SDK loaded by a
:cpp:class:`VendorLibrary` subclass, declare the dependency at static
registration time:

.. code-block:: cpp

   REGISTER_HARDWARE_META(MyDriver, "...")
   REGISTER_HARDWARE_PROTOCOLS(MyDriver, CommunicationProtocol::Custom)
   REGISTER_LIBRARY(MyDriver, MyVendorLibrary)

The registry uses the ``REGISTER_LIBRARY`` linkage to know which
drivers must be torn down before the library is reloaded — that is how
the *Library Status* tab in the Hardware Configuration dialog can
change a vendor library's path without leaving live consumers holding
a stale handle. Authoring a new :cpp:class:`VendorLibrary` subclass is
the topic of :doc:`/developer_guide/vendor_libraries`; a driver that
only consumes an existing one needs nothing more than
``REGISTER_LIBRARY`` and the corresponding header include.

Smoke testing
-------------

Most of a new driver cannot be meaningfully exercised without the
physical hardware it implements: the ``Virtual`` communication
protocol returns no real data, so a profile created against
``Virtual`` only confirms that the driver's static registration is
intact and that the application starts. Genuine verification of
``testConnection``, ``prepareForExperiment``, the ``hw*`` overrides,
and the aux-data path requires connecting to the actual device. Plan
for development time on the instrument itself, and instrument the
driver accordingly:

- **Use** :cpp:func:`HardwareObject::hwDebug` **liberally** while the
  driver is being brought up. Log every command sent and every
  response received — the raw bytes, not just the parsed result —
  including hex dumps for any non-ASCII payload.
- **Enable debug logging at runtime** so the ``hwDebug`` output reaches
  the *Log* tab and the on-disk log file. The toggle is an
  application-level configuration item (see
  :doc:`/user_guide/log_tab`); this is the single most useful tool for
  diagnosing protocol-level mismatches against a vendor manual.
- **Be deliberate about which debug calls survive once the driver is
  stable.** ``hwDebug`` is *not* compiled out of release builds: if
  debug logging is disabled the call returns without writing anything,
  but the arguments are still evaluated. A ``hwDebug`` whose argument
  builds a multi-kilobyte hex dump on every read is still doing that
  work in production, even if no one ever sees the output. Keep the
  calls that are cheap and diagnostically valuable (a one-line
  ``*IDN?`` echo, a pre-acquisition handshake summary); prune the
  ones that allocate aggressively or run inside per-shot inner loops.
  Prefer to keep calls that provide diagnostic information about why
  an error occurred rather than simply logging every command and response.
- **Consider writing a Python driver first** when the protocol is
  novel or the documentation is incomplete. The Python trampoline
  path lets you iterate on command syntax, parsing, and the per-state
  control flow without a rebuild on every change, then convert to a
  C++ driver once the protocol behavior is confirmed. The Python
  hardware architecture is on :doc:`/developer_guide/python_hardware`
  and the user-facing workflow on :doc:`/user_guide/python_hardware`.

What you *can* verify without the hardware is that the driver
builds, registers correctly, and does not break the rest of the
application:

#. **Build with tests.** The default build option ``BC_BUILD_TESTS=ON``
   compiles the test executables; rebuild and run the relevant ones:

   .. code-block:: bash

      cmake . -B build/tests
      cmake --build build/tests --target tests -j$(nproc)
      ctest --test-dir build/tests

   The hardware-side tests that are most likely to surface issues with
   a new driver:

   - ``tst_hardwareregistrytest`` — exercises registration macros,
     factory invocation, supported-protocol lookup, and inheritance
     chain construction. A typo in ``REGISTER_HARDWARE_META`` or a
     missing ``Q_OBJECT`` typically shows up here.
   - ``tst_runtimehardwareconfigtest`` — exercises the active-selection
     map, validation, and threading override.
   - ``tst_hardwareprofilemanagertest`` — exercises profile create /
     activate / deactivate / delete and the system-profile guarantee.
   - ``tst_hardwarekeys`` — catches collisions in the static key
     declarations under ``BC::Key::``. If you added new keys to
     ``hardwarekeys.h`` or to a per-driver namespace, this is where a
     duplicate or shadowed key surfaces.

#. **Launch the application** and confirm the driver is registered.
   A debug build under ``build/Desktop-Debug/`` is fastest. The new
   driver should appear in the *Hardware Configuration* dialog's
   right-hand Configuration panel under its hardware type, with the
   description string from ``REGISTER_HARDWARE_META`` and the
   protocol(s) from ``REGISTER_HARDWARE_PROTOCOLS`` rendered in the
   *Add Profile* dialog. Settings you registered with
   ``REGISTER_HARDWARE_SETTINGS`` should be present and editable in
   :cpp:class:`HwSettingsWidget`.

Beyond that, take the driver to the bench. Create a profile against
the real communication protocol, point it at the device, watch the
debug log while
:cpp:func:`HardwareObject::testConnection` runs, and iterate from
there. Drivers that emit :cpp:func:`HardwareObject::hardwareFailure`
will mark themselves disconnected, and — if ``d_critical`` is true
(the default) — block the *Start Experiment* state machine until the
test passes; the *Hardware Menu* surface
(:doc:`/user_guide/hardware_menu`) shows that state at a glance.
Once the connection is solid, run a short experiment that exercises
``prepareForExperiment``, ``beginAcquisition`` / ``endAcquisition``,
and any ``readAuxData`` / ``readValidationData`` overrides — for
Pattern A and Pattern C drivers the experiment is the only place
those code paths run.
