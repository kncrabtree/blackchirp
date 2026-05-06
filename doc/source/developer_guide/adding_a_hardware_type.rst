.. index::
   single: adding a hardware type
   single: hardware type; recipe
   single: HardwareObject; new type
   single: REGISTER_HARDWARE_BASE; new type
   single: state-management patterns; new type
   single: HeaderStorage; optional hardware config
   single: addOptHwConfig
   single: HardwareManager; type fan-out
   single: status box; new type
   single: control widget; new type
   single: experiment setup page; new type
   single: HARDWARE_TYPE_HEADERS
   single: blackchirp-test-hardware; new type

Adding a New Hardware Type
==========================

Adding a new abstract *hardware type* — a new interface class that no
existing driver matches, alongside :cpp:class:`AWG`,
:cpp:class:`Clock`, :cpp:class:`FtmwScope`, :cpp:class:`FlowController`,
and the eight other types Blackchirp ships — is the rarer and broader
contributor task. It is a coordinated change across ``hardware/``,
``data/experiment/``, and ``gui/`` rather than a self-contained file
drop, so the recipe for it is correspondingly larger than the one for
:doc:`/developer_guide/adding_a_driver`.

This page walks the design and integration steps end to end: deciding
that a new type is genuinely warranted, picking the state-management
shape, sketching the interface class, wiring it into the build,
authoring the optional config object that travels with the experiment,
hooking the GUI surfaces (status box, control widget, experiment-setup
page) so the device is usable, plumbing the per-type fan-out through
:cpp:class:`HardwareManager`, and recommending the Python trampoline
and test fixtures that round out the type. Before reading further,
make sure you have already read
:doc:`/developer_guide/adding_a_driver` — the C++ surface a new
driver carries (constructor signature, registration macros,
:cpp:func:`HardwareObject::initialize` /
:cpp:func:`HardwareObject::testConnection`, aux/validation data) is
the same surface every driver of the new type will carry, and
this page does not repeat it.

When this applies vs. adding a driver
-------------------------------------

A new hardware type is justified when no existing interface class
captures the *role* the new device will play in Blackchirp, even
loosely. The litmus test is the existing types' interfaces: if any
one of them could be adapted with reasonable effort — by adding a
new ``hw*`` virtual, a new aux-data key, or a small extension to its
config — that is the right move. Adding a new type pays for itself
only when the role is genuinely new; otherwise you are creating a
new branch in the dispatch logic of every cross-cutting subsystem
(:cpp:class:`HardwareManager`, the Hardware menu in
:cpp:class:`MainWindow`, :cpp:class:`ExperimentSetupDialog`, the
test-hardware library) for a device that an existing type would
have absorbed.

Concretely:

- A new model of an existing role — a different vendor's mass flow
  controller, a different GPIB-attached synthesizer, a different
  AWG — is a new *driver* against an existing type. Use
  :doc:`/developer_guide/adding_a_driver`.
- A new role that does not yet exist — a beam blocker, a magnetic
  field coil, a sample-loading robot — is a new *type*. This page
  applies.

Drivers carry no cross-system blast radius beyond their own
``.cpp``/``.h`` pair (and ``hardwarekeys.h`` if they declare new
keys). Types carry it everywhere. Plan accordingly.

Designing the interface
-----------------------

Before writing code, decide six things. Each one shapes the
interface class and is awkward to revisit later because every
driver of the type has to be reworked alongside it.

State-management pattern
~~~~~~~~~~~~~~~~~~~~~~~~

The C++ patterns from
:doc:`/developer_guide/adding_a_driver` (*State-management
patterns*) are not driver choices — they are *type* choices. The
interface class commits to one pattern; every driver follows it.
The same A/B/C taxonomy describes the Python trampolines on
:doc:`/developer_guide/python_hardware`, so a new type's pattern
also decides what its Python side will look like.

- **Pattern A (Bulk Configure)** when the type owns or carries a
  complex config object — channel maps, trigger settings,
  per-output state — and the experiment hands the device a fully-
  formed configuration in one shot. The interface class typically
  inherits its config object via multiple inheritance (the way
  :cpp:class:`IOBoard` inherits from :cpp:class:`IOBoardConfig`,
  and :cpp:class:`FtmwScope` from
  :cpp:class:`FtmwDigitizerConfig`), final-overrides
  :cpp:func:`HardwareObject::hwPrepareForExperiment` to pull the
  desired config out of the experiment, dispatch a pure-virtual
  ``configure(config&)`` to the driver, and write the validated
  config back through :cpp:func:`Experiment::addOptHwConfig`.
- **Pattern B (Granular methods)** when interaction is per-channel
  or per-parameter — each call setting or reading one value. The
  interface class contains a config object as a member, exposes
  public ``setX`` / ``readX`` slots, owns the polling sequence on
  a :cpp:any:`QTimer`, and delegates the per-call hardware I/O to
  ``hw*`` pure virtuals. :cpp:class:`FlowController`,
  :cpp:class:`PulseGenerator`,
  :cpp:class:`PressureController`, and
  :cpp:class:`TemperatureController` all follow this pattern.
- **Pattern C (Stateless / pass-through)** when the type carries
  no internal config to manage and is configured exclusively at
  experiment time. The interface class is intentionally thin; each
  driver overrides :cpp:func:`HardwareObject::prepareForExperiment`
  directly to read the per-experiment data out of the
  :cpp:class:`Experiment`, program the hardware, and return.
  :cpp:class:`AWG` and :cpp:class:`Clock` follow this pattern.

The pattern interacts with how readily users can supply Python
drivers for the new type: Pattern A wants a single ``configure``
JSON dispatch per experiment, Pattern B wants one IPC round trip
per ``hw*`` call, Pattern C wants a single
``prepare_for_experiment`` dispatch with the experiment payload.
The trampoline contract on :doc:`/developer_guide/python_hardware`
maps each pattern onto an explicit recipe.

Threading and criticality defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``d_threaded`` in the interface-class constructor body. Most
hardware types are threaded — vendor I/O is expensive enough that
running it on the manager thread starves every other device — and
the interface class is the right place to set the default so every
driver inherits it. The user can still override the per-
profile threading at profile creation time
(:doc:`/developer_guide/hardware_runtime` covers the runtime side).
The same is true for ``d_critical``, which defaults to ``true``;
override only when the device is non-essential by nature (the
existing :cpp:class:`Clock` / :cpp:class:`FtmwScope` types use the
default; an instrument whose absence should never abort an
experiment is the rare exception).

Once a type sets ``d_threaded = true``, the
**threaded-hardware constructor restriction** from
:doc:`/developer_guide/hardware_runtime` applies to every
driver: no ``QObject`` parent at construction, and no
child :cpp:any:`QObject` constructed in the constructor. Construct
children inside :cpp:func:`HardwareObject::initialize`. Document
this once at the top of the new interface header so driver authors
do not have to rediscover it.

Supported communication protocols
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Decide which subset of
:cpp:enumerator:`CommunicationProtocol::Rs232`,
:cpp:enumerator:`CommunicationProtocol::Tcp`,
:cpp:enumerator:`CommunicationProtocol::Gpib`,
:cpp:enumerator:`CommunicationProtocol::Custom`, and
:cpp:enumerator:`CommunicationProtocol::Virtual` the type can
support across drivers. Each driver further narrows the
set with its own ``REGISTER_HARDWARE_PROTOCOLS`` invocation. Most
new types should at least allow ``Virtual`` so the system-profile
fall-back pattern (every required type carries a ``virtual``
profile, see
:doc:`/developer_guide/hardware_configuration`) extends to the
new type.

Shared settings and validation keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every setting that every driver of the new type will share
— channel counts, range tables, polling intervals — is registered
on the interface class with ``REGISTER_HARDWARE_BASE`` (or
``REGISTER_HARDWARE_BASE_ARRAY`` for array settings). Drivers
re-register a key with ``REGISTER_HARDWARE_SETTINGS`` only to
override the default, the bounds, or the priority for their own
driver. The merge is described in
:doc:`/developer_guide/hardware_configuration` (*Base /
driver override pattern*). Pick which keys go in
``BC::Key::<TypeName>::`` versus which belong on a per-driver
sub-namespace at the same time you draft the registration block.

If the type produces values that should be range-checked during
acquisition (a temperature that must stay below a threshold, a
flow that must stay above one), enumerate them in an override of
:cpp:func:`HardwareObject::validationKeys`. The validator side of
the abort path is documented on
:doc:`/developer_guide/experiment_lifecycle`.

Optional config object: decide if you need one
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the type carries experiment-time configuration that should
travel with the experiment record, plan a dedicated
:cpp:class:`HeaderStorage` subclass — the role
:cpp:class:`FlowConfig` plays for :cpp:class:`FlowController`,
:cpp:class:`IOBoardConfig` for :cpp:class:`IOBoard`,
:cpp:class:`LifDigitizerConfig` for :cpp:class:`LifScope`. The
config object owns the storeValues / retrieveValues / prepareChildren
contract from :doc:`/developer_guide/persistence`; the interface
class registers the validated config with
:cpp:func:`Experiment::addOptHwConfig` from
:cpp:func:`HardwareObject::hwPrepareForExperiment` (Pattern A) or
:cpp:func:`HardwareObject::prepareForExperiment` (Pattern B/C).
:cpp:class:`Experiment` owns the registered copy via
``std::shared_ptr<HeaderStorage>``, keyed by header key, and
:cpp:func:`Experiment::getOptHwConfig` hands it back to the GUI
and to driver code that needs to inspect it.

The interface class
-------------------

Sketch for a new type ``BeamBlocker`` that uses Pattern B (per-
channel granular methods). Adapt the pattern to A or C by removing
the ``hw*`` virtuals and the public-slot wrapper, and overriding
either :cpp:func:`HardwareObject::hwPrepareForExperiment` (Pattern A)
or :cpp:func:`HardwareObject::prepareForExperiment` (Pattern C)
instead.

.. code-block:: cpp

   // beamblocker.h
   #include <hardware/core/hardwareobject.h>
   #include <data/settings/hardwarekeys.h>

   namespace BC::Key::BeamBlocker {
   inline constexpr QLatin1StringView numChannels{"numChannels"};
   inline constexpr QLatin1StringView pollInterval{"pollInterval"};
   }

   class BeamBlocker : public HardwareObject
   {
       Q_OBJECT
   public:
       BeamBlocker(const QString& impl, const QString& label,
                   QObject *parent = nullptr);
       ~BeamBlocker() override;

       QStringList validationKeys() const override;

   public slots:
       void   setBlocked(int channel, bool blocked);
       bool   readBlocked(int channel);

   signals:
       void blockedUpdate(int channel, bool blocked, QPrivateSignal);

   protected:
       virtual void hwSetBlocked(int channel, bool blocked) = 0;
       virtual int  hwReadBlocked(int channel) = 0;

       AuxDataStorage::AuxDataMap readAuxData() override;

   private:
       int d_numChannels{0};
   };

.. code-block:: cpp

   // beamblocker.cpp
   #include "beamblocker.h"
   #include <hardware/core/hardwareregistration.h>

   REGISTER_HARDWARE_BASE(BeamBlocker,
       {BC::Key::BeamBlocker::numChannels, "Channels",
        "Number of beam blocker channels.",
        2, 1, 16, HwSettingPriority::Required},
       {BC::Key::BeamBlocker::pollInterval, "Poll Interval (ms)",
        "Interval between blocker readbacks in milliseconds.",
        500, 1, QVariant{}, HwSettingPriority::Optional}
   )

   BeamBlocker::BeamBlocker(const QString& impl, const QString& label,
                            QObject *parent) :
       HardwareObject(QString(BeamBlocker::staticMetaObject.className()),
                      impl, label, parent),
       d_numChannels(get(BC::Key::BeamBlocker::numChannels, 2))
   {
       d_threaded = true;
       // d_critical defaults to true; leave alone unless intentionally optional.
   }

A few pieces are worth calling out:

- The base-class constructor takes ``hwType`` from
  ``staticMetaObject.className()``. Every driver calls the base-
  class constructor with its own
  ``staticMetaObject.className()`` for ``hwImpl``, so the
  combination yields a unique ``d_key`` of
  ``"BeamBlocker.<label>"`` for the type and
  ``"<DriverName>"`` for the driver. Renaming the class
  renames the registry key.
- ``REGISTER_HARDWARE_BASE`` is the type-level counterpart of
  ``REGISTER_HARDWARE_SETTINGS``. The settings declared here are
  applied to every driver through
  :cpp:func:`HardwareObject::applyRegisteredSettings` from the
  base-class constructor; a driver that needs different defaults
  re-registers the same keys with ``REGISTER_HARDWARE_SETTINGS``,
  and the driver-level entry wins. The macro reference is
  on :doc:`/classes/hardwareregistry`.
- The public ``setBlocked`` / ``readBlocked`` slots are the
  surface :cpp:class:`HardwareManager` and the GUI consume; they
  call the protected ``hw*`` pure virtuals, emit
  ``blockedUpdate`` so observers can react, and handle errors.
  Keep the ``hw*`` virtuals as narrow as possible — a single
  command/response interaction per call — so each driver can be
  written without re-reasoning about the polling cadence or the
  signal protocol.

Wiring into the build
---------------------

A new hardware type touches the build system in three places. The
:doc:`/developer_guide/build_system` page covers the cmake layout
in detail; the points specific to a new type are:

1. **Place the interface source files.** Pick whether the type is
   *core* (required to run an FTMW experiment, like
   :cpp:class:`Clock` or :cpp:class:`FtmwScope`) or *optional*
   (everything else). Put the interface ``.cpp``/``.h`` in a new
   subdirectory under ``src/hardware/core/<type>/`` or
   ``src/hardware/optional/<type>/``. Drivers of the
   new type will live alongside the interface in the same
   directory.

2. **Add the interface .cpp to** ``HARDWARE_TYPES_SOURCES``. The
   list in ``cmake/BlackchirpHardware.cmake`` enumerates every
   interface ``.cpp`` explicitly. Append the new type's interface
   path; the driver source files are picked up by the glob
   pattern in step 4.

3. **Add the interface .h to** ``HARDWARE_TYPE_HEADERS``. The same
   file enumerates every interface header explicitly so the
   generated ``hw_base.h`` aggregator picks them up. The aggregator
   is what gives :cpp:class:`HardwareRegistry` access to every
   interface metaobject at static-init time.

4. **Add the new directory's glob patterns to**
   ``HARDWARE_IMPLEMENTATIONS_SOURCES`` **and**
   ``HARDWARE_IMPLEMENTATION_HEADERS``. ``BlackchirpHardware.cmake``
   discovers driver source files by directory and filename prefix —
   ``virtual*``, ``mks*``, ``awg*``, and so on. A new directory
   means a new pair of glob patterns. Use the existing per-type
   blocks as templates; the entries for ``flowcontroller/`` are a
   minimal model.

After adding source files, **re-run cmake** so the globs are
re-evaluated; ``cmake --build`` alone will not pick up new files.

Optional config object
----------------------

When the type needs experiment-time configuration that should
travel with the experiment record, the convention is one
``HeaderStorage`` subclass per type, owning all of the type's
configurable state. Convention:

- File: ``src/data/experiment/hardware/<core|optional>/<type>/<typename>config.{cpp,h}``.
- Class: ``<TypeName>Config``, inheriting from
  :cpp:class:`HeaderStorage`. Constructor takes the parent
  hardware key so :cpp:func:`HeaderStorage::headerKey` resolves
  correctly.
- Override :cpp:func:`HeaderStorage::storeValues` /
  :cpp:func:`HeaderStorage::retrieveValues` to round-trip the
  scalar fields. Override :cpp:func:`HeaderStorage::prepareChildren`
  if the config has its own nested ``HeaderStorage`` children
  (rare for hardware configs).
- Add value keys in a ``BC::Store::<TypeName>::`` sub-namespace
  next to the class, per :doc:`/developer_guide/persistence`
  (*Key namespaces*).

The interface class registers the validated config on the
experiment by calling :cpp:func:`Experiment::addOptHwConfig`
from inside its experiment-preparation hook (see *Lifecycle hooks*
below). :cpp:class:`Experiment` owns the registered copy via
``std::shared_ptr<HeaderStorage>``, keyed by the config's header
key. Reading the config back — from a GUI page, from a Python
trampoline, from another :cpp:class:`HardwareObject` — goes
through :cpp:func:`Experiment::getOptHwConfig`, which returns a
``std::weak_ptr`` typed to the requested config class.

If the config object should be configurable from the experiment-
setup wizard, plan a corresponding
:cpp:class:`ExperimentConfigPage` subclass at the same time;
*GUI integration* below covers the page side.

Lifecycle hooks
---------------

The interface class is also where the type-level lifecycle
overrides land. Most types do **not** need to override every hook
— pick the minimal set that delivers the type's contract.

- :cpp:func:`HardwareObject::hwPrepareForExperiment` for **Pattern
  A** types. Final-override the wrapper, pull the desired config
  out of :cpp:class:`Experiment`, dispatch a pure-virtual
  ``configure(config&)`` to the driver, write the validated
  config back via :cpp:func:`Experiment::addOptHwConfig`. The
  :cpp:class:`IOBoard` implementation in
  ``src/hardware/optional/ioboard/ioboard.cpp`` is the canonical
  template.
- :cpp:func:`HardwareObject::prepareForExperiment` for **Pattern
  B/C** types. The base class's
  :cpp:func:`HardwareObject::hwPrepareForExperiment` already
  reattempts a connection if disconnected and dispatches to
  :cpp:func:`HardwareObject::prepareForExperiment`; a Pattern B
  type final-overrides the inner virtual to push experiment-time
  setpoints to the device, register aux-data keys with
  :cpp:func:`AuxDataStorage::registerKey`, and call
  :cpp:func:`Experiment::addOptHwConfig`. The
  :cpp:class:`FlowController` implementation in
  ``src/hardware/optional/flowcontroller/flowcontroller.cpp`` is
  the canonical Pattern B template; for Pattern C, each driver
  overrides :cpp:func:`HardwareObject::prepareForExperiment`
  directly and the interface class itself stays out of the way
  (see :cpp:class:`AWG`).
- :cpp:func:`HardwareObject::beginAcquisition` /
  :cpp:func:`HardwareObject::endAcquisition` when the type needs
  to start or stop hardware actions at experiment boundaries — an
  AWG starting waveform playback, a digitizer arming for
  triggers. Default is a no-op.
- :cpp:func:`HardwareObject::sleep` when the hardware supports a
  low-power state and you want :cpp:class:`HardwareManager` to
  put the device into it between experiments. Default is a
  no-op.
- :cpp:func:`HardwareObject::readSettings` when the user editing
  settings in :cpp:class:`HWDialog` should refresh in-driver cached
  state (a re-read of channel-count after a Required setting
  change, a re-validation of array sizes). The base class calls
  the override from :cpp:func:`HardwareObject::bcReadSettings`
  after it has reloaded settings from disk and refreshed
  ``d_critical`` and ``d_commType``.

The full bring-up sequence (``bcInitInstrument`` →
``buildCommunication`` → ``initialize`` → ``bcTestConnection``)
is described on :doc:`/developer_guide/hardware_runtime`.

GUI integration
---------------

A new hardware type usually contributes three GUI surfaces. Each
is optional, but each is what makes the new type usable from the
running application; do not skip them unless the type is
genuinely headless.

Status box
~~~~~~~~~~

The :cpp:class:`HardwareStatusBox` subclass that shows live
device state on the Hardware Status panel.

- File: ``src/gui/widget/<typename>statusbox.{cpp,h}``.
- Inherits :cpp:class:`HardwareStatusBox` (which is a
  :cpp:any:`QFrame` carrying the configure-requested signal so
  clicking the box opens the per-device dialog).
- Subscribes to the type-specific update signals on
  :cpp:class:`HardwareManager` (see *HardwareManager fan-out*
  below) and renders them into the layout.

Status boxes are not auto-discovered. The dispatch site is the
``hwType`` if/else chain inside
:cpp:func:`MainWindow::buildHardwareUI`. Add a new ``else if``
branch keyed on
``QString(BeamBlocker::staticMetaObject.className())``: construct
the status box, add it to ``ui->hwStatusLayout``, wire its
``configureRequested`` signal to the menu action, connect every
type-specific :cpp:class:`HardwareManager` update signal to the
box's update slots. Use the existing
:cpp:class:`PressureStatusBox` and :cpp:class:`PulseStatusBox`
branches as templates.

Control widget
~~~~~~~~~~~~~~

The widget that occupies the **Control** tab of
:cpp:class:`HWDialog` for live device interaction.

- File: ``src/gui/widget/<typename>controlwidget.{cpp,h}``.
- Inherits :cpp:any:`QWidget` (and
  :cpp:class:`SettingsStorage` if the widget itself needs
  persistent state — see :cpp:class:`GasControlWidget` for the
  multiple-inheritance pattern).
- Communicates with the live :cpp:class:`HardwareObject` only
  through :cpp:class:`HardwareManager` slots, never directly.
  The threaded-hardware threading rules from
  :doc:`/developer_guide/hardware_runtime` mean every interaction
  must be queued through a manager slot so the call lands on the
  device's thread. This is also why the widget takes a manager
  pointer (or no hardware-side reference at all, with the
  manager-side connections wired by :cpp:class:`MainWindow`)
  instead of a hardware-object pointer.

Like the status box, the control widget is wired in
:cpp:func:`MainWindow::buildHardwareUI` — inside the same
``else if`` branch you added for the status box. Construct the
control widget on the menu action's ``triggered`` slot, connect
it to the manager's update signals and to its own
type-specific slots, and pass it to ``createHWDialog`` so it
appears as the Control tab. The
:cpp:class:`GasControlWidget` plus
:cpp:func:`HardwareManager::flowSetpointUpdate` plumbing in the
``FlowController`` branch is a clean template.

Experiment-setup page
~~~~~~~~~~~~~~~~~~~~~

The page in :cpp:class:`ExperimentSetupDialog` that lets the
user configure the type's experiment-time settings before the
experiment starts. Only relevant when the type contributes an
optional config object.

- File: ``src/gui/expsetup/experiment<typename>configpage.{cpp,h}``.
- Inherits :cpp:class:`ExperimentConfigPage` (which is itself a
  :cpp:any:`QWidget` plus :cpp:class:`SettingsStorage`).
- Constructor signature
  ``(const QString hwKey, const QString title, Experiment *exp,
  QWidget *parent = nullptr)`` to match the
  ``addOptHwPages<T>`` template that
  :cpp:class:`ExperimentSetupDialog` uses to instantiate one
  page per active profile of the type.
- Implements the slots :cpp:func:`ExperimentConfigPage::initialize`,
  :cpp:func:`ExperimentConfigPage::validate`, and
  :cpp:func:`ExperimentConfigPage::apply`. ``apply`` is the hook
  that calls :cpp:func:`Experiment::addOptHwConfig` with the
  page's edited config so the experiment record carries the
  user's choices into the acquisition.

The dispatch site is the constructor of
:cpp:class:`ExperimentSetupDialog`, which already calls
``addOptHwPages<PageT>(hwTypeName, expTypeItem)`` once per
page-bearing hardware type. Add a new line for the new type:

.. code-block:: cpp

   addOptHwPages<ExperimentBeamBlockerConfigPage>(
       QString(BeamBlocker::staticMetaObject.className()), expTypeItem);

The ``addOptHwPages`` template walks the active hardware map,
filters to the requested type, and instantiates one page per
matching profile, so the dialog automatically shows one
configuration page per profile of the new type without further
plumbing. ``ExperimentFlowConfigPage`` in
``src/gui/expsetup/experimentflowconfigpage.{cpp,h}`` is a
minimal Pattern B template; ``ExperimentIOBoardConfigPage`` is
the Pattern A counterpart.

Auxiliary and validation data
-----------------------------

The aux/validation pipeline is the same one
:doc:`/developer_guide/adding_a_driver` describes for new
drivers. From the type's perspective:

- :cpp:func:`HardwareObject::readAuxData` returns the per-experiment
  readings the type produces — temperatures, flows, blocker
  positions, anything worth plotting on the Aux/Rolling tabs and
  persisting in :cpp:class:`AuxDataStorage`. For Pattern B types,
  implement this on the interface class so every driver gets the
  aux-data emission for free; the base class's polling sequence
  already populates the cached state ``readAuxData`` reads from.
  See ``FlowController::readAuxData`` for the convention.
- :cpp:func:`HardwareObject::readValidationData` returns the
  subset of readings the validator should range-check. The keys
  must be a subset of the values returned by
  :cpp:func:`HardwareObject::validationKeys`.

Both run from the wrapper
:cpp:func:`HardwareObject::bcReadAuxData`. The
:cpp:class:`HardwareManager` prefixes the per-device map keys
with the source object's ``hwKey`` before fanning the data out
to consumers, so multiple instances of the new type
distinguish themselves automatically. The full plumbing is on
:doc:`/developer_guide/hardware_runtime` (*Auxiliary, validation,
and rolling data*); the persistence side is on
:doc:`/classes/auxdatastorage`.

HardwareManager fan-out
-----------------------

:cpp:class:`HardwareManager` already handles the generic
``auxData`` / ``validationData`` / ``rollingData`` signals with
``hwKey`` prefixing — no per-type code needed there.

Type-specific signals — per-channel updates, configuration
broadcasts, mode changes — go through the convention already
established for the existing types: a ``HardwareManager`` signal
named ``<typename>Update(QString hwKey, …)`` (or a small set of
related signals) that the manager forwards from the source
:cpp:class:`HardwareObject`'s signal of the same shape. The
forwarding installs in the manager's
``setupHardwareObjectWithTracking`` helper or in the per-type
branch of :cpp:func:`HardwareManager::syncWithRuntimeConfig`,
depending on whether the signal is generic or type-specific.

The pattern to follow is
:cpp:func:`HardwareManager::flowUpdate` ``(hwKey, channel,
value)`` and
:cpp:func:`HardwareManager::pressureControlMode` ``(hwKey,
mode)`` — every signal carries the source hwKey as its first
argument, so consumers (status boxes, control widgets, the Aux
tab) can disambiguate readings from multiple instances of the
new type without enumerating the active profiles in advance. The
existing per-type signal lists on :doc:`/classes/hardwaremanager`
are the ground truth.

Virtual driver
--------------

Every hardware type in Blackchirp ships with a ``Virtual<TypeName>``
driver, and a new type is no exception. The virtual driver
is not a test-only artifact — it is the driver Blackchirp
itself runs whenever the user has not configured a real device, and
it is what makes the new type visible in the running application
from the moment the type lands. Plan and write it alongside the
interface class, not after.

Three responsibilities the virtual driver carries inside
the application:

- **System-profile fall-back.** :cpp:class:`HardwareProfileManager`
  guarantees that every required hardware type carries a profile
  labeled ``virtual`` backed by the type's virtual driver, so
  :cpp:func:`HardwareManager::initialize` always has something to
  instantiate even when no real hardware is configured (see
  :doc:`/developer_guide/hardware_configuration`, *System profiles*).
  Without a virtual driver, a new required type would leave
  Blackchirp in a state where it cannot start; without a virtual
  driver for a new optional type, users have no way to exercise
  the new GUI surfaces, the experiment-setup page, or the
  aux-data plumbing without first acquiring the real hardware.
- **Live exercise of the new GUI surfaces.** The status box,
  control widget, and experiment-setup page added in
  *GUI integration* above need a live :cpp:class:`HardwareObject`
  to talk to during development. Wire the virtual driver up
  first, then iterate on the GUI against synthesized readings
  before chasing vendor protocol details on the bench.
- **End-to-end experiment runs.** A user evaluating Blackchirp,
  writing a Python trampoline, or setting up a new instrument
  configuration can run a full experiment end-to-end against the
  virtual driver — chirp generation, FID acquisition, aux-data
  recording, validation — without owning the real hardware. The
  virtual driver is what keeps that workflow possible
  for the new type.

Conventions for the virtual driver:

- File: ``src/hardware/<core|optional>/<type>/virtual<typename>.{cpp,h}``,
  alongside the interface class. Class name ``Virtual<TypeName>``
  (matching the existing :cpp:class:`VirtualAwg`,
  :cpp:class:`VirtualFlowController`,
  :cpp:class:`VirtualIOBoard`, :cpp:class:`VirtualFtmwScope`
  pattern).
- Inherits from the new interface class. Implements every pure
  virtual the interface declares — the ``hw*`` slots for Pattern
  B types, the ``configure(config&)`` virtual for Pattern A
  types, and either :cpp:func:`HardwareObject::prepareForExperiment`
  or whatever per-driver hook the interface delegates to for
  Pattern C.
- Synthesizes plausible readings rather than returning fixed
  values. Use :cpp:any:`QRandomGenerator` for noise (the existing
  :cpp:class:`VirtualFlowController` is the canonical model:
  flows wander around their setpoints, pressure drifts within
  bounds), and let the synthesized values track any user-driven
  setpoints so control-widget round trips behave the way they
  would against real hardware.
- Registers with :cpp:enumerator:`CommunicationProtocol::Virtual`
  in its ``REGISTER_HARDWARE_PROTOCOLS`` invocation. The
  ``Virtual`` protocol carries no real :cpp:any:`QIODevice`, so
  the driver's :cpp:func:`HardwareObject::testConnection` simply
  returns ``true`` — the synthesis itself is the hardware
  contract.

The virtual driver also doubles as the canonical fixture for
unit tests; the test-side wiring is in *Tests* below.

Optional Python trampoline
--------------------------

A new hardware type should ship with a Python trampoline so
users can write drivers in Python without recompiling. The
trampoline is one C++ class plus one Python template script;
the C++ class is small.

- Subclass both the new interface and
  :cpp:class:`PythonHardwareBase`. Initialize the mixin in the
  initializer list with ``d_key`` and ``d_model``:

  .. code-block:: cpp

     PythonBeamBlocker::PythonBeamBlocker(const QString &label,
                                           QObject *parent) :
         BeamBlocker(QString(PythonBeamBlocker::staticMetaObject.className()),
                     label, parent),
         PythonHardwareBase(d_key, d_model)
     { d_threaded = true; save(); }

- Pick the matching state-management pattern. The trampoline
  uses the same A/B/C taxonomy as the C++ side; see
  :doc:`/developer_guide/python_hardware`
  (*Three state-management patterns*) for the IPC shape per
  pattern.
- Wire the mixin in the type-specific initialize/test hooks.
  For a plain :cpp:class:`HardwareObject` subclass that is
  :cpp:func:`HardwareObject::initialize` and
  :cpp:func:`HardwareObject::testConnection`; for hardware bases
  that final-override those (such as :cpp:class:`Clock` and
  :cpp:class:`FlowController`) it is the typed helper virtual
  the base class calls into (``initializeClock`` /
  ``testClockConnection``, ``fcInitialize`` / ``fcTestConnection``,
  …). Decide which the new type will use at the same time you
  draft the interface class.
- Provide ``python_<typename>_template.py`` next to the trampoline
  source. The host script's generic dispatch picks methods up by
  name; the template defines a class with the canonical name
  ``<TypeName>Driver`` (``BeamBlockerDriver``) that works out of
  the box on the ``Virtual`` protocol.

Tests
-----

A new hardware type warrants three test additions:

- **Round-trip serialization for the optional config object.**
  Add a fixture in ``tst_headerstoragetest`` (or a new
  ``tst_<typename>configtest`` if the round-trip logic is
  non-trivial) that exercises :cpp:func:`HeaderStorage::storeValues`
  and :cpp:func:`HeaderStorage::retrieveValues` for the new
  config class. The existing ``HeaderStorage`` fixtures are the
  template.
- **Wire the virtual driver into** ``blackchirp-test-hardware``.
  The ``Virtual<TypeName>`` driver authored in *Virtual
  driver* above is the canonical test fixture; add its
  ``.cpp`` to the explicit list of test-hardware sources in the
  top-level ``CMakeLists.txt`` (alongside
  ``virtualflowcontroller.cpp``, ``virtualawg.cpp``, and the
  rest) so the test executables link against it. Tests built
  against ``blackchirp-test-hardware`` rely on the virtual driver
  being present for every active hardware type.
- **A registration-pipeline assertion.** Extend
  ``tst_hardwareregistrytest`` with a check that the new type's
  factory is registered, that its protocol set is non-empty, and
  that the inheritance chain from
  :cpp:class:`HardwareObject` is what the type expects. A typo
  in ``REGISTER_HARDWARE_BASE`` or a missing ``Q_OBJECT`` will
  surface here. ``tst_hardwarekeys`` similarly catches collisions
  in the new ``BC::Key::<TypeName>::`` namespace.

Beyond the unit tests, follow the smoke-testing checklist in
:doc:`/developer_guide/adding_a_driver` (*Smoke testing*): build
with ``BC_BUILD_TESTS=ON``, run the existing hardware suite,
launch the application, confirm the new type appears in the
*Add Profile* dialog under its hardware-type entry, and create a
profile against the ``Virtual`` protocol to verify the static
registration is intact.

Documentation follow-up
-----------------------

A new hardware type is a documentation event in three places.
Plan all three at the same time you draft the interface; doing
them in one pass keeps the type's vocabulary consistent across
chapters.

- **API reference.** Add a class page at
  ``doc/source/classes/<typename>.rst`` following
  :ref:`api-reference-style`. The page carries a 1–3
  paragraph orientation intro and a final ``API Reference``
  section with the ``.. doxygenclass::`` directive. Doxygen
  comments in the header are the source of truth for member-
  level prose.
- **User guide.** Add a per-device page under
  ``doc/source/user_guide/hw/<typename>.rst`` once at least one
  concrete driver exists, mirroring the existing pages for AWG,
  flow controllers, pulse generators, and the rest. The
  user-facing pages are how operators discover that the new type
  is available.
- **Developer guide.** If the new type introduces a pattern not
  covered here — a new threading model, a new mid-experiment
  hook, a fourth state-management shape — flag it for a refresh
  of this page and of
  :doc:`/developer_guide/adding_a_driver` so future contributors
  do not have to reverse-engineer the precedent from your code.
