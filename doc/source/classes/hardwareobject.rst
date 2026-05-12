HardwareObject
==============

``HardwareObject`` is the abstract base class for every device
Blackchirp talks to: digitizers, AWGs, clocks, pulse generators, flow
controllers, pressure controllers, temperature sensors, and so on. It
provides the common identity, communication, settings, and lifecycle
machinery that the rest of the program relies on; concrete drivers add
the per-vendor command translation and any device-specific behavior.

Hardware profiles, loadouts, and the registry that supplies setting
defaults are described in :doc:`/user_guide/hardware_config`; the
runtime ``Hardware`` menu and connection workflow are covered in
:doc:`/user_guide/hardware_menu` and :doc:`/user_guide/hwdialog`. The
:cpp:any:`REGISTER_HARDWARE_META`, :cpp:any:`REGISTER_HARDWARE_PROTOCOLS`,
and :cpp:any:`REGISTER_HARDWARE_SETTINGS` macros that every driver uses
to register itself with the framework are documented on
:doc:`/classes/hardwareregistry`.

.. highlight:: cpp

Identity
--------

Each instance has a *hardware type* (e.g. ``"AWG"``, ``"FtmwDigitizer"``,
``"PulseGenerator"``) naming the abstract role and a *label* — a
free-form, user-supplied string distinguishing multiple instances of
the same type (e.g. ``"main"``, ``"secondary"``). The hardware type
matches the interface class's metaobject name by convention.

Type and label combine to form ``d_key`` (e.g.
``"PulseGenerator.main"``), which uniquely identifies the instance.
``d_key`` is also the :cpp:class:`SettingsStorage` group that holds
this instance's persistent settings, which is why the same key cannot
be reused for two different drivers: an instance's driver is fixed at
profile creation, and changing it requires deleting the profile and
rebuilding it under the same (or a new) label.

``d_model`` is a separate string carrying just the driver class name
(e.g. ``"AWG70002a"``, ``"PythonAwg"``); it is recorded inside the
settings group and used for display purposes — for example, to show
the user which driver backs a profile in the hardware settings dialog.
``d_model`` is not part of ``d_key`` and is not used as a dispatch
identifier.

Constructing a driver
---------------------

Two-layer inheritance is the convention: an *interface* class
(:cpp:class:`AWG`, :cpp:class:`FtmwDigitizer`,
:cpp:class:`PulseGenerator`, …) declares the slot and signal API the
rest of Blackchirp consumes, and one or more *driver* classes
(``AWG70002a``, ``VirtualAwg``, …) translate that API into vendor
commands. A typical pair:

.. code-block:: cpp

   class Analyzer : public HardwareObject
   {
       Q_OBJECT
   public:
       Analyzer(const QString& impl, const QString& label, QObject *parent = nullptr)
           : HardwareObject(QString(Analyzer::staticMetaObject.className()),
                            impl, label, parent) {}

   public slots:
       bool setFoo(double f) {
           auto ok = hwSetFoo(f);
           if(!ok) emit hardwareFailure();
           else    readFoo();
           return ok;
       }
       double readFoo() {
           d_foo = hwReadFoo();
           emit fooUpdated(d_foo, QPrivateSignal());
           return d_foo;
       }

   signals:
       void fooUpdated(double, QPrivateSignal);

   private:
       double d_foo;
       virtual bool   hwSetFoo(double f) = 0;
       virtual double hwReadFoo()        = 0;
   };

   class VendorAnalyzer : public Analyzer
   {
   public:
       VendorAnalyzer(const QString& label, QObject *parent = nullptr)
           : Analyzer(QString(VendorAnalyzer::staticMetaObject.className()),
                      label, parent) {}
   private:
       bool   hwSetFoo(double f) override;
       double hwReadFoo()        override;
   };

Each driver is also registered at static initialization with
:cpp:any:`REGISTER_HARDWARE_META`,
:cpp:any:`REGISTER_HARDWARE_PROTOCOLS`, and (where the driver
introduces settings) :cpp:any:`REGISTER_HARDWARE_SETTINGS`; see
``hardwareregistration.h`` and :doc:`/classes/hardwareregistry`.

Configuration flags
-------------------

Drivers may set the following flags in their constructors:

- ``d_threaded`` — if ``true``, the :cpp:class:`HardwareManager` moves
  the object to a dedicated ``QThread``. Threaded objects must not
  have a ``QObject`` parent at construction; this is also why their
  child ``QObject`` members are constructed in ``initialize()`` rather
  than in the constructor itself.
- ``d_critical`` — if ``true`` (the default), a ``hardwareFailure()``
  emission aborts any active experiment and blocks new ones until
  ``bcTestConnection()`` succeeds again. If ``false``, the object is
  simply marked disconnected and the experiment continues. The flag
  is also user-editable through the hardware settings dialog.
- ``d_commType`` — the active :cpp:class:`CommunicationProtocol`
  type. The set of protocols a driver supports is declared via
  :cpp:any:`REGISTER_HARDWARE_PROTOCOLS`; the user picks one at
  profile creation time and the persisted value is loaded by the base
  constructor.

Drivers whose ``CommunicationProtocol`` type is
``CommunicationProtocol::Custom`` additionally declare any user-supplied
connection parameters (device path, serial number, etc.) following the
convention documented in :cpp:class:`CustomInstrument`.

Lifecycle
---------

``HardwareObject`` instances are not bound to program lifetime. The
:cpp:class:`HardwareManager` creates and destroys them in response to
loadout activation, profile edits, and protocol changes. After
construction the manager calls ``bcInitInstrument()``, which builds the
:cpp:class:`CommunicationProtocol` object, calls ``initialize()``, and
wires up ``hardwareFailure`` routing. Drivers must implement
``initialize()``: it is the place to construct child ``QObject`` s
(the constructor cannot, for threaded drivers) and perform any
one-shot setup. Per-connection work belongs in ``testConnection()``
instead, since connection attempts happen many times over the lifetime
of an instance.

Connection testing
------------------

Drivers must implement ``testConnection()``: it should attempt some
cheap interaction with the device (typically an ``*IDN?`` query) to
confirm both that something is responding and that it is the expected
hardware. On failure, store a descriptive message in ``d_errorString``
and return ``false``. ``testConnection()`` is called from the
``bcTestConnection()`` wrapper, which first reloads settings from
:cpp:class:`SettingsStorage`; the override therefore sees up-to-date
settings and may read them freely.

Optional virtual hooks
----------------------

Drivers may override these hooks to participate in the experiment
lifecycle and the auxiliary-data system:

- ``validationKeys()`` — keys whose values appear on the experiment
  setup dialog's Validation page.
- ``sleep()`` — switch the device to/from a low-power state.
- ``beginAcquisition()`` / ``endAcquisition()`` — start- and
  end-of-experiment hooks.
- ``prepareForExperiment()`` — validate and stage per-experiment
  settings; called via the ``hwPrepareForExperiment()`` wrapper.
- ``readAuxData()`` — values to plot on the Aux/Rolling tabs and to
  persist as auxiliary data.
- ``readValidationData()`` — values to verify post-acquisition,
  separate from the plotted aux data.
- ``readSettings()`` — refresh cached state when the user accepts the
  hardware settings dialog.

Each is documented at the per-method level below.

Threading
---------

Functions intended to be invoked from outside the object — for example
by :cpp:class:`HardwareManager` during an acquisition — must be
declared as Qt slots so they reach the object via queued connection
when the driver is threaded.

API Reference
-------------

.. doxygenclass:: HardwareObject
   :members:
   :protected-members:
   :undoc-members:
