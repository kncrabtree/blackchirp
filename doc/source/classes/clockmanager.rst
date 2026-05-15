.. index::
   single: ClockManager
   single: clocks; RF clock routing
   single: clocks; frequency configuration
   single: RfConfig; ClockType routing
   single: hardware; clock subsystem

ClockManager
============

``ClockManager`` owns the live :cpp:class:`Clock` instances for the active hardware
loadout, maps each :cpp:enum:`RfConfig::ClockType` role to the physical clock object
that serves it, and mediates all configure and read operations that flow between
:doc:`hardwaremanager` and the underlying oscillator hardware.  It inherits from
``SettingsStorage`` and persists the available clock output table so that the RF
configuration UI can present the correct options without querying live hardware.

``ClockManager`` is created by :doc:`hardwaremanager` and runs on the
``HardwareManager`` thread.  ``HardwareManager`` holds it via
``std::unique_ptr<ClockManager> pu_clockManager`` and calls
``setClocksFromHardwareManager()`` to hand off the current set of :cpp:class:`Clock`
pointers after any hardware sync cycle.  When the runtime hardware map changes,
``HardwareManager`` calls ``updateClockManager()`` to rebuild the internal clock list
and re-emit routing signals.  Direct calls from other threads must go through
``HardwareManager``'s queued connections — ``ClockManager``'s public slots are not
safe to invoke directly across thread boundaries.

Primary collaborators: :doc:`hardwaremanager` (owner, signal forwarder, and caller
of ``setClocks`` / ``configureClocks``); :doc:`rfconfig` (source of the
``RfConfig::ClockType`` enum used as the routing key and of the ``RfConfig::ClockFreq``
descriptor passed in clock maps); :doc:`hardwareloadout` (clock assignments are
part of the loadout's RF section and are persisted per FTMW preset).  The
:cpp:class:`Clock` base class is a :doc:`hardwareobject`; ``ClockManager`` does not
own the ``Clock`` objects — it borrows raw pointers from ``HardwareManager``.

Clock-routing model
-------------------

Each :cpp:enum:`RfConfig::ClockType` role (``UpLO``, ``DownLO``, ``AwgRef``,
``DRClock``, ``DigRef``, ``ComRef``) is satisfied by exactly one
:cpp:class:`Clock` object paired with an output index on that clock.  A single
:cpp:class:`Clock` device may expose multiple independent outputs (for example the
Valon 5009 has two independently tunable channels), so one physical device can
simultaneously satisfy several distinct roles — each on a different output port.

Internally, ``d_clockRoles`` maps every currently active ``ClockType`` to the
:cpp:class:`Clock` that serves it.  ``d_clockList`` holds the full set of
:cpp:class:`Clock` objects that ``HardwareManager`` has provided; entries in
``d_clockRoles`` are always a subset of that list.  When ``configureClocks()`` is
called, the existing role map is cleared, each entry in the incoming
``QHash<ClockType, ClockFreq>`` is resolved against ``d_clockList`` by hardware
key, the matching clock's output is registered via ``Clock::addRole()``, and the
multiplication factor is written to the output via ``Clock::setMultFactor()``.
Roles that were present before the call but absent from the new map receive a
``clockHardwareUpdate(type, "", -1)`` signal so consumers know the role is no
longer active.

The multiplication factor in ``RfConfig::ClockFreq`` accounts for clocks whose
oscillator output frequency differs from the logical RF frequency by a fixed
multiply-or-divide relationship (e.g., a ×2 doubler stage).  When the factor is
less than 1.0 the operation is ``RfConfig::Divide``; otherwise it is
``RfConfig::Multiply``.

The available clock outputs are persisted in ``SettingsStorage`` under the
``BC::Key::ClockManager::hwClocks`` array so that the RF configuration dialog can
populate its output selector without polling live hardware.

Configuration flow
------------------

The typical configure path during experiment preparation is:

1. ``HardwareManager::initializeExperiment()`` calls
   ``ClockManager::prepareForExperiment(Experiment &exp)``.
2. ``prepareForExperiment()`` extracts the clock map from
   ``exp.ftmwConfig()->d_rfConfig.getClocks()`` and passes it to
   ``configureClocks()``.
3. ``configureClocks()`` resolves each ``ClockType`` → ``Clock*`` binding,
   sets the multiplication factor, calls ``Clock::setFrequency()`` on each
   affected output, and emits ``clockHardwareUpdate()`` for every newly assigned
   role.
4. After ``configureClocks()`` returns, ``prepareForExperiment()`` writes the
   achieved frequencies back into ``exp`` via
   ``RfConfig::setCurrentClocks(getCurrentClocks())``.
5. When a mid-acquisition clock step requires a frequency change,
   ``MainWindow::clockPrompt`` calls ``HardwareManager::setClocks()`` via
   ``QMetaObject::invokeMethod``; ``HardwareManager`` gates the FTMW digitizer
   during the transition and emits :cpp:func:`HardwareManager::allClocksReady`
   when all clocks have settled.  That signal is connected directly to
   ``AcquisitionManager::clockSettingsComplete``, which gates progression to
   the next acquisition step.

Frequency reads (``readClockFrequency()``, ``readActiveClocks()``) forward
directly to ``Clock::readFrequency()``.  Each :cpp:class:`Clock` emits
``frequencyUpdate(ClockType, double)``; ``setupClocks()`` connects that signal
to ``ClockManager::clockFrequencyUpdate``, which ``HardwareManager`` then
re-emits to the rest of the system.

For the end-user view of clock assignment see :doc:`/user_guide/ftmw_configuration/rf_configuration`
and :doc:`/user_guide/ftmw_configuration/presets`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: ClockManager
   :members:
   :protected-members:
   :undoc-members:
