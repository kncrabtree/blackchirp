.. index::
   single: RuntimeHardwareConfig
   single: hardware; runtime configuration
   single: HardwareValidationResult

RuntimeHardwareConfig
=====================

``RuntimeHardwareConfig`` is the singleton that records which hardware
drivers are *active* at any given moment. It maps each
``"<Type>.<label>"`` key to a driver key (e.g.,
``"FtmwDigitizer.default"`` → ``"m4i2220x8"``), validates those selections
against the :cpp:class:`HardwareRegistry`, and exposes the configuration to
the rest of Blackchirp for experiment setup.

Read access is unrestricted and available through ``constInstance()``. Write
access is restricted to ``HardwareManager`` and ``RuntimeHardwareConfigDialog``
via the friend-class pattern, enforcing that only the hardware management layer
can change the active configuration.

The class provides both string-based and template-based query methods. The
template variants use Qt's ``staticMetaObject`` to derive the hardware-type
key at compile time, so callers avoid raw string keys:

.. code-block:: cpp

   auto labels = RuntimeHardwareConfig::constInstance().getActiveLabels<FtmwDigitizer>();
   QString impl = RuntimeHardwareConfig::constInstance()
                      .getHardwareImplementation<FtmwDigitizer>("default");

Validation
----------

``validateConfiguration()`` checks every active selection against the
``HardwareRegistry`` and returns a map of per-type
:cpp:struct:`HardwareValidationResult` structs. The static overload
``validateHardwareConfiguration(map)`` validates an arbitrary hardware map
without requiring the singleton, useful for preview operations in the
configuration dialog. Neither method performs automatic fallbacks; callers
must handle errors explicitly.

Relationship to HardwareProfileManager
---------------------------------------

``RuntimeHardwareConfig`` is the active-selection layer on top of
:cpp:class:`HardwareProfileManager`, which owns profile metadata and
persistence. When profiles are activated or deactivated, the runtime
configuration is refreshed to match the current active profiles. The loadout system
described in :doc:`/user_guide/hardware_config` works at the
``RuntimeHardwareConfig`` layer: switching a loadout replaces the active
hardware map.

``createHardwareDataContainer()`` packages the active configuration into a
``BC::Data::HardwareDataContainer`` for use by data-layer classes such as
``Experiment``.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: RuntimeHardwareConfig
   :members:
   :undoc-members:

.. doxygenstruct:: HardwareValidationResult
   :members:
   :undoc-members:
