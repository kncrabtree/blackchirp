CustomInstrument
================

``CustomInstrument`` is the :cpp:class:`CommunicationProtocol`
subclass for hardware that does not communicate over RS-232, TCP, or
GPIB. It carries no ``QIODevice``: ``initialize()`` and
``testConnection()`` are no-ops, and ``_device()`` returns ``nullptr``.
What makes the class useful is the convention it establishes for
collecting the connection parameters that *are* needed for such
hardware — file paths, device handles, serial numbers, USB IDs — from
the user.

A :cpp:class:`HardwareObject` whose communication type is
``CommunicationProtocol::Custom`` declares its connection parameters
at static initialization time using the ``REGISTER_CUSTOM_COMM`` macro
from ``hardwareregistration.h``. Each ``CustomCommDef`` descriptor
specifies the settings key, user-facing label, description, type
(``String``, ``Int``, or ``FilePath``), and optional bounds. The
HardwareRegistry makes these descriptors available to the UI before
any hardware object is constructed, so both the AddProfileDialog
(new profiles) and the CommunicationDialog (existing profiles) can
render the appropriate input widgets without instantiating the
driver. The driver reads user-supplied values back from the
``BC::Key::Comm::custom`` group of its SettingsStorage inside
``testConnection()``.

See also :doc:`/classes/settingsstorage` for the ``REGISTER_HARDWARE_SETTINGS``
macro family, which follows the same registration pattern for
hardware configuration parameters.

.. highlight:: cpp

.. doxygenclass:: CustomInstrument
   :members:
   :undoc-members:
