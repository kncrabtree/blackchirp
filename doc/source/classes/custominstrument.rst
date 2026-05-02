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
``CommunicationProtocol::Custom`` declares its parameters by writing
a ``BC::Key::Custom::comm`` array into its :cpp:class:`SettingsStorage`
on first construction. Each entry in that array describes one input
field (label, settings key, value type, optional bounds), and the
hardware connection dialog described in :doc:`/user_guide/hwdialog`
renders the appropriate widget for each. The implementation reads the
user-supplied values back from ``SettingsStorage`` inside its own
``testConnection()`` override, where the actual handshake with the
device occurs. Python-backed hardware (see
:doc:`/user_guide/python_hardware`) uses this mechanism for any
parameter the Python driver requires.

.. highlight:: cpp

.. doxygenclass:: CustomInstrument
   :members:
   :undoc-members:
