HardwareObject
==============

``HardwareObject`` is the abstract base class for every device Blackchirp
talks to: digitizers, AWGs, clocks, pulse generators, flow controllers,
pressure controllers, temperature sensors, and so on. A subclass binds a
device to a :cpp:class:`CommunicationProtocol`, registers any persistent
settings via the hardware-settings registry, and implements the small
set of pure virtuals that drive connection, experiment preparation, and
auxiliary-data sampling. Each instance is identified by a hardware-type
string and a user-supplied label, combined to form the settings key
``"<hwType>.<label>"`` (for example ``"PulseGenerator.Default"``).

Subclasses typically come in two layers: an *interface* class (e.g.
``AWG``, ``FtmwDigitizer``) that declares the slot/signal API the rest
of Blackchirp uses, and one or more *implementation* classes that
inherit the interface and translate it into vendor-specific commands.
The ``d_critical`` flag in the constructor decides whether a
communication failure aborts the active experiment, and ``d_threaded``
selects between in-thread and dedicated-thread execution. Hardware
profiles, loadouts, and the registry that supplies setting defaults
are described in :doc:`/user_guide/hardware_config`; the runtime
``Hardware`` menu and connection workflow are covered in
:doc:`/user_guide/hardware_menu` and :doc:`/user_guide/hwdialog`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: HardwareObject
   :members:
   :protected-members:
   :undoc-members:
