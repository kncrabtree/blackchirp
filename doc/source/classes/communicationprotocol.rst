CommunicationProtocol
=====================

``CommunicationProtocol`` is the thin wrapper that sits between a
:cpp:class:`HardwareObject` and the operating system's I/O facilities.
It exposes a uniform ``writeCmd`` / ``writeBinary`` / ``queryCmd`` /
``readBytes`` API on top of an underlying ``QIODevice`` (or, for
buses that have no ``QIODevice`` representation, no device at all).
Concrete subclasses provide the actual transport: ``Rs232Instrument``
wraps a ``QSerialPort``, ``TcpInstrument`` wraps a ``QTcpSocket``,
``GpibInstrument`` proxies through the GPIB controller, and
``VirtualInstrument`` and :cpp:class:`CustomInstrument` keep the
device pointer null when no socket-style interface is appropriate.

Read behavior — timeout and termination characters — is shared across
transports and is loaded from settings via ``loadCommReadOptions()``.
The ``device<T>()`` template is the recommended way to reach the
underlying ``QIODevice`` when a hardware driver needs functionality
that the wrapper does not expose. The user-facing controls for
selecting and configuring a transport per device live in
:doc:`/user_guide/hardware_config` and the connection dialog described
in :doc:`/user_guide/hwdialog`.

.. highlight:: cpp

.. doxygenclass:: CommunicationProtocol
   :members:
   :protected-members:
   :undoc-members:
