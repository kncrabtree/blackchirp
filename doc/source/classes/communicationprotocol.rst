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
The convenience helpers ``writeCmd()``, ``writeBinary()``, and
``queryCmd()`` cover the common ASCII / binary / query patterns and
may be extended or overridden by subclasses; the read termination and
timeout used by ``queryCmd()`` are configured through
``setReadOptions()``. The user-facing controls for selecting and
configuring a transport per device live in
:doc:`/user_guide/hardware_config` and the connection dialog described
in :doc:`/user_guide/hwdialog`.

.. highlight:: cpp

Reaching the underlying device
------------------------------

``_device()`` returns the raw ``QIODevice*`` (or ``nullptr`` when no
``QIODevice`` representation exists). When a driver needs transport
functionality that the wrapper does not expose, the ``device<T>()``
template casts to the requested derived type and returns ``nullptr``
if the cast is not appropriate, so the wrong-type case is safe to
detect at the call site:

.. code-block:: cpp

   CommunicationProtocol *comm = new TcpInstrument("key");
   comm->initialize();

   auto socket = comm->device<QTcpSocket>();
   // socket is a QTcpSocket*

   auto serial = comm->device<QSerialPort>();
   // serial is nullptr — comm is a TcpInstrument

API Reference
-------------

.. doxygenclass:: CommunicationProtocol
   :members:
   :protected-members:
   :undoc-members:
