.. index::
   single: PythonProcess
   single: Python hardware; PythonProcess
   single: JSON IPC
   single: Python subprocess
   single: Proxy injection; Python
   single: Waveform push

PythonProcess
=============

``PythonProcess`` owns the child Python interpreter and the
JSON-lines IPC channel to the user driver script behind every
Python hardware trampoline. The Python heap stays in a separate
process, so a script crash cannot take down the Qt application.

There are two directions of traffic. **C++ → Python** uses the
synchronous ``sendRequest()`` API, which writes a JSON method call
on the subprocess's stdin and runs a nested ``QEventLoop`` until
the matching response arrives. **Python → C++** uses three
channels: relay requests that service ``self.comm`` and
``self.settings`` against the trampoline's
:cpp:class:`CommunicationProtocol` and
:cpp:class:`SettingsStorage`; log lines forwarded to the global
Blackchirp log via ``bcLog()``; and base64-encoded waveform pushes
emitted on the ``waveformReceived`` signal for digitizer
trampolines to drain into the WaveformBuffer.

The standard ``self.comm``, ``self.settings``, and ``self.log``
proxies are injected on every subprocess start. Optional,
hardware-type-specific proxies — currently ``self.digi`` for
digitizer push — are gated by ``setEnabledProxies()``: trampolines
that need them call it between ``initPythonProcess()`` and the
first ``sendRequest()``.

For the trampoline contract and the user-side script API, see
:doc:`/user_guide/python_hardware/overview` and its sub-pages; the
contributor-level architecture of the IPC host and the proxy system
is covered in :doc:`/developer_guide/python_hardware`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: PythonProcess
   :members:
   :protected-members:
   :undoc-members:
