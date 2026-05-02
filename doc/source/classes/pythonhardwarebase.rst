.. index::
   single: PythonHardwareBase
   single: Python hardware; trampoline base
   single: Trampoline; Python

PythonHardwareBase
==================

``PythonHardwareBase`` turns a :cpp:class:`HardwareObject` subclass
into a Python-backed *trampoline*: a C++ shim whose virtuals forward
to a child Python interpreter over JSON IPC. A concrete trampoline
inherits *both* its hardware base class — ``AWG``, ``Clock``,
``FtmwDigitizer``, and so on — *and* this mixin via multiple
inheritance. The hardware base supplies the Qt slot/signal API the
rest of Blackchirp uses; the mixin owns the subprocess and the IPC
plumbing.

A subclass routes its hardware-base lifecycle hooks through the
mixin: ``initialize`` calls ``initPythonProcess()`` to bind the comm
pointer and the settings get/set callbacks; ``testConnection`` calls
``testPythonConnection()``, which lazily starts the subprocess on
the first invocation and dispatches the script's
``test_connection`` method on every call; ``sleep`` and
``readSettings`` delegate to ``pythonSleep()`` and
``pythonReadSettings()``. Hardware-specific virtuals translate to
JSON dispatches through ``pu_process->sendRequest()``. Push-style
hardware additionally calls ``pu_process->setEnabledProxies()``
and connects ``waveformReceived`` to a shot handler.

The script path, class name, and Python environment directory are
read on demand from :cpp:class:`HardwareProfileManager`.
``startPythonProcess()`` refuses to start the subprocess if either
the script path or the class name is empty rather than substituting
a default. ``resolvePythonExecutable()`` probes the configured
environment for the standard venv and conda layouts and falls back
to the literal ``"python3"`` (resolved through ``PATH``) when no
environment is set; ``findHostScript()`` locates the IPC host script
``python_hw_host.py`` in the application directory or the
``share/blackchirp/`` install location. The profile workflow and
the user-side script API are documented in
:doc:`/user_guide/python_hardware/selecting` and
:doc:`/user_guide/python_hardware/writing_a_driver`.

.. TODO: cross-link to the contributor-level walk-through at
   /developer_guide/python_hardware once that page exists.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: PythonHardwareBase
   :members:
   :protected-members:
   :undoc-members:
