.. index::
   single: Python hardware
   single: Python driver
   single: JSON IPC
   single: Subprocess
   single: Hardware driver; Python
   single: Trampoline; Python

.. _python-hardware-overview:

Overview
========

Blackchirp lets you write hardware drivers in Python without modifying
or recompiling the application. A Python driver is an ordinary ``.py``
file containing a class with snake-case methods that match the C++
hardware interface (``initialize``, ``test_connection``,
``prepare_for_experiment``, and so on). At runtime, Blackchirp launches
the script in a separate process and forwards calls to it. Any
hardware type that has a corresponding Python trampoline can be backed
by a user script.

Why Python hardware
-------------------

Writing a driver in Python is appropriate when:

- A new instrument needs to be supported quickly, without going through
  the C++ build cycle.
- The vendor ships a Python SDK or a ``pyvisa`` wrapper that already
  handles the wire protocol.
- A driver must be tuned, patched, or swapped in the field by someone
  who is not a C++ developer.
- A user-contributed implementation can be shared as a single ``.py``
  file rather than a forked source tree.

Python drivers coexist with built-in C++ drivers. The choice of
implementation is a per-profile setting (see
:doc:`selecting`); a single loadout may mix Python-backed and
native-C++ profiles freely.

Architecture
------------

Each Python driver runs in its own subprocess, launched and managed
by Blackchirp through Qt's ``QProcess``. The C++ side communicates
with the subprocess over standard input and output using a
JSON-lines protocol: one JSON object per line for requests,
responses, log messages, and relayed communication. This design
keeps the Python heap completely isolated from the Qt application
(a script crash cannot corrupt Blackchirp), avoids any compile-time
dependency on a particular Python version, and removes the need to
ship pybind11 or its toolchain on the user's machine. Python is
required only at runtime; ``python3`` on ``PATH`` is enough for the
default case, and a per-profile environment field supports venv and
conda layouts (see :doc:`selecting`).

Within the subprocess, an injected ``self.comm`` proxy relays
communication-protocol calls back to Blackchirp's C++ side, so a
Python driver uses the same RS-232, TCP, GPIB, or Virtual protocol
that the rest of the application uses. The **Custom** protocol is
also exposed: selecting it tells Blackchirp that the driver does not
use the C++ ``self.comm`` transport at all (typical when the script
talks to its hardware through a vendor-supplied Python package or
USB-HID library), and that any connection parameters live inside the
``.py`` script itself. See :ref:`python-hardware-custom-protocol` for
the convention. Settings reads and writes go through
``self.settings``, and log messages routed through ``self.log``
appear in the hardware log panel alongside messages from C++ drivers.

.. _python-hardware-security:

Security
--------

.. warning::

   **Python hardware scripts run with full system access.** Scripts
   can access files, network resources, and hardware devices with the
   same permissions as Blackchirp. Only use scripts from sources you
   trust.

This warning is also shown by the **Add Profile** dialog when you
create a Python-backed profile. Blackchirp does not sandbox Python
scripts; they are loaded and executed in a normal interpreter
process. Treat a third-party Python driver the same way you would
treat any other executable downloaded from the internet.

Supported hardware types
------------------------

Each hardware type that supports Python implementations ships with a
trampoline class on the C++ side and a starter template script.
The default class name is the name the driver class uses inside the
template; the **Add Profile** dialog scans the chosen script for
class definitions and offers them in a dropdown, so any class name
can be used in your own scripts.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Trampoline
     - Default class name
     - Template file
   * - ``PythonAwg``
     - ``AwgDriver``
     - ``python_awg_template.py``
   * - ``PythonClock``
     - ``ClockDriver``
     - ``python_clock_template.py``
   * - ``PythonFlowController``
     - ``FlowControllerDriver``
     - ``python_flowcontroller_template.py``
   * - ``PythonFtmwScope``
     - ``FtmwScopeDriver``
     - ``python_ftmwscope_template.py``
   * - ``PythonGpibController``
     - ``GpibControllerDriver``
     - ``python_gpibcontroller_template.py``
   * - ``PythonIOBoard``
     - ``IOBoardDriver``
     - ``python_ioboard_template.py``
   * - ``PythonLifLaser``
     - ``LifLaserDriver``
     - ``python_liflaser_template.py``
   * - ``PythonLifScope``
     - ``LifScopeDriver``
     - ``python_lifscope_template.py``
   * - ``PythonPressureController``
     - ``PressureControllerDriver``
     - ``python_pressurecontroller_template.py``
   * - ``PythonPulseGenerator``
     - ``PulseGeneratorDriver``
     - ``python_pulsegenerator_template.py``
   * - ``PythonTemperatureController``
     - ``TemperatureControllerDriver``
     - ``python_temperaturecontroller_template.py``

What each trampoline expects from a Python class — which methods are
required, which are optional, and what data flows through
``prepare_for_experiment`` — is summarized in
:doc:`per_type_capabilities`. The mechanics of writing a class
against the template are covered in :doc:`writing_a_driver`. The
profile-creation flow that ties a Python script to a hardware
profile is described in :doc:`selecting`, and the in-application
controls for editing and reloading a script are covered in
:doc:`hot_reload`.
