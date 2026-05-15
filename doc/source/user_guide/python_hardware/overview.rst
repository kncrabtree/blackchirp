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

A Python driver is an ordinary ``.py`` file containing a class whose
snake-case methods match the C++ hardware interface (``initialize``,
``test_connection``, ``prepare_for_experiment``, and so on). The
driver class is selected per profile and may be mixed with native
C++ profiles in the same loadout. Any hardware type with a
corresponding Python trampoline can be backed by a user script.

Architecture
------------

Each Python driver runs in its own subprocess, launched and managed
through Qt's ``QProcess``. The C++ side communicates with the
subprocess over standard input and output using a JSON-lines protocol:
one JSON object per line for requests, responses, log messages, and
relayed communication. ``python3`` on ``PATH`` is enough for the
default case; a per-profile environment field supports venv and conda
layouts (see :doc:`selecting`).

Within the subprocess, an injected ``self.comm`` proxy relays
communication-protocol calls back to the C++ side, so a Python driver
uses the same RS-232, TCP, GPIB, or Virtual protocol the rest of the
application uses. The **Custom** protocol is also exposed; see
:ref:`Custom protocol <python-hardware-custom-protocol>` for the
convention. Settings reads and writes go through ``self.settings``,
and log messages routed through ``self.log`` appear in the hardware
log panel alongside messages from C++ drivers.

.. _python-hardware-security:

Security
--------

.. warning::

   **Python hardware scripts run with full system access.** Scripts
   can access files, network resources, and hardware devices with the
   same permissions as Blackchirp. Only use scripts from sources you
   trust.

The same warning appears in the **Add Profile** dialog when a
Python-backed profile is created. Python scripts are not sandboxed;
treat a third-party Python driver the same way you would treat any
other executable downloaded from the internet.

Supported hardware types
------------------------

Each Python-capable hardware type ships with a trampoline class on
the C++ side and a starter template script named
``python_<type>_template.py``. The full list of trampolines, their
default class names, and the methods each one expects is in
:ref:`python-hardware-trampoline-overview`.
