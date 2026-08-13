.. index::
   single: Python hardware; writing a driver
   single: Python hardware; lifecycle methods
   single: Python hardware; injected proxies
   single: Python hardware; self.comm
   single: Python hardware; self.settings
   single: Python hardware; self.log
   single: Python hardware; self.digi
   single: Driver class name
   single: Driver template

.. _python-hardware-writing-a-driver:

Writing a Python Driver
=======================

A Python hardware driver is a single ``.py`` file containing one
class. Blackchirp instantiates the class, injects a small set of
proxy objects onto the instance, and dispatches lifecycle and
hardware-specific calls to its methods. This page covers the rules
every driver follows; the per-type method list lives on
:doc:`per_type_capabilities`.

.. _python-hardware-driver-class:

Driver Class
------------

Each Python hardware type expects a driver class with a specific
default name (``AwgDriver``, ``IOBoardDriver``, and so on). Keeping
the default name means the **Python Class** dropdown in the Hardware
Configuration dialog selects the correct entry without further input.
If the class is renamed — for example, to host two drivers in one
file — pick the desired one from the dropdown.

The class must have a no-argument constructor. Initialization that
depends on the hardware connection or persistent settings belongs in
:meth:`initialize`, not in ``__init__``: ``self.comm``,
``self.settings``, and ``self.log`` are not available until after
construction.

.. _python-hardware-injected-proxies:

Injected Proxies
----------------

Three proxy objects are attached to every driver instance before
:meth:`initialize` runs. They are the driver's interface back to
Blackchirp: hardware I/O, persistent settings, and the log panel
are all relayed across the IPC pipe.

.. _python-hardware-comm-proxy:

``self.comm`` — communication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Routes through the C++ communication protocol that the user selected
for this profile (RS-232, TCP, GPIB, or Virtual). The driver does not
need to know which protocol is in use.

.. code-block:: python

   response: str = self.comm.query(cmd: str)
   ok: bool      = self.comm.write(cmd: str)
   data: bytes   = self.comm.read_bytes(n: int)
   ok: bool      = self.comm.write_binary(data: bytes)

``query`` sends a command and returns the response as a string.
``write`` sends a command without expecting a response. ``read_bytes``
returns ``n`` raw bytes from the device. ``write_binary`` sends a raw
byte sequence (for AWG waveform uploads, digitizer block transfers,
etc.). All four raise ``ConnectionError`` if the underlying transport
fails.

.. _python-hardware-custom-protocol:

The **Custom** protocol option signals that the driver bypasses
``self.comm`` entirely — typical when the script talks to its
hardware through a vendor Python package, a USB-HID library, or a
memory-mapped device. Connection parameters in this case live
inside the ``.py`` script (as constants near the top of the file
or as constructor arguments); the Communication Settings panel
shows a note to that effect instead of input fields.

.. _python-hardware-settings-proxy:

``self.settings`` — persistent settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reads and writes settings from the profile's persistent storage, the
same mechanism the C++ side uses. Useful for configuration values that
the user can adjust through the Hardware Settings dialog without
restarting Blackchirp.

.. code-block:: python

   value = self.settings.get(key: str, default=None)
   self.settings.set(key: str, value)
   self.settings.key    # read-only: full hardware key, e.g. "PythonAwg.Default"
   self.settings.model  # read-only: model name string

Values returned by :meth:`get` are JSON-typed: integers, floats,
booleans, strings, lists, and dicts pass through cleanly. Cast to the
expected Python type if the stored value may have come from a Qt
``QVariant`` whose runtime type does not match.

The native key string that :meth:`get` and :meth:`set` expect for any
registered setting is shown in the tooltip of that setting's row on
the :ref:`Settings tab <hwdialog-settings>` of the Hardware Dialog.
Hover the label or value cell to read the key off the bottom of the
tooltip.

.. _python-hardware-log-proxy:

``self.log`` — logging
~~~~~~~~~~~~~~~~~~~~~~

Sends messages to the Blackchirp log panel and the rolling log file.
Each method maps to a log level recognized by the GUI:

.. code-block:: python

   self.log.log(msg)         # Normal
   self.log.debug(msg)       # Debug (visible only when debug logging is on)
   self.log.warning(msg)     # Warning (yellow)
   self.log.error(msg)       # Error (red)
   self.log.highlight(msg)   # Highlight (bold)

Log messages are buffered and delivered out-of-band, so they do not
block other IPC traffic. They are safe to call from any thread,
including a digitizer acquisition thread.

.. _python-hardware-digi-proxy:

``self.digi`` — push-style waveforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optional ``self.digi`` proxy is injected only for digitizer
trampolines (FTMW Digitizer and LIF Digitizer). It exposes one method:

.. code-block:: python

   self.digi.emit_shot(raw_bytes: bytes, shots: int = 1)

``emit_shot`` pushes a single record (or pre-accumulated block) of
waveform bytes to the C++ side, which forwards them to the appropriate
acquisition manager. The byte layout must match the digitizer
configuration the driver applied in :meth:`configure`: record length,
bytes per point, byte order, and (for multi-record acquisitions)
number of records. Pass ``shots=N`` when the bytes already represent
``N`` averaged shots; this lets the C++ shot counter advance correctly.

The push model lets the driver run its own acquisition loop on a
background thread — typically a daemon thread started in
:meth:`begin_acquisition` and stopped in :meth:`end_acquisition`. The
main thread must remain free to process incoming IPC messages such as
``end_acquisition``, ``sleep``, and ``read_settings``. ``emit_shot``
itself is thread-safe.

.. _python-hardware-lifecycle-methods:

Lifecycle Methods
-----------------

The methods below are invoked on every driver, regardless of hardware
type. Each is called at a specific point in the experiment lifecycle.
All methods other than :meth:`initialize` are optional; missing methods
return a safe default value, listed in :ref:`python-hardware-defaults`
below.

``initialize(self)``
   Called once, immediately after the proxy objects are injected. Use
   it to set up internal state — channel dictionaries, calibration
   tables, default values — that does not require hardware
   communication. ``self.comm`` is available, but ``test_connection``
   has not yet run, so do not assume the hardware responds. Returns
   nothing.

``test_connection(self) -> bool``
   Verifies that the hardware responds. Blackchirp calls this when the
   user clicks **Test** in the Hardware menu, when an experiment
   starts, and after a profile change. Return ``True`` if
   communication succeeds, ``False`` otherwise. May be called multiple
   times during a session.

``prepare_for_experiment(self, config: dict) -> bool``
   Called once at the start of every experiment, before
   :meth:`begin_acquisition`. The ``config`` dictionary is the
   experiment configuration relevant to this hardware type — chirp and
   RF parameters for an AWG, clock-role assignments for a clock,
   digitizer settings for a scope, and so on. The exact shape is
   documented in the corresponding template's docstring. Return
   ``True`` to proceed, ``False`` to abort the experiment with an
   error.

   For some hardware types (IOBoard, FtmwDigitizer, LifDigitizer), the C++
   trampoline calls a separate ``configure(config: dict) -> dict``
   method instead, with a ``{"success": bool, "config": dict}`` return
   value that lets the driver report values clamped or substituted by
   the hardware. See :doc:`per_type_capabilities` for which method
   applies to each type.

``begin_acquisition(self)``
   Called when acquisition starts (after
   :meth:`prepare_for_experiment` succeeds and all hardware is ready).
   Use it to enable outputs, arm digitizers, or start a background
   acquisition thread. Returns nothing.

``end_acquisition(self)``
   Called when acquisition stops (normally, or because the experiment
   was aborted). Use it to disable outputs, stop the acquisition
   thread, and return the hardware to an idle state. Returns nothing.

``sleep(self, sleeping: bool)``
   Called when Blackchirp's global sleep state changes. ``sleeping``
   is ``True`` when entering sleep, ``False`` when waking. Use it to
   park the hardware in a safe low-power state and to restore it on
   wake. Returns nothing.

``read_settings(self)``
   Called when the user accepts changes in the Hardware Settings
   dialog. The subprocess is **not** restarted, so any cached values
   in the driver remain in place; re-read whichever settings the
   driver depends on through ``self.settings.get``. Returns nothing.

``read_aux_data(self) -> dict[str, float]``
   Returns auxiliary data displayed in rolling-data plots and recorded
   to the experiment's aux-data file. Each key becomes a plot trace
   (and a column in the aux-data CSV). Return an empty dict if the
   driver has no auxiliary readings at this time.

``read_validation_data(self) -> dict[str, float]``
   Returns values that the experiment validation rules can compare
   against thresholds (for example, an interlock voltage that must
   stay above 4.5 V). The dict shape is identical to
   :meth:`read_aux_data`; the difference is purely how Blackchirp uses
   the values.

In addition to these common methods, each hardware type defines its
own granular methods (``hw_read_flow``, ``hw_set_frequency``,
``read_analog_channels``, ``configure``, ``set_address``, and so on).
:doc:`per_type_capabilities` lists them for every trampoline.

.. _python-hardware-defaults:

Defaults for Unimplemented Methods
----------------------------------

A driver that omits an optional method does not break the IPC
protocol: the host script returns a safe default. The default depends
on which method was called.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Method
     - Default return
     - Effect
   * - ``test_connection``
     - ``True``
     - Connection assumed good (use this only for stub drivers).
   * - | ``read_aux_data``
       | ``read_validation_data``
     - ``{}``
     - No aux or validation data is recorded.
   * - ``prepare_for_experiment``
     - ``True``
     - Experiment proceeds without driver-specific configuration.
   * - | ``initialize``
       | ``begin_acquisition``
       | ``end_acquisition``
       | ``sleep``
       | ``read_settings``
     - ``None``
     - The call is a no-op.
   * - | Hardware-specific methods
       | (``hw_*``, ``configure``,
       | ``read_analog_channels``,
       | etc.)
     - ``None``
     - Effectively unimplemented; the C++ trampoline treats the
       missing return as an error or empty data, depending on the
       method.

The required methods for each hardware type are the ones whose
``None`` default would leave the trampoline unable to function. Those
are called out per type in :doc:`per_type_capabilities`.

.. _python-hardware-error-handling:

Error Handling
--------------

When a method raises an exception, the host script catches it,
forwards the exception type, message, and traceback to Blackchirp,
and continues running. The traceback appears in the log panel
(level **Error**), and the C++ trampoline treats the call as failed.
The subprocess does not exit — subsequent calls dispatch normally.

This means a driver can let exceptions propagate from
``self.comm.query`` (for example, ``ConnectionError`` when the device
times out) without special handling. For methods that must return a
sentinel rather than raise — for example, ``hw_read_flow`` returning
``-1.0`` on a transient read failure — catch the exception explicitly
and return the documented sentinel value.

Method calls are dispatched serially on the subprocess's main thread.
A long-running method blocks subsequent calls. For acquisition loops,
use a background thread (see :ref:`python-hardware-digi-proxy`).

.. _python-hardware-driver-skeleton:

Driver Skeleton
---------------

The following skeleton applies to most hardware types. Replace the
class name and add hardware-specific methods from
:doc:`per_type_capabilities`.

.. code-block:: python

   class MyDriver:
       def initialize(self):
           self.log.log("driver initialized")
           self._state = {}

       def test_connection(self):
           response = self.comm.query("*IDN?\n")
           ok = bool(response.strip())
           if not ok:
               self.log.error("no response to *IDN?")
           return ok

       def prepare_for_experiment(self, config):
           # configure hardware from config dict
           return True

       def begin_acquisition(self):
           pass

       def end_acquisition(self):
           pass

       def sleep(self, sleeping):
           pass

       def read_settings(self):
           pass

       def read_aux_data(self):
           return {}

The :doc:`overview` page covers how Blackchirp finds, launches, and
reloads the script. The :doc:`hot_reload` page covers the
**Reload Script** and **Open in Editor** controls used while a driver
is being developed.
