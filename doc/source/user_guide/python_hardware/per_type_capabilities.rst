.. index::
   single: Python hardware; per-type capabilities
   single: Python hardware; trampolines
   single: Python hardware; state-management patterns
   single: AwgDriver
   single: ClockDriver
   single: FlowControllerDriver
   single: TemperatureControllerDriver
   single: PressureControllerDriver
   single: IOBoardDriver
   single: PulseGeneratorDriver
   single: FtmwDigitizerDriver
   single: LifDigitizerDriver
   single: GpibControllerDriver
   single: LifLaserDriver

.. _python-hardware-per-type-capabilities:

Per-Type Capabilities
=====================

Each Python hardware trampoline expects a different set of methods
depending on how the C++ base class manages hardware state. This page
catalogs every trampoline that ships with Blackchirp, the default
class name its template defines, the state-management pattern it uses,
and the methods a driver must implement for the trampoline to operate.

The lifecycle methods listed in :doc:`writing_a_driver`
(:meth:`initialize`, :meth:`test_connection`,
:meth:`prepare_for_experiment`, :meth:`begin_acquisition`,
:meth:`end_acquisition`, :meth:`sleep`, :meth:`read_settings`,
:meth:`read_aux_data`, :meth:`read_validation_data`) are common to
every type and are not repeated below. The methods listed here are the
hardware-specific ones the trampoline dispatches in addition to the
common lifecycle.

.. _python-hardware-state-patterns:

State-Management Patterns
-------------------------

Blackchirp's hardware base classes fall into one of three patterns,
which determine the shape of the Python interface:

**Pattern A — Bulk configure.**
   The base class inherits from a complex config object (a digitizer
   config with channel maps, trigger settings, sample rates, and so
   on). Before each experiment, the trampoline serializes the entire
   config into a dict and calls ``configure(...)`` on the driver. The
   driver applies the settings, reads back actual values, and returns
   ``{"success": bool, "config": dict}``. The returned ``config``
   dict is deserialized back into the C++ side, so any clamped or
   substituted values are preserved.

**Pattern B — Granular methods.**
   The base class owns a config object and updates it through
   individual setter and getter methods. Each method delegates to a
   ``hw_*`` virtual that the driver implements. The driver only ever
   sees one value at a time; the base class decides when and in what
   order to call the methods. Polling, validation, and signal
   emission are handled C++-side.

**Pattern C — Stateless / experiment-data pass-through.**
   The base class has no internal config to manage between calls.
   Each experiment delivers its data — chirp segments and markers,
   clock-frequency assignments — through
   :meth:`prepare_for_experiment`, and the driver programs the
   hardware with that data. There are no granular setters for the
   driver to implement.

Digitizer trampolines (FtmwDigitizer, LifDigitizer) combine Pattern A
configuration with a **push** acquisition model: the driver runs an
acquisition loop on its own thread and pushes waveforms back through
``self.digi.emit_shot``. See :ref:`python-hardware-digi-proxy`.

.. _python-hardware-trampoline-overview:

Trampoline Overview
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 15 35

   * - Trampoline
     - Default class
     - Pattern
     - Driver entry points
   * - PythonFtmwDigitizer
     - ``FtmwDigitizerDriver``
     - A + push
     - :meth:`configure`,
       :meth:`begin_acquisition`,
       :meth:`end_acquisition`,
       ``self.digi.emit_shot``
   * - PythonIOBoard
     - ``IOBoardDriver``
     - A
     - :meth:`configure`,
       :meth:`read_analog_channels`,
       :meth:`read_digital_channels`
   * - PythonLifDigitizer
     - ``LifDigitizerDriver``
     - A + push
     - :meth:`configure`,
       :meth:`begin_acquisition`,
       :meth:`end_acquisition`,
       ``self.digi.emit_shot``
   * - PythonFlowController
     - ``FlowControllerDriver``
     - B
     - 8 ``hw_*`` flow / pressure methods
   * - PythonGpibController
     - ``GpibControllerDriver``
     - B
     - :meth:`read_address`,
       :meth:`set_address`
   * - PythonLifLaser
     - ``LifLaserDriver``
     - B
     - :meth:`read_pos`, :meth:`set_pos`,
       :meth:`read_fl`, :meth:`set_fl`
   * - PythonPressureController
     - ``PressureControllerDriver``
     - B
     - 7 ``hw_*`` pressure / valve methods
   * - PythonPulseGenerator
     - ``PulseGeneratorDriver``
     - B
     - 22 ``set_*`` / ``read_*`` channel and global methods
   * - PythonTemperatureController
     - ``TemperatureControllerDriver``
     - B
     - :meth:`hw_read_temperature`
   * - PythonAwg
     - ``AwgDriver``
     - C
     - :meth:`prepare_for_experiment`
   * - PythonClock
     - ``ClockDriver``
     - C
     - :meth:`hw_set_frequency`,
       :meth:`hw_read_frequency`

The remainder of this page lists the hardware-specific entry points
for each trampoline along with their signatures and return-type
expectations. Method docstrings in the corresponding template script
document the full argument shape (especially the ``config`` dict
contents for Pattern A trampolines).

.. _python-hardware-pattern-a:

Pattern A — Bulk Configure (and Push)
-------------------------------------

For Pattern A trampolines, the C++ trampoline calls ``configure(...)``
in place of ``prepare_for_experiment``. The full configuration is
delivered as keyword arguments; the driver applies the settings,
reads back actual values, and returns:

.. code-block:: python

   {"success": True, "config": validated_config_dict}

Setting ``success`` to ``False`` aborts the experiment. The
``config`` dict is deserialized back into the C++ side, so any values
the hardware clamped or substituted are reflected in subsequent
operations. Omitted keys leave the C++ value unchanged.

The shape of the input keywords and the validated dict is the same
between FtmwDigitizer and LifDigitizer; LifDigitizer adds LIF-specific keys
(``lif_channel``, ``ref_channel``, ``ref_enabled``,
``channel_order``). Refer to the corresponding template script for
the complete keyword list.

.. _python-hardware-ioboard:

IO Board (``IOBoardDriver``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def configure(self, analog_channels=None, digital_channels=None,
                 trigger=None, **kwargs) -> dict: ...
   def read_analog_channels(self, channels=None) -> dict[int, float]: ...
   def read_digital_channels(self, channels=None) -> dict[int, bool]: ...

``read_analog_channels`` is called periodically (via
``read_aux_data``); ``read_digital_channels`` is called for validation.
Both receive a ``channels`` list of enabled channel indices and must
return a dict keyed by those indices. Returning data for
channels that are not enabled is harmless but wasteful; omitting an
enabled channel leaves a gap in the aux-data row for that polling
tick.

.. _python-hardware-ftmwdigitizer:

FTMW Digitizer (``FtmwDigitizerDriver``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def configure(self, analog_channels=None, digital_channels=None,
                 trigger=None, sample_rate=0.0, record_length=1000,
                 bytes_per_point=1, byte_order=0,
                 block_average=False, num_averages=1,
                 multi_record=False, num_records=1,
                 fid_channel=0, **kwargs) -> dict: ...

``begin_acquisition`` should start the acquisition loop; the
recommended pattern is a daemon thread that calls
``self.digi.emit_shot(raw_bytes)`` for each waveform.
``end_acquisition`` signals the thread to stop and joins it.

The byte layout of the data passed to ``emit_shot`` must match the
applied configuration: ``record_length`` × ``bytes_per_point`` ×
``num_records`` bytes for multi-record acquisitions, signed integers,
and the configured ``byte_order``. ``fid_channel`` selects the analog
channel that carries the FID; multi-channel digitizers may interleave
all enabled channels in the byte stream depending on the model.

Pre-accumulated data — multiple shots already averaged on the
hardware — is reported through the ``shots`` argument:

.. code-block:: python

   self.digi.emit_shot(raw_bytes, shots=num_averages)

``readWaveform`` is **not** dispatched. The acquisition is
push-driven from Python.

.. _python-hardware-lifdigitizer:

LIF Digitizer (``LifDigitizerDriver``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def configure(self, analog_channels=None, digital_channels=None,
                 trigger=None, sample_rate=0.0, record_length=1000,
                 bytes_per_point=1, byte_order=0,
                 block_average=False, num_averages=1,
                 multi_record=False, num_records=1,
                 lif_channel=1, ref_channel=2, ref_enabled=False,
                 channel_order=0, **kwargs) -> dict: ...

The acquisition pattern is identical to the FTMW Digitizer: start a
daemon thread in ``begin_acquisition``, push waveforms with
``self.digi.emit_shot``, and stop the thread in
``end_acquisition``. The byte layout depends on ``channel_order``:
sequential layout writes the LIF record followed by the reference
record (when enabled); interleaved layout writes alternating LIF and
reference samples sample-by-sample. Each sample is signed
(``int8`` for ``bytes_per_point=1``, ``int16`` for
``bytes_per_point=2``); ``byte_order`` selects little-endian (``0``)
or big-endian (``1``).

.. _python-hardware-pattern-b:

Pattern B — Granular Methods
----------------------------

.. _python-hardware-flowcontroller:

Flow Controller (``FlowControllerDriver``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The base class polls each flow channel and the chamber pressure on a
timer; it also calls the setters when the user changes a value or an
experiment applies a flow configuration.

.. code-block:: python

   def hw_read_flow(self, channel: int) -> float: ...
   def hw_read_flow_setpoint(self, channel: int) -> float: ...
   def hw_set_flow_setpoint(self, channel: int, value: float) -> None: ...
   def hw_read_pressure(self) -> float: ...
   def hw_read_pressure_setpoint(self) -> float: ...
   def hw_set_pressure_setpoint(self, value: float) -> None: ...
   def hw_read_pressure_control_mode(self) -> int: ...
   def hw_set_pressure_control_mode(self, enabled: bool) -> None: ...

Read methods return ``-1.0`` on error (``-1`` for
``hw_read_pressure_control_mode``). The number of flow channels comes
from the ``flowChannels`` setting.

.. _python-hardware-tempcontroller:

Temperature Controller (``TemperatureControllerDriver``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single granular read method, called once per enabled channel by the
poll timer:

.. code-block:: python

   def hw_read_temperature(self, channel: int) -> float: ...

Return ``float('nan')`` to signal a transient error; the base class
discards the reading and continues polling. The number of channels
comes from the ``numChannels`` setting.

.. _python-hardware-pressurecontroller:

Pressure Controller (``PressureControllerDriver``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def hw_read_pressure(self) -> float: ...
   def hw_read_pressure_setpoint(self) -> float: ...
   def hw_set_pressure_setpoint(self, value: float) -> float: ...
   def hw_read_pressure_control_mode(self) -> int: ...
   def hw_set_pressure_control_mode(self, enabled: bool) -> None: ...
   def hw_open_gate_valve(self) -> None: ...
   def hw_close_gate_valve(self) -> None: ...

Pressure reads return ``math.nan`` on error;
``hw_read_pressure_control_mode`` returns ``-1`` on error.
``hw_set_pressure_setpoint`` returns the actual setpoint applied (or
``math.nan`` on failure) so the GUI displays the value the hardware
accepted. When the profile is configured **read-only**, the four
setter methods (``hw_set_pressure_setpoint``,
``hw_set_pressure_control_mode``, ``hw_open_gate_valve``,
``hw_close_gate_valve``) are not dispatched.

.. _python-hardware-pulsegenerator:

Pulse Generator (``PulseGeneratorDriver``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pulse generator has the largest method surface — eight setter and
eight getter methods per channel, plus six global methods — because
each pulse-generator parameter (width, delay, active level, sync
source, duty-cycle) is exposed as an independent UI control.

Per-channel setters and getters:

.. code-block:: python

   def set_ch_width(self, channel: int, width: float) -> bool: ...
   def set_ch_delay(self, channel: int, delay: float) -> bool: ...
   def set_ch_active_level(self, channel: int, level: int) -> bool: ...
   def set_ch_enabled(self, channel: int, enabled: bool) -> bool: ...
   def set_ch_sync_ch(self, channel: int, sync_ch: int) -> bool: ...
   def set_ch_mode(self, channel: int, mode: int) -> bool: ...
   def set_ch_duty_on(self, channel: int, pulses: int) -> bool: ...
   def set_ch_duty_off(self, channel: int, pulses: int) -> bool: ...

   def read_ch_width(self, channel: int) -> float: ...
   def read_ch_delay(self, channel: int) -> float: ...
   def read_ch_active_level(self, channel: int) -> int: ...
   def read_ch_enabled(self, channel: int) -> bool: ...
   def read_ch_sync_ch(self, channel: int) -> int: ...
   def read_ch_mode(self, channel: int) -> int: ...
   def read_ch_duty_on(self, channel: int) -> int: ...
   def read_ch_duty_off(self, channel: int) -> int: ...

Global setters and getters:

.. code-block:: python

   def set_hw_rep_rate(self, rep_rate: float) -> bool: ...
   def set_hw_pulse_mode(self, mode: int) -> bool: ...
   def set_hw_pulse_enabled(self, enabled: bool) -> bool: ...

   def read_hw_rep_rate(self) -> float: ...
   def read_hw_pulse_mode(self) -> int: ...
   def read_hw_pulse_enabled(self) -> bool: ...

Enum integers used in these calls:

- ``active_level``: ``0`` = ActiveLow, ``1`` = ActiveHigh.
- ``mode`` (per channel): ``0`` = Normal, ``1`` = DutyCycle.
- ``pulse_mode`` (global): ``0`` = Continuous,
  ``1`` = Triggered_Rising, ``2`` = Triggered_Falling.

The C++ base class declares ``sleep`` ``final`` and calls
``set_hw_pulse_enabled(False)`` on entry. A driver that simply
disables outputs on sleep does not need its own :meth:`sleep` method.

The number of channels comes from the ``numChannels`` setting. Methods
that depend on hardware capability flags
(``set_ch_enabled``, ``set_ch_sync_ch``, ``set_ch_mode``,
``set_ch_duty_on``, ``set_ch_duty_off``, ``set_hw_pulse_mode``) are
only dispatched when the corresponding capability is set in the
profile configuration.

.. _python-hardware-gpibcontroller:

GPIB Controller (``GpibControllerDriver``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The C++ base class handles bus arbitration and address bookkeeping;
the driver supplies the two operations that vary by hardware (Prologix,
NI USB-GPIB, etc.):

.. code-block:: python

   def read_address(self) -> bool: ...
   def set_address(self, address: int) -> bool: ...

``read_address`` queries the controller for the active
talker/listener address and returns ``True`` on success. If the
hardware does not support reading the address back, return ``True``
without modifying state. ``set_address`` is called before each
``writeCmd``, ``writeBinary``, or ``queryCmd`` whose target address
differs from the cached current address; ``address`` is the GPIB
primary address (0–30) of the device to select.

.. _python-hardware-liflaser:

LIF Laser (``LifLaserDriver``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The base class clamps position requests to the configured
``[minPos, maxPos]`` range and emits GUI-update signals when values
change. The driver implements four granular methods:

.. code-block:: python

   def read_pos(self) -> float: ...
   def set_pos(self, pos: float) -> None: ...
   def read_fl(self) -> bool: ...
   def set_fl(self, enabled: bool) -> bool: ...

``read_pos`` returns the current laser position (in the configured
units, default nm) or ``-1.0`` on error. ``set_pos`` is invoked by the
GUI; the base class calls ``read_pos`` afterwards to confirm the
result. ``set_fl`` returns ``True`` if the flashlamp command was
accepted (not the new state — the base class re-reads with
``read_fl``). Position units, range, and the auto-disable behavior at
end-of-experiment are configured through the profile's settings.

.. _python-hardware-pattern-c:

Pattern C — Stateless
---------------------

.. _python-hardware-awg:

AWG (``AwgDriver``)
~~~~~~~~~~~~~~~~~~~

Receives the full chirp configuration through
:meth:`prepare_for_experiment`. The ``config`` dict contains a
``chirp`` sub-dict with chirp segments, markers, sample rate, and
repetition parameters, plus an ``rf_config`` sub-dict with AWG and
chirp multipliers, sideband choices, and clock-role assignments. No
``hw_*`` methods are dispatched.

The AWG template provides three optional NumPy helper functions for
memory-based AWGs:

.. code-block:: python

   times_us, amplitudes = AwgDriver._compute_waveform(config['chirp'])
   indices, markers     = AwgDriver._compute_markers(config['chirp'])
   packed               = AwgDriver._compute_markers_packed(config['chirp'])

These mirror the C++ ``ChirpConfig::getChirpMicroseconds``,
``getMarkerData``, and ``getPackedMarkerData`` methods. DDS-style AWGs
typically program the segment parameters directly and do not need
these helpers.

.. _python-hardware-clock:

Clock (``ClockDriver``)
~~~~~~~~~~~~~~~~~~~~~~~

Frequency assignments per role are applied through two granular
methods that the C++ base class drives:

.. code-block:: python

   def hw_set_frequency(self, freq_mhz: float, output: int = 0) -> bool: ...
   def hw_read_frequency(self, output: int = 0) -> float: ...

``freq_mhz`` is the **raw hardware frequency** — the requested
frequency already divided by the output's external multiplication
factor. ``output`` is a zero-based hardware output index. The base
class performs range checking and applies the multiplier;
``hw_read_frequency`` must return the raw hardware reading, also
without applying the multiplier. Return ``-1.0`` from
``hw_read_frequency`` to signal an error; ``readAll()`` stops
iterating outputs at the first negative return.
