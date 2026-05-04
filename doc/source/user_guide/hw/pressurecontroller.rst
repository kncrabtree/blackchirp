Pressure Controller
===================

* Overview_
* Settings_
* Drivers_

Overview
--------

A PressureController monitors and optionally controls a pressure by means of a PID loop, and may also operate a gate valve.

.. note::
   The PressureController interface was implemented with a narrow scope in mind: regulating a vacuum chamber pressure for CRESU experiments on a chamber with a programmable pendulum valve. A natural extension is to support multiple channels and valves, or to split the role into separate controller and monitor classes so that thermocouple or ion gauges can be exposed without a control loop. Contributions are welcome.

Settings
--------

Most settings are exposed in the :doc:`hardware dialog </user_guide/hwdialog>` with inline labels and tooltips supplied by the settings registry. A few items are worth highlighting:

* ``min`` and ``max`` set the display range for the pressure readout and the bounds enforced by the setpoint spin box; ``decimal`` controls the number of decimal places.
* ``units`` (Pressure Units) is registered as an Important setting because it must match the unit system reported by the device. The base-class default is ``Torr``.
* ``readInterval`` is the polling period in milliseconds. Faster intervals smooth the rolling-data trace at the cost of more serial traffic.
* ``hasValve`` controls whether the dialog and gas-control widget expose explicit open/close gate-valve actions. Read-only monitor configurations should leave this off.

Pressure values are reported as both rolling data and auxiliary data, so the channel can be used as a validation setting to terminate an acquisition that drifts outside an acceptable window.

Drivers
-------

Virtual
.................

A dummy driver that returns a pressure equal to the setpoint. Useful for offline UI testing.

Intellisys IQplus
..............................

The `Intellisys IQplus <https://www.idealvac.com/files/manuals/08-Nor-CalProductsDownstreamPressureControlCatalog2018.pdf>`_ is an adaptive pressure controller that combines a pressure sensor, PID loop, and pendulum valve to regulate the pressure in a process chamber. The communication protocol is RS232. The driver overrides the base-class default for ``min`` to ``0.0`` to match the device's downstream-control range.
