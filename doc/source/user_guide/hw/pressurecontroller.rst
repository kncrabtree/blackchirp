Pressure Controller
===================

* Overview_
* Settings_
* Implementations_

Overview
--------

A PressureController is a device which monitors and/or controls a pressure by means of a PID control, and which optionally controls a valve.

.. note::
   The PressureController was implemented with a limited scope of functionality in mind: regulating a vacuum chamber pressure for CRESU experiments on a chambber with a programmable pendulum valve. In the future, it would be better to expand this implementation to support multiple channels and valves, or to split the functionality to create one controller class and one monitor class. This would enable reading, e.g., pressures from TC or ion gauges.

Settings
--------

- ``decimal`` (int): Number of decimal places to display on UI.
- ``hasValve`` (bool): If true, adds controls for opening/closing a valve.
- ``intervalMs`` (int): Time between pressure readings, in ms.
- ``max`` (double): Maximum pressure setting.
- ``min`` (double): Minimum pressure setting.
- ``units`` (string): Units for pressure setting. Displayed on UI


Implementations
---------------

Virtual (virtual)
.................

A dummy implementation which returns a pressure equal to the setpoint.

Intellisys IQplus (intellisys)
..............................

The `Intellisys IQplus <https://www.idealvac.com/files/manuals/08-Nor-CalProductsDownstreamPressureControlCatalog2018.pdf>`_ is an adaptice pressure controller that uses a pressure sensor, PID controller, and pendulum valve to regulate the pressure in a process chamber.

