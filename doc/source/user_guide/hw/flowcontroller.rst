Flow Controller
===============

* Overview_
* Settings_
* Implementations_

Overview
--------

A FlowController is a device that oversees the operation of a series of mass flow controllers with an optional pressure sensor. They typically support two modes of operation: controlling gases at fixed flow rates, or maintaining a constant ratio of flow rates while regulating the total flow rate to sustain a target pressure read by a pressure sensor. Blackchirp records the names and setpoints of all enabled gases at the start of the experiment, and reads the flow rates and/or pressure as Auxiliary data.

Settings
--------

* ``channels`` (menu): Settings for the individual flow channels.
  - ``decimals`` (int): Number of decimal places to display for channel N.
  - ``max`` (double): Maximum flow rate for channel N.
  - ``units`` (string): Units to be displayed on UI and recorded in files for channel N.
* ``intervalMs`` (int): The polling interval, in ms. Enabled channels are polled sequentually, one per polling interval.
* ``pressureDecimals`` (int): Number of decimal places to be displayed for the pressure.
* ``pressureMax`` (double): Maximum pressure for pressure sensor.
* ``pressureUnits`` (string): Units to be displayed on UI and recorded in files for pressure.


Implementations
---------------

Virtual (virtual)
.................

A dummy 4-channel implementation.

MKS 647C (mks647c)
..................

A 4-channel device with a built-in PID controller for maintaining a target pressure. The communication protocol is RS232.

MKS 946 (mks946)
................

A newer model from MKS which is a modular platform supporting different combinations of mass flow controllers and pressure sensors. The Blackchirp implementation assumes 4 flow channels and 1 pressure channel. The communication protocol is RS232.
