LIF Laser
=========

* Overview_
* Settings_
* Implementations_

Overview
--------

The LIF Laser is the laser whose frequency is stepped during an LIF acquisition. It is assumed to be a pulsed, tunable laser with the ability to toggle on and off a flashlamp (though this behavior is optional).

Settings
--------

- ``decimals`` (int): Number of decimal places to display on UI.
- ``maxPos`` (double): Maximum laser position setting. Interpretation of the units is up to the implementation; may be wavelength, frequency, stepper motor position, etc.
- ``maxPos`` (double): Minimum laser position setting.
- ``units`` (string): Units for laser position. Only used for display on UI and in text output files.


Implementations
---------------

Virtual (virtual)
.................

A dummy implementation.

Opolette (opolette)
...................

The OPOTek Opolette is a tunable pulsed OPO laser. Operation of the laser requires a Windows-based SDK. Blackchirp communicates using a custom-written TCP server which runs on a Windows machine, and communicates with Blackchirp through a TCP socket connection. At the time of writing, the socket server code is not in the Blackchirp repository, though it can be added in the unlikely event anyone else uses the same laser.
